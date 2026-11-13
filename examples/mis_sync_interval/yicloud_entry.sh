#!/usr/bin/env bash
# YiCloud's native multi-node environment starts a standard Ray cluster.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
: "${RUN:?Set an independent shared output directory}"
: "${MODEL_DIR:?Set Base model assets}"
: "${TRAIN_DATA:?Set IF70/DAPO30 JSONL}"
: "${CHECKPOINT_DIR:?Set an independent shared checkpoint directory}"
[[ "$NODE_COUNT" == 16 && "$PROC_PER_NODE" == 8 ]]
NODE_DIR="$RUN/node-$NODE_RANK"
mkdir -p "$NODE_DIR" "$RUN/logs" "$RUN/tensorboard" "$CHECKPOINT_DIR"
exec > >(tee -a "$NODE_DIR/entry.log") 2>&1
cd "$ROOT"
# The immutable image contains this exact Git archive; no install or patch here.
cat "$ROOT/COMMIT"
python3 -c 'import slime; print("slime package:", slime.__file__)'
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RAY_USAGE_STATS_ENABLED=0
export TENSORBOARD_DIR="$RUN/tensorboard"
export HF_HOME="/tmp/coop-$NODE_RANK/hf"
export TORCH_EXTENSIONS_DIR="/tmp/coop-$NODE_RANK/extensions"
export TRITON_CACHE_DIR="/tmp/coop-$NODE_RANK/triton"
DRIVER_PID=
cleanup() {
  local status=$?
  trap - EXIT TERM INT
  if [[ -n "$DRIVER_PID" ]] && kill -0 "$DRIVER_PID" 2>/dev/null; then
    kill -TERM "$DRIVER_PID" 2>/dev/null || true
  fi
  printf '%s\n' "$status" > "$NODE_DIR/exit-code.txt"
  if [[ "$NODE_RANK" == 0 ]]; then
    printf '%s\n' "$status" > "$RUN/driver-exit-code.txt"
  elif (( status != 0 )); then
    printf '%s\n' "$status" > "$RUN/node-failed-$NODE_RANK"
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 124' TERM INT
NODE_IP=$(python3 -c 'import ray; print(ray.util.get_node_ip_address())')
HEAD_IP=$(python3 -c 'import socket,os; print(socket.gethostbyname(os.environ["MASTER_ADDR"]))')
export no_proxy="${no_proxy:+$no_proxy,}localhost,127.0.0.1,$NODE_IP,$HEAD_IP"
export NO_PROXY="$no_proxy"
if [[ "$NODE_RANK" == 0 ]]; then
  timeout 180s bash "$ROOT/examples/mis_sync_interval/run_qwen35.sh" --check-args > "$RUN/logs/resolved-arguments.log" 2>&1
  timeout 180s ray start --head --node-ip-address "$NODE_IP" --port 6379 \
    --num-gpus 8 --num-cpus 120 --object-store-memory 893353197568 \
    --system-config '{"plasma_store_usage_trigger_gc_threshold":0.9}' \
    --dashboard-host 127.0.0.1 --dashboard-port 8265 --disable-usage-stats \
    --temp-dir "/tmp/coop-$NODE_RANK" > "$NODE_DIR/ray-start.log" 2>&1
  python3 - <<'PY'
import ray,time
ray.init(address='auto')
deadline=time.monotonic()+3600
while time.monotonic()<deadline:
    nodes=[n for n in ray.nodes() if n['Alive']]
    if len(nodes)==16 and sum(n['Resources'].get('GPU',0) for n in nodes)==128:
        break
    time.sleep(5)
else:
    raise RuntimeError('Expected16 live Ray nodes/128GPUs within3600s')
ray.shutdown()
PY
  bash "$ROOT/examples/mis_sync_interval/run_qwen35.sh" > >(tee -a "$RUN/logs/train.log") 2>&1 &
  DRIVER_PID=$!
  wait "$DRIVER_PID"
else
  python3 - "$HEAD_IP" <<'PY'
import socket,sys,time
deadline=time.monotonic()+3600
while time.monotonic()<deadline:
    try:
        with socket.create_connection((sys.argv[1],6379),timeout=3): break
    except OSError: time.sleep(5)
else: raise RuntimeError('Ray head not reachable within3600s')
PY
  timeout 180s ray start --address "$HEAD_IP:6379" --node-ip-address "$NODE_IP" \
    --num-gpus 8 --num-cpus 120 --object-store-memory 137438953472 \
    --disable-usage-stats > "$NODE_DIR/ray-start.log" 2>&1
  while [[ ! -f "$RUN/driver-exit-code.txt" ]]; do sleep 15; done
  exit "$(cat "$RUN/driver-exit-code.txt")"
fi
