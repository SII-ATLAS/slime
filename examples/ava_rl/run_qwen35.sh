#!/usr/bin/env bash
# U: same recipe as T r2; SC off, weight publication interval 64.
set -euo pipefail
: "${CHECKPOINT_DIR:?Set unique checkpoint path on dmodel-large-file}"
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
: "${RUN:?Set a fresh absolute run directory}"
: "${MODEL_DIR:?Set Qwen3.5-35B-A3B-Base directory}"
: "${TRAIN_DATA:?Set verified Ava Math/Terminal mixture}"

RUN_NAME="${RUN_NAME:-ava-math-terminal}"
source "$ROOT/scripts/models/qwen3.5-35B-A3B.sh"

ARGS=(
  --seq-length 131072
  --max-position-embeddings 131072
  --train-backend megatron
  --qwen-gdn-implementation megatron
  --qwen-gdn-backend fla
  --actor-num-nodes 4
  --no-offload-train
  --no-offload-rollout
  --actor-num-gpus-per-node 8
  --num-gpus-per-node 8
  "${MODEL_ARGS[@]}"
  --hf-checkpoint "${MODEL_DIR}"
  --load "${LOAD_DIR:-$MODEL_DIR}"
  --prompt-data "${TRAIN_DATA}"
  --input-key prompt
  --label-key label
  --rollout-shuffle
  --rm-type math
  --custom-generate-function-path examples.ava_rl.generate.generate
  --custom-rollout-log-function-path examples.ava_rl.metrics.log_rollout_data
  --advantage-estimator grpo
  --pg-loss-type reinforce
  --disable-grpo-std-normalization
  --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
  --flush-cache-interval 8
  --sglang-server-concurrency 2
  --fully-async-group-prefetch-factor 3
  --use-distributed-post
  --save "$CHECKPOINT_DIR"
  --save-interval 20
  --num-rollout "${NUM_ROLLOUT:-1000}"
  --rollout-batch-size 32
  --n-samples-per-prompt 8
  --global-batch-size 256
  --rollout-stop-token-ids 248044 248046
  --rollout-max-response-len 129024
  --rollout-max-context-len 131072
  --rollout-temperature 1.0
  --rollout-top-p 1.0
  --rollout-top-k -1
  --rollout-data-transport object-store
  --balance-data
  --use-tis
  --custom-tis-function-path slime.backends.megatron_utils.loss.icepop_function
  --tis-clip-low 0.5
  --tis-clip 2
  --kl-loss-coef 0.0
  --kl-loss-type low_var_kl
  --entropy-coef 0.0
  --skip-eval-before-train
  --n-samples-per-eval-prompt 1
  --tensor-model-parallel-size 8
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 2
  --expert-model-parallel-size 8
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu 32768
  --log-probs-max-tokens-per-gpu 32768
  --log-probs-chunk-size 1024
  --calculate-per-token-loss
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --use-distributed-optimizer
  --overlap-grad-reduce
  --use-precision-aware-optimizer
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
  --rollout-num-gpus-per-engine 2
  --sglang-dtype bfloat16
  --sglang-mem-fraction-static 0.65
  --sglang-context-length 262144
  --sglang-max-running-requests 32
  --sglang-cuda-graph-max-bs 32
  --sglang-mamba-ssm-dtype float32
  --use-tensorboard
  --tb-project-name slime-qwen35-base
  --tb-experiment-name "${RUN_NAME}"
  --seed 1234
  --rollout-num-gpus 96
  --sglang-reasoning-parser qwen3
  --sglang-ep-size 2
  --sglang-attention-backend fa3
  --sglang-enable-mfu-metrics
  --sglang-enable-metrics
  --eval-max-context-len 131072
)
case "${USE_R3:-1}" in
  1) ARGS+=(--use-rollout-routing-replay) ;;
  0) ;;
  *) echo 'USE_R3 must be 0 or 1.' >&2; exit 2 ;;
esac
if [[ "${WEIGHT_INTERVAL:-1}" != 1 ]]; then
  ARGS+=(--update-weights-interval "$WEIGHT_INTERVAL")
fi
if [[ -n "${CKPT_STEP:-}" ]]; then
  ARGS+=(--ckpt-step "$CKPT_STEP" --start-rollout-id "$((CKPT_STEP + 1))")
else
  ARGS+=(--start-rollout-id 0)
fi
if [[ -n "${EVAL_CONFIG:-}" ]]; then ARGS+=(--eval-config "$EVAL_CONFIG"); fi
if [[ "${1:-}" == --dry-run ]]; then
  printf '%q ' python3 train.py "${ARGS[@]}"; printf '\n'; exit 0
fi
[[ $# == 0 ]] || { echo 'Only --dry-run is supported; set the documented configuration variables.' >&2; exit 2; }
[[ "$ROOT" = /* && "$RUN" = /* ]] || { echo 'ROOT and RUN must be absolute.' >&2; exit 2; }
[[ -f "$ROOT/train.py" && -f "$MODEL_DIR/config.json" && -f "$TRAIN_DATA" ]]
mkdir -p "$RUN/logs" "$RUN/tensorboard"
mkdir "$RUN/driver-started"
cd "$ROOT"
export TENSORBOARD_DIR="$RUN/tensorboard"
printf '%q ' python3 train.py "${ARGS[@]}" > "$RUN/command.sh"
printf '\n' >> "$RUN/command.sh"
RUNTIME_ENV=$(python3 - <<'PYENV'
import json,os
names=['CUDA_DEVICE_MAX_CONNECTIONS','TENSORBOARD_DIR','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_HOME','TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR','OMP_NUM_THREADS','no_proxy','NO_PROXY','PYTHONDONTWRITEBYTECODE']
names += [k for k in os.environ if k.startswith('AVA_')]
env={k:os.environ[k] for k in names if k in os.environ}
env['TENSORBOARD_DIR']=env['TENSORBOARD_DIR']+'/train'
print(json.dumps({'env_vars':env}))
PYENV
)
# Run once on the head of an already initialized 16-node / 128-GPU Ray cluster.
# Before this step, validate args/topology in the chosen image as described in the reference.
ray job submit --address http://127.0.0.1:8265 --working-dir "$ROOT" --runtime-env-json "$RUNTIME_ENV" -- python3 train.py "${ARGS[@]}"
