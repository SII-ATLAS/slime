#!/usr/bin/env bash
# M2: M algorithm/data recipe, delayed publication instead of a large ready buffer.
set -euo pipefail
: "${CHECKPOINT_DIR:?Set unique checkpoint path on dmodel-large-file}"
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
: "${RUN:?Set a fresh absolute run directory}"
: "${MODEL_DIR:?Set Qwen3.5-35B-A3B-Base directory}"
: "${TRAIN_DATA:?Set verified 32K IF70/DAPO30 mixed JSONL}"
: "${EVAL_CONFIG:?Set valid eval JSON; this recipe does not run online eval}"
RUN_NAME="${RUN_NAME:-long09-reference}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$ROOT/scripts/models/qwen3.5-35B-A3B.sh"

ARGS=(
   --spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"

   --disable-bias-linear
   --qk-layernorm
   --group-query-attention
   --num-attention-heads 16
   --num-query-groups 2
   --kv-channels 256
   --num-layers 40
   --hidden-size 2048
   --ffn-hidden-size 512
   --use-gated-attention

   --normalization RMSNorm
   --apply-layernorm-1p
   --position-embedding-type rope
   --norm-epsilon 1e-6
   --rotary-percent 0.25
   --swiglu
   --untie-embeddings-and-output-weights
   --vocab-size 248320

   --rotary-base 10000000

   # moe
   --moe-ffn-hidden-size 512
   --moe-shared-expert-intermediate-size 512
   --moe-router-score-function softmax
   --moe-token-dispatcher-type alltoall
   --moe-router-topk 8
   --moe-layer-freq "$MOE_LAYER_FREQ"
   --num-experts 256
   --moe-grouped-gemm
   --moe-token-drop-policy probs
   --moe-router-dtype fp32
   --moe-permute-fusion
   --moe-aux-loss-coeff 0

   # qwen3.5 specific
   --attention-output-gate
   --moe-shared-expert-gate
)

ARGS=(
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
  --load "${MODEL_DIR}"
  --start-rollout-id 0
  --update-weights-interval 128
  --ref-load "${MODEL_DIR}"
  --prompt-data "${TRAIN_DATA}"
  --input-key prompt
  --label-key label
  --apply-chat-template
  --rollout-shuffle
  --rm-type math
  --custom-rm-path examples.mis_sync_interval.if_mix_support.reward
  --custom-rollout-log-function-path examples.mis_sync_interval.if_mix_support.log_sources
  --advantage-estimator grpo
  --pg-loss-type reinforce
  --use-score-centering
  --score-centering-top-k 32
  --disable-grpo-std-normalization
  --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
  --flush-cache-interval 8
  --partial-rollout
  --sglang-server-concurrency 16
  --fully-async-group-prefetch-factor 1
  --use-distributed-post
  --save "$CHECKPOINT_DIR"
  --save-interval 50
  --num-rollout 2000
  --rollout-batch-size 64
  --n-samples-per-prompt 8
  --global-batch-size 512
  --rollout-max-response-len 30720
  --rollout-max-context-len 32768
  --rollout-temperature 1.0
  --rollout-top-p 1.0
  --rollout-top-k -1
  --rollout-data-transport object-store
  --balance-data
  --use-tis
  --custom-tis-function-path slime.backends.megatron_utils.loss.icepop_function
  --tis-clip-low 0.5
  --tis-clip 2
  --use-kl-loss
  --kl-loss-coef 0.0
  --kl-loss-type low_var_kl
  --entropy-coef 0.0
  --eval-config "${EVAL_CONFIG}"
  --skip-eval-before-train
  --n-samples-per-eval-prompt 1
  --tensor-model-parallel-size 8
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 8
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu 49152
  --log-probs-chunk-size 1024
  --calculate-per-token-loss
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --override-opt-param-scheduler
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
  --sglang-context-length 32768
  --sglang-allow-auto-truncate
  --sglang-max-running-requests 32
  --sglang-cuda-graph-max-bs 32
  --sglang-mamba-ssm-dtype float32
  --use-tensorboard
  --tb-project-name slime-qwen35-base
  --tb-experiment-name "${RUN_NAME}"
  --log-passrate
  --seed 1234
  --rollout-num-gpus 96
  --sglang-ep-size 2
  --sglang-attention-backend fa3
  --sglang-enable-mfu-metrics
  --sglang-enable-metrics
  --eval-max-context-len 32768
)
if [[ "${1:-}" == --check-args ]]; then
  python3 -m examples.mis_sync_interval.check_args "${ARGS[@]}"; exit
fi
if [[ "${1:-}" == --dry-run ]]; then
  printf '%q ' python3 train.py "${ARGS[@]}"; printf '\n'; exit 0
fi
[[ $# == 0 ]] || { echo 'Only --dry-run or --check-args is supported.' >&2; exit 2; }
[[ "$ROOT" = /* && "$RUN" = /* ]] || { echo 'ROOT and RUN must be absolute.' >&2; exit 2; }
[[ -f "$ROOT/train.py" && -f "$MODEL_DIR/config.json" && -f "$TRAIN_DATA" && -f "$EVAL_CONFIG" ]]
[[ -f "$SCRIPT_DIR/if_mix_support.py" ]]
mkdir -p "$RUN/logs" "$RUN/tensorboard"
mkdir "$RUN/driver-started"
cd "$ROOT"
export TENSORBOARD_DIR="$RUN/tensorboard"
printf '%q ' python3 train.py "${ARGS[@]}" > "$RUN/command.sh"
printf '\n' >> "$RUN/command.sh"
RUNTIME_ENV=$(python3 - <<'PYENV'
import json,os
names=['CUDA_DEVICE_MAX_CONNECTIONS','TENSORBOARD_DIR','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_HOME','TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR','OMP_NUM_THREADS','no_proxy','NO_PROXY','PYTHONDONTWRITEBYTECODE']
env={k:os.environ[k] for k in names if k in os.environ}
env['TENSORBOARD_DIR']=env['TENSORBOARD_DIR']+'/train'
print(json.dumps({'env_vars':env}))
PYENV
)
# Run once on the head of an already initialized 16-node / 128-GPU Ray cluster.
# Before this step, validate args/topology in the chosen image as described in the reference.
ray job submit --address http://127.0.0.1:8265 --working-dir "$ROOT" --runtime-env-json "$RUNTIME_ENV" -- python3 train.py "${ARGS[@]}"
