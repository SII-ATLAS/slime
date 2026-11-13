"""Validate the native GPU argument contract before any Ray training starts."""

import os

from examples.mis_sync_interval.if_mix_support import log_sources, reward

from slime.utils.arguments import parse_args

a = parse_args()
assert a.start_rollout_id == 0 and a.load == os.environ["MODEL_DIR"]
assert a.no_load_optim and a.no_load_rng and a.finetune
assert a.update_weights_interval == 128 and a.num_rollout == 2000 and a.save_interval == 50
assert a.use_score_centering and a.score_centering_top_k == 32 and a.use_tis and a.use_rollout_routing_replay
assert a.global_batch_size == 512 and a.rollout_batch_size == 64 and a.n_samples_per_prompt == 8
assert a.tensor_model_parallel_size == 8 and a.context_parallel_size == 1 and a.pipeline_model_parallel_size == 1
assert a.rollout_num_gpus == 96 and a.actor_num_nodes == 4 and a.actor_num_gpus_per_node == 8
assert a.rollout_max_context_len == 32768 and a.rollout_max_response_len == 30720
assert a.max_tokens_per_gpu == 49152 and a.sglang_server_concurrency == 16
assert a.fully_async_group_prefetch_factor == 1 and a.flush_cache_interval == 8 and a.partial_rollout
assert not a.no_save_optim and not a.no_save_rng and a.eval_interval is None
assert a.ref_load == os.environ["MODEL_DIR"] and a.use_kl_loss and a.kl_loss_coef == 0
assert not a.grpo_std_normalization and a.rollout_data_transport == "object-store"
assert not hasattr(a, "fully_async_ready_buffer_size")
print("M2_ARGS_VALIDATED: Base/start0, sync128, MIS+SC32+R3, M data/resource flags, 2000/save50", flush=True)
