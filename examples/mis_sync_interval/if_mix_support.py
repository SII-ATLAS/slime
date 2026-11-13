"""Slime public hooks: unmodified official verifier dispatch and read-only source metrics."""

import logging
import math
from collections import defaultdict

from open_instruct.ground_truth_utils import IFEvalVerifier

from slime.rollout.rm_hub.math_utils import grade_answer_verl

_IF_VERIFIER = IFEvalVerifier()


async def reward(args, sample, **kwargs):
    source = sample.metadata["source_name"]
    if source == "ifbench":
        value = _IF_VERIFIER([], sample.response or "", sample.label).score
    elif source == "dapo":
        value = float(grade_answer_verl(sample.response or "", sample.label))
    else:
        raise ValueError(f"Unexpected training source: {source}")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid reward from {source}: {value}")
    return float(value)


def log_sources(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    from slime.observability import logging_utils
    from slime.observability.metric_utils import compute_rollout_step, compute_statistics

    groups = defaultdict(list)
    for sample in samples:
        groups[sample.metadata["source_name"]].append(sample)
    metrics = {"rollout/step": compute_rollout_step(args, rollout_id)}
    for source, items in groups.items():
        values = [s.get_reward_value(args) for s in items]
        prefix = f"rollout/{source}"
        metrics[f"{prefix}/reward"] = sum(values) / len(values)
        metrics[f"{prefix}/all_correct"] = sum(v == 1 for v in values) / len(values)
        metrics[f"{prefix}/samples"] = len(values)
        metrics[f"{prefix}/groups"] = len({s.group_index for s in items})
        metrics[f"{prefix}/truncated"] = sum(s.status == s.Status.TRUNCATED for s in items) / len(items)
        for name, value in compute_statistics([s.response_length for s in items]).items():
            metrics[f"{prefix}/response_len/{name}"] = value
        rewards_by_group = defaultdict(list)
        for sample, value in zip(items, values):
            rewards_by_group[sample.group_index].append(value)
        metrics[f"{prefix}/groups_with_reward_variation"] = sum(
            min(v) != max(v) for v in rewards_by_group.values()
        ) / len(rewards_by_group)
    logging.getLogger(__name__).info("source metrics %s: %s", rollout_id, metrics)
    logging_utils.log(args, metrics, step_key="rollout/step")
    return False  # Keep all native rollout metrics and staleness logging.
