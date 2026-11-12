"""Native slime logging hook: episode and segment metrics kept distinct."""

from collections import defaultdict


def summarize(samples):
    episodes = {}
    segments = defaultdict(list)
    for sample in samples:
        md = sample.metadata or {}
        info = md.get("ava_episode")
        if info is None:
            continue
        episodes.setdefault(info["sid"], info)
        for source in ("all", info["source"]):
            segments[source].append(sample.status == sample.Status.TRUNCATED)
    result = {}
    for source, flags in segments.items():
        cohort = [e for e in episodes.values() if source == "all" or e["source"] == source]
        prefix = f"ava/{source}/"
        result[prefix + "episode_count"] = len(cohort)
        result[prefix + "segment_count"] = len(flags)
        result[prefix + "segment_truncated_ratio"] = sum(flags) / len(flags)
        calls = sum(e.get("model_calls", 0) for e in cohort)
        if calls:
            result[prefix + "model_call_count"] = calls
            result[prefix + "call_output_cap_ratio"] = sum(e.get("output_cap_calls", 0) for e in cohort) / calls
            result[prefix + "episode_any_output_cap_ratio"] = sum(
                e.get("output_cap_calls", 0) > 0 for e in cohort
            ) / len(cohort)
        for key in ("truncated", "timeout", "context_limit", "max_turns", "terminal_exited", "last_call_length"):
            result[prefix + "episode_" + key + "_ratio"] = sum(bool(e[key]) for e in cohort) / len(cohort)
        result[prefix + "episode_destructive_action_ratio"] = sum(
            bool(e.get("destructive_action")) for e in cohort
        ) / len(cohort)
    return result


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    from slime.observability import logging_utils
    from slime.observability.metric_utils import compute_rollout_step

    values = summarize(samples)
    if values:
        values["rollout/step"] = compute_rollout_step(args, rollout_id)
        logging_utils.log(args, values, step_key="rollout/step")
    return False  # Preserve all native metrics and existing training semantics.
