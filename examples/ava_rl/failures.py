"""Ava's explicit trial-failure contract, separate from partial continuation."""

import os
import re

from slime.rollout.base_types import RecoverableRolloutError


def fail_attempt(reason, message):
    """Opt in per run; never reinterpret arbitrary Python/transport exceptions."""
    if os.environ.get("AVA_TOLERATE_ROLLOUT_FAILURES", "0") == "1":
        raise RecoverableRolloutError(reason, message)
    raise RuntimeError(message)


def validate_verifier(metadata):
    error = metadata.get("verifier_error")
    if isinstance(error, dict) and error.get("type") == "VerifierTimeout":
        fail_attempt("verifier_timeout", f"Native verifier exceeded {error['timeout_seconds']}s; no usable reward")
    if metadata.get("verifier_error"):
        fail_attempt("verifier_error", "Ava verifier infrastructure error: not a valid zero reward")
    reward = metadata.get("reward", {})
    reason = reward.get("reason")
    # The supplied test.sh can exit 0 and write reward=0 after pytest failed
    # during collection. Inspect only native verifier output, never model text.
    output = reason.get("pytest_log", "") if isinstance(reason, dict) else reason
    if isinstance(output, str) and re.search(r"(?m)^\s*(?:_+\s*)?ERROR collecting\s", output):
        fail_attempt(
            "verifier_collection_error", "Native verifier could not collect its tests; inspect original pytest log"
        )
