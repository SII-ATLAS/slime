"""Keep the existing IFBench verifier for the unchanged held-out evaluation."""

from open_instruct.ground_truth_utils import IFEvalVerifier

_VERIFIER = IFEvalVerifier()


async def ifbench_reward(args, sample, **kwargs):
    return float(_VERIFIER([], sample.response or "", sample.label).score)
