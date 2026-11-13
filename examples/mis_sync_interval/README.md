# M2: delayed-publication staleness

The M reference uses IF70/DAPO30, Qwen3.5-35B-A3B Base, 128 GPUs (32 training,
96 rollout), TP8/CP1/DP4, B64G8/global512, 32K context/30720 output,
MIS IcePop[0.5,2] + SC top32 + R3, zero-coefficient Reference path retained,
LR1e-6/WD0.1, BF16/SSM FP32, C16/P1/F8, native running32/graph32,
packing49152, 2000 updates/save50 with optimizer/RNG/rollout state.

M2 disables R3 by explicit user instruction and replaces M's ready4160/fill4032/oldest buffer experiment with native
`--update-weights-interval128`. No ready-buffer/type1/type2 implementation is
imported. Startup publishes version1, cumulative update128 publishes129, then
257 at256; saves/final do not force publications. F8 is per scheduled publication
(1024 updates per periodic full refresh). Staleness varies naturally across the
cycle; this is not fixed age128 or a claim of equivalence to M's age64 queue.

Start from Base rather than resuming M's final policy. Set MODEL_DIR, TRAIN_DATA,
EVAL_CONFIG, RUN, CHECKPOINT_DIR and RUN_NAME. `run_qwen35.sh --dry-run` prints
arguments; `--check-args` performs native GPU-environment validation.
`yicloud_entry.sh` uses native 16x8 platform variables, 120CPU/1800GiB per node,
head832GiB Plasma/worker128GiB as M; no package installs or external code hooks.
IF and DAPO reward/source metrics are ordinary public hooks owned in this folder.
Data/model paths are assets only; do not commit data or credentials.

Build Dockerfile with fixed standard dependency SLIME_IMAGE digest and clean
`git archive` named slime context + SLIME_COMMIT. No Ava runtime dependency is
used by M2. The upstream fork includes normal correctness/resource fixes absent
in the historical snapshot, so results are a cross-version comparison as well
as a staleness-mechanism comparison; scientific recipe flags are retained.

Acceptance: actual native arguments, original rewards, finite gradients/SC, R3 off,
initial-only publication before update128, real engine129 at the boundary,
complete model/optimizer/RNG/rollout save50 and continued training.
