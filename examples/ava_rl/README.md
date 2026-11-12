# Ava Math and Terminal rollouts

This example uses slime's `--custom-generate-function-path` and native
OpenAIAdapter. The adapter captures exact sampled token IDs, log probabilities,
R3 routes and optional score-centering heads. AvaCore owns the original Math
LMaaJ and Terminal agent/verifier lifecycle. The example does not reconstruct
model tokens from text or replace dataset tests.

## Reproducible dependencies

`dependencies.json` pins the private AvaCore repository and commit. Obtain
access through its maintainers. Build the separate Python3.12.12 CPU environment
with Ava's own `uv.lock`, `uv sync --frozen --no-dev --no-editable`. Do not mix
its dependencies into the training Python environment.

`Dockerfile` takes immutable `SLIME_IMAGE`, clean named `avacore` and `slime`
build contexts, and `SLIME_COMMIT`. Prepare those contexts using `git archive`
of the pinned dependency and the tested slime commit, then build the runtime
stage. The source archive and installed package are included in the image;
launchers do not install packages, apply patches, or load external helper code.
A matching standard slime image must supply Megatron, SGLang and their CUDA
libraries. Record its digest and verify GPU initialization before training.

```bash
docker buildx build --load --provenance=false \
  --build-context avacore=/build/avacore \
  --build-context slime=/build/slime \
  --build-arg SLIME_IMAGE=registry/image@sha256:... \
  --build-arg SLIME_COMMIT=<tested-commit> \
  -f examples/ava_rl/Dockerfile -t slime-ava:<unique-tag> examples/ava_rl
```

## Data and credentials

`python -m examples.ava_rl.prepare_data --math math.jsonl --terminal terminal.jsonl
--output train.jsonl` retains all original rows, tests and runtime specs and
mixes by original row counts. slime owns shuffling. The JSONL contract is
`prompt` (user messages), `label`, and `metadata` containing `source`,
`source_index` and `ava_row`. Supported sources are `ava_math` and `ava_terminal`.
Math `ava_row` has `prompt`/`label`; Terminal carries its original row unchanged.
Do not commit data, prompts, credentials, traces, checkpoints or application
cache contents to this repository.

Set `AVA_ENV_FILE` to a private literal KEY=VALUE file (mode0600). The isolated
CPU service reads only AWS_ENDPOINT_URL/AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY,
OPEN_SANDBOX_DOMAIN/OPEN_SANDBOX_API_KEY and AVACORE_APPLICATION_CACHE. No shell
is executed and no credentials are passed to sandbox environments. Set the
native sandbox API key to the authorized public key + `:` + secret key. The
application cache is read by native Ava; do not modify shared cache assets.

| Configuration | Meaning |
|---|---|
| `AVA_PYTHON` | Pinned CPU interpreter, baked into the example image |
| `AVA_JUDGE_ORIGIN`, `AVA_JUDGE_MODEL` | Authorized LMaaJ endpoint/model |
| `AVA_JUDGE_MAX_TOKENS` | Judge completion budget; default16384 |
| `AVA_RPC_CONCURRENCY` | Complete generate→reward transactions; default4 |
| `AVA_POLICY_READ_TIMEOUT`, `AVA_RPC_TIMEOUT` | Policy HTTP and outer RPC deadlines |
| `AVA_SANDBOX_MEMORY` | Override native RuntimeSpec memory (e.g.16Gi); image/tests/CPU unchanged |
| `AVA_TERMINAL_MAX_OUTPUT_TOKENS` | Per-call output cap, default8192 |
| `AVA_VERIFIER_TIMEOUT_SECONDS` | Optional positive deadline for the complete native after_run phase |
| `AVA_RUNTIME_LOG_DIR`, `AVA_TRACE_DIR`, `AVA_TIMING_DIR` | Private service/errors, optional trajectory and timing evidence |

Math uses the supplied LMaaJ prompt and strict native verdict parser. Empty or
invalid verdicts never become invented zero rewards. Raising the completion
budget does not guarantee a valid verdict. Terminal retains the native128-turn,
900s agent loop; the verifier has its own optional deadline, independent of the
agent budget. Cancellation interrupts the same native execution ID and leaves
sandbox cleanup to the existing serving context. Interruption/cleanup can take
additional SDK time; inspect exact cleanup failures rather than assuming deletion.

## Explicit compatibility policies

The following flags default off and are recorded in each run:

- `AVA_VALIDATE_TERMINAL_ACTIONS=1`: reject decoded NUL, isolated Unicode
  surrogates and invalid durations through native parser-error feedback. Keep
  the model output unchanged and ask for a correction before executing commands.
- `AVA_END_ON_DESTRUCTIVE_ACTIONS=1`: terminate before a limited set of direct
  execd-killing commands executes, preserve the trace and ask the original
  verifier to score the existing task state. No fixed reward or shell recreation.
  This is an action restriction, not a general shell security boundary.
- `AVA_TMUX_COMMAND_STATUS=1`: when using the relay runtime, submit tmux writes
  once through native detached execution-ID status/wait. Require explicit exit0;
  unknown/nonzero states propagate and ambiguous commands are not replayed.
- Confirmed missing tmux sessions only end the episode after proving the command
  channel is healthy; transport failures propagate. Native verifier still runs.
- `AVA_TOLERATE_ROLLOUT_FAILURES=1`: classify only native recoverable/TaskEnvError,
  invalid judge verdicts and unusable verifier outcomes. Fully-async drops the
  entire prompt group, settles its siblings and fills another group. No fake0,
  in-place command replay or silent evaluation replacement. Unknown exceptions
  remain fatal. `--fully-async-max-consecutive-failed-groups` defaults32 and bounds
  persistent failure. Monitor source/reason failure ratios and effective mix.
  The failure contract currently supports the in-process fully-async producer;
  do not configure distributed producers with this example.

R3 replay preparation now also uses the upstream bounded microbatch path for
resident tensors. This avoids expanding an entire long agent batch per training
rank, while preserving exact route values, CP/TP layout and replay order. It does
not guarantee every128K batch fits; measure actual memory and finite updates.

## Optional network relay

When GPU nodes cannot reach the authorized sandbox gateway directly, run
`python -m examples.ava_rl.relay --bind <private-IP> --port <port> --token-file
<private-token>` on a reachable host. It permits authenticated CONNECT to the
single configured gateway, preserves end-to-end TLS and never handles YiCloud
signing keys. Configure `AVA_RELAY_URL` and `AVA_RELAY_TOKEN_FILE`; the explicit
native RuntimeSpec subclass uses the SDK's public HTTP transport injection.
No MiMo package or external relay script is imported at runtime. Use a finite
process lifetime and record the owning PID separately from GPU jobs.

## Example launch and validation

`run_qwen35.sh` uses the repository's Qwen model config and expects MODEL_DIR,
TRAIN_DATA, RUN and CHECKPOINT_DIR. LOAD_DIR/CKPT_STEP select native resume;
NUM_ROLLOUT defaults1000, save20, B32G8/128K, MIS+R3, SCoff. WEIGHT_INTERVAL
(default1) can select the separate delayed-publication feature when installed.
EVAL_CONFIG can retain the existing held-out datasets; this recipe does not run
online evaluation. Review all flags before selecting this experiment recipe.
`yicloud_entry.sh` starts16×8GPU Ray nodes with the platform's native rank/address
variables. Code is shipped by Ray's standard working_dir from the baked source,
with no runtime package installs or external executable hooks.

Native metrics remain enabled. `metrics.py` additionally reports deduplicated
whole-episode timeout/context/turn/exit ratios separately from segment truncation
and per-call output limits. Concurrent RPC durations cannot be summed as walltime.

CPU core regressions are in existing registered tests. Private dependency
contracts are manual: `/opt/ava/bin/python -m examples.ava_rl.test_terminal`.
Run them in the image before GPU use. Real GPU acceptance must show original
Math/Terminal scoring, finite nonzero gradients, expected publication versions,
failed-group replacement, full model/optimizer/RNG/rollout save and continued
training. Submission and a readable checkpoint directory alone do not pass.

A contract-by-contract review of native error handling, omissions fixed by this
adapter, and the R3 memory/time tradeoff is recorded in [AUDIT.md](AUDIT.md).
