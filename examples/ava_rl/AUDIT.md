# Ava integration boundary review

Reviewed against AvaCore `6d4f0dd2b9e7d7e830445b64e019ae5ad98d759f`.
This is a review of the pinned integration, not a claim about all Ava deployments
or future versions. Preserve original traces, task tests and verdicts.

| Boundary | Native behavior and integration policy | Regression |
|---|---|---|
| Malformed action JSON | Native parser errors feed back into the original Terminus loop. Also translate parser numeric overflow, JSON integer/depth limits into that same feedback; do not catch unrelated programmer/runtime exceptions. | `test_terminal.py` numeric/depth and real native correction loop |
| Non-executable keystrokes | Parser checks decoded NUL, isolated UTF-8 surrogates and negative/nonfinite durations before executing any command. Valid Unicode and literal shell escapes are unchanged. | `test_terminal.py` invalid/valid actions |
| Context budget | Adapter emits explicit empty length completion when exhausted; budget client maps only the exhausted-context signal to native ContextLengthExceededError. Per-call output cap is not episode exhaustion. | Existing adapter budget tests; native context termination remains inherited |
| Shell exit | Missing tmux alone is insufficient: healthy command channel plus confirmed missing session ends the episode; still invoke native verifier. Unknown transport/status errors propagate. | `test_terminal.py` confirmed exit and unhealthy channel |
| Destructive action policy | Optional narrow pre-execution guard ends the episode with the original action in trace and invokes native verifier. No fixed score, recreation or command replay. | `test_terminal.py` verifier zero/one preserved |
| Verifier | Native after_run retained, with optional total deadline. Cancellation interrupts native execution ID; verifier errors/timeouts and pytest collection failures are unusable outcomes, not invented zero rewards. | Native deadline/score tests and fully-async failure-policy tests |
| Local RPC transport | Timeouts, disconnects and incomplete payloads discard a classified failed group when opted in, never retry an ambiguously executed action. Dead service, authentication errors and unknown failures remain fatal. | Real HTTP tests in `tests/test_agent/test_adapters.py` |
| Native trial errors | Honor explicit recoverable/TaskEnvError classification. Only definitive pre-generation ready timeout is retried, at most three attempts, per the existing policy. Native command-too-long fallback and SDK command status/cancellation are inherited. | Real HTTP tests in `tests/test_fully_async_rollout.py` |
| Group lifecycle | Settle/cancel every sibling before whole-group disposal and refill; no reward fabrication or partial group training. Bounded consecutive failed groups, source/reason metrics; ABORTED continuation unchanged. | Fully-async group disposition/cancellation/unknown-error tests |
| Admission | Bound whole generate-to-reward transaction; reject nonpositive concurrency before starting services instead of waiting forever. | Adapter configuration tests |

Known boundaries: this is the in-process producer integration; unknown errors
are intentionally not swallowed. Native sandbox cleanup may itself fail and must
remain visible. A timeout is not proof the remote process was deleted. No CPU
contract suite proves platform/network reliability or all model outputs safe.

## Resident R3 memory and time

The host OOM occurred in slime's training actor, outside Ava's lifecycle.
Resident routes now reuse upstream's lazy RoutedExpertsMicrobatch and bounded
prefetch path. Original route tensors still exist; the bounded component is the
additional prepared replay layout, not total process or node RAM.

- No rollout inference, sandbox, verifier or network operation is added.
- No disk spill or new file transport is introduced for resident tensors.
- CPU layout preparation happens near consumption, with default one-microbatch
  prefetch; a late prefetch can stall the training consumer.
- Separate actor-logprob and training passes may prepare a microbatch twice.
  Training retains its prepared layout through recompute/backward and then frees
  it; this does not add model forward/backward passes or change route values.
- GPU H2D replay still happens per layer. CPU scheduling, pinning and bandwidth
  costs need real GPU measurements; no promised percentage speedup/slowdown.

Resident/disk layout equivalence and bounded release are tested in
`tests/test_routed_experts_disk_layout.py`. Before accepting the U test, compare
handoff/R3 setup, actor logprob and training durations, GPU idle gaps, node RSS
and memory pressure, with token lengths and segment counts alongside timings.
A successful first update does not establish long-run memory safety.
