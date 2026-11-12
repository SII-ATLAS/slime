"""Standard slime generation hook backed by native Ava generation and rewards.

slime captures sampled IDs, probabilities and replay via its OpenAIAdapter;
Ava owns the agent, sandbox and grader. No text-to-token reconstruction.
"""

import asyncio
import atexit
import json
import logging
import math
import os
import secrets
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

import aiohttp
from examples.ava_rl.diagnostics import record
from examples.ava_rl.failures import fail_attempt, validate_verifier

from slime.agent.adapters import OpenAIAdapter
from slime.agent.aiohttp_threaded import run_app_in_thread
from slime.rollout.base_types import RecoverableRolloutError
from slime.utils.misc import SingletonMeta
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample


class Service(metaclass=SingletonMeta):
    def __init__(self, args):
        # Reserve capacity for a complete generate -> reward transaction, not
        # just its first HTTP request, so queued generation cannot starve reward.
        self.admission = asyncio.Semaphore(int(os.environ.get("AVA_RPC_CONCURRENCY", "4")))
        if args.apply_chat_template:
            raise ValueError("Ava's model adapter renders the template; disable --apply-chat-template")
        self.turn_counts = {}
        self.adapter = OpenAIAdapter(
            tokenizer=load_tokenizer(args.hf_checkpoint, trust_remote_code=True),
            sglang_url=f"http://{args.sglang_router_ip}:{args.sglang_router_port}",
            tool_parser=getattr(args, "sglang_tool_call_parser", None),
            reasoning_parser=getattr(args, "sglang_reasoning_parser", None),
            rollout_args=args,
            request_read_timeout=float(os.environ.get("AVA_POLICY_READ_TIMEOUT", "900")),
            debug_callback=self.capture_turn,
        )
        self.handle = run_app_in_thread(
            self.adapter.app,
            host="127.0.0.1",
            port=0,
            runner_kwargs={"handler_cancellation": True, "access_log": None},
        )
        self.model = {"kind": "openai", "name": "policy", "endpoint": f"http://127.0.0.1:{self.handle.port}"}
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.token = secrets.token_hex(32)
        self.url = f"http://127.0.0.1:{port}"
        env = os.environ.copy()
        env.update(AVA_RPC_PORT=str(port), AVA_RPC_TOKEN=self.token)
        # Avoid importing the GPU environment's packages into native Ava.
        env.pop("PYTHONPATH", None)
        logs = Path(os.environ.get("AVA_RUNTIME_LOG_DIR", "ava-services"))
        logs.mkdir(parents=True, exist_ok=True)
        self.log = (logs / f"ava-{socket.gethostname()}-{os.getpid()}.log").open("a")
        self.process = subprocess.Popen(
            [os.environ["AVA_PYTHON"], "-m", "examples.ava_rl.service"],
            env=env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        atexit.register(self.close)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("Native Ava service exited; inspect its dedicated log")
            try:
                with opener.open(self.url + "/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(0.2)
        else:
            self.close()
            raise TimeoutError("Native Ava service did not start")

    def capture_turn(self, sid, messages, tools, assistant, turn):
        """Optional native debug hook: text/history only, never replay tensors."""
        counts = self.turn_counts.setdefault(sid, {"calls": 0, "length_outputs": {}})
        counts["calls"] += 1
        if turn.finish_reason == "length":
            size = len(turn.output_ids)
            counts["length_outputs"][size] = counts["length_outputs"].get(size, 0) + 1
        record(
            "policy_turn",
            sid=sid,
            prompt_tokens=len(turn.prompt_ids),
            completion_tokens=len(turn.output_ids),
            finish_reason=turn.finish_reason,
        )
        if not os.environ.get("AVA_TRACE_DIR"):
            return
        directory = Path(os.environ["AVA_TRACE_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{sid}.json"
        temporary = path.with_suffix(".tmp")
        with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as stream:
            json.dump(
                {
                    "messages": messages,
                    "assistant": assistant,
                    "finish_reason": turn.finish_reason,
                    "prompt_tokens": len(turn.prompt_ids),
                    "completion_tokens": len(turn.output_ids),
                },
                stream,
            )
        temporary.replace(path)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log.close()
        self.handle.stop()

    async def call(self, operation, payload):
        call_started = time.monotonic()
        timeout = aiohttp.ClientTimeout(total=float(os.environ.get("AVA_RPC_TIMEOUT", "1500")), connect=10)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
            for attempt in range(3):  # Ava runner.Retry.attempts_per_trial default.
                async with session.post(
                    self.url + "/v1/" + operation, json=payload, headers={"Authorization": "Bearer " + self.token}
                ) as response:
                    if response.status < 400:
                        result = await response.json()
                        record(
                            "ava_rpc",
                            operation=operation,
                            key=payload.get("key"),
                            seconds=time.monotonic() - call_started,
                        )
                        return result
                    error = await response.json()
                    # Retry only a definitive pre-generation readiness failure.
                    # Never replay ambiguous network failures or agent commands.
                    if (
                        operation == "generate"
                        and response.status == 500
                        and attempt < 2
                        and error.get("error_type") == "SandboxReadyTimeoutException"
                        and error.get("retryable") is True
                    ):
                        logging.getLogger(__name__).warning(
                            "Ava sandbox readiness retry %d/3 for %s", attempt + 2, payload.get("key")
                        )
                        continue
                    # Ava runner also drops terminal TaskEnvError trials; they
                    # have no usable outcome and should not be retried in place.
                    if response.status == 500 and (
                        error.get("recoverable") is True or error.get("error_type") == "TaskEnvError"
                    ):
                        fail_attempt(
                            f"{operation}_{error.get('error_type', 'recoverable')}",
                            f"Native Ava {operation}: {error.get('message', '')}",
                        )
                    raise RuntimeError(f"Native Ava {operation} HTTP {response.status}; inspect Ava service log")


def config_for(source, model, context_limit=32768):
    if source == "ava_math":
        return {
            "kind": "single_turn",
            "model": model,
            "instance": {
                "messages": [
                    {"role": "user", "content": "{{ prompt }}\nRemember to put your final answer within \\boxed{}."}
                ]
            },
        }
    if source == "ava_terminal":
        verifier_timeout = os.environ.get("AVA_VERIFIER_TIMEOUT_SECONDS")
        verifier_timeout = float(verifier_timeout) if verifier_timeout else None
        if verifier_timeout is not None and (not math.isfinite(verifier_timeout) or verifier_timeout <= 0):
            raise ValueError("AVA_VERIFIER_TIMEOUT_SECONDS must be finite and positive")
        return {
            "kind": "ava_terminal_exit",
            "model": {**model, "kind": "ava_budgeted_openai", "context_limit": context_limit},
            "parser_name": "json",
            "user": "root",
            "max_turns": 128,
            "max_output_bytes": 10000,
            "recording_path": None,
            "end_on_destructive_actions": os.environ.get("AVA_END_ON_DESTRUCTIVE_ACTIONS", "0") == "1",
            "validate_terminal_actions": os.environ.get("AVA_VALIDATE_TERMINAL_ACTIONS", "0") == "1",
            "verifier_timeout_seconds": verifier_timeout,
            "instance": {
                "prompt": "{{ prompt.split('\\n## Recommended Workflow', 1)[0] }}",
                "cwd": {"field": "cwd"},
                "tests": {"field": "tests"},
                "tests_binary": {"field": "tests_binary"},
                "runtime_spec": {"field": "runtime_spec"},
                "timeout_seconds": 900,
            },
        }
    raise ValueError(f"Unsupported Ava source: {source}")


async def generate(args, sample, sampling_params, evaluation=False):
    if evaluation:
        from slime.rollout.sglang_rollout import generate as native_generate

        return await native_generate(args, sample, sampling_params)
    state = Service(args)
    waiting = time.monotonic()
    async with state.admission:
        record("admission", source=sample.metadata["source"], seconds=time.monotonic() - waiting)
        return await _generate(args, sample, sampling_params, state)


async def _generate(args, sample, sampling_params, state):
    source = sample.metadata["source"]
    row = sample.metadata["ava_row"]
    if source == "ava_terminal" and os.environ.get("AVA_SANDBOX_MEMORY"):
        spec = row["runtime_spec"]
        resource = {"cpu": "2", "memory": "4Gi", **spec.get("resource", {})}
        resource["memory"] = os.environ["AVA_SANDBOX_MEMORY"]
        row = {**row, "runtime_spec": {**spec, "resource": resource}}
    if source == "ava_terminal" and os.environ.get("AVA_RELAY_URL"):
        spec = row["runtime_spec"]
        if spec["kind"] != "opensandbox":
            raise ValueError("Relay supports only the original OpenSandbox runtime")
        row = {**row, "runtime_spec": {**spec, "kind": "ava_relay_opensandbox"}}
    sid = secrets.token_hex(16)
    started = time.monotonic()
    metadata = {}
    state.adapter.open_session(
        sid,
        sampling_defaults=sampling_params.copy(),
        max_context_tokens=args.rollout_max_context_len,
        apply_chat_template_kwargs=getattr(args, "apply_chat_template_kwargs", None) or {},
    )
    try:
        output_limit = (
            int(os.environ.get("AVA_TERMINAL_MAX_OUTPUT_TOKENS", "8192"))
            if source == "ava_terminal"
            else sampling_params["max_new_tokens"]
        )
        if output_limit <= 0:
            raise ValueError("AVA_TERMINAL_MAX_OUTPUT_TOKENS must be positive")
        params = {
            **sampling_params,
            "metadata": {"session_id": sid},
            "max_tokens": min(sampling_params["max_new_tokens"], output_limit),
        }
        # Internal SGLang-only keys are supplied by adapter sampling_defaults.
        params = {k: v for k, v in params.items() if k in {"temperature", "top_p", "top_k", "max_tokens", "metadata"}}
        data = {"row": row, "key": sid}
        result = await state.call(
            "generate",
            {
                **data,
                "generate": config_for(source, state.model, args.rollout_max_context_len),
                "sampling_params": params,
                "kwargs": {},
            },
        )
        traces = result["traces"]
        if traces.get("format") != "ava.trace" or traces.get("version") != 2 or len(traces["tips"]) != 1:
            raise ValueError("Expected one native Ava v2 trace tip")
        metadata = traces["nodes"][traces["tips"][0]].get("metadata", {})
        if source == "ava_terminal":
            validate_verifier(metadata)
        if source == "ava_math":
            reward_cfg = {
                "kind": "lmaaj",
                "model": {
                    "kind": "ava_observed_judge",
                    "endpoint": os.environ["AVA_JUDGE_ORIGIN"],
                    "name": os.environ["AVA_JUDGE_MODEL"],
                    "timeout": 600,
                    "sampling_params": {
                        "temperature": 0,
                        "top_p": 1,
                        "reasoning_effort": "low",
                        "max_tokens": int(os.environ.get("AVA_JUDGE_MAX_TOKENS", "16384")),
                    },
                },
                "reference": {"question": {"field": "prompt"}, "answer": {"field": "label"}},
            }
            # Exact verdict-pairing guidance from Ava's supplied Math recipe;
            # keep the new native judge criteria and strict parser unchanged.
            reward_cfg["system_prompt"] = (
                "You are a helpful assistant who evaluates the correctness and quality of models' outputs. "
                "INCORRECT always uses B; CORRECT always uses A. The letter is not an abbreviation for Answer. "
                "Never output A: INCORRECT or B: CORRECT."
            )
        else:
            reward_cfg = {"kind": "recorded"}
        reward = (await state.call("reward", {**data, "traces": traces, "reward": reward_cfg}))["reward"]
        if reward.get("metadata", {}).get("grader_invalid"):
            path = Path(os.environ["AVA_RUNTIME_LOG_DIR"]) / f"invalid-judge-{sid}.json"
            with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as evidence:
                json.dump(
                    {
                        "source_index": sample.metadata.get("source_index"),
                        "reward": reward,
                        "traces": traces,
                        "judge": reward_cfg,
                    },
                    evidence,
                    ensure_ascii=False,
                )
            fail_attempt("grader_invalid", "Invalid native judge verdict is not a valid training reward")
        samples = await state.adapter.finish_session(sid, base_sample=sample, reward=float(reward["score"]))
        if not samples:
            raise ValueError("No token-exact samples were captured")
        termination = metadata.get("termination", {})
        reason = metadata.get("terminus_2", {}).get("termination_reason")
        episode_info = {
            "sid": sid,
            "source": source,
            "timeout": termination.get("reason") == "timeout",
            "context_limit": reason == "context_limit",
            "max_turns": reason == "max_turns",
            "terminal_exited": reason == "terminal_exited",
            "destructive_action": reason == "destructive_control_action",
            "last_call_length": any(s.metadata.get("truncated", False) for s in samples),
        }
        counts = state.turn_counts.get(sid, {"calls": 0, "length_outputs": {}})
        episode_info.update(
            model_calls=counts["calls"],
            output_cap_calls=counts["length_outputs"].get(params["max_tokens"], 0),
            output_limit=params["max_tokens"],
        )
        episode_info["truncated"] = bool(
            termination.get("budget_exceeded") or episode_info["context_limit"] or episode_info["last_call_length"]
        )
        for segment in samples:
            segment.metadata = {**(segment.metadata or {}), "source": source, "ava_episode": episode_info}
            if (
                termination.get("budget_exceeded")
                or metadata.get("terminus_2", {}).get("termination_reason") == "context_limit"
                or segment.metadata.get("truncated")
            ):
                segment.status = Sample.Status.TRUNCATED
        record(
            "episode",
            sid=sid,
            source=source,
            source_index=sample.metadata.get("source_index"),
            seconds=time.monotonic() - started,
            segments=len(samples),
            termination=episode_info,
        )
        return samples
    except Exception as error:
        if isinstance(error, RecoverableRolloutError):
            error.source = source
            record(
                "failed_episode",
                sid=sid,
                source=source,
                source_index=sample.metadata.get("source_index"),
                reason=error.reason,
                seconds=time.monotonic() - started,
            )
        if os.environ.get("AVA_TRACE_DIR"):
            path = Path(os.environ["AVA_TRACE_DIR"]) / f"{sid}-error.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
                json.dump(
                    {
                        "source": source,
                        "source_index": sample.metadata.get("source_index"),
                        "error": str(error),
                        "error_type": type(error).__name__,
                        "reason": getattr(error, "reason", None),
                        "verifier_reward": metadata.get("reward") if source == "ava_terminal" else None,
                        "elapsed_seconds": time.monotonic() - started,
                    },
                    stream,
                )
        raise
    finally:
        await state.adapter.drop_session(sid, wait_timeout=30)
        state.turn_counts.pop(sid, None)
