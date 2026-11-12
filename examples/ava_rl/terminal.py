"""Ava Terminus: confirmed shell exit ends the episode, not the verifier.

The native model loop and token trace are inherited. This module owns session
lifecycle and an opt-in destructive-action termination rule; all episodes retain
the native verifier. No shell restart, retry or replacement sample.
"""

import asyncio
import math
import shlex
import time
from contextlib import aclosing, nullcontext
from dataclasses import dataclass, replace
from pathlib import PurePosixPath

from ava_core.application.service.service import serving
from ava_core.core.model import kind
from ava_core.generate.terminal import TerminalBenchTerminus2
from ava_core.generate.tools.terminus.terminus_json_plain_parser import TerminusJSONPlainParser
from ava_core.generate.tools.terminus.tmux_session import TmuxSession
from ava_core.runtime.runtime import RuntimeExecutionError
from examples.ava_rl.diagnostics import record


class ValidatedTerminalJSONParser(TerminusJSONPlainParser):
    """Validate native tmux input constraints before any command is executed."""

    def parse_response(self, response):
        result = super().parse_response(response)
        if not result.error:
            if any("\0" in command.keystrokes for command in result.commands):
                return replace(
                    result,
                    commands=[],
                    error="commands[].keystrokes must not contain NUL; use textual shell escapes or tmux key names",
                )
            try:
                for command in result.commands:
                    command.keystrokes.encode("utf-8")
            except UnicodeEncodeError:
                return replace(
                    result,
                    commands=[],
                    error="commands[].keystrokes must be valid UTF-8; isolated Unicode surrogates are not executable",
                )
            if any(not math.isfinite(command.duration) or command.duration < 0 for command in result.commands):
                return replace(result, commands=[], error="commands[].duration must be finite and nonnegative")
        return result


def destructive_control_command(keystrokes):
    """Recognize only a direct, unconditional first command; never infer shell code."""
    try:
        lexer = shlex.shlex(keystrokes.splitlines()[0], posix=True, punctuation_chars=";&|<>")
        lexer.whitespace_split = True
        words = list(lexer)
    except (ValueError, IndexError):
        return False
    for i, word in enumerate(words):
        if word in (";", "&", "&&", "||", "|"):
            words = words[:i]
            break
    if words[-3:] == ["2", ">", "/dev/null"]:
        words = words[:-3]
    if len(words) < 2:
        return False
    command = PurePosixPath(words[0]).name
    if command in ("pkill", "killall"):
        flags = {"-9", "-15", "-TERM", "-KILL", "--"}
        flags |= {"-f", "--full", "-x", "--exact"} if command == "pkill" else {"-q", "--quiet"}
        return words[-1] in ("execd", "/opt/opensandbox/bin/execd") and set(words[1:-1]) <= flags
    return command == "fuser" and words[1:] == ["-k", "44772/tcp"]


class DestructiveControlAction(ValueError):
    """A parsed action rejected before any commands in that action execute."""


def missing_terminal(error):
    return (
        isinstance(error, RuntimeExecutionError)
        and error.exit_code == 1
        and error.argv[:1] == ("tmux",)
        and any(
            text in error.output.stderr for text in ("no server running on", "can't find session:", "can't find pane:")
        )
    )


class ExitAwareSession(TmuxSession):
    terminal_exited = False

    async def confirm_exit(self):
        # A healthy command channel is required; transport/VM failures propagate.
        await self.runtime.run(["true"])
        try:
            await self.runtime.run(["tmux", "has-session", "-t", f"={self.session_name}"])
        except RuntimeExecutionError as error:
            if not missing_terminal(error):
                raise
            self.terminal_exited = True
            return True
        return False

    async def close(self):
        try:
            await super().close()
        except RuntimeExecutionError as error:
            if not self.terminal_exited or not missing_terminal(error):
                raise
            await self.runtime.run(["true"])


@kind("ava_terminal_exit")
@dataclass(frozen=True)
class ExitAwareTerminal(TerminalBenchTerminus2):
    end_on_destructive_actions: bool = False
    validate_terminal_actions: bool = False
    verifier_timeout_seconds: float | None = None

    def _parser(self):
        if self.validate_terminal_actions and self.parser_name == "json":
            return ValidatedTerminalJSONParser()
        return super()._parser()

    async def _execute_commands(self, commands, session):
        if self.end_on_destructive_actions:
            for command in commands:
                if destructive_control_command(command.keystrokes):
                    raise DestructiveControlAction(command.keystrokes)
        started = time.monotonic()
        try:
            return await super()._execute_commands(commands, session)
        finally:
            record(
                "terminal_commands",
                session=session.session_name,
                seconds=time.monotonic() - started,
                count=len(commands),
            )

    async def after_run(self, runtime, instance, trace):
        limit = self.verifier_timeout_seconds
        if limit is not None and (not math.isfinite(limit) or limit <= 0):
            raise ValueError("verifier_timeout_seconds must be finite and positive")
        started = time.monotonic()
        deadline = asyncio.timeout(limit)
        timed_out = False
        try:
            async with deadline:
                return await super().after_run(runtime, instance, trace)
        except TimeoutError:
            if not deadline.expired():
                raise
            # Native OpenSandbox.run handles cancellation by interrupting the
            # same execution ID. The serving context then stops the sandbox.
            # No test replay, fabricated reward, or detached verifier task.
            timed_out = True
            return trace.with_metadata(
                verifier_error={
                    "type": "VerifierTimeout",
                    "timeout_seconds": limit,
                }
            )
        finally:
            record("terminal_verifier", seconds=time.monotonic() - started, timeout=timed_out, timeout_seconds=limit)

    async def run_stream(self, session, instance, **kwargs):
        last = None
        try:
            async with aclosing(super().run_stream(session, instance, **kwargs)) as stream:
                async for trace in stream:
                    last = trace
                    yield trace
        except DestructiveControlAction as error:
            if last is None:
                raise
            yield last.with_metadata(
                rejected_action={
                    "kind": "destructive_control_action",
                    "command": str(error),
                    "rejected_before_execution": True,
                },
                terminus_2={**last.metadata.get("terminus_2", {}), "termination_reason": "destructive_control_action"},
            )
        except RuntimeExecutionError as error:
            if last is None or not missing_terminal(error) or not await session.confirm_exit():
                raise
            yield last.with_metadata(
                terminus_2={**last.metadata.get("terminus_2", {}), "termination_reason": "terminal_exited"}
            )

    async def __call__(self, instance, *, sampling_params=None, **kwargs):
        # Same native lifecycle, with the explicit session class above. No global
        # monkeypatch of Ava's TmuxSession or duplicated model/command loop.
        async with nullcontext() if self.starting is None else self.starting:
            started = time.monotonic()
            async with serving(
                instance.runtime_spec.runtime(), self.services, user=self.user, cache=self.cache
            ) as runtime:
                record("sandbox_start", seconds=time.monotonic() - started)
                await runtime.run(["test", "-d", instance.cwd])
                await self.before_run(runtime, instance)
                session = ExitAwareSession(
                    runtime, cwd=instance.cwd, logging_path=self.logging_path, recording_path=self.recording_path
                )
                await session.start()
                try:
                    trace = await self.run(session, instance, sampling_params=sampling_params, **kwargs)
                finally:
                    try:
                        await session.close()
                    finally:
                        await self.after_session(runtime, instance)
                return (await self.after_run(runtime, instance, trace),)
