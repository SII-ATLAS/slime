"""Manual integration contracts: run in the pinned private Ava dependency image.

Not registered in public CI because AvaCore is a private dependency.
"""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from ava_core.core import ChatMessage, Trace
from ava_core.generate.terminal import TerminalBenchTerminus2
from ava_core.runtime.runtime import CommandOutput, RuntimeExecutionError
from examples.ava_rl.terminal import ExitAwareSession, ExitAwareTerminal, ValidatedTerminalJSONParser


class TerminalContracts(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_utf8_is_feedback_then_only_corrected_action_executes(self):
        original = json.dumps(
            {"analysis": "test", "plan": "test", "commands": [{"keystrokes": "\udc03", "duration": 0}]}
        )
        corrected = "echo corrected\n"
        replies = [
            original,
            json.dumps({"analysis": "test", "plan": "test", "commands": [{"keystrokes": corrected, "duration": 0}]}),
            json.dumps({"analysis": "done", "plan": "done", "commands": [], "task_complete": True}),
        ]
        feedback = []

        class Model:
            async def step(self, trace, **kwargs):
                feedback.append(trace.messages[-1].text_content)
                return Trace.from_messages(
                    [ChatMessage(role="assistant", content=(replies.pop(0) if len(replies) > 1 else replies[0]))],
                    prev=trace,
                )

        session = SimpleNamespace(
            session_name="test",
            send_keys=AsyncMock(),
            get_incremental_output=AsyncMock(return_value="real terminal output"),
        )
        agent = ExitAwareTerminal(model=Model(), validate_terminal_actions=True, recording_path=None, max_turns=4)
        trace = await agent.run(session, SimpleNamespace(prompt="test", timeout_seconds=5))
        self.assertIn("UTF-8", feedback[1])
        session.send_keys.assert_awaited_once_with(corrected, block=False, min_timeout_sec=0)
        self.assertTrue(any(m.role == "assistant" and m.text_content == original for m in trace.messages))
        self.assertEqual(trace.metadata["terminus_2"]["termination_reason"], "task_complete")

    async def test_bad_actions_rejected_without_sanitizing(self):
        parser = ValidatedTerminalJSONParser()
        for text, duration in [("\0", 0), ("\ud800", 0), ("true", -1), ("true", float("nan"))]:
            result = parser.parse_response(
                json.dumps({"analysis": "x", "plan": "x", "commands": [{"keystrokes": text, "duration": duration}]})
            )
            self.assertTrue(result.error)
            self.assertFalse(result.commands)
        valid = "printf '\\000'; echo 😀\n"
        result = parser.parse_response(
            json.dumps({"analysis": "x", "plan": "x", "commands": [{"keystrokes": valid, "duration": 0}]})
        )
        self.assertFalse(result.error)
        self.assertEqual(result.commands[0].keystrokes, valid)

    async def test_parser_numeric_and_depth_limits_are_feedback(self):
        responses = [
            json.dumps({"analysis": "x", "plan": "x", "commands": [{"keystrokes": "true", "duration": 10**400}]}),
            '{"analysis":"x","plan":"x","commands":[],"unused":' + "[" * 20000 + "0" + "]" * 20000 + "}",
            '{"analysis":"x","plan":"x","commands":[{"keystrokes":"true","duration":' + "1" * 5000 + "}]}",
        ]
        parser = ValidatedTerminalJSONParser()
        for response in responses:
            with self.subTest(size=len(response)):
                result = parser.parse_response(response)
                self.assertIn("limits", result.error)
                self.assertFalse(result.commands)
                self.assertFalse(result.is_task_complete)
        # Programmer errors are not converted into model feedback.
        with patch.object(ValidatedTerminalJSONParser.__bases__[0], "parse_response", side_effect=TypeError("bug")):
            with self.assertRaisesRegex(TypeError, "bug"):
                parser.parse_response("{}")

    async def test_confirmed_shell_exit_and_destructive_action_still_use_native_verifier(self):
        for destructive in (False, True):
            command = "killall execd\n" if destructive else "exit\n"
            response = json.dumps({"analysis": "x", "plan": "x", "commands": [{"keystrokes": command, "duration": 0}]})

            class Model:
                async def step(self, trace, **kwargs):
                    return Trace.from_messages([ChatMessage(role="assistant", content=response)], prev=trace)

            missing = RuntimeExecutionError(
                ["tmux", "send-keys"], exit_code=1, output=CommandOutput("", "can't find session: test")
            )
            runtime = SimpleNamespace(run=AsyncMock(side_effect=["", missing]))
            session = SimpleNamespace(
                session_name="test",
                runtime=runtime,
                send_keys=AsyncMock(side_effect=missing),
                get_incremental_output=AsyncMock(return_value="shell output"),
            )
            session.confirm_exit = lambda: ExitAwareSession.confirm_exit(session)
            agent = ExitAwareTerminal(model=Model(), end_on_destructive_actions=True, max_turns=2, recording_path=None)
            trace = await agent.run(session, SimpleNamespace(prompt="x", timeout_seconds=5))
            reason = "destructive_control_action" if destructive else "terminal_exited"
            self.assertEqual(trace.metadata["terminus_2"]["termination_reason"], reason)
            self.assertEqual(session.send_keys.await_count, 0 if destructive else 1)
            self.assertTrue(any(m.text_content == response for m in trace.messages))
            for score in (0, 1):
                expected = trace.with_metadata(reward={"score": score})
                with patch.object(TerminalBenchTerminus2, "after_run", AsyncMock(return_value=expected)) as verifier:
                    result = await agent.after_run(runtime, None, trace)
                verifier.assert_awaited_once_with(runtime, None, trace)
                self.assertEqual(result.metadata["reward"]["score"], score)

    async def test_shell_exit_requires_healthy_command_channel(self):
        failure = RuntimeExecutionError(["true"], exit_code=None, output=CommandOutput("", "upstream closed"))
        session = SimpleNamespace(runtime=SimpleNamespace(run=AsyncMock(side_effect=failure)), session_name="test")
        with self.assertRaises(RuntimeExecutionError) as raised:
            await ExitAwareSession.confirm_exit(session)
        self.assertIs(raised.exception, failure)

    async def test_verifier_timeout_cancels_and_does_not_make_reward(self):
        trace = Trace.from_messages([ChatMessage(role="assistant", content="original")])
        cancelled = asyncio.Event()

        async def verify(*args):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        with patch.object(TerminalBenchTerminus2, "after_run", verify):
            result = await ExitAwareTerminal(model=None, verifier_timeout_seconds=0.01).after_run(None, None, trace)
        self.assertTrue(cancelled.is_set())
        self.assertEqual(result.metadata["verifier_error"]["type"], "VerifierTimeout")
        self.assertNotIn("reward", result.metadata)

    async def test_verifier_preserves_native_zero_and_one(self):
        trace = Trace.from_messages([ChatMessage(role="assistant", content="original")])
        for score in (0, 1):
            expected = trace.with_metadata(reward={"score": score})
            with patch.object(TerminalBenchTerminus2, "after_run", AsyncMock(return_value=expected)):
                result = await ExitAwareTerminal(model=None, verifier_timeout_seconds=1).after_run(None, None, trace)
            self.assertIs(result, expected)


if __name__ == "__main__":
    unittest.main()
