"""Ava RuntimeSpec extension using OpenSandbox's public transport injection.

Native lifecycle, command payloads, files and verifier remain Ava's. Each runtime
owns its HTTP transport. Opt-in tmux writes use native execution-ID status
instead of foreground completion events; commands are never replayed.
"""

import asyncio
import os
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import httpx
from ava_core.core.model import kind
from ava_core.runtime.runtime import CommandOutput, RuntimeExecutionError
from ava_core.runtime.sandbox import OpenSandbox, OpenSandboxSpec


class RelayedOpenSandbox(OpenSandbox):
    async def start(self):
        path = Path(os.environ["AVA_RELAY_TOKEN_FILE"])
        if path.stat().st_mode & 0o077:
            raise PermissionError("Relay token must be private")
        proxy = httpx.Proxy(os.environ["AVA_RELAY_URL"], auth=("slime", path.read_text().strip()))
        self.relay_transport = httpx.AsyncHTTPTransport(proxy=proxy)
        self.connection_config = self.connection_config.model_copy(update={"transport": self.relay_transport})
        try:
            await super().start()
        except BaseException:
            await self.relay_transport.aclose()
            raise

    async def run(self, argv, env=None, *, cwd=None, user=None, full_output=False):
        if os.environ.get("AVA_TMUX_COMMAND_STATUS", "0") != "1" or tuple(argv[:2]) != ("tmux", "send-keys"):
            return await super().run(argv, env, cwd=cwd, user=user, full_output=full_output)
        # Submit once using Ava's native detached API. A lost foreground SSE
        # completion must never cause a second send-keys (duplicate keystrokes).
        process = await super().run_detached(argv, env, cwd=cwd, user=user)
        try:
            status = await process.wait()
            if not status.running and status.exit_code == 0:
                return CommandOutput("", "") if full_output else ""
            output = await process.logs()
            raise RuntimeExecutionError(
                argv, exit_code=status.exit_code, output=CommandOutput("", output), error=status.error
            )
        except asyncio.CancelledError:
            with suppress(Exception):
                await process.kill()
            raise

    async def stop(self):
        try:
            await super().stop()
        finally:
            if hasattr(self, "relay_transport"):
                await self.relay_transport.aclose()


@kind("ava_relay_opensandbox")
@dataclass(frozen=True)
class RelayedOpenSandboxSpec(OpenSandboxSpec):
    def runtime(self):
        return RelayedOpenSandbox(
            self.image,
            environment_id=self.environment_id,
            project_name=self.project_name,
            connection_config=self.connection_config,
            entrypoint=self.entrypoint,
            envs=self.envs,
            timeout=self.timeout,
            resource=self.resource,
            ready_timeout=self.ready_timeout,
            skip_health_check=self.skip_health_check,
            network_policy=self.network_policy,
            layout=self.layout,
        )
