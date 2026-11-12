"""Retain empty judge responses without changing native request or grading."""

import json
import os
import secrets
from pathlib import Path

from ava_core.core.client import OpenAIClient
from ava_core.core.model import kind


@kind("ava_observed_judge")
class ObservedJudgeClient(OpenAIClient):
    async def post(self, path, payload, headers):
        response = await super().post(path, payload, headers)
        if not response["choices"][0]["message"].get("content"):
            destination = Path(os.environ["AVA_RUNTIME_LOG_DIR"]) / f"empty-judge-{secrets.token_hex(16)}.json"
            with os.fdopen(os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as out:
                json.dump({"request": payload, "response": response}, out, ensure_ascii=False)
        return response
