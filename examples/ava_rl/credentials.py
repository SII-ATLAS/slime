"""Read literal, private runtime configuration without executing shell code."""

import os
import shlex
from pathlib import Path

ALLOWED = {
    "AWS_ENDPOINT_URL",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "OPEN_SANDBOX_DOMAIN",
    "OPEN_SANDBOX_API_KEY",
    "AVACORE_APPLICATION_CACHE",
}


def configure():
    """Load only Ava runtime credentials in the isolated CPU service process."""
    if not (filename := os.environ.get("AVA_ENV_FILE")):
        return
    path = Path(filename)
    if path.stat().st_mode & 0o077:
        raise PermissionError("AVA_ENV_FILE must not be readable by group or other")
    for line in path.read_text().splitlines():
        words = shlex.split(line, comments=True)
        if words and words[0] == "export":
            words = words[1:]
        for word in words:
            name, sep, value = word.partition("=")
            if sep and name in ALLOWED:
                if "$" in value or "`" in value:
                    raise ValueError("AVA_ENV_FILE accepts literal values only")
                os.environ[name] = value
