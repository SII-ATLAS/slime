"""Optional append-only timing observations; unset AVA_TIMING_DIR to disable."""

import json
import os
import time
from pathlib import Path


def record(event, **fields):
    directory = os.environ.get("AVA_TIMING_DIR")
    if not directory:
        return
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        data = json.dumps({"time": time.time(), "event": event, **fields}) + "\n"
        fd = os.open(Path(directory) / f"timing-{os.getpid()}.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, data.encode())
        finally:
            os.close(fd)
    except OSError:
        # Optional observability must not invalidate a training trajectory.
        pass
