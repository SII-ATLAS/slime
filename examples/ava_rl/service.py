"""Run the pinned Ava package and this example's explicit extensions."""

import os

from examples.ava_rl.credentials import configure

if __name__ == "__main__":
    configure()
    from ava_core.remote_service import serve

    from examples.ava_rl import budget_client, judge_client, terminal  # noqa: F401

    if os.environ.get("AVA_RELAY_URL"):
        from examples.ava_rl import sandbox_relay  # noqa: F401

    serve(
        "127.0.0.1",
        int(os.environ["AVA_RPC_PORT"]),
        os.environ["AVA_RPC_TOKEN"],
        concurrency=int(os.environ.get("AVA_RPC_CONCURRENCY", "4")),
    )
