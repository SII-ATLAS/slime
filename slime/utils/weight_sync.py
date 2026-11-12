"""Cache refresh policy shared by the weight transports."""


def should_flush_cache(
    flush_cache_interval: int,
    weight_version: int,
    initial_weight_version: int = 0,
    *,
    update_weights_interval: int = 1,
) -> bool:
    """Refresh at startup and every N subsequent syncs; nonpositive preserves KV.

    The first publication flushes even when resuming with a nonzero version.
    Refresh frequency counts scheduled publications, relative to version 1.
    A full refresh aborts requests; other updates pause them in place.
    """
    return weight_version == initial_weight_version + 1 or (
        flush_cache_interval > 0 and ((weight_version - 1) // update_weights_interval) % flush_cache_interval == 0
    )
