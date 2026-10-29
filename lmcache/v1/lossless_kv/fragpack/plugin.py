# SPDX-License-Identifier: Apache-2.0
"""vLLM general plugin: register the FragPack backend as the CUSTOM backend.

Expose it through a ``vllm.general_plugins`` entry point
(``lmcache_fragpack = lmcache.v1.lossless_kv.fragpack.plugin:register``).
"""

# Future
from __future__ import annotations


def register() -> None:
    """Register ``backend.FragPackAttentionBackend`` as CUSTOM.

    Returns:
        None.

    Async/thread-safety:
        Called by vLLM in every process at startup; only records a class path,
        so it imports nothing heavy.
    """
    # Third Party
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    register_backend(
        AttentionBackendEnum.CUSTOM,
        "lmcache.v1.lossless_kv.fragpack.backend.FragPackAttentionBackend",
    )
