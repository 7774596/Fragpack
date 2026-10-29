# SPDX-License-Identifier: Apache-2.0
"""Hand-off between the FragPack attention backend and the KV connector.

Both run in a vLLM worker without importing each other. The backend records
its page geometry when vLLM sizes the KV cache (before the cache is registered
with LMCache) and publishes its store once it creates it, on the first
metadata build. The connector reads both: the geometry becomes layout hints
at registration, and the store's blob tables are shared with the cache server
before the first store or retrieve.

This module imports no CUDA extension, so the connector may import it
whether or not FragPack is in use.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector.kv_format.types import LayoutHints
    from lmcache.v1.lossless_kv.fragpack.hbm_kv import HbmKvStore

# ``kv_codec`` layout-hint values of FragPack pages with BF16 / FP8 records.
KV_CODEC_BF16 = "fragpack"
KV_CODEC_FP8 = "fragpack-fp8"

_GEOMETRY: tuple[int, bool] | None = None
_STORE: HbmKvStore | None = None


def record_geometry(kv_heads: int, fp8: bool) -> None:
    """Record that this process runs FragPack attention, and its pages.

    Args:
        kv_heads: KV heads per layer on this rank.
        fp8: Whether records hold FP8 E4M3 KV.

    Returns:
        None.

    Async/thread-safety:
        Called while vLLM sizes the KV cache; plain write.
    """
    global _GEOMETRY
    _GEOMETRY = (kv_heads, fp8)


def layout_hints() -> LayoutHints:
    """Return the layout hints announcing FragPack pages, if any.

    Returns:
        ``{"kv_codec": ..., "num_kv_heads": H}`` once
        :func:`record_geometry` ran in this process, else ``{}``.

    Async/thread-safety:
        Lock-free read.
    """
    if _GEOMETRY is None:
        return {}
    kv_heads, fp8 = _GEOMETRY
    return {
        "kv_codec": KV_CODEC_FP8 if fp8 else KV_CODEC_BF16,
        "num_kv_heads": kv_heads,
    }


def register_store(store: HbmKvStore) -> None:
    """Publish the store attention now reads; replaces any earlier one.

    Args:
        store: The new store.

    Returns:
        None.

    Async/thread-safety:
        Call from the forward thread when the store is (re)created.
    """
    global _STORE
    _STORE = store


def active_store() -> HbmKvStore | None:
    """Return the store published by :func:`register_store`.

    Returns:
        The current store, or None before attention created one.

    Async/thread-safety:
        Lock-free read; the reference is replaced only on KV re-binding.
    """
    return _STORE


__all__ = [
    "KV_CODEC_BF16",
    "KV_CODEC_FP8",
    "active_store",
    "layout_hints",
    "record_geometry",
    "register_store",
]
