# SPDX-License-Identifier: Apache-2.0
"""FragPack-owned server module loaded by stock LMCache 0.5.6."""

from __future__ import annotations

from typing import Any

from fragpack.integrations.lmcache.client import (
    MAX_PREFIX_KEYS,
    RPC_METHOD,
    BlobClient,
    require_lmcache_056,
)
from fragpack.integrations.lmcache.identity import BlockKey, CacheIdentity
from fragpack.integrations.lmcache.protocol import decode_message, encode_message
from fragpack.integrations.lmcache.storage import LMCacheL1Store


class FragPackServerModule:
    """Validate a bounded RPC envelope and delegate to a compressed-object store.

    The caller owns the shared server context. CPU-only handlers return after
    storage completion; callers should submit them from a background worker.
    Blob bytes are never interpreted as ordinary dense engine KV tensors.
    """

    def __init__(
        self, context: Any, store: BlobClient, identity: CacheIdentity
    ) -> None:
        self._context = context
        self._store = store
        self._identity = identity

    @property
    def context(self) -> Any:
        """Return the shared LMCache context required by EngineModule."""
        return self._context

    def handle(self, data: bytes) -> bytes:
        """Handle one bounded namespaced request; exceptions become RPC errors."""
        header, payload = decode_message(data, self._identity.max_blob_bytes)
        op = header.get("op")
        if op in ("put", "get"):
            key = BlockKey.from_dict(header.get("key"))
            if key.namespace != self._identity.namespace:
                raise ValueError("Block key belongs to another cache identity")
            if op == "put":
                return encode_message({"stored": self._store.put(key, payload)})
            if payload:
                raise ValueError("Get request must not contain object bytes")
            blob = self._store.get(key)
            return encode_message({"found": blob is not None}, blob or b"")
        if op in ("lookup", "probe"):
            raw_keys = header.get("keys")
            if (
                payload
                or not isinstance(raw_keys, list)
                or len(raw_keys) > MAX_PREFIX_KEYS
            ):
                raise ValueError("Lookup requires a bounded key list and no payload")
            keys = [BlockKey.from_dict(value) for value in raw_keys]
            if any(key.namespace != self._identity.namespace for key in keys):
                raise ValueError("Prefix contains a different cache identity")
            if op == "probe":
                return encode_message({"present": list(self._store.probe_many(keys))})
            return encode_message({"prefix_blocks": self._store.lookup_prefix(keys)})
        raise ValueError("Unknown FragPack blob operation")

    def report_status(self) -> dict:
        """Expose the explicitly limited integration mode to LMCache status."""
        return {
            "fragpack": {
                "namespace": self._identity.namespace,
                "lmcache_version": "0.5.6",
                "storage_mode": "cpu-l1-only",
                "transport": "bounded-bytes-rpc",
                "storage": (
                    self._store.report_status()
                    if isinstance(self._store, LMCacheL1Store)
                    else {}
                ),
            }
        }

    def close(self) -> None:
        """Release only plugin-owned state; the shared manager belongs to LMCache."""
        if isinstance(self._store, LMCacheL1Store):
            self._store.close()


def build_server_modules(build_context: Any) -> FragPackServerModule:
    """Build a module using LMCache's official external-module factory hook.

    ``config`` requires an ``identity`` object with CacheIdentity fields and
    optionally a positive ``timeout_s``. Only CPU L1 without configured L2 is
    supported. Installing this module does not register a vLLM connector.
    """
    require_lmcache_056()
    from lmcache.v1.multiprocess.ext_server_module import server_module_handler

    config = build_context.config
    unknown = set(config) - {"identity", "timeout_s"}
    if unknown:
        raise ValueError(f"Unsupported FragPack plugin options: {sorted(unknown)}")
    identity = CacheIdentity(**config["identity"])
    context = build_context.server_context
    store = LMCacheL1Store(
        context.storage_manager, identity, config.get("timeout_s", 30)
    )

    class RegisteredFragPackModule(FragPackServerModule):
        @server_module_handler(RPC_METHOD)
        def dispatch(self, payload: bytes) -> bytes:
            """Route one public LMCache extension call to the FragPack protocol."""
            return self.handle(payload)

    return RegisteredFragPackModule(context, store, identity)
