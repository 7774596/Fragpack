# SPDX-License-Identifier: Apache-2.0
"""Blocking blob client for LMCache's public server-module RPC envelope.

Calls belong on the transfer worker, not the model's CUDA producer path.
The caller owns the messaging client and its shutdown. This transport copies
bounded compressed bytes; it does not register dense KV or claim zero-copy I/O.
"""

from __future__ import annotations

from importlib import metadata
import math
import threading
from typing import Any, Protocol, Sequence

from fragpack.integrations.lmcache.identity import BlockKey, CacheIdentity
from fragpack.integrations.lmcache.protocol import (
    canonicalize_blob,
    decode_message,
    encode_message,
)

RPC_METHOD = "fragpack.blob.v1"
# A 128-character salt can expand to 1,536 JSON bytes (escaped surrogate pairs).
# This bound fits even those keys into the protocol's 64 KiB metadata frame.
MAX_PREFIX_KEYS = 32


def require_lmcache_056() -> None:
    """Reject missing or differently versioned LMCache installations."""
    try:
        installed = metadata.version("lmcache")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("This integration requires LMCache 0.5.6") from exc
    if installed != "0.5.6":
        raise RuntimeError(f"Expected LMCache 0.5.6, found {installed}")


def _load_request_type() -> Any:
    from lmcache.v1.multiprocess.protocols.server_module import ServerModuleCallRequest

    return ServerModuleCallRequest


class BlobClient(Protocol):
    """Synchronous compressed-object operations used by a background worker."""

    def put(self, key: BlockKey, blob: bytes) -> bool:
        """Return true only after matching bytes are confirmed resident."""
        ...

    def get(self, key: BlockKey) -> bytes | None:
        """Copy a resident object, or return None for a miss."""
        ...

    def lookup_prefix(self, keys: Sequence[BlockKey]) -> int:
        """Return the number of consecutive resident keys from the beginning."""
        ...


class RemoteBlobClient:
    """Use a caller-owned LMCache 0.5.6 RequestClient for FPKB objects.

    Args:
        messaging_client: Public LMCache RequestClient supporting
            ``server_module_call`` and MessagingFuture.result.
        identity: Exact model, format, table and shard identity.
        timeout_s: Positive timeout per RPC. A timeout is an uncertain outcome,
            never publication or import completion.

    Calls are blocking and serialized: at most one RPC may be in flight per
    RemoteBlobClient. An uncertain send/wait failure permanently stops this
    wrapper because a MessagingFuture timeout does not cancel the request.
    The caller must close the underlying messaging client before reconnecting;
    making a new wrapper over that same unclosed client does not clean it up.
    Application-level rejection of a completed RPC does not disable the client.
    No torch or LMCache imports occur until construction.
    """

    def __init__(
        self,
        messaging_client: Any,
        identity: CacheIdentity,
        timeout_s: float = 30.0,
    ) -> None:
        require_lmcache_056()
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError("RPC timeout must be finite and positive")
        self._messaging = messaging_client
        self._request_type = _load_request_type()
        self._identity = identity
        self._timeout_s = timeout_s
        self._rpc_lock = threading.Lock()
        self._failed = threading.Event()
        self._pending_request: Any | None = None
        self._pending_future: Any | None = None

    @property
    def failed(self) -> bool:
        """Whether an uncertain transport outcome has disabled further RPCs.

        Reading this state never waits for an in-flight RPC. A true value does
        not imply cancellation or rollback; the caller still owns shutdown of
        the underlying messaging client and any request it retains.
        """
        return self._failed.is_set()

    def _check_active(self) -> None:
        if self.failed:
            raise RuntimeError(
                "FragPack RPC client failed; close its messaging client "
                "before reconnecting"
            )

    def _call(self, header: dict, payload: bytes = b"") -> tuple[dict, bytes]:
        with self._rpc_lock:
            self._check_active()
            request = self._request_type(
                method=RPC_METHOD, payload=encode_message(header, payload)
            )
            self._pending_request = request
            try:
                self._pending_future = self._messaging.server_module_call(request)
                response = self._pending_future.result(self._timeout_s)
            except BaseException:
                # Preserve the single uncertain request/future until caller
                # shutdown. Retrying here could accumulate requests inside
                # LMCache even after the transfer worker releases its slot.
                self._failed.set()
                raise
            self._pending_request = None
            self._pending_future = None
            if not response.success:
                raise RuntimeError(f"FragPack server rejected RPC: {response.error}")
            return decode_message(response.payload, self._identity.max_blob_bytes)

    def _key(self, key: BlockKey) -> dict:
        if key.namespace != self._identity.namespace:
            raise ValueError("Block key belongs to another cache identity")
        return key.to_dict()

    def put(self, key: BlockKey, blob: bytes) -> bool:
        """Publish validated bytes and report the server's confirmed outcome."""
        self._check_active()
        blob = canonicalize_blob(blob, self._identity)
        header, payload = self._call({"op": "put", "key": self._key(key)}, blob)
        if payload or type(header.get("stored")) is not bool:
            raise ValueError("Malformed put response")
        return header["stored"]

    def get(self, key: BlockKey) -> bytes | None:
        """Return canonical compressed bytes after independent validation."""
        self._check_active()
        header, payload = self._call({"op": "get", "key": self._key(key)})
        if type(header.get("found")) is not bool:
            raise ValueError("Malformed get response")
        if not header["found"]:
            if payload:
                raise ValueError("A cache miss must not carry a payload")
            return None
        return canonicalize_blob(payload, self._identity)

    def lookup_prefix(self, keys: Sequence[BlockKey]) -> int:
        """Query bounded groups, stopping at the first missing prefix block.

        A hit is advisory: eviction may occur before get. Retrieval must still
        handle a miss and must not report readiness before successful import.
        """
        self._check_active()
        found = 0
        for start in range(0, len(keys), MAX_PREFIX_KEYS):
            group = keys[start : start + MAX_PREFIX_KEYS]
            header, payload = self._call(
                {"op": "lookup", "keys": [self._key(key) for key in group]}
            )
            count = header.get("prefix_blocks")
            if payload or type(count) is not int or not 0 <= count <= len(group):
                raise ValueError("Malformed prefix lookup response")
            found += count
            if count < len(group):
                break
        return found
