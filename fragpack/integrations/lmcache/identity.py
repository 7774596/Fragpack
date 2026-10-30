# SPDX-License-Identifier: Apache-2.0
"""Stable identities for complete FragPack prefix blocks.

These pure host-side helpers do not import LMCache, vLLM, or CUDA. Namespace
identity describes the representation; the tenant salt remains a separate
field when a key is adapted to an external storage implementation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import struct

from fragpack.compression import attn_page as ap

_TOKEN = struct.Struct("<Q")
_HASH_DOMAIN = b"fragpack-prefix-v1\x00"


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    if any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _salt(value: object) -> str:
    # Match ObjectKey's public 0.5.6 salt constraints without importing LMCache.
    if not isinstance(value, str) or len(value) > 128:
        raise ValueError("cache_salt must be a string of at most 128 characters")
    if any(c in value for c in "@/\\\x00"):
        raise ValueError("cache_salt contains a storage-key separator")
    return value


@dataclass(frozen=True)
class BlockKey:
    """One representation-specific, tenant-scoped chained prefix key.

    Args:
        namespace: SHA-256 of the complete :class:`CacheIdentity`.
        prefix_hash: SHA-256 of all complete token blocks through this block.
        cache_salt: Tenant isolation label, preserved by LMCache adaptation.

    Instances are immutable and safe to share between transfer threads.
    """

    namespace: str
    prefix_hash: str
    cache_salt: str = ""

    def __post_init__(self) -> None:
        _digest(self.namespace, "namespace")
        _digest(self.prefix_hash, "prefix_hash")
        _salt(self.cache_salt)

    def to_dict(self) -> dict[str, str]:
        """Return a fresh JSON-compatible key description without I/O."""
        return {
            "namespace": self.namespace,
            "prefix_hash": self.prefix_hash,
            "cache_salt": self.cache_salt,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> BlockKey:
        """Parse exactly the three public key fields, rejecting malformed keys."""
        if not isinstance(value, Mapping) or set(value) != {
            "namespace",
            "prefix_hash",
            "cache_salt",
        }:
            raise ValueError("block key requires namespace, prefix_hash and cache_salt")
        return cls(
            namespace=_digest(value["namespace"], "namespace"),
            prefix_hash=_digest(value["prefix_hash"], "prefix_hash"),
            cache_salt=_salt(value["cache_salt"]),
        )


@dataclass(frozen=True)
class CacheIdentity:
    """Immutable FPKB v1 identity for a model's 128-token/128-dimension KV.

    Args:
        model_revision: Exact model and relevant execution configuration revision;
            a mutable model alias alone is insufficient to isolate incompatible KV.
        table_digest: Signed int64 digest exposed by ``HbmKvStore.table_digest``.
        num_layers: Layers represented in each exported blob.
        kv_heads: KV heads in this TP shard, from 1 through the core limit of 16.
        rank: This shard's zero-based TP rank.
        world_size: Number of TP shards participating in the model.

    Computing keys performs only host hashing. It neither reads tensors nor
    synchronizes CUDA. Token IDs must be nonnegative Python integers.
    """

    model_revision: str
    table_digest: int
    num_layers: int
    kv_heads: int
    rank: int = 0
    world_size: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.model_revision, str) or not self.model_revision.strip():
            raise ValueError("model_revision must be a nonempty string")
        _integer(self.table_digest, "table_digest", -(1 << 63), (1 << 63) - 1)
        _integer(self.num_layers, "num_layers", 1, (1 << 31) - 1)
        _integer(self.kv_heads, "kv_heads", 1, 16)
        _integer(self.world_size, "world_size", 1, (1 << 31) - 1)
        _integer(self.rank, "rank", 0, self.world_size - 1)

    @property
    def namespace(self) -> str:
        """Return the stable format/model/table/geometry/shard namespace."""
        value = {
            "format": "FPKB",
            "version": 1,
            "block_tokens": ap.PAGE_TOKENS,
            "head_dim": ap.HEAD_DIM,
            "model_revision": self.model_revision,
            "table_digest": self.table_digest,
            "num_layers": self.num_layers,
            "kv_heads": self.kv_heads,
            "rank": self.rank,
            "world_size": self.world_size,
        }
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def max_blob_bytes(self) -> int:
        """Return the all-raw FPKB upper bound, including alignment padding."""
        return ap.blob_slot_bytes(self.num_layers, self.kv_heads)

    def block_keys(
        self, token_ids: Sequence[int], cache_salt: str = ""
    ) -> list[BlockKey]:
        """Hash only complete blocks, chaining every block to its full prefix.

        Args:
            token_ids: Prompt tokens in causal order. Incomplete tails produce
                no key; all supplied IDs are validated, including the tail.
            cache_salt: Tenant label retained in each returned key.

        Returns:
            One key per complete 128-token block, in prefix order. Changing a
            token changes the key of its block and every later block.
        """
        _salt(cache_salt)
        for token in token_ids:
            _integer(token, "token ID", 0, (1 << 64) - 1)
        namespace = self.namespace
        salt_bytes = cache_salt.encode("utf-8")
        previous = hashlib.sha256(
            _HASH_DOMAIN + bytes.fromhex(namespace) + salt_bytes
        ).digest()
        keys = []
        for start in range(0, len(token_ids) - ap.PAGE_TOKENS + 1, ap.PAGE_TOKENS):
            digest = hashlib.sha256(_HASH_DOMAIN + previous)
            for token in token_ids[start : start + ap.PAGE_TOKENS]:
                digest.update(_TOKEN.pack(token))
            previous = digest.digest()
            keys.append(BlockKey(namespace, previous.hex(), cache_salt))
        return keys
