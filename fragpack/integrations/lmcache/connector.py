# SPDX-License-Identifier: Apache-2.0
"""Explicit prefix-cache operations over FragPack's closed block interface.

This connector is called after all layers have encoded a block. It does not
register a vLLM scheduler hook or infer ownership from GPU address tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from fragpack.integrations.lmcache.identity import BlockKey, CacheIdentity

if TYPE_CHECKING:
    import torch

    from fragpack.integrations.lmcache.client import BlobClient
    from fragpack.integrations.lmcache.transfer import (
        BlockTransfer,
        RestoreTicket,
        StoreTicket,
    )


@dataclass(frozen=True)
class PrefixMatch:
    """Contiguous external hits, not permission to read destination KV yet.

    ``keys`` retain prefix order. A lookup is advisory: eviction may turn a
    later restore into a miss. Only successful restore completion permits use.
    """

    keys: tuple[BlockKey, ...]

    @property
    def token_count(self) -> int:
        """Number of complete tokens represented by this matched prefix."""
        return 128 * len(self.keys)


class PrefixCacheConnector:
    """Connect token-prefix identity to explicit GPU block transfers.

    Args:
        identity: Exact model, table, geometry and shard identity.
        client: LMCache byte client. Lookups may block on CPU transport and
            belong outside the model's per-layer attention path.
        transfer: Bounded transfer worker for the same identity and GPU store.

    This facade does not own the caller's GPU blocks. Store callers retain
    source blocks until ``ticket.source_ready`` completes. Restore callers
    retain exclusive destination ownership until ``ticket.completion`` resolves,
    and use only its successful contiguous prefix. A hit alone is insufficient.
    """

    def __init__(
        self, identity: CacheIdentity, client: BlobClient, transfer: BlockTransfer
    ) -> None:
        self.identity = identity
        self.client = client
        self.transfer = transfer

    def lookup(self, token_ids: Sequence[int], *, cache_salt: str = "") -> PrefixMatch:
        """Look up only complete blocks and return the contiguous prefix.

        Tokens after the first miss cannot be reused even if individually
        present. The remote implementation must return a bounded prefix count.
        """
        keys = self.identity.block_keys(token_ids, cache_salt=cache_salt)
        if not keys:
            return PrefixMatch(())
        count = self.client.lookup_prefix(keys)
        if type(count) is not int or not 0 <= count <= len(keys):
            raise ValueError("invalid contiguous-prefix count from cache service")
        return PrefixMatch(tuple(keys[:count]))

    def store_complete(
        self,
        token_ids: Sequence[int],
        block_ids: torch.Tensor,
        *,
        cache_salt: str = "",
        start_block: int = 0,
        max_blocks: int | None = None,
    ) -> StoreTicket | None:
        """Submit complete source blocks, leaving an optional final tail local.

        ``block_ids`` is the caller's ordered CUDA int32 physical block list
        for these tokens, including a final partial block if present. All full
        blocks must already be encoded across all layers. ``None`` means no
        full block or immediate pool refusal; no source ownership transfers in
        that case. ``start_block`` and ``max_blocks`` select a bounded batch
        from this full prefix. Keep the complete token and physical-block lists
        when selecting a later batch: hashes must include preceding tokens.
        The caller schedules subsequent batches after pool space becomes free.
        """
        expected = (len(token_ids) + 127) // 128
        if block_ids.ndim != 1 or block_ids.numel() != expected:
            raise ValueError("one ordered physical block is required per token block")
        keys = self.identity.block_keys(token_ids, cache_salt=cache_salt)
        if type(start_block) is not int or not 0 <= start_block <= len(keys):
            raise ValueError("start_block must select a complete block boundary")
        if max_blocks is not None and (type(max_blocks) is not int or max_blocks < 1):
            raise ValueError("max_blocks must be a positive integer or None")
        stop = len(keys) if max_blocks is None else start_block + max_blocks
        selected = keys[start_block:stop]
        if not selected:
            return None
        return self.transfer.submit_store(
            selected, block_ids[start_block : start_block + len(selected)]
        )

    def restore(
        self, match: PrefixMatch, destination_blocks: torch.Tensor
    ) -> RestoreTicket | None:
        """Restore a matched prefix into allocated, exclusive closed destinations.

        Destination blocks must be non-open and unavailable to attention until
        the returned completion succeeds. A partial failure requires recompute
        from the first unsuccessful block, even when earlier layers were copied.
        No GPU operation is submitted for an empty match or pool refusal.
        """
        if any(key.namespace != self.identity.namespace for key in match.keys):
            raise ValueError("prefix match belongs to a different cache identity")
        if destination_blocks.ndim != 1 or destination_blocks.numel() != len(
            match.keys
        ):
            raise ValueError("one destination block is required per matched key")
        if not match.keys:
            return None
        return self.transfer.submit_restore(match.keys, destination_blocks)
