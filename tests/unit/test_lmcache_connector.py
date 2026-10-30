# SPDX-License-Identifier: Apache-2.0
"""Prefix and ownership boundaries of the explicit cache connector."""

from types import SimpleNamespace

import pytest
import torch

from fragpack.integrations.lmcache.connector import PrefixCacheConnector, PrefixMatch
from fragpack.integrations.lmcache.identity import CacheIdentity


def test_prefix_match_stops_at_gap_and_retains_tenant_identity():
    identity = CacheIdentity("model-rev", 7, 1, 1)
    seen = []

    def lookup(keys):
        seen.extend(keys)
        return 1

    connector = PrefixCacheConnector(
        identity, SimpleNamespace(lookup_prefix=lookup), None
    )
    match = connector.lookup(list(range(300)), cache_salt="tenant-a")
    assert match.token_count == 128
    assert len(seen) == 2
    assert match.keys == (seen[0],)
    assert all(key.cache_salt == "tenant-a" for key in seen)
    assert connector.lookup([1] * 127).keys == ()


def test_store_excludes_partial_tail_and_preserves_refusal():
    identity = CacheIdentity("model-rev", 7, 1, 1)
    submitted = []

    def submit(keys, blocks):
        submitted.append((keys, blocks.tolist()))
        return None

    connector = PrefixCacheConnector(
        identity, None, SimpleNamespace(submit_store=submit)
    )
    assert connector.store_complete(list(range(257)), torch.tensor([9, 3, 6])) is None
    assert submitted[0][1] == [9, 3]
    assert len(submitted[0][0]) == 2
    with pytest.raises(ValueError, match="physical block"):
        connector.store_complete(list(range(257)), torch.tensor([9, 3]))


def test_restore_rejects_foreign_identity_and_incomplete_destinations():
    identity = CacheIdentity("model-rev", 7, 1, 1)
    foreign = CacheIdentity("other-rev", 7, 1, 1)
    connector = PrefixCacheConnector(identity, None, None)
    with pytest.raises(ValueError, match="different cache identity"):
        connector.restore(
            PrefixMatch(tuple(foreign.block_keys([1] * 128))), torch.tensor([1])
        )
    with pytest.raises(ValueError, match="destination"):
        connector.restore(
            PrefixMatch(tuple(identity.block_keys([1] * 128))), torch.tensor([])
        )


def test_later_store_batch_preserves_full_prefix_hash_and_physical_mapping():
    identity = CacheIdentity("model-rev", 7, 1, 1)
    submitted = []
    connector = PrefixCacheConnector(
        identity,
        None,
        SimpleNamespace(
            submit_store=lambda keys, blocks: submitted.append((keys, blocks.tolist()))
        ),
    )
    tokens = list(range(513))
    physical = torch.tensor([9, 4, 7, 2, 1])
    connector.store_complete(tokens, physical, start_block=2, max_blocks=2)
    assert submitted == [(identity.block_keys(tokens)[2:4], [7, 2])]
    assert connector.store_complete(tokens, physical, start_block=4) is None
    assert len(submitted) == 1
    for options in ({"start_block": -1}, {"start_block": 5}, {"max_blocks": 0}):
        with pytest.raises(ValueError):
            connector.store_complete(tokens, physical, **options)


@pytest.mark.parametrize("count", [-1, 3, True, "1"])
def test_lookup_rejects_invalid_remote_prefix_count(count):
    identity = CacheIdentity("model-rev", 7, 1, 1)
    connector = PrefixCacheConnector(
        identity, SimpleNamespace(lookup_prefix=lambda keys: count), None
    )
    with pytest.raises(ValueError, match="prefix count"):
        connector.lookup([1] * 256)
