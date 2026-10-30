# SPDX-License-Identifier: Apache-2.0
"""Benchmark-only immutable native-layout Raw archive and direct page copies."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np
import torch


def native_archive_bits(content: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Return owned contiguous [L,2,N,T,H,D] BF16 bits from captured native blocks."""
    if (
        content.dtype != np.uint16
        or content.ndim != 6
        or content.shape[2] != 2
        or any(size < 1 for size in content.shape)
        or indices.ndim != 1
        or len(indices) < 1
        or not np.issubdtype(indices.dtype, np.integer)
        or np.any(indices < 0)
        or np.any(indices >= len(content))
    ):
        raise ValueError("expected uint16 [N,L,2,T,H,D] and valid integer indices")
    return np.ascontiguousarray(content[indices].transpose(1, 2, 0, 3, 4, 5))


def plan_native_runs(
    slots: tuple[int, ...], blocks: int, capacity: int
) -> tuple[tuple[int, int, int], ...]:
    """Validate all targets, then return (source block, target block, run length)."""
    if (
        blocks < 1
        or capacity < 1
        or len(slots) != blocks
        or any(
            isinstance(slot, (bool, np.bool_))
            or not isinstance(slot, Integral)
            or not 0 <= slot < capacity
            for slot in slots
        )
        or len(set(slots)) != blocks
    ):
        raise ValueError("targets must be complete, integral, in range and disjoint")
    runs = []
    begin = 0
    for end in range(1, blocks + 1):
        if end == blocks or slots[end] != slots[end - 1] + 1:
            runs.append((begin, int(slots[begin]), end - begin))
            begin = end
    return tuple(runs)


def _checked_runs(
    caches: list[torch.Tensor], archive: torch.Tensor, slots: tuple[int, ...]
) -> tuple[tuple[int, int, int], ...]:
    if (
        archive.ndim != 6
        or archive.shape[1] != 2
        or any(size < 1 for size in archive.shape)
        or archive.dtype != torch.bfloat16
        or not archive.is_contiguous()
        or len(caches) != archive.shape[0]
    ):
        raise ValueError("archive must be contiguous BF16 [L,2,N,T,H,D]")
    capacity = caches[0].shape[1] if caches[0].ndim == 5 else 0
    expected = (2, capacity, *archive.shape[3:])
    owners = {(archive.device, archive.untyped_storage().data_ptr())}
    for cache in caches:
        owner = (cache.device, cache.untyped_storage().data_ptr())
        if (
            tuple(cache.shape) != expected
            or cache.dtype != torch.bfloat16
            or not cache.is_contiguous()
            or cache.device != caches[0].device
            or owner in owners
        ):
            raise ValueError(
                "native targets must have matching disjoint contiguous storage"
            )
        owners.add(owner)
    return plan_native_runs(slots, archive.shape[2], capacity)


def _submit_copies(
    caches: list[torch.Tensor],
    archive: torch.Tensor,
    runs: tuple[tuple[int, int, int], ...],
) -> int:
    for layer, cache in enumerate(caches):
        for kv in range(2):
            for source, target, count in runs:
                cache[kv, target : target + count].copy_(
                    archive[layer, kv, source : source + count], non_blocking=True
                )
    return len(caches) * 2 * len(runs)


def copy_native_runs(
    caches: list[torch.Tensor], archive: torch.Tensor, slots: tuple[int, ...]
) -> int:
    """Enqueue validated copies; no GPU storage allocation or host wait.

    Return the number of copy submissions; planning and tensor views use CPU objects.

    CPU tensors are supported for layout tests. CUDA callers retain every owner
    until the current stream completes; pinned CPU sources enable asynchronous H2D.
    All layouts and targets are validated before any target write.
    """
    return _submit_copies(caches, archive, _checked_runs(caches, archive, slots))


def restore_native_raw(
    runtime: Any, session: Any, slots: tuple[int, ...], *, validate: bool
) -> dict[str, int]:
    """Restore one immutable archive on the runtime stream, then wait once."""
    raw = session.raw
    if (
        session.raw_layout != "native-direct"
        or raw.device.type != "cpu"
        or raw.dtype != torch.uint8
        or raw.ndim != 1
        or not raw.is_contiguous()
        or not raw.is_pinned()
    ):
        raise ValueError("direct restore requires owned pinned native-layout raw bytes")
    caches = runtime.pages.caches
    shape = (len(caches), 2, session.blocks, *caches[0].shape[2:])
    if raw.numel() != int(np.prod(shape)) * 2:
        raise ValueError("native raw archive length does not match complete pages")
    archive = raw.view(torch.bfloat16).view(shape)
    runs = _checked_runs(caches, archive, slots)
    try:
        with torch.cuda.stream(runtime.stream):
            if validate:
                runtime.read_ids_host.numpy()[: session.blocks] = slots
                runtime.read_ids[: session.blocks].copy_(
                    runtime.read_ids_host[: session.blocks], non_blocking=True
                )
                runtime.poison(runtime.read_ids[: session.blocks])
            calls = _submit_copies(caches, archive, runs)
            runtime.restored.record(runtime.stream)
            runtime.restored.synchronize()
    except BaseException:
        # Drain partially submitted copies before archive/target owners can unwind.
        runtime.stream.synchronize()
        raise
    if validate:
        for begin in range(0, session.blocks, runtime.chunk):
            count = min(runtime.chunk, session.blocks - begin)
            runtime.validate_chunk(
                session, begin, count, ids=runtime.read_ids[begin : begin + count]
            )
    return {
        "payload_h2d_bytes": raw.numel(),
        "restore_metadata_h2d_bytes": 0,
        "status_d2h_bytes": 0,
        "chunks": 0,
        "restore_host_waits": 1,
        "raw_transfer_runs": len(runs),
        "raw_copy_calls": calls,
    }


def validate_direct_restore(runtime: Any) -> dict[str, Any]:
    """Probe changing complete archives/maps and inactive canaries outside timing."""
    probes = []
    for index, session in enumerate(runtime.sessions[:3]):
        count = session.blocks
        if index % 3 == 0:
            slots = tuple(range(count))
        elif index % 3 == 1:
            slots = tuple(reversed(range(count)))
        else:
            slots = tuple((value + 1) % runtime.blocks for value in range(count))
        inactive = next((i for i in range(runtime.blocks) if i not in slots), None)
        before = (
            runtime.snapshot_inactive_block(inactive) if inactive is not None else []
        )
        runtime.output.fill_(float("nan"))
        result = restore_native_raw(runtime, session, slots, validate=True)
        after = (
            runtime.snapshot_inactive_block(inactive) if inactive is not None else []
        )
        if not all(torch.equal(a, b) for a, b in zip(before, after, strict=True)):
            raise AssertionError("direct restore changed inactive destination")
        if not bool(torch.isnan(runtime.output).all()):
            raise AssertionError("restore modified attention output before attention")
        probes.append(
            {
                "blocks": count,
                "context": session.context,
                "map": list(slots),
                "inactive_block": inactive,
                "inactive_intact": True if inactive is not None else None,
                "all_layer_kv_bits_exact": True,
                **result,
            }
        )
    return {
        "status": "passed",
        "effective_dispatch": "native-direct-copy",
        "probes": probes,
        "scope": "functional, excluded from timed replay",
    }


def validate_direct_rejection(runtime: Any) -> dict[str, Any]:
    """Reject bad CPU metadata before any DMA and preserve all poisoned pages."""
    session = runtime.sessions[0]
    good = tuple(range(session.blocks))
    bad = [good[:-1] + (runtime.blocks,), good[:-1] + (True,), good[:-1]]
    if session.blocks > 1:
        bad.append(good[:-1] + (good[0],))
    for cache in runtime.pages.caches:
        cache.view(torch.int16).fill_(0x6B6B)
    runtime.output.fill_(float("nan"))
    for slots in bad:
        try:
            restore_native_raw(runtime, session, slots, validate=False)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid direct map was accepted")
    if not all(
        bool(torch.all(c.view(torch.int16) == 0x6B6B)) for c in runtime.pages.caches
    ):
        raise AssertionError("invalid direct map wrote a target")
    if not bool(torch.isnan(runtime.output).all()):
        raise AssertionError("invalid direct map changed attention output")
    return {
        "status": "passed",
        "invalid_maps_checked": len(bad),
        "all_target_pages_unchanged": True,
        "attention_output_remained_poisoned": True,
        "error_boundary": "CPU validation before first DMA; no codec GPU status",
        "scope": "functional, excluded from timed replay",
    }
