# SPDX-License-Identifier: Apache-2.0
"""Independent BF16 page codec; no attention, HBM store or cache-service dependency.

Initialization may allocate and copy metadata. Operations enqueue on the current
CUDA stream, never read device results on the host, and borrow their arguments
until completion. A workspace serves one batch at a time. Callers provide stream
dependencies, retain source/target pages, and inspect device status before using
restored KV. Status zero means success; bits 1/2/4/8 denote header, length,
page-mapping and record-payload errors respectively.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib

import numpy as np
from numpy.typing import NDArray
import torch

from fragpack.compression import attn_page as ap
from fragpack.compression.attn_tables import device_tables
from fragpack.ops.native_codec_ext import extension


@dataclass(frozen=True)
class NativeTensorLayout:
    """Validated native page geometry; strides count BF16 elements."""

    num_pages: int
    page_tokens: int
    kv_heads: int
    strides: tuple[int, ...]
    span_bytes: int


def validate_native_tensor(tensor: torch.Tensor) -> NativeTensorLayout:
    """Validate geometry without accessing data or initializing CUDA.

    Accept CPU or CUDA BF16 ``[2, pages, tokens, heads, 128]`` tensors with
    positive non-overlapping strides, contiguous head dimensions and page sizes
    16/32/64/128. Padding between dimensions is allowed. Return the geometry and
    byte span needed for detecting overlapping layer views.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("native cache must be a torch tensor")
    if tensor.dtype != torch.bfloat16 or tensor.ndim != 5:
        raise ValueError("native cache must be BF16 [2, pages, tokens, heads, 128]")
    kv, pages, tokens, heads, dim = tensor.shape
    if kv != 2 or pages < 1 or tokens not in (16, 32, 64, 128):
        raise ValueError("native cache requires two KV planes and 16/32/64/128 tokens")
    if not 1 <= heads <= 16 or dim != ap.HEAD_DIM:
        raise ValueError("native cache requires 1..16 KV heads and head dimension 128")
    strides = tuple(tensor.stride())
    if min(strides) <= 0 or strides[-1] != 1:
        raise ValueError("native cache requires positive strides and contiguous dim")
    span = 1
    for stride, size in sorted(zip(strides, tensor.shape, strict=True)):
        if size == 1:
            continue
        if stride < span:
            raise ValueError("native cache dimensions must not overlap")
        span += (size - 1) * stride
    return NativeTensorLayout(pages, tokens, heads, strides, span * 2)


def validate_tables(table: NDArray[np.uint8]) -> None:
    """Check per-layer/KV/head table shape and invertibility on the CPU."""
    if not isinstance(table, np.ndarray) or table.dtype != np.uint8:
        raise ValueError("tables must be a uint8 numpy array")
    if (
        table.ndim != 4
        or table.shape[0] < 1
        or table.shape[1] != 2
        or not 1 <= table.shape[2] <= 16
        or table.shape[3] != ap.TABLE_ENTRIES
    ):
        raise ValueError("tables must have shape [layers, 2, heads, 22]")
    ordered = np.sort(table, axis=-1)
    if np.any(ordered[..., 1:] == ordered[..., :-1]):
        raise ValueError("each code table must contain distinct high bytes")


class NativeKVPages:
    """Own native layer tensors and device pointer/stride descriptors.

    Layer views must have disjoint byte spans. Retain this object while work is
    pending. Its pointer descriptors are immutable: never resize or rebind the
    registered storage, even while idle; construct a new registration instead.
    Construction copies a small descriptor; operations only borrow it.
    Cross-stream callers must wait for construction and KV production before use.
    """

    def __init__(self, caches: Sequence[torch.Tensor]) -> None:
        if not caches:
            raise ValueError("at least one native KV layer is required")
        self.caches = tuple(caches)
        layouts = [validate_native_tensor(cache) for cache in self.caches]
        first = layouts[0]
        self.device = self.caches[0].device
        if self.device.type != "cuda":
            raise ValueError("native KV pages must be on CUDA")
        if any(cache.device != self.device for cache in self.caches):
            raise ValueError("all native KV layers must be on the same CUDA device")
        if any(
            (layout.page_tokens, layout.kv_heads) != (first.page_tokens, first.kv_heads)
            for layout in layouts
        ):
            raise ValueError("all native KV layers must have the same token/head shape")
        spans = sorted(
            (cache.data_ptr(), cache.data_ptr() + layout.span_bytes)
            for cache, layout in zip(self.caches, layouts, strict=True)
        )
        if any(
            left[1] > right[0] for left, right in zip(spans, spans[1:], strict=False)
        ):
            raise ValueError("native KV layer views must not overlap")
        self.num_layers = len(self.caches)
        self.page_tokens = first.page_tokens
        self.kv_heads = first.kv_heads
        self.layouts = tuple(layouts)
        rows = [
            [cache.data_ptr(), *layout.strides[:4], layout.num_pages]
            for cache, layout in zip(self.caches, layouts, strict=True)
        ]
        self.descriptors = torch.tensor(rows, dtype=torch.int64, device=self.device)

    def memory_usage(self) -> dict[str, int]:
        """Return unique owned cache-storage and descriptor allocation bytes."""
        storage = {
            cache.untyped_storage().data_ptr(): cache.untyped_storage().nbytes()
            for cache in self.caches
        }
        return {
            "cache_storage": sum(storage.values()),
            "descriptors": self.descriptors.numel() * self.descriptors.element_size(),
        }


class NativeCodec:
    """Bounded workspace for independent native pages / FPKB round trips.

    ``table`` is the CPU uint8 [layers, 2, heads, 22] calibration table.
    ``max_blocks`` bounds the number of logical 128-token blocks per batch.
    Output views belong to this workspace and are overwritten by the next batch.
    One instance cannot process concurrent batches, including on different
    streams. No operation synchronizes or checks GPU status on the host.
    """

    def __init__(
        self,
        table: NDArray[np.uint8],
        *,
        max_blocks: int,
        device: torch.device | str,
    ) -> None:
        validate_tables(table)
        if not isinstance(max_blocks, int) or max_blocks < 1:
            raise ValueError("max_blocks must be a positive integer")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("native codec requires a CUDA device")
        self.device = (
            torch.device("cuda", torch.cuda.current_device())
            if (device.index is None)
            else device
        )
        self.num_layers, _, self.kv_heads, _ = table.shape
        self.max_blocks = max_blocks
        self.tables = device_tables(np.ascontiguousarray(table), self.device)
        ranks = np.full((*table.shape[:3], 256), 255, dtype=np.uint8)
        for index in np.ndindex(table.shape[:3]):
            ranks[index][table[index]] = np.arange(ap.TABLE_ENTRIES, dtype=np.uint8)
        self.digest = int.from_bytes(
            hashlib.blake2b(ranks.tobytes(), digest_size=8).digest(),
            "little",
            signed=True,
        )
        shape = (max_blocks, self.num_layers, 2, self.kv_heads)
        self._records = torch.empty(
            (*shape, ap.RAW_RECORD_BYTES), dtype=torch.uint8, device=self.device
        )
        self._rec_lens = torch.empty(shape, dtype=torch.int32, device=self.device)
        self._rec_offsets = torch.empty(shape, dtype=torch.int64, device=self.device)
        self._offsets = torch.empty(
            max_blocks + 1, dtype=torch.int64, device=self.device
        )
        self._lengths = torch.empty(max_blocks, dtype=torch.int64, device=self.device)
        self._status = torch.empty(max_blocks, dtype=torch.int32, device=self.device)
        self.buffer = torch.empty(
            max_blocks * ap.blob_slot_bytes(self.num_layers, self.kv_heads),
            dtype=torch.uint8,
            device=self.device,
        )
        self._count = 0
        self._stage = "idle"

    @property
    def records(self) -> torch.Tensor:
        """Borrow active fixed-size encoded-record scratch slots."""
        return self._records[: self._count]

    @property
    def rec_lens(self) -> torch.Tensor:
        """Borrow active int32 record lengths past the dense region."""
        return self._rec_lens[: self._count]

    @property
    def rec_offsets(self) -> torch.Tensor:
        """Borrow active int64 record offsets relative to each FPKB blob."""
        return self._rec_offsets[: self._count]

    @property
    def offsets(self) -> torch.Tensor:
        """Borrow batch offsets; the last device value is total transfer bytes."""
        return self._offsets[: self._count + 1]

    @property
    def lengths(self) -> torch.Tensor:
        """Borrow active aligned FPKB blob lengths."""
        return self._lengths[: self._count]

    @property
    def status(self) -> torch.Tensor:
        """Borrow per-block status; read it only after the operation completes."""
        return self._status[: self._count]

    def _bind(self, pages: NativeKVPages, page_map: torch.Tensor) -> None:
        if (pages.device, pages.num_layers, pages.kv_heads) != (
            self.device,
            self.num_layers,
            self.kv_heads,
        ):
            raise ValueError("native pages and codec must match device/layers/heads")
        if (
            page_map.device != self.device
            or page_map.dtype != torch.int32
            or page_map.ndim != 2
            or not page_map.is_contiguous()
            or page_map.shape[1] != ap.PAGE_TOKENS // pages.page_tokens
            or not 1 <= page_map.shape[0] <= self.max_blocks
        ):
            raise ValueError(
                "page_map must be contiguous CUDA int32 [N, 128/page_tokens]"
            )
        self._count = page_map.shape[0]
        self._stage = "idle"
        self.status.zero_()

    def encode_records(self, pages: NativeKVPages, page_map: torch.Tensor) -> None:
        """Read selected native pages into encoded scratch on the current stream."""
        self._bind(pages, page_map)
        extension().encode_records(
            pages.descriptors,
            page_map,
            self.tables.rank_of,
            self.records,
            self.rec_lens,
            self.status,
            pages.page_tokens,
            self.kv_heads,
        )
        self._stage = "records"

    def plan(self) -> None:
        """Plan lengths/offsets after encoding on the same stream."""
        if self._stage not in ("records", "planned", "packed"):
            raise RuntimeError("encode_records must establish a batch before planning")
        extension().plan_blobs(
            self.rec_lens,
            self.status,
            self.offsets,
            self.lengths,
            self.rec_offsets,
            self.num_layers,
            self.kv_heads,
        )
        self._stage = "planned"

    def pack(self) -> None:
        """Pack planned records into contiguous aligned FPKB blobs asynchronously."""
        if self._stage not in ("planned", "packed"):
            raise RuntimeError("encode_records and plan must precede packing")
        extension().pack_blobs(
            self.records,
            self.rec_lens,
            self.status,
            self.offsets,
            self.lengths,
            self.rec_offsets,
            self.buffer,
            self.num_layers,
            self.kv_heads,
            self.digest,
        )
        self._stage = "packed"

    def encode(self, pages: NativeKVPages, page_map: torch.Tensor) -> torch.Tensor:
        """Enqueue encode, plan and pack; return device status without host waiting."""
        self.encode_records(pages, page_map)
        self.plan()
        self.pack()
        return self.status

    def decode(
        self,
        pages: NativeKVPages,
        page_map: torch.Tensor,
        *,
        src: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Validate blobs then scatter to native pages, returning device status.

        ``src`` is the contiguous uint8 payload after actual H2D completion;
        ``offsets`` is contiguous int64 [N] or [N+1]. Defaults borrow the last
        encoded batch for a GPU-only round trip. External-transfer callers must
        supply their received buffers. Destination pages may not overlap.
        """
        self._bind(pages, page_map)
        source = self.buffer if src is None else src
        starts = self.offsets if offsets is None else offsets
        if (
            source.device != self.device
            or source.dtype != torch.uint8
            or not source.is_contiguous()
            or source.data_ptr() % 16
        ):
            raise ValueError("blob source must be aligned contiguous CUDA uint8")
        if (
            starts.device != self.device
            or starts.dtype != torch.int64
            or starts.ndim != 1
            or not starts.is_contiguous()
            or starts.numel() not in (self._count, self._count + 1)
        ):
            raise ValueError("blob offsets must be contiguous CUDA int64 [N] or [N+1]")
        extension().decode_blobs(
            pages.descriptors,
            page_map,
            self.tables.plut,
            self.tables.slut,
            source.view(-1),
            starts,
            self.rec_offsets,
            self.status,
            pages.page_tokens,
            self.kv_heads,
            self.digest,
        )
        return self.status

    def _raw_buffer(self, tensor: torch.Tensor) -> torch.Tensor:
        size = self._count * self.num_layers * 2 * self.kv_heads * ap.RAW_RECORD_BYTES
        if (
            tensor.device != self.device
            or tensor.dtype not in (torch.uint8, torch.bfloat16)
            or not tensor.is_contiguous()
            or tensor.data_ptr() % 16
            or tensor.numel() * tensor.element_size() < size
        ):
            raise ValueError("raw buffer must be aligned contiguous CUDA bytes or BF16")
        return tensor.view(torch.uint8).view(-1)[:size]

    def raw_gather(
        self, pages: NativeKVPages, page_map: torch.Tensor, out: torch.Tensor
    ) -> torch.Tensor:
        """Gather raw [N,L,2,H,128,128] bytes for an uncompressed transfer."""
        self._bind(pages, page_map)
        extension().raw_gather(
            pages.descriptors,
            page_map,
            self._raw_buffer(out),
            self.status,
            pages.page_tokens,
            self.kv_heads,
        )
        return self.status

    def raw_scatter(
        self, pages: NativeKVPages, page_map: torch.Tensor, src: torch.Tensor
    ) -> torch.Tensor:
        """Restore transferred raw bytes to native pages on the current stream."""
        self._bind(pages, page_map)
        extension().raw_scatter(
            pages.descriptors,
            page_map,
            self._raw_buffer(src),
            self.status,
            pages.page_tokens,
            self.kv_heads,
        )
        return self.status

    def memory_usage(self) -> dict[str, int]:
        """Return allocated workspace/table bytes, excluding native/transfer buffers."""
        tensors = {
            "record_scratch": self._records,
            "record_lengths": self._rec_lens,
            "record_offsets": self._rec_offsets,
            "blob_offsets": self._offsets,
            "blob_lengths": self._lengths,
            "status": self._status,
            "blob_buffer": self.buffer,
            "rank_tables": self.tables.rank_of,
            "primary_tables": self.tables.plut,
            "secondary_tables": self.tables.slut,
        }
        return {name: t.numel() * t.element_size() for name, t in tensors.items()}


__all__ = [
    "NativeCodec",
    "NativeKVPages",
    "NativeTensorLayout",
    "validate_native_tensor",
    "validate_tables",
]
