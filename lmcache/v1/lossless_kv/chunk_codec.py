# SPDX-License-Identifier: Apache-2.0

"""GPU codec adapter for engine-owned kv-major KV chunks.

LMCache stages one chunk per object as a contiguous ``[2, layers, tokens,
heads, head_dim]`` tensor (K/V outermost) of BF16 or FP8 E4M3 scalars and copies
it to host objects. This adapter lets it store the strict-lossless online
record instead of the raw chunk without touching the codec kernels: it
permutes chunks into a small private ``BLNHC`` scratch cache, runs
``FusedOnlineKVPacker`` / ``FusedCompressedKVDecoder`` on that scratch, and
writes records back into the first ``capacity_bytes`` of the engine's own
staging buffer. Engines whose staging buffer already fuses K
and V inside each head (``[layers, tokens, heads, 2, head_dim]``, as LMCache
stages vLLM's ``kv_size == 1`` layouts) pass ``fused_kv=True`` and skip the
permute.

Records are variable-length by default: each chunk stores its own 4 KiB
aligned record, and a chunk whose record would not be smaller than raw keeps
its raw bytes, so a stored length equal to ``raw_bytes`` marks a raw chunk.
Engines then size each host object from the length ``encode`` returns.
Engines that can only allocate one object size pass ``capacity_ratio`` to pad
every record to the capacity class ``align4K(raw * capacity_ratio)``; a chunk
that does not fit raises ``ChunkCapacityError`` and callers fail the store
closed, because the class cannot hold a raw fallback.
"""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
import math
import threading

# Third Party
import torch

# First Party
from lmcache.v1.lossless_kv.calibration import load_calibrated_codebooks
from lmcache.v1.lossless_kv.codec import default_online_codebooks
from lmcache.v1.lossless_kv.compressed_kv import (
    FusedCompressedKVDecoder,
    FusedOnlineKVPacker,
    KVRestorePlan,
    PendingPack,
    kv_dtype_bytes,
    warm_fused_online_kv_packer,
)
from lmcache.v1.lossless_kv.format import (
    ONLINE_TILE_SCALARS,
    CompressedStoreGeometry,
    SlotMode,
    align_up,
)
from lmcache.v1.lossless_kv.kv_layout import (
    allocate_cross_layer_kv_cache,
    physical_kv_view,
)

# KV dtypes the codec stores; vLLM keeps FP8 E4M3 caches as uint8 bytes.
_KV_DTYPE_BYTES = {torch.bfloat16: 2, torch.float8_e4m3fn: 1, torch.uint8: 1}


class ChunkCapacityError(RuntimeError):
    """A chunk's compressed record exceeds the configured capacity class."""


def chunk_capacity_bytes(raw_bytes: int, capacity_ratio: float) -> int:
    """Return the fixed host-object size for compressed chunk records.

    Args:
        raw_bytes (int): Raw chunk bytes; must be 4 KiB aligned.
        capacity_ratio (float): Capacity class as a fraction of raw, in
            ``(0, 1]``.

    Returns:
        ``align4K(ceil(raw_bytes * capacity_ratio))``, never above raw.

    Raises:
        ValueError: If either argument is out of range.

    Async/thread-safety:
        Pure; safe from any thread.
    """
    if raw_bytes <= 0 or raw_bytes % align_up(1):
        raise ValueError("raw chunk bytes must be positive and 4 KiB aligned")
    if not 0.0 < capacity_ratio <= 1.0:
        raise ValueError("capacity ratio must be in (0, 1]")
    return min(raw_bytes, align_up(math.ceil(raw_bytes * capacity_ratio)))


def kv_major_to_physical(chunk: torch.Tensor) -> torch.Tensor:
    """View a ``[2, L, T, H, D]`` chunk in physical ``[L, T, H, 2, D]`` order.

    Args:
        chunk (torch.Tensor): Kv-major chunk tensor.

    Returns:
        A permuted view sharing ``chunk``'s memory.

    Raises:
        ValueError: If ``chunk`` is not five-dimensional with K/V outermost.
    """
    if chunk.dim() != 5 or chunk.shape[0] != 2:
        raise ValueError(f"unsupported kv-major chunk shape {tuple(chunk.shape)}")
    return chunk.permute(1, 2, 3, 0, 4)


def physical_to_kv_major(block: torch.Tensor) -> torch.Tensor:
    """View one ``[L, T, H, 2, D]`` block in kv-major ``[2, L, T, H, D]`` order.

    Args:
        block (torch.Tensor): One physical ``BLNHC`` block.

    Returns:
        A permuted view sharing ``block``'s memory.

    Raises:
        ValueError: If ``block`` is not five-dimensional with K/V at dim 3.
    """
    if block.dim() != 5 or block.shape[3] != 2:
        raise ValueError(f"unsupported physical block shape {tuple(block.shape)}")
    return block.permute(3, 0, 1, 2, 4)


class KVMajorChunkCodec:
    """Encode and decode kv-major KV chunks in their engine staging buffers.

    Args:
        num_layers (int): KV layers in one chunk.
        chunk_tokens (int): Tokens in one chunk.
        num_kv_heads (int): KV heads per layer on this rank.
        head_dim (int): Scalars per head.
        kv_dtype (torch.dtype): Scalar type of the chunks: ``torch.bfloat16``,
            or FP8 E4M3 as ``torch.float8_e4m3fn`` or its ``torch.uint8``
            bytes.
        device (torch.device): CUDA device holding the engine staging buffers.
        max_chunks (int): Largest chunk batch passed to one call.
        capacity_ratio (float | None): Fixed capacity class as a fraction of
            raw bytes; None stores variable-length records.
        codebook_dir (str | Path | None): Calibration artifact directory
            for this model; the model-independent generic palette when None.
        kv_head_start (int): First calibrated KV head this rank holds.
        fused_kv (bool): Chunks hold K and V side by side in each head,
            ``[layers, tokens, heads, 2, head_dim]``, instead of kv-major.
            This is the codec's physical block order, so no permute is needed.
        decode_sets (int): Record buffer sets that decode batches rotate
            through, so the next batches' stored bytes can be copied in while
            an earlier batch decodes.

    Raises:
        ValueError: If the device is not CUDA, the dtype or geometry is
            unsupported, or the calibrated codebooks were fitted for another geometry.
        FileNotFoundError: If ``codebook_dir`` lacks the artifact files.

    Async/thread-safety:
        Encode and decode own separate scratch and may run concurrently on
        different threads; concurrent ``encode`` or ``decode`` calls serialize
        on an internal lock, while ``record_buffers`` / ``decode_records``
        must come from one thread at a time. Every call enqueues on the
        caller's stream; a decode batch may be filled on another stream than
        it decodes on. ``encode`` synchronizes that stream because record
        lengths are read back to the host; decoding never synchronizes.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        chunk_tokens: int,
        num_kv_heads: int,
        head_dim: int,
        kv_dtype: torch.dtype,
        device: torch.device,
        max_chunks: int,
        capacity_ratio: float | None = None,
        codebook_dir: str | Path | None = None,
        kv_head_start: int = 0,
        fused_kv: bool = False,
        decode_sets: int = 4,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("kv-major chunk codec requires a CUDA device")
        if max_chunks <= 0 or decode_sets <= 0:
            raise ValueError("kv-major chunk codec batch must be positive")
        dtype_bytes = _KV_DTYPE_BYTES.get(kv_dtype)
        if dtype_bytes is None:
            raise ValueError(f"kv-major chunk codec does not support {kv_dtype}")
        # Scratch keeps FP8 as bytes, the form the codec kernels read.
        self._scratch_dtype = torch.bfloat16 if dtype_bytes == 2 else torch.uint8
        self._geometry = CompressedStoreGeometry(
            num_slots=max_chunks,
            slot_size=num_layers
            * 2
            * chunk_tokens
            * num_kv_heads
            * head_dim
            * dtype_bytes,
            block_tokens=chunk_tokens,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tile_scalars=ONLINE_TILE_SCALARS,
            dtype_bytes=dtype_bytes,
        )
        self._fixed_capacity = capacity_ratio is not None
        # Variable records are bounded by raw: anything larger stays raw.
        self._capacity = (
            chunk_capacity_bytes(self._geometry.slot_size, capacity_ratio)
            if capacity_ratio is not None
            else self._geometry.slot_size
        )
        self._max_chunks = max_chunks
        self._decode_sets = decode_sets
        self._fused_kv = fused_kv
        self._chunk_shape = (
            (num_layers, chunk_tokens, num_kv_heads, 2, head_dim)
            if fused_kv
            else (2, num_layers, chunk_tokens, num_kv_heads, head_dim)
        )
        self._codebooks = (
            default_online_codebooks(self._geometry)
            if codebook_dir is None
            else load_calibrated_codebooks(codebook_dir, self._geometry, kv_head_start)
        )
        with torch.cuda.device(device), torch.inference_mode(False):
            self._encode_scratch = self._scratch_cache(device)
            self._decode_scratch = self._scratch_cache(device)
            self._pack_staging = torch.empty(
                max_chunks * self._geometry.slot_size, dtype=torch.uint8, device=device
            )
            # Per set, one raw-sized record buffer per chunk, so any stored
            # length fits.
            self._decode_staging = torch.empty(
                decode_sets * max_chunks * self._geometry.slot_size,
                dtype=torch.uint8,
                device=device,
            )
            # Kv-major chunks need their own output; fused chunks are the
            # decoder's physical blocks.
            self._decode_out = (
                None
                if fused_kv
                else torch.empty(
                    max_chunks * self._geometry.slot_size,
                    dtype=torch.uint8,
                    device=device,
                )
            )
            # Compile the kernels here so no store or retrieve pays TileLang JIT.
            warm_fused_online_kv_packer(
                self._encode_scratch, max_chunks, tile_scalars=ONLINE_TILE_SCALARS
            )
            self._packer = FusedOnlineKVPacker(
                kv_cache=self._encode_scratch,
                codebooks=self._codebooks,
                tile_scalars=ONLINE_TILE_SCALARS,
                max_slots_per_buffer=max_chunks,
            )
            self._decoder = FusedCompressedKVDecoder(
                kv_cache=self._decode_scratch,
                codebooks=self._codebooks,
                tile_scalars=ONLINE_TILE_SCALARS,
                max_slots_per_buffer=max_chunks,
            )
        # Restore plans by record set and the batch indices holding
        # compressed records; each uploads its metadata once, so decoding
        # never touches the host.
        self._restore_plans: dict[tuple[int, tuple[int, ...]], KVRestorePlan] = {}
        # Decoder into the engine's own KV blocks, and its plans by record
        # set and batch size.
        self._engine_decoder: FusedCompressedKVDecoder | None = None
        # Packer reading the engine's own KV blocks (variable records only).
        self._engine_packer: FusedOnlineKVPacker | None = None
        self._engine_plans: dict[tuple[int, int], KVRestorePlan] = {}
        # Scratch and staging are shared by calls that may use other streams.
        self._encode_done: torch.cuda.Event | None = None
        # The record set of the current decode batch, the stream filling it,
        # and per set the point after which its buffers may be refilled.
        self._decode_set = decode_sets - 1
        self._set_free: list[torch.cuda.Event | None] = [None] * decode_sets
        self._fill_stream: torch.cuda.Stream | None = None
        self._decode_stream: torch.cuda.Stream | None = None
        self._encode_lock = threading.Lock()
        self._decode_lock = threading.Lock()

    @property
    def raw_bytes(self) -> int:
        """Return the raw bytes of one kv-major chunk."""
        return self._geometry.slot_size

    @property
    def capacity_bytes(self) -> int:
        """Return the largest stored length of one chunk.

        The fixed capacity class when ``capacity_ratio`` was given (every
        stored length then equals it), otherwise ``raw_bytes``.
        """
        return self._capacity

    @property
    def fixed_capacity(self) -> bool:
        """Return whether every record is padded to ``capacity_bytes``."""
        return self._fixed_capacity

    def encode(
        self, chunks: list[torch.Tensor], stream: torch.cuda.Stream
    ) -> list[int]:
        """Replace each raw chunk's leading bytes with its compressed record.

        Args:
            chunks (list[torch.Tensor]): Contiguous CUDA uint8 staging buffers
                of ``raw_bytes`` each, holding chunks in the codec's chunk
                layout (kv-major, or fused when ``fused_kv``). Once
                ``stream`` passes this call, the first ``capacity_bytes`` of
                each hold its record followed by unspecified padding.
            stream (torch.cuda.Stream): Stream ordered after the producer of
                ``chunks``; consumers of the records must also use it.

        Returns:
            Stored length of each chunk: the 4 KiB aligned record length,
            ``raw_bytes`` for a chunk kept raw, or ``capacity_bytes`` for
            every chunk under a fixed capacity class. Engines copy and later
            restore exactly this prefix of each buffer.

        Raises:
            ValueError: If buffers do not match the codec geometry.
            ChunkCapacityError: Under a fixed capacity class, if a record
                does not fit it. The chunks' contents are then unspecified.

        Async/thread-safety:
            Synchronizes ``stream`` to read record lengths.
        """
        self._validate(chunks)
        raw = self._geometry.slot_size
        with self._encode_lock:
            self._order_after(self._encode_done, stream)
            physical = physical_kv_view(self._encode_scratch)
            with torch.cuda.stream(stream):
                for index, chunk in enumerate(chunks):
                    physical[index].copy_(self._physical_view(chunk))
            records = self._packer.pack_into(
                staging=self._pack_staging,
                block_ids=list(range(len(chunks))),
                logical_slots=list(range(len(chunks))),
                slot_stride=raw,
                stream=stream,
            )
            lengths: list[int] = []
            for record in records:
                compressed = (
                    record.mode is SlotMode.COMPRESSED and record.stored_length < raw
                )
                if self._fixed_capacity and (
                    not compressed or record.stored_length > self._capacity
                ):
                    raise ChunkCapacityError(
                        f"chunk record needs {record.stored_length} bytes "
                        f"(mode {record.mode.name}), capacity is {self._capacity}"
                    )
                lengths.append(record.stored_length if compressed else raw)
            with torch.cuda.stream(stream):
                for chunk, record, length in zip(chunks, records, lengths, strict=True):
                    # A raw chunk already holds its stored bytes.
                    if length < raw:
                        start = record.source_offset
                        chunk[:length].copy_(self._pack_staging[start : start + length])
                self._encode_done = torch.cuda.Event()
                self._encode_done.record(stream)
            if self._fixed_capacity:
                return [self._capacity] * len(chunks)
            return lengths

    def record_buffers(
        self, count: int, stream: torch.cuda.Stream
    ) -> list[torch.Tensor]:
        """Return the codec's buffers for the stored bytes of a decode batch.

        Fill ``buffer[i][:length]`` with chunk ``i``'s stored bytes on
        ``stream``, then call :meth:`decode_records`. Batches rotate through
        ``decode_sets`` buffer sets, so with a fill stream other than the
        decode stream the next batches fill while earlier ones decode.

        Args:
            count (int): Chunks in the batch, at most ``max_chunks``.
            stream (torch.cuda.Stream): Stream of the fills.

        Returns:
            list[torch.Tensor]: ``count`` uint8 device buffers of
            ``raw_bytes`` each.

        Raises:
            ValueError: If ``count`` is out of range.

        Async/thread-safety:
            Orders ``stream`` after the decode stream's work up to the last
            batch that used this set, including reads of its outputs
            enqueued before this call; never synchronizes. One thread at a
            time.
        """
        if not 0 < count <= self._max_chunks:
            raise ValueError("kv-major chunk batch size is out of range")
        if self._decode_stream is not None:
            # The batch staged last is done once its readers so far are.
            done = torch.cuda.Event()
            done.record(self._decode_stream)
            self._set_free[self._decode_set] = done
        self._decode_set = (self._decode_set + 1) % self._decode_sets
        free = self._set_free[self._decode_set]
        if free is not None:
            stream.wait_event(free)
        self._fill_stream = stream
        return self._staged(count)

    def decode_records(
        self, lengths: list[int], stream: torch.cuda.Stream
    ) -> list[torch.Tensor]:
        """Decode the batch staged in :meth:`record_buffers`.

        Args:
            lengths (list[int]): Stored length of each staged chunk, as
                returned by ``encode``; ``raw_bytes`` marks a raw chunk.
            stream (torch.cuda.Stream): Stream of the decode and of every
                reader of its outputs; it is ordered after the fills.

        Returns:
            list[torch.Tensor]: Each chunk's raw bytes in the codec's chunk
            layout (uint8, ``raw_bytes``), valid on ``stream`` until the next
            ``record_buffers`` call.

        Raises:
            ValueError: If a length is out of range, or no batch is staged.

        Async/thread-safety:
            Enqueues on ``stream`` without host work beyond the launches; the
            first batch of each new mix of raw and compressed chunks builds
            its restore plan, which synchronizes a private stream once.
        """
        self._validate_lengths(lengths)
        self._order_decode(stream)
        raw = self._geometry.slot_size
        compressed = tuple(
            index
            for index, length in enumerate(lengths)
            if length < raw or self._fixed_capacity
        )
        staged = self._staged(len(lengths))
        if not compressed:
            return staged
        key = (self._decode_set, compressed)
        plan = self._restore_plans.get(key)
        if plan is None:
            plan = self._decoder.plan(
                staging=self._set_staging(),
                staging_offsets=[index * raw for index in compressed],
                block_ids=list(compressed),
                modes=[int(SlotMode.COMPRESSED)] * len(compressed),
            )
            self._restore_plans[key] = plan
        plan.submit(stream)
        physical = physical_kv_view(self._decode_scratch)
        outputs = staged
        for index in compressed:
            if self._decode_out is None:
                # Fused chunks are byte-identical to the physical block.
                outputs[index] = physical[index].reshape(-1).view(torch.uint8)
                continue
            out = self._decode_out[index * raw : (index + 1) * raw]
            with torch.cuda.stream(stream):
                self._physical_view(out).copy_(physical[index])
            outputs[index] = out
        return outputs

    @property
    def engine_store(self) -> bool:
        """Return whether :meth:`encode_engine_blocks` is available."""
        return self._engine_packer is not None

    def attach_engine_cache(self, kv_cache: torch.Tensor) -> None:
        """Compile a decoder that restores records straight into engine blocks.

        Args:
            kv_cache (torch.Tensor): Logical ``[blocks, layers, 2, tokens,
                heads, head_dim]`` view of the engine's paged KV cache, in a
                layout ``kernel_kv_view`` accepts, whose blocks each hold one
                chunk.

        Raises:
            ValueError: If the codec is not fused, or a block does not match
                the chunk geometry.

        Async/thread-safety:
            Compiles kernels; call once before serving traffic.
        """
        if not self._fused_kv:
            raise ValueError("engine-direct decode requires fused K/V chunks")
        if (
            tuple(kv_cache.shape[1:]) != tuple(self._decode_scratch.shape[1:])
            or kv_cache.device != self._decode_scratch.device
            or kv_dtype_bytes(kv_cache) != self._geometry.dtype_bytes
        ):
            raise ValueError("engine KV blocks do not match the codec chunks")
        with torch.cuda.device(kv_cache.device), torch.inference_mode(False):
            self._engine_decoder = FusedCompressedKVDecoder(
                kv_cache=kv_cache,
                codebooks=self._codebooks,
                tile_scalars=ONLINE_TILE_SCALARS,
                max_slots_per_buffer=self._max_chunks,
            )
            # A capacity class pads records in place, which needs raw-sized
            # buffers per chunk; only variable records pack from the engine.
            self._engine_packer = (
                None
                if self._fixed_capacity
                else FusedOnlineKVPacker(
                    kv_cache=kv_cache,
                    codebooks=self._codebooks,
                    tile_scalars=ONLINE_TILE_SCALARS,
                    max_slots_per_buffer=self._max_chunks,
                )
            )
        self._engine_plans.clear()

    def encode_engine_blocks(
        self, blocks: list[int], staging: torch.Tensor, stream: torch.cuda.Stream
    ) -> PendingPack:
        """Enqueue the records of chunks held in engine blocks.

        Args:
            blocks (list[int]): Engine block ID of each chunk, at most
                ``max_chunks``.
            staging (torch.Tensor): 16-byte aligned CUDA uint8 buffer of at
                least ``len(blocks) * raw_bytes`` bytes; it receives the
                records and must stay untouched until their consumers finish.
            stream (torch.cuda.Stream): Stream ordered after the blocks'
                producer. Every call must use the same stream.

        Returns:
            PendingPack: Pass it to :meth:`engine_records` exactly once.

        Raises:
            ValueError: If no engine cache is attached, records have a fixed
                capacity class, or the arguments do not match.

        Async/thread-safety:
            Never synchronizes; the blocks may be reused once ``stream``
            passes this call.
        """
        if self._engine_packer is None:
            raise ValueError("no engine KV cache is attached for variable records")
        return self._engine_packer.launch_pack(
            staging=staging,
            block_ids=blocks,
            logical_slots=list(range(len(blocks))),
            slot_stride=self._geometry.slot_size,
            stream=stream,
        )

    def engine_records(
        self, pending: PendingPack, staging: torch.Tensor, stream: torch.cuda.Stream
    ) -> list[torch.Tensor]:
        """Wait for :meth:`encode_engine_blocks` and return the stored bytes.

        Args:
            pending (PendingPack): The pack of ``staging``.
            staging (torch.Tensor): The buffer given to ``encode_engine_blocks``.
            stream (torch.cuda.Stream): Stream ordered after the pack.

        Returns:
            list[torch.Tensor]: Each chunk's stored bytes as a uint8 view of
            ``staging``, exactly what :meth:`encode` stores: the record, or
            ``raw_bytes`` of the raw chunk in the codec's chunk layout.

        Raises:
            ValueError: If no engine cache is attached.

        Async/thread-safety:
            Waits on the host for the pack; raw chunks are converted to the
            chunk layout on ``stream``, so read the views on ``stream``. One
            thread at a time.
        """
        if self._engine_packer is None:
            raise ValueError("no engine KV cache is attached for variable records")
        raw = self._geometry.slot_size
        outputs: list[torch.Tensor] = []
        for record in self._engine_packer.collect(pending):
            start = record.source_offset
            if record.mode is SlotMode.COMPRESSED:
                if record.stored_length >= raw:
                    raise RuntimeError("compressed engine record is not below raw")
                outputs.append(staging[start : start + record.stored_length])
                continue
            chunk = staging[start : start + raw]
            self._planes_to_chunk(chunk, stream)
            outputs.append(chunk)
        return outputs

    def decode_records_to_engine(
        self, lengths: list[int], blocks: torch.Tensor, stream: torch.cuda.Stream
    ) -> bool:
        """Decode the batch staged in :meth:`record_buffers` into engine blocks.

        Args:
            lengths (list[int]): Stored length of each staged chunk, as
                returned by ``encode``.
            blocks (torch.Tensor): Contiguous device int32 engine block ID of
                each chunk, ready on ``stream``.
            stream (torch.cuda.Stream): Stream of the decode; it is ordered
                after the fills.

        Returns:
            bool: Whether the batch was enqueued. False, with nothing
            enqueued, if a chunk is stored raw; decode it with
            :meth:`decode_records` instead.

        Raises:
            ValueError: If no engine cache is attached, a length is out of
                range, ``blocks`` does not match, or no batch is staged.

        Async/thread-safety:
            As :meth:`decode_records`; ``blocks`` must stay alive until the
            launch runs.
        """
        if self._engine_decoder is None:
            raise ValueError("no engine KV cache is attached")
        self._validate_lengths(lengths)
        raw = self._geometry.slot_size
        if not self._fixed_capacity and any(length >= raw for length in lengths):
            return False
        self._order_decode(stream)
        key = (self._decode_set, len(lengths))
        plan = self._engine_plans.get(key)
        if plan is None:
            plan = self._engine_decoder.plan_dynamic(
                staging=self._set_staging(),
                staging_offsets=[index * raw for index in range(len(lengths))],
                modes=[int(SlotMode.COMPRESSED)] * len(lengths),
            )
            self._engine_plans[key] = plan
        plan.submit(stream, blocks)
        return True

    def decode(
        self,
        chunks: list[torch.Tensor],
        lengths: list[int],
        stream: torch.cuda.Stream,
    ) -> None:
        """Restore raw chunks from records at their buffer starts.

        Convenience wrapper over :meth:`record_buffers` and
        :meth:`decode_records` for callers that own the chunk buffers.

        Args:
            chunks (list[torch.Tensor]): Contiguous CUDA uint8 staging buffers
                of ``raw_bytes`` each whose first ``lengths[i]`` bytes hold
                what ``encode`` stored. They are overwritten in place.
            lengths (list[int]): Stored length of each chunk as returned by
                ``encode``; ``raw_bytes`` marks a chunk kept raw, which is
                left untouched.
            stream (torch.cuda.Stream): Stream ordered after the copies that
                filled ``chunks``.

        Raises:
            ValueError: If buffers do not match the codec geometry, a length
                is out of range, or a record header is malformed.

        Async/thread-safety:
            Enqueues work on ``stream`` without synchronizing it; the chunks
            are valid once ``stream`` reaches this point.
        """
        self._validate(chunks)
        if len(lengths) != len(chunks):
            raise ValueError("kv-major chunk stored lengths are out of range")
        self._validate_lengths(lengths)
        raw = self._geometry.slot_size
        with self._decode_lock:
            buffers = self.record_buffers(len(chunks), stream)
            with torch.cuda.stream(stream):
                for buf, chunk, length in zip(buffers, chunks, lengths, strict=True):
                    if length < raw or self._fixed_capacity:
                        buf[:length].copy_(chunk[:length])
            outputs = self.decode_records(lengths, stream)
            with torch.cuda.stream(stream):
                for chunk, out, length in zip(chunks, outputs, lengths, strict=True):
                    if length < raw or self._fixed_capacity:
                        chunk.copy_(out)

    @staticmethod
    def _order_after(
        previous: torch.cuda.Event | None, stream: torch.cuda.Stream
    ) -> None:
        """Make ``stream`` wait for the previous call's use of shared scratch."""
        if previous is not None:
            stream.wait_event(previous)

    def _set_staging(self) -> torch.Tensor:
        """Return the current record set's staging, as the decoders expect."""
        size = self._max_chunks * self._geometry.slot_size
        base = self._decode_set * size
        return self._decode_staging[base : base + size]

    def _staged(self, count: int) -> list[torch.Tensor]:
        """Return the current set's first ``count`` record buffers."""
        raw = self._geometry.slot_size
        staging = self._set_staging()
        return [staging[i * raw : (i + 1) * raw] for i in range(count)]

    def _order_decode(self, stream: torch.cuda.Stream) -> None:
        """Order a decode on ``stream`` after its fills and the last decode."""
        if self._fill_stream is None:
            raise ValueError("no decode batch is staged in record_buffers")
        if self._fill_stream != stream:
            filled = torch.cuda.Event()
            filled.record(self._fill_stream)
            stream.wait_event(filled)
        # Decodes share scratch, so one on a new stream waits for the last.
        previous = self._decode_stream
        if previous is not None and previous != stream:
            done = torch.cuda.Event()
            done.record(previous)
            stream.wait_event(done)
        self._decode_stream = stream

    def _scratch_cache(self, device: torch.device) -> torch.Tensor:
        """Allocate one private logical ``BLNHC`` cache of ``max_chunks`` blocks."""
        geometry = self._geometry
        return allocate_cross_layer_kv_cache(
            self._max_chunks,
            geometry.num_layers,
            geometry.block_tokens,
            geometry.num_kv_heads,
            geometry.head_dim,
            dtype=self._scratch_dtype,
            device=device,
        )

    def _planes_to_chunk(self, chunk: torch.Tensor, stream: torch.cuda.Stream) -> None:
        """Rewrite a packer raw record (planes back to back) as a fused chunk."""
        geometry = self._geometry
        rows = geometry.block_tokens * geometry.num_kv_heads
        shape = (geometry.num_layers, 2, rows, geometry.head_dim)
        with self._encode_lock:
            self._order_after(self._encode_done, stream)
            planes = self._pack_staging[: geometry.slot_size]
            with torch.cuda.stream(stream):
                planes.copy_(chunk)
                chunk.view(self._scratch_dtype).view(
                    geometry.num_layers, rows, 2, geometry.head_dim
                ).copy_(planes.view(self._scratch_dtype).view(shape).transpose(1, 2))
                self._encode_done = torch.cuda.Event()
                self._encode_done.record(stream)

    def _physical_view(self, chunk: torch.Tensor) -> torch.Tensor:
        """View a raw uint8 chunk buffer in the codec's physical block order."""
        view = chunk.view(self._scratch_dtype).view(self._chunk_shape)
        return view if self._fused_kv else kv_major_to_physical(view)

    def _validate(self, chunks: list[torch.Tensor]) -> None:
        """Reject batches that do not match the compiled geometry."""
        if not 0 < len(chunks) <= self._max_chunks:
            raise ValueError("kv-major chunk batch size is out of range")
        for chunk in chunks:
            if (
                chunk.dtype is not torch.uint8
                or chunk.device != self._pack_staging.device
                or chunk.numel() != self._geometry.slot_size
                or not chunk.is_contiguous()
                or chunk.data_ptr() % 4
            ):
                raise ValueError("kv-major chunk buffer does not match the codec")

    def _validate_lengths(self, lengths: list[int]) -> None:
        """Reject stored lengths that no ``encode`` batch could return."""
        if not 0 < len(lengths) <= self._max_chunks or any(
            not 0 < length <= self._capacity for length in lengths
        ):
            raise ValueError("kv-major chunk stored lengths are out of range")
