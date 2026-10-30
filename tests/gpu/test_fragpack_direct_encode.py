# SPDX-License-Identifier: Apache-2.0
"""Complete-new-block input bypass, representation, and lifecycle checks.

The reference is the supplied BF16 bits and the CPU record encoder. Tests
inspect the record bytes only after stream completion; production hot paths
must not perform these readbacks. No serving-engine imports are required.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from fragpack.compression import attn_page as ap
from fragpack.compression.attn_tables import device_tables
from fragpack.ops.hbm_kv import HbmKvStore, StoreError, page_bytes

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]

T, D, H, L, NB = 128, 128, 2, 2, 12
POISON = 0xA5


def _bits(values: np.ndarray) -> np.ndarray:
    return (values.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _host_bits(tensor: torch.Tensor) -> np.ndarray:
    return tensor.view(torch.int16).cpu().numpy().view(np.uint16)


def _input(bits: np.ndarray, strided: bool) -> torch.Tensor:
    """Build contiguous or deliberately two-byte-offset strided BF16 input."""
    source = torch.from_numpy(bits.view(np.int16).copy()).cuda().view(torch.bfloat16)
    if not strided:
        return source
    rows, heads, dims = bits.shape
    head_stride = dims + 4
    row_stride = heads * head_stride + 33
    owner = torch.empty(1 + rows * row_stride, dtype=torch.bfloat16, device="cuda")
    result = owner.as_strided(
        (rows, heads, dims), (row_stride, head_stride, 1), storage_offset=1
    )
    result.copy_(source)
    if rows:
        assert result.data_ptr() % 4 == 2
    return result


class _Harness:
    """Own a two-layer store and a host copy of every written BF16 bit."""

    def __init__(
        self,
        *,
        direct: bool = True,
        page: int | None = None,
        staging_slots: int = 2,
        overflow_records: int = 64,
    ) -> None:
        self.page = page_bytes(H) if page is None else page
        rng = np.random.default_rng(2718)
        calibration = _bits(rng.standard_normal((L, 2, H, 4 * T, D)) * 0.5)
        self.table = np.empty((L, 2, H, ap.TABLE_ENTRIES), dtype=np.uint8)
        for index in np.ndindex(L, 2, H):
            self.table[index] = ap.calibrate_table(
                (calibration[index] >> 8).astype(np.uint8)
            )
        self.caches = [
            torch.empty((NB, self.page), dtype=torch.uint8, device="cuda")
            for _ in range(L)
        ]
        self.store = HbmKvStore(
            self.caches,
            H,
            device_tables(self.table, torch.device("cuda")),
            max_tokens=1024,
            staging_slots=staging_slots,
            overflow_records=overflow_records,
            direct_new_blocks=direct,
        )
        self.direct = direct
        self.ref = np.zeros((L, 2, NB, T, H, D), dtype=np.uint16)

    def inputs(
        self, slots: list[int], seed: int, *, strided: bool, raw: bool = False
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Populate the reference and return inputs retained until encoding."""
        rng = np.random.default_rng(seed)
        source = _bits(rng.standard_normal((L, 2, len(slots), H, D)) * 0.5)
        if raw:
            # Exercise every high byte, including non-finite BF16 bit patterns.
            source[:] = rng.integers(0, 65536, source.shape, dtype=np.uint16)
        for row, slot in enumerate(slots):
            if slot >= 0:
                self.ref[:, :, slot // T, slot % T] = source[:, :, row]
        return [
            (_input(source[layer, 0], strided), _input(source[layer, 1], strided))
            for layer in range(L)
        ]

    def check_gather(
        self,
        layer: int,
        filled: dict[int, int],
        inputs: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        """Check both source-aware pre-encode and ordinary post-commit reads."""
        # Non-monotonic page order catches confusion between block and row ids.
        blocks = list(reversed(filled))
        ids = torch.tensor(blocks, dtype=torch.int32, device="cuda")
        wk = torch.empty((len(blocks), T, H, D), dtype=torch.bfloat16, device="cuda")
        wv = torch.empty_like(wk)
        kwargs = {} if inputs is None else {"key": inputs[0], "value": inputs[1]}
        self.store.gather(layer, ids, wk, wv, **kwargs)
        actual = [_host_bits(wk), _host_bits(wv)]
        for position, block in enumerate(blocks):
            for kv in (0, 1):
                n = filled[block]
                np.testing.assert_array_equal(
                    actual[kv][position, :n], self.ref[layer, kv, block, :n]
                )

    def check_records(self, blocks: list[int]) -> None:
        """Compare coded bytes exactly; raw fallbacks must reconstruct exactly."""
        addresses = self.store.rec_addr.cpu().numpy()
        lengths = self.store.rec_len.cpu().numpy()
        overflow = self.store.overflow.cpu().numpy().reshape(-1)
        overflow_base = self.store.overflow.data_ptr()
        for layer in range(L):
            arena = self.caches[layer].cpu().numpy().reshape(-1)
            arena_base = self.caches[layer].data_ptr()
            for block in blocks:
                intervals = []
                for kv, head in np.ndindex(2, H):
                    expected_bits = self.ref[layer, kv, block, :, head]
                    table = self.table[layer, kv, head]
                    expected_record = ap.encode_record(expected_bits, table, kv)
                    address = int(addresses[layer, block, kv, head])
                    size = ap.DENSE_BYTES + int(lengths[layer, block, kv, head])
                    if overflow_base <= address < overflow_base + overflow.size:
                        assert size == ap.RAW_RECORD_BYTES
                        offset = address - overflow_base
                        record = overflow[offset : offset + size].tobytes()
                    else:
                        start = arena_base + block * self.page
                        assert start <= address < start + self.page
                        assert address + size <= start + self.page
                        assert address % 16 == 0
                        offset = address - arena_base
                        record = arena[offset : offset + size].tobytes()
                        assert record == expected_record
                        intervals.append((address, address + size))
                    np.testing.assert_array_equal(
                        ap.decode_record(record, table, kv), expected_bits
                    )
                intervals.sort()
                assert all(
                    left[1] <= right[0]
                    for left, right in zip(intervals, intervals[1:], strict=False)
                )

    def step(
        self,
        slots: list[int],
        *,
        seed: int,
        filled: dict[int, int],
        live: list[int] | None = None,
        strided: bool = True,
        raw: bool = False,
        expected_direct: dict[int, int] | None = None,
    ) -> None:
        """Run a mixed step and prove bypassed scratch is never populated."""
        slot_tensor = torch.tensor(slots, dtype=torch.int64, device="cuda")
        inputs = self.inputs(slots, seed, strided=strided, raw=raw)
        closes = len({s // T for s in slots if s >= 0 and s % T == T - 1})
        self.store.scratch.fill_(POISON)
        self.store.prepare(
            slot_tensor, len(slots), np.array(live or [], dtype=np.int32)
        )
        direct_map = self.store.direct_rows.cpu().tolist()
        if expected_direct is not None:
            assert {b: row for b, row in enumerate(direct_map) if row >= 0} == (
                expected_direct if self.direct else {}
            )
        direct_blocks = [b for b, row in enumerate(direct_map) if row >= 0]
        wslots = self.store.wslot.cpu().tolist()
        scratch_slots = [wslots[b] - self.store.staging.shape[0] for b in direct_blocks]
        for layer, (key, value) in enumerate(inputs):
            kwargs = {"key": key, "value": value} if self.direct else {}
            self.store.write(layer, key, value, slot_tensor)
            self.check_gather(layer, filled, (key, value) if self.direct else None)
            self.store.encode(layer, closes, **kwargs)
            for scratch_slot in scratch_slots:
                assert scratch_slot >= 0
                assert bool(torch.all(self.store.scratch[scratch_slot] == POISON))
        self.store.commit()
        assert not bool(torch.any(self.store.direct_rows >= 0))
        for layer in range(L):
            self.check_gather(layer, filled)


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("kind", ["compressed", "raw", "spill"])
def test_complete_new_blocks_match_reference_and_skip_scratch(
    strided: bool, kind: str
) -> None:
    """Full new blocks at a nonzero batch row bypass scratch for both layouts."""
    page = ap.DENSE_BYTES + 1280 if kind == "spill" else None
    h = _Harness(page=page)
    # A short request precedes two complete blocks of another request.
    slots = list(range(3 * T, 3 * T + 7)) + list(range(T, 3 * T))
    h.step(
        slots,
        seed=17,
        filled={3: 7, 1: T, 2: T},
        strided=strided,
        raw=kind == "raw",
        expected_direct={1: 7, 2: 7 + T},
    )
    h.check_records([1, 2])
    stats = h.store.stats()
    assert stats.encoded == 2 * L * 2 * H
    assert stats.err_staging == stats.err_overflow == 0
    if kind == "raw":
        assert stats.fallbacks == stats.encoded
    elif kind == "spill":
        assert stats.spills > 0
    else:
        assert stats.fallbacks == 0


def test_old_tail_and_heir_keep_staging_while_new_full_block_bypasses() -> None:
    """Closing an old tail and opening its heir preserves all layers' bits."""
    h = _Harness(staging_slots=1)
    h.step(list(range(19)), seed=21, filled={0: 19}, expected_direct={})
    # Block 0 was already open. Block 1 is new and complete. Block 2 must
    # inherit block 0's sole persistent slot after using transient storage.
    h.step(
        list(range(19, 2 * T + 11)),
        seed=22,
        filled={0: T, 1: T, 2: 11},
        live=[0],
        expected_direct={1: T - 19},
    )
    h.check_records([0, 1])
    h.step(
        list(range(2 * T + 11, 3 * T)),
        seed=23,
        filled={0: T, 1: T, 2: T},
        live=[2],
        expected_direct={},
    )
    h.check_records([0, 1, 2])
    stats = h.store.stats()
    assert stats.err_staging == stats.err_overflow == 0
    assert stats.encoded == 3 * L * 2 * H


def test_prepare_resets_every_layer_of_reordered_claims_and_reuses_overflow() -> None:
    """Reclaim mixed records without confusing layer, claim, and head indices."""
    layers, heads, blocks, persistent = 5, 4, 13, 2
    per_block = layers * 2 * heads
    table = np.broadcast_to(
        np.arange(0x30, 0x30 + ap.TABLE_ENTRIES, dtype=np.uint8),
        (layers, 2, heads, ap.TABLE_ENTRIES),
    ).copy()
    caches = [
        torch.empty((blocks, page_bytes(heads)), dtype=torch.uint8, device="cuda")
        for _ in range(layers)
    ]
    store = HbmKvStore(
        caches,
        heads,
        device_tables(table, torch.device("cuda")),
        max_tokens=4 * T,
        staging_slots=persistent,
        overflow_records=4 * per_block,
        direct_new_blocks=True,
    )
    rng = np.random.default_rng(721)
    first_slots = [b * T + offset for b in [11, 2, 9, 7] for offset in range(T)]
    first_mapping = torch.tensor(first_slots, dtype=torch.int64, device="cuda")
    first_bits = rng.integers(
        0, 65536, (layers, 2, len(first_slots), heads, D), dtype=np.uint16
    )
    # Block 2 has coded lengths and nonzero arena fill. Blocks 11, 9, and 7
    # occupy overflow, so reclaiming 11/9 must release all layers' old slots.
    first_bits[:, :, T : 2 * T] &= 0xFF
    first_bits[:, :, T : 2 * T] |= 0x3000
    store.prepare(first_mapping, len(first_slots), np.array([], dtype=np.int32))
    for layer in range(layers):
        key, value = (_input(first_bits[layer, kv], False) for kv in (0, 1))
        store.write(layer, key, value, first_mapping)
        store.encode(layer, 4, key=key, value=value)
    store.commit()
    assert store.stats().fallbacks == 3 * per_block
    assert sum((int(w) & 0xFFFFFFFF).bit_count() for w in store.ovf_bits.cpu()) == (
        3 * per_block
    )

    claims = [(9, T), (5, 11), (11, T), (2, 19), (0, T)]
    slots = [b * T + offset for b, count in claims for offset in range(count)]
    mapping = torch.tensor(slots, dtype=torch.int64, device="cuda")
    # Poison claimed fills, including raw/new records, to expose missed resets.
    claimed_ids = [b for b, _ in claims]
    store.fill[:, claimed_ids] = torch.arange(
        1, layers * len(claims) + 1, dtype=torch.int32, device="cuda"
    ).reshape(layers, len(claims))
    expected_addr = store.rec_addr.cpu().numpy().copy()
    expected_len = store.rec_len.cpu().numpy().copy()
    expected_fill = store.fill.cpu().numpy().copy()
    retained_addresses = expected_addr[:, 7].copy()
    store.prepare(mapping, len(slots), np.array([], dtype=np.int32))
    wslots = store.wslot.cpu().tolist()
    direct = store.direct_rows.cpu().tolist()
    source_row = 0
    for block, count in claims:
        wslot = wslots[block]
        assert wslot >= 0
        assert (wslot >= persistent) == (count == T)
        assert direct[block] == (source_row if count == T else -1)
        source_row += count
        for layer in range(layers):
            owner = (
                store.staging[wslot, layer]
                if wslot < persistent
                else store.scratch[wslot - persistent]
            )
            for kv, head in np.ndindex(2, heads):
                expected_addr[layer, block, kv, head] = owner[kv, head].data_ptr()
            expected_len[layer, block] = ap.RAW_RECORD_BYTES - ap.DENSE_BYTES
            expected_fill[layer, block] = 0
    np.testing.assert_array_equal(store.rec_addr.cpu().numpy(), expected_addr)
    np.testing.assert_array_equal(store.rec_len.cpu().numpy(), expected_len)
    np.testing.assert_array_equal(store.fill.cpu().numpy(), expected_fill)
    # Only unclaimed block 7 may retain overflow ownership after prepare.
    assert sum((int(w) & 0xFFFFFFFF).bit_count() for w in store.ovf_bits.cpu()) == (
        per_block
    )

    next_bits = rng.integers(
        0, 65536, (layers, 2, len(slots), heads, D), dtype=np.uint16
    )
    for layer in range(layers):
        key, value = (_input(next_bits[layer, kv], False) for kv in (0, 1))
        store.write(layer, key, value, mapping)
        store.encode(layer, 3, key=key, value=value)
    store.commit()
    stats = store.stats()  # A missed release exhausts this deliberately exact pool.
    assert stats.encoded == 7 * per_block
    assert stats.fallbacks == 6 * per_block
    assert sum((int(w) & 0xFFFFFFFF).bit_count() for w in store.ovf_bits.cpu()) == (
        4 * per_block
    )
    np.testing.assert_array_equal(
        store.rec_addr[:, 7].cpu().numpy(), retained_addresses
    )


@pytest.mark.parametrize("layout", ["permuted", "negative_gap"])
def test_nonconsecutive_input_slots_fall_back_to_staging(layout: str) -> None:
    """Seeing offsets zero and 127 is insufficient to identify a direct block."""
    h = _Harness()
    slots = list(range(T))
    if layout == "permuted":
        slots[9], slots[10] = slots[10], slots[9]
    else:
        slots.insert(11, -1)
    h.step(slots, seed=31, filled={0: T}, expected_direct={})
    h.check_records([0])


@pytest.mark.parametrize("source_row", [31, 32, 127, 128])
def test_direct_source_rows_cross_warp_and_block_boundaries(source_row: int) -> None:
    """A full source run need not start on a warp or block-size boundary."""
    h = _Harness()
    slots = [-1] * source_row + list(range(2 * T, 3 * T)) + [-1] * 17
    h.step(
        slots,
        seed=81 + source_row,
        filled={2: T},
        expected_direct={2: source_row},
    )
    h.check_records([2])


@pytest.mark.parametrize("mismatch", [31, 63, 95])
def test_interior_mismatch_rejects_direct_even_when_endpoints_match(
    mismatch: int,
) -> None:
    """Each 32-token portion must participate in the complete-run decision."""
    h = _Harness()
    slots = list(range(T))
    slots[mismatch], slots[mismatch + 1] = slots[mismatch + 1], slots[mismatch]
    h.step(slots, seed=201 + mismatch, filled={0: T}, expected_direct={})
    h.check_records([0])


@pytest.mark.parametrize("num_tokens", [0, 1, 127])
def test_direct_detection_respects_active_length_before_trailing_slot_storage(
    num_tokens: int,
) -> None:
    """Inactive buffer contents cannot turn a short input into a complete block."""
    h = _Harness()
    # Even though the buffer contains all offsets, the active input is shorter.
    slots = torch.arange(T + 32, dtype=torch.int64, device="cuda")
    inputs = h.inputs(list(range(num_tokens)), 310 + num_tokens, strided=True)
    h.store.prepare(slots, num_tokens, np.array([], dtype=np.int32))
    assert not bool(torch.any(h.store.direct_rows >= 0))
    filled = {0: num_tokens} if num_tokens else {}
    for layer, (key, value) in enumerate(inputs):
        h.store.write(layer, key, value, slots)
        h.check_gather(layer, filled, (key, value))
        h.store.encode(layer, 0, key=key, value=value)
    h.store.commit()
    assert h.store.stats().encoded == 0
    for layer in range(L):
        h.check_gather(layer, filled)


def test_missing_or_unpaired_original_inputs_are_rejected_on_host() -> None:
    h = _Harness()
    slots = torch.arange(T, dtype=torch.int64, device="cuda")
    key, value = h.inputs(list(range(T)), 41, strided=True)[0]
    h.store.prepare(slots, T, np.array([], dtype=np.int32))
    h.store.write(0, key, value, slots)
    pages = torch.tensor([0], dtype=torch.int32, device="cuda")
    wk = torch.empty((1, T, H, D), dtype=torch.bfloat16, device="cuda")
    wv = torch.empty_like(wk)
    with pytest.raises(ValueError):
        h.store.encode(0, 1)
    with pytest.raises(ValueError):
        h.store.encode(0, 0)
    with pytest.raises(ValueError):
        h.store.gather(0, pages, wk, wv, key=key)
    with pytest.raises(ValueError):
        h.store.encode(0, 1, value=value)


def _gather_unwritten_direct_block_without_inputs() -> None:
    """Intentionally fault only a subprocess's CUDA context, never pytest's."""
    h = _Harness()
    slots = torch.arange(T, dtype=torch.int64, device="cuda")
    key, value = h.inputs(list(range(T)), 42, strided=True)[0]
    h.store.prepare(slots, T, np.array([], dtype=np.int32))
    h.store.write(0, key, value, slots)
    pages = torch.tensor([0], dtype=torch.int32, device="cuda")
    wk = torch.empty((1, T, H, D), dtype=torch.bfloat16, device="cuda")
    wv = torch.empty_like(wk)
    h.store.gather(0, pages, wk, wv)
    torch.cuda.synchronize()


def test_gathering_unwritten_direct_block_without_inputs_fails_on_device() -> None:
    """Device source metadata protects graph replays from stale host state."""
    script = (
        "import runpy, sys; "
        "runpy.run_path(sys.argv[1])"
        "['_gather_unwritten_direct_block_without_inputs']()"
    )
    # Inherit the GPU mask, extension cache and source path selected by the
    # authorized launcher. A fresh interpreter isolates the expected assertion.
    result = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).resolve())],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    output = (result.stdout + result.stderr).lower()
    assert result.returncode != 0, output
    assert "cuda" in output and "device-side assert" in output, output


def test_active_direct_step_can_gather_only_old_encoded_blocks_without_inputs() -> None:
    """Unrelated pending direct blocks do not invalidate published history."""
    h = _Harness()
    h.step(list(range(T)), seed=43, filled={0: T}, expected_direct={0: 0})
    slots_list = list(range(T, 2 * T))
    slots = torch.tensor(slots_list, dtype=torch.int64, device="cuda")
    inputs = h.inputs(slots_list, 44, strided=True)
    h.store.prepare(slots, T, np.array([], dtype=np.int32))
    for layer, (key, value) in enumerate(inputs):
        h.store.write(layer, key, value, slots)
        h.check_gather(layer, {0: T})
        h.store.encode(layer, 1, key=key, value=value)
    h.store.commit()
    h.check_records([0, 1])


def test_overflow_exhaustion_remains_an_explicit_error() -> None:
    h = _Harness(overflow_records=0)
    slots = torch.arange(T, dtype=torch.int64, device="cuda")
    inputs = h.inputs(list(range(T)), 51, strided=True, raw=True)
    h.store.prepare(slots, T, np.array([], dtype=np.int32))
    for layer, (key, value) in enumerate(inputs):
        h.store.write(layer, key, value, slots)
        h.store.encode(layer, 1, key=key, value=value)
    h.store.commit()
    with pytest.raises(StoreError, match=f"overflow={L * 2 * H}"):
        h.store.stats()


@pytest.mark.parametrize("commit_in_graph", [False, True])
def test_graph_replay_uses_current_original_inputs_and_clears_direct_rows(
    commit_in_graph: bool,
) -> None:
    """Capture per-layer work; prepare remains outside capture as documented."""
    h = _Harness()
    slots_list = list(range(2 * T))
    slots = torch.tensor(slots_list, dtype=torch.int64, device="cuda")
    inputs = h.inputs(slots_list, 61, strided=True)
    pages = torch.tensor([1, 0], dtype=torch.int32, device="cuda")
    workspaces = [
        (
            torch.empty((2, T, H, D), dtype=torch.bfloat16, device="cuda"),
            torch.empty((2, T, H, D), dtype=torch.bfloat16, device="cuda"),
        )
        for _ in range(L)
    ]

    def layer_work() -> None:
        for layer, (key, value) in enumerate(inputs):
            h.store.write(layer, key, value, slots)
            h.store.gather(layer, pages, *workspaces[layer], key=key, value=value)
            h.store.encode(layer, 2, key=key, value=value)

    # Compile and initialize before capture on an ordered side stream.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        h.store.prepare(slots, slots.numel(), np.array([], dtype=np.int32))
        layer_work()
        h.store.commit()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    h.store.prepare(slots, slots.numel(), np.array([], dtype=np.int32))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        layer_work()
        if commit_in_graph:
            h.store.commit()
    h.store.commit()
    for seed in (62, 63, 64):
        updated = h.inputs(slots_list, seed, strided=True)
        for old, new in zip(inputs, updated, strict=True):
            old[0].copy_(new[0])
            old[1].copy_(new[1])
        h.store.scratch.fill_(POISON)
        h.store.prepare(slots, slots.numel(), np.array([], dtype=np.int32))
        graph.replay()
        if not commit_in_graph:
            h.store.commit()
        assert not bool(torch.any(h.store.direct_rows >= 0))
        assert bool(torch.all(h.store.scratch == POISON))
        for layer, (wk, wv) in enumerate(workspaces):
            for kv, tensor in enumerate((wk, wv)):
                np.testing.assert_array_equal(
                    _host_bits(tensor), h.ref[layer, kv, [1, 0]]
                )
            h.check_gather(layer, {0: T, 1: T})
        h.check_records([0, 1])
