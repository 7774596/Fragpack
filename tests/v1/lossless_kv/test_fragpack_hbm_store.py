# SPDX-License-Identifier: Apache-2.0
"""HbmKvStore + fused decode attention against the attn_page reference.

Drives four steps of a small batch through the store's public API (prepare,
write, encode, commit), for BF16 and FP8 E4M3 records, and checks: coded
records bit-identical to ``attn_page.encode_record`` and packed disjointly
inside their block's page, raw records (one fallback; records that no longer
fit a tight page) holding the written KV in their page or in overflow, gather
output equal to the written KV, staging released by GC and commit, and decode
attention over coded, raw, and still-open blocks against an fp32 reference.
FP8 KV has channels at different magnitudes, so the records exercise the
per-channel exponent offsets.
Block blobs exported from one store and imported into other blocks of a second
store must pass the same checks, free the overflow slots they replace, and
reject corrupt or foreign blobs without touching the destination. A port whose
arenas and overflow pool sit at other addresses than the tables record (a
cache server's CUDA IPC mappings) must move the same blobs, and exporting an
open or never-written block must fail. FP8 writes
must produce the bytes of vLLM's FP8 cache write. A request whose chunk closes
its open block and opens a new tail must not need a second persistent slot.
"""

# Standard
from dataclasses import dataclass

# Third Party
import numpy as np
import pytest
import torch

# First Party
from lmcache.v1.lossless_kv.fragpack import attn_page as ap
from lmcache.v1.lossless_kv.fragpack.attn_tables import device_tables, fp8_bytes
from lmcache.v1.lossless_kv.fragpack.fused_attn_decode import (
    GQA,
    decode_attention,
    decode_units,
    workspace_elems,
)
from lmcache.v1.lossless_kv.fragpack.hbm_kv import (
    BlobPort,
    BlobTables,
    HbmKvStore,
    blob_failures,
    page_bytes,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]

L, H, D, T = 2, 2, ap.HEAD_DIM, ap.PAGE_TOKENS
HQ = H * GQA
NB = 12
SCALE = D**-0.5
WIDE = (9, 0, 0, 1)  # (block, layer, kv, head) made incompressible
# FP8 cache scales; powers of two, so the quantization is exact in any
# rounding of the division and the CPU reference can follow it bit for bit.
FP8_SCALES = (2.0, 0.5)


def _bits(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _bf16(bits: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(bits.view(np.int16)).cuda().view(torch.bfloat16)


def _widen(fp8: np.ndarray) -> np.ndarray:
    """FP8 E4M3 bytes -> BF16 bit patterns of the same (exact) values."""
    x = torch.from_numpy(np.ascontiguousarray(fp8)).view(torch.float8_e4m3fn)
    return x.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


def _overflow_used(store: HbmKvStore, records: int) -> int:
    bits = np.unpackbits(store.ovf_bits.cpu().numpy().view(np.uint8))
    return int(bits.sum()) - (bits.size - records)


@dataclass(frozen=True)
class Placement:
    """Where check_arena found the records of the checked blocks.

    Attributes:
        fallbacks: Raw records that do not code (only WIDE may).
        spills: Coded records stored raw because their page was full.
        overflow: Records held in the overflow pool (raw or coded).
    """

    fallbacks: int
    spills: int
    overflow: int


class _Harness:
    def __init__(
        self,
        rng: np.random.Generator,
        page: int,
        overflow_records: int = 16,
        staging_slots: int = 3,
        fp8: bool = False,
        scales: tuple[float, float] = FP8_SCALES,
    ) -> None:
        self.rng = rng
        self.page = page
        self.fp8 = fp8
        self.fmt = ap.record_format(fp8)
        self.scales = scales if fp8 else (1.0, 1.0)
        # Per-channel magnitudes [L, 2, H, D] (FP8 only, which codes them).
        self.chan = np.exp2(rng.integers(-3, 4, (L, 2, H, D))) if fp8 else np.ones(1)
        cal = rng.standard_normal((L, 2, H, 4 * T, D)) * 0.5
        cal = _bits(cal * (self.chan[:, :, :, None] if fp8 else 1.0))
        self.table = np.empty((L, 2, H, ap.TABLE_ENTRIES), np.uint8)
        self.offset = np.zeros((L, 2, H, D), np.uint8)
        for idx in np.ndindex(L, 2, H):
            if fp8:
                self.table[idx], self.offset[idx] = ap.calibrate_fp8(
                    fp8_bytes(cal[idx], self.scales[idx[1]])
                )
            else:
                self.table[idx] = ap.calibrate_table((cal[idx] >> 8).astype(np.uint8))
        # Garbage, like an engine's uninitialized KV cache.
        self.caches = [
            torch.randint(0, 256, (NB, page), dtype=torch.uint8, device="cuda")
            for _ in range(L)
        ]
        self.store = HbmKvStore(
            self.caches,
            H,
            device_tables(self.table, torch.device("cuda"), fp8, self.offset),
            max_tokens=1024,
            staging_slots=staging_slots,
            overflow_records=overflow_records,
        )
        self.overflow_records = overflow_records
        # ref[layer, kv, block, offset, head, dim]: stored scalars of written
        # tokens (BF16 bits, or FP8 bytes after the cache scale).
        self.ref = np.zeros((L, 2, NB, T, H, D), np.uint8 if fp8 else np.uint16)
        # src[layer, kv, block, offset, head, dim]: BF16 bits given to write.
        self.src = np.zeros((L, 2, NB, T, H, D), np.uint16)

    def step(self, rows: list[tuple[list[int], int, int]], live: list[int]) -> None:
        """Run one step; a row is (blocks, first position, token count)."""
        slots = [
            blocks[p // T] * T + p % T
            for blocks, start, n in rows
            for p in range(start, start + n)
        ]
        n = len(slots)
        slot_t = torch.tensor(slots, dtype=torch.int64, device="cuda")
        closes = sum((s + c) // T - s // T for _, s, c in rows)
        self.store.prepare(slot_t, n, np.array(live, np.int32))
        blk, off = np.array(slots) // T, np.array(slots) % T
        for layer in range(L):
            kv = self.rng.standard_normal((2, n, H, D)) * 0.5
            if self.fp8:
                kv *= self.chan[layer][:, None]
            kv = _bits(kv)
            hit = blk == WIDE[0]
            if layer == WIDE[1] and hit.any():
                if self.fp8:
                    # Every E4M3 exponent and both signs: too many escapes.
                    mag = np.exp2(self.rng.uniform(-8, 8, (int(hit.sum()), D)))
                    sign = self.rng.choice([-1.0, 1.0], mag.shape)
                    kv[WIDE[2], hit, WIDE[3]] = _bits(mag * sign * self.scales[0])
                else:
                    wide = np.exp2(self.rng.uniform(-40, 4, (int(hit.sum()), D)))
                    kv[WIDE[2], hit, WIDE[3]] = _bits(wide)
            self.src[layer, :, blk, off] = kv.transpose(1, 0, 2, 3)
            if self.fp8:
                stored = np.stack([fp8_bytes(kv[i], self.scales[i]) for i in range(2)])
            else:
                stored = kv
            self.ref[layer, :, blk, off] = stored.transpose(1, 0, 2, 3)
            # K / V as strided slices of one fused qkv row, like vLLM.
            qkv = torch.empty(n, HQ + 2 * H, D, dtype=torch.bfloat16, device="cuda")
            qkv[:, HQ : HQ + H] = _bf16(kv[0])
            qkv[:, HQ + H :] = _bf16(kv[1])
            key, value = qkv[:, HQ : HQ + H], qkv[:, HQ + H :]
            self.store.write(layer, key, value, slot_t, *self.scales)
            self.store.encode(layer, closes)
        self.store.commit()

    def check_arena(self, blocks: list[int]) -> Placement:
        """Check closed blocks' records and report where they live."""
        addr = self.store.rec_addr.cpu().numpy()
        length = self.store.rec_len.cpu().numpy()
        overflow = self.store.overflow.cpu().numpy().reshape(-1)
        ovf_base = self.store.overflow.data_ptr()
        fmt = self.fmt
        fallbacks = spills = in_overflow = 0
        for b in blocks:
            for layer in range(L):
                arena = self.caches[layer].cpu().numpy().reshape(-1)
                page = self.caches[layer].data_ptr() + b * self.page
                used = []
                for kv in range(2):
                    for h in range(H):
                        ref = self.ref[layer, kv, b, :, h]
                        table = self.table[layer, kv, h]
                        offset = self.offset[layer, kv, h] if self.fp8 else None
                        want = ap.encode_record(ref, table, kv, offset)
                        a = int(addr[layer, b, kv, h])
                        n = fmt.dense_bytes + int(length[layer, b, kv, h])
                        raw = n == fmt.raw_bytes
                        if ovf_base <= a < ovf_base + overflow.size:
                            assert (a - ovf_base) % fmt.raw_bytes == 0
                            got = overflow[a - ovf_base :][:n]
                            in_overflow += 1
                        else:
                            assert a % 16 == 0 and page <= a <= page + self.page - n
                            o = a - self.caches[layer].data_ptr()
                            got = arena[o : o + n]
                            used.append((a, a + n))
                        if not raw:
                            assert got.tobytes() == want, (b, layer, kv, h)
                            continue
                        np.testing.assert_array_equal(
                            ap.decode_record(
                                got.tobytes(), table, kv, self.fp8, offset
                            ),
                            ref,
                        )
                        if len(want) == fmt.raw_bytes:
                            fallbacks += 1
                            assert (b, layer, kv, h) == WIDE
                        else:
                            spills += 1
                used.sort()
                ends, starts = [e for _, e in used[:-1]], [s for s, _ in used[1:]]
                assert all(e <= s for e, s in zip(ends, starts, strict=True))
        return Placement(fallbacks, spills, in_overflow)

    def gather_bits(self, blocks: list[int], layer: int) -> list[np.ndarray]:
        """Gather blocks of one layer; return BF16 bits ``[K, V]``."""
        src = torch.tensor(blocks, dtype=torch.int32, device="cuda")
        wk = torch.empty(len(blocks), T, H, D, dtype=torch.bfloat16, device="cuda")
        wv = torch.empty_like(wk)
        self.store.gather(layer, src, wk, wv)
        return [w.view(torch.int16).cpu().numpy().view(np.uint16) for w in (wk, wv)]

    def stored_bits(self, x: np.ndarray) -> np.ndarray:
        """BF16 bits of stored scalars (FP8 widened, unscaled)."""
        return _widen(x) if self.fp8 else x

    def check_gather(self, filled: dict[int, int]) -> None:
        """Gather blocks of every layer and compare the written tokens."""
        blocks = sorted(filled)
        for layer in range(L):
            got = self.gather_bits(blocks, layer)
            for i, b in enumerate(blocks):
                n = filled[b]
                for kv in range(2):
                    np.testing.assert_array_equal(
                        got[kv][i, :n],
                        self.stored_bits(self.ref[layer, kv, b, :n]),
                        err_msg=f"{b}",
                    )

    def check_decode(
        self, tables: list[list[int]], seqs: list[int], group: int = GQA
    ) -> None:
        """Fused decode vs an fp32 reference over the written KV.

        Runs with the default unit count (one page per chunk here) and with
        fewer units, so chunks span several pages and sequences end mid-chunk.
        ``group`` is the number of query heads per KV head. FP8 records pass
        the K scale in the softmax scale and the V scale as the output scale,
        like the attention backend.
        """
        B = len(seqs)
        width = max(len(t) for t in tables)
        bt = np.zeros((B, width), np.int32)
        for i, t in enumerate(tables):
            bt[i, : len(t)] = t
        block_table = torch.from_numpy(bt).cuda()
        seq_lens = torch.tensor(seqs, dtype=torch.int32, device="cuda")
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        k_scale, v_scale = self.scales
        for units in (decode_units(B, H, sms), B, B + 1):
            opart, ml, plan = workspace_elems(H, units)
            ws = (
                torch.empty(opart, device="cuda"),
                torch.empty(ml, device="cuda"),
                torch.empty(plan, dtype=torch.int32, device="cuda"),
            )
            for layer in range(L):
                q = torch.randn(B, H * group, D, dtype=torch.bfloat16, device="cuda")
                out = torch.empty_like(q)
                decode_attention(
                    q,
                    self.store.layer_tables(layer),
                    block_table,
                    seq_lens,
                    SCALE * k_scale,
                    units,
                    out,
                    ws,
                    layer == 0,
                    out_scale=v_scale,
                )
                kv = _bf16(
                    self.stored_bits(self.ref[layer])
                ).float()  # [2, NB, T, H, D]
                kv[0] *= k_scale
                kv[1] *= v_scale
                ref = torch.empty(B, H * group, D, device="cuda")
                for i, (t, s) in enumerate(zip(tables, seqs, strict=True)):
                    pg = torch.tensor(t, device="cuda")
                    for h in range(H):
                        k = kv[0, pg, :, h].reshape(-1, D)[:s]
                        v = kv[1, pg, :, h].reshape(-1, D)[:s]
                        qh = q[i, group * h : group * (h + 1)].float()
                        ref[i, group * h : group * (h + 1)] = (
                            torch.softmax(qh @ k.T * SCALE, -1) @ v
                        )
                torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


def _tight(fp8: bool) -> int:
    """A page with room for three of a block-layer's four coded records.

    Raw plus two coded records always fit and raw plus three never do, so
    every block-layer places exactly one record in overflow, whichever record
    claims last.
    """
    return 2 * H * ap.record_format(fp8).dense_bytes - 16


FORMATS = [False, True]
PAGES = [
    pytest.param(
        fp8, tight, id=f"{'fp8' if fp8 else 'bf16'}-{'tight' if tight else 'page'}"
    )
    for fp8 in FORMATS
    for tight in (False, True)
]


def _page(fp8: bool, tight: bool) -> int:
    return _tight(fp8) if tight else page_bytes(H, fp8)


@pytest.mark.parametrize("fp8, tight", PAGES)
def test_store_steps_match_reference(fp8: bool, tight: bool) -> None:
    h = _Harness(np.random.default_rng(0), _page(fp8, tight), fp8=fp8)
    assert h.offset.any() == fp8
    a, b, c, d = [0, 1, 4, 5], [8, 9, 10], [2], [3]
    closed = [0, 1, 4, 8, 9]
    # Prefill: A and B close blocks 0, 1, 8, 9 in the same step (transient
    # staging); A's block 4 and C's block 2 stay open (persistent).
    h.step([(a, 0, 300), (b, 0, 256), (c, 0, 10)], a + b + c)
    # A's chunk closes persistent block 4; B opens block 10.
    h.step([(a, 300, 84), (b, 256, 1), (c, 10, 1)], a + b + c)
    torch.cuda.synchronize()
    placed = h.check_arena(closed)
    assert placed.fallbacks == 1
    if tight:
        assert placed.overflow == len(closed) * L
    else:
        # The WIDE block-layer may be one record over its page; whichever of
        # its records claims last (raw or coded) goes to overflow.
        assert placed.overflow <= 1 and placed.spills <= placed.overflow
    # C finished: GC must free block 2's slot, else D / A's claims overflow
    # the 3 persistent slots (10, 3, 5).
    h.step([(a, 384, 1), (b, 257, 1), (d, 0, 10)], a + b + d)
    h.step([(a, 385, 1), (b, 258, 1), (d, 10, 1)], a + b + d)
    torch.cuda.synchronize()
    stats = h.store.stats()
    assert stats.encoded == len(closed) * L * 2 * H
    assert stats.fallbacks == 1 + placed.spills and stats.spills == placed.spills
    assert stats.err_staging == 0 and stats.err_overflow == 0
    assert _overflow_used(h.store, h.overflow_records) == placed.overflow
    # Block 11 was never written: it must read as zeros, not arena garbage.
    h.check_gather({0: T, 1: T, 4: T, 5: 2, 8: T, 9: T, 10: 3, 3: 11, 11: T})
    for group in (GQA, 4, 1):
        h.check_decode([a, b, d, [11]], [386, 259, 11, T], group)


def test_chunk_reopening_tail_inherits_closing_slot() -> None:
    """One slot per request suffices when chunks close and reopen blocks.

    Each chunk of step 2 completes its request's open block and opens a new
    tail, so both requests need two slots within the step while staging has
    one each; the tails must inherit the closing blocks' slots.
    """
    h = _Harness(np.random.default_rng(1), page_bytes(H), staging_slots=2)
    a, b = [0, 1, 2], [3, 4, 5]
    h.step([(a, 0, 200), (b, 0, 200)], [])
    h.step([(a, 200, 100), (b, 200, 100)], [1, 4])
    torch.cuda.synchronize()
    assert h.store.stats().err_staging == 0
    assert h.check_arena([0, 1, 3, 4]) == Placement(0, 0, 0)
    h.check_gather({2: 44, 5: 44})
    h.step([(a, 300, 1), (b, 300, 1)], [2, 5])
    h.step([(a, 301, 83), (b, 301, 83)], [2, 5])
    torch.cuda.synchronize()
    stats = h.store.stats()
    assert stats.err_staging == 0 and stats.err_overflow == 0
    assert stats.encoded == 6 * L * 2 * H
    assert h.check_arena([0, 1, 2, 3, 4, 5]) == Placement(0, 0, 0)
    h.check_gather({b: T for b in range(6)})
    h.check_decode([a, b], [384, 384])


# Source blocks closed by _prefill, and where the import puts them (the WIDE
# block keeps its id so check_arena can recognise its fallback).
SRC, DST = [0, 1, 4, 8, 9], [6, 7, 2, 3, 9]


def _prefill(h: _Harness) -> None:
    a, b, c = [0, 1, 4, 5], [8, 9, 10], [2]
    h.step([(a, 0, 300), (b, 0, 256), (c, 0, 10)], a + b + c)
    h.step([(a, 300, 84), (b, 256, 1), (c, 10, 1)], a + b + c)


class _Port:
    """A store's blob port, optionally through copies at other addresses.

    The remote port works on copies of the arenas and the overflow pool
    while the tables keep recording the store's own addresses, as a cache
    server does over CUDA IPC mappings. The copies are refreshed from the
    store before each call and written back after it.
    """

    def __init__(self, h: _Harness, remote: bool) -> None:
        self.store = h.store
        self.remote = remote
        if not remote:
            self.port = h.store.blob_port()
            return
        own = h.store.blob_tables()
        self.arenas = [a.clone() for a in own.arenas]
        self.overflow = own.overflow.clone()
        self.port = BlobPort(
            BlobTables(
                rec_addr=own.rec_addr,
                rec_len=own.rec_len,
                ovf_bits=own.ovf_bits,
                fill=own.fill,
                overflow=self.overflow,
                arenas=self.arenas,
                arena_stride=own.arena_stride,
                ovf_rec=own.ovf_rec,
                arena_rec=own.arena_rec,
                digest=own.digest,
            )
        )

    def export(self, blocks: list[int]) -> tuple[torch.Tensor, list[int], list[int]]:
        self._pull()
        stride = self.port.blob_stride
        out = torch.zeros(len(blocks) * stride, dtype=torch.uint8, device="cuda")
        offsets = [i * stride for i in range(len(blocks))]
        lengths, status = self.port.export_blocks(blocks, out, offsets)
        return out, lengths.cpu().tolist(), status.cpu().tolist()

    def import_(self, src: torch.Tensor, offsets: list[int], blocks: list[int]):
        self._pull()
        status = self.port.import_blocks(src, offsets, blocks).cpu().tolist()
        if self.remote:
            own = self.store.blob_tables()
            for arena, copy in zip(own.arenas, self.arenas, strict=True):
                arena.copy_(copy)
            own.overflow.copy_(self.overflow)
        return status

    def _pull(self) -> None:
        if self.remote:
            own = self.store.blob_tables()
            for arena, copy in zip(own.arenas, self.arenas, strict=True):
                copy.copy_(arena)
            self.overflow.copy_(own.overflow)


def _export(h: _Harness, blocks: list[int]) -> tuple[torch.Tensor, list[int]]:
    out, lengths, status = _Port(h, remote=False).export(blocks)
    assert status == [0] * len(blocks)
    return out, lengths


def _import(h: _Harness, src: torch.Tensor, offsets: list[int], blocks: list[int]):
    return _Port(h, remote=False).import_(src, offsets, blocks)


@pytest.mark.parametrize("remote", [False, True], ids=["local", "remote"])
@pytest.mark.parametrize("fp8, tight", PAGES)
def test_blob_round_trip_matches_reference(
    fp8: bool, tight: bool, remote: bool
) -> None:
    page = _page(fp8, tight)
    src = _Harness(np.random.default_rng(0), page, fp8=fp8)
    _prefill(src)
    torch.cuda.synchronize()
    src_spills = src.check_arena(SRC).spills
    blobs, lengths, status = _Port(src, remote).export(SRC)
    assert status == [0] * len(SRC)
    if remote:
        assert torch.equal(blobs, _export(src, SRC)[0])
    stride = src.store.blob_port().blob_stride
    assert stride == ap.blob_slot_bytes(L, H, fp8)
    head = ap.blob_head_bytes(L, H)
    dense = src.fmt.dense_bytes
    rec_len = src.store.rec_len.cpu().numpy()
    for b, n in zip(SRC, lengths, strict=True):
        body = sum(dense + int(v) for v in rec_len[:, b].ravel())
        assert n == -(-(head + body) // ap.BLOB_ALIGN) * ap.BLOB_ALIGN <= stride

    # Same seed: same codec tables, so the blobs are accepted. An import
    # places its records before freeing the ones it replaces, so the
    # re-import below needs room for both copies.
    dst = _Harness(np.random.default_rng(0), page, overflow_records=32, fp8=fp8)
    for s_b, d_b in zip(SRC, DST, strict=True):
        dst.ref[:, :, d_b] = src.ref[:, :, s_b]
    offsets = [i * stride for i in range(len(SRC))]
    port = _Port(dst, remote)
    assert port.import_(blobs, offsets, DST) == [0] * len(DST)
    torch.cuda.synchronize()
    placed = dst.check_arena(DST)
    # Records keep their coding: the import neither codes nor spills.
    assert placed.fallbacks == 1 and placed.spills == src_spills
    used = _overflow_used(dst.store, dst.overflow_records)
    assert used == placed.overflow
    # Re-importing frees the overflow records it replaces.
    assert port.import_(blobs, offsets, DST) == [0] * len(DST)
    assert _overflow_used(dst.store, dst.overflow_records) == used
    dst.check_gather({b: T for b in DST})
    dst.check_decode([[6, 7], [2], [3, 9]], [256, 128, 256])


def test_blob_import_rejects_without_side_effects() -> None:
    src = _Harness(np.random.default_rng(0), page_bytes(H))
    _prefill(src)
    blobs, _ = _export(src, [8, 9])
    stride = src.store.blob_port().blob_stride
    dst = _Harness(np.random.default_rng(0), page_bytes(H), overflow_records=1)
    before = dst.store.rec_addr.clone(), dst.store.rec_len.clone()

    bad = blobs.clone()
    bad[0] ^= 1  # magic
    bad[stride + ap.BLOB_FIXED_BYTES] = 8  # first var length: not a record size
    assert _import(dst, bad, [0, stride], [5, 6]) == [1, 2]
    # Tables of another calibration have another digest.
    other = _Harness(np.random.default_rng(1), page_bytes(H))
    assert _import(other, blobs, [0], [5]) == [1]
    # FP8 records are not BF16 records.
    fp8 = _Harness(np.random.default_rng(0), page_bytes(H, True), fp8=True)
    _prefill(fp8)
    fp8_blobs, _ = _export(fp8, [8])
    assert _import(dst, fp8_blobs, [0], [5]) == [1]
    torch.cuda.synchronize()
    assert torch.equal(dst.store.rec_addr, before[0])
    assert torch.equal(dst.store.rec_len, before[1])

    # Block 9's WIDE block-layer (raw record + three coded ones) is one record
    # over its page, which takes the only overflow slot; a second copy finds
    # the pool full in that layer and must release what it took.
    assert _import(dst, blobs, [stride], [9]) == [0]
    assert _import(dst, blobs, [stride], [10]) == [4]
    assert _overflow_used(dst.store, 1) == 1
    torch.cuda.synchronize()
    wide_layer = WIDE[1]
    assert torch.equal(dst.store.rec_addr[wide_layer, 10], before[0][wide_layer, 10])


@pytest.mark.parametrize("remote", [False, True], ids=["local", "remote"])
def test_blob_export_flags_open_blocks(remote: bool) -> None:
    h = _Harness(np.random.default_rng(0), page_bytes(H))
    _prefill(h)
    # 10 and 2 are open (in staging), 5 was never written, 8 is closed.
    _, _, status = _Port(h, remote).export([10, 2, 5, 8])
    assert [s != 0 for s in status] == [True, True, True, False]
    assert blob_failures(status[0]) == "block not closed"


@pytest.mark.parametrize(
    "scales", [(1.0, 1.0), (0.37, 1.9)], ids=["unscaled", "scaled"]
)
def test_fp8_records_hold_vllm_cache_bytes(scales: tuple[float, float]) -> None:
    """Stored FP8 scalars equal vLLM's FP8 cache write of the same KV."""
    # Third Party
    from vllm import _custom_ops as ops

    h = _Harness(np.random.default_rng(0), page_bytes(H, True), fp8=True, scales=scales)
    _prefill(h)
    torch.cuda.synchronize()
    # Closed (coded or raw) and still-open (staging) blocks.
    blocks = {0: T, 1: T, 4: T, 5: 0, 8: T, 9: T, 10: 1, 2: 11}
    ids = sorted(b for b, n in blocks.items() if n)
    k_scale = torch.tensor(scales[0], device="cuda")
    v_scale = torch.tensor(scales[1], device="cuda")
    for layer in range(L):
        got = h.gather_bits(ids, layer)
        slots = torch.tensor(
            [b * T + o for b in ids for o in range(blocks[b])],
            dtype=torch.int64,
            device="cuda",
        )
        src = h.src[layer][:, ids].reshape(2, -1, H, D)
        tok = [
            np.concatenate(
                [src[kv][i * T : i * T + blocks[b]] for i, b in enumerate(ids)]
            )
            for kv in range(2)
        ]
        caches = [
            torch.zeros(NB, T, H, D, dtype=torch.uint8, device="cuda") for _ in range(2)
        ]
        ops.reshape_and_cache_flash(
            _bf16(tok[0]),
            _bf16(tok[1]),
            caches[0].view(torch.float8_e4m3fn),
            caches[1].view(torch.float8_e4m3fn),
            slots,
            "fp8",
            k_scale,
            v_scale,
        )
        want = [c.cpu().numpy() for c in caches]
        for i, b in enumerate(ids):
            n = blocks[b]
            for kv in range(2):
                np.testing.assert_array_equal(
                    got[kv][i, :n], _widen(want[kv][b, :n]), err_msg=f"{layer} {b} {kv}"
                )
