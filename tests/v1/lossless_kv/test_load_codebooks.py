# SPDX-License-Identifier: Apache-2.0
"""Codebook layout consumed by the compressed-KV load kernels."""

# Third Party
import pytest

# First Party
from lmcache.v1.lossless_kv import CompressedStoreGeometry, default_online_codebooks
from lmcache.v1.lossless_kv.format import CODEBOOK_ENTRIES

tilelang_kv = pytest.importorskip("lmcache.v1.lossless_kv.tilelang_kernels")

_PACKED_ROW_BYTES = 16


def _codebooks() -> tuple[bytes, int]:
    geometry = CompressedStoreGeometry(
        num_slots=1,
        slot_size=2 * 2 * 16 * 2 * 128 * 2,
        block_tokens=16,
        num_layers=2,
        num_kv_heads=2,
        head_dim=128,
    )
    rows = geometry.num_layers * 2 * geometry.num_kv_heads
    return default_online_codebooks(geometry)[: rows * CODEBOOK_ENTRIES], rows


def test_rows_place_each_level_in_one_byte_permute_pair() -> None:
    """Primary entries fill bytes 0-6 and secondary entries bytes 8-14."""
    codebooks, rows = _codebooks()
    packed = tilelang_kv.pack_load_codebooks(codebooks, rows)
    assert len(packed) == rows * _PACKED_ROW_BYTES
    for plane in range(rows):
        row = codebooks[plane * CODEBOOK_ENTRIES : (plane + 1) * CODEBOOK_ENTRIES]
        out = packed[plane * _PACKED_ROW_BYTES : (plane + 1) * _PACKED_ROW_BYTES]
        assert out[:7] == row[:7]
        assert out[8:15] == row[7:14]
        assert out[7] == 0 and out[15] == 0


def test_packing_rejects_mismatched_codebook_length() -> None:
    """A codebook blob for another row count is refused."""
    with pytest.raises(ValueError, match="codebook bytes"):
        tilelang_kv.pack_load_codebooks(bytes(CODEBOOK_ENTRIES * 3), 4)


@pytest.mark.parametrize("compile_kernel", ["load", "store"])
def test_kernels_reject_tiles_other_than_the_online_size(compile_kernel: str) -> None:
    """Only the online 256-scalar tile has a TileLang codec kernel."""
    geometry = {"num_blocks": 2, "num_planes": 4, "plane_scalars": 2048}
    with pytest.raises(ValueError, match="tile_scalars"):
        if compile_kernel == "load":
            tilelang_kv.compile_load_kernel(
                **geometry,
                head_dim=128,
                heads=2,
                tile_scalars=1024,
                staging_bytes=4096,
            )
        else:
            tilelang_kv.compile_store_kernels(
                **geometry, head_dim=128, heads=2, tile_scalars=1024
            )


def test_fp8_shifts_pack_a_lane_per_word() -> None:
    """Channels c and 4 + c of a lane share byte c; encode negates."""
    offsets = bytes([0, 1, 7, 15, 2, 3, 4, 5])
    assert tilelang_kv.pack_fp8_shifts(offsets, decode=True) == bytes(
        [0x20, 0x31, 0x47, 0x5F]
    )
    assert tilelang_kv.pack_fp8_shifts(offsets, decode=False) == bytes(
        [0xE0, 0xDF, 0xC9, 0xB1]
    )
    with pytest.raises(ValueError, match="lanes"):
        tilelang_kv.pack_fp8_shifts(bytes(4), decode=True)
