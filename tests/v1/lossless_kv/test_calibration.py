# SPDX-License-Identifier: Apache-2.0
"""One calibration serves tier records and FragPack records."""

# Standard
import json

# Third Party
import numpy as np
import pytest

# First Party
from lmcache.v1.lossless_kv import (
    CompressedStoreGeometry,
    SlotMode,
    calibrate_codebooks,
    decode_slot,
    encode_slot,
    load_calibrated_codebooks,
    write_calibration_artifact,
)
from lmcache.v1.lossless_kv.calibration import (
    TABLES_FILE,
    load_artifact_tables,
    tier_codebooks,
)
from lmcache.v1.lossless_kv.format import CODEBOOK_ENTRIES, split_codec_tables
from lmcache.v1.lossless_kv.fragpack.attn_tables import fp8_bytes

_LAYERS, _HEADS, _TOKENS, _DIM = 2, 2, 64, 128


def _dump(seed: int, slots: int = 2) -> np.ndarray:
    """Return BF16 bits ``[slots, layers, 2, tokens, heads, dim]`` whose
    heads and channels have different scales."""
    rng = np.random.default_rng(seed)
    shape = (slots, _LAYERS, 2, _TOKENS, _HEADS, _DIM)
    scale = 2.0 ** rng.integers(-4, 4, (1, _LAYERS, 2, 1, _HEADS, _DIM))
    values = (rng.standard_normal(shape) * scale).astype(np.float32)
    return (values.view(np.uint32) >> 16).astype(np.uint16)


def _geometry(fp8: bool, heads: int = _HEADS) -> CompressedStoreGeometry:
    dtype_bytes = 1 if fp8 else 2
    return CompressedStoreGeometry(
        num_slots=1,
        slot_size=_LAYERS * 2 * _TOKENS * heads * _DIM * dtype_bytes,
        block_tokens=_TOKENS,
        num_layers=_LAYERS,
        num_kv_heads=heads,
        head_dim=_DIM,
        dtype_bytes=dtype_bytes,
    )


def _raw(dump: np.ndarray, fp8: bool, heads: slice | None = None) -> bytes:
    slot = np.ascontiguousarray(dump[0][:, :, :, heads or slice(None)])
    return (fp8_bytes(slot) if fp8 else slot).tobytes()


@pytest.mark.parametrize("fp8", [False, True])
def test_artifact_round_trips_both_dtypes(tmp_path, fp8: bool) -> None:
    dump = _dump(1)
    artifact = calibrate_codebooks(
        dump,
        model_id="example/model",
        fp8=fp8,
        rope_theta=1e6 if fp8 else None,
        max_positions=4096 if fp8 else 0,
        source_sha256="source-digest",
    )
    metadata = artifact.metadata()
    assert metadata["model_id"] == "example/model"
    assert metadata["source_sha256"] == "source-digest"
    assert metadata["fp8"] is fp8
    assert artifact.table.shape == (_LAYERS, 2, _HEADS, 22)
    assert artifact.offset.shape == (_LAYERS, 2, _HEADS, _DIM)
    assert artifact.offset.any() == fp8

    write_calibration_artifact(artifact, tmp_path)
    assert json.loads((tmp_path / "metadata.json").read_text()) == metadata
    table, offset, stored_fp8 = load_artifact_tables(tmp_path)
    assert np.array_equal(table, artifact.table)
    assert np.array_equal(offset, artifact.offset)
    assert stored_fp8 == fp8

    geometry = _geometry(fp8)
    codebooks = load_calibrated_codebooks(tmp_path, geometry)
    raw = _raw(dump, fp8)
    encoded = encode_slot(raw, slot_id=0, geometry=geometry, codebooks=codebooks)
    assert encoded.mode is SlotMode.COMPRESSED
    assert (
        decode_slot(
            encoded.payload,
            mode=encoded.mode,
            slot_id=0,
            geometry=geometry,
            codebooks=codebooks,
        )
        == raw
    )


@pytest.mark.parametrize("fp8", [False, True])
def test_tier_tables_are_the_leading_fragpack_entries(fp8: bool) -> None:
    """Tier rows are each table's 14 most frequent symbols plus a sentinel."""
    artifact = calibrate_codebooks(_dump(2), model_id="m", fp8=fp8)
    books, offsets = split_codec_tables(
        tier_codebooks(artifact.table, artifact.offset, fp8),
        num_layers=_LAYERS,
        num_kv_heads=_HEADS,
        head_dim=_DIM,
        fp8=fp8,
    )
    table = artifact.table.reshape(-1, 22)
    leading = (table >> 3) if fp8 else table
    assert np.array_equal(books[:, : CODEBOOK_ENTRIES - 1], leading[:, :14])
    assert np.array_equal(offsets, artifact.offset.reshape(-1, _DIM))


@pytest.mark.parametrize("fp8", [False, True])
def test_rank_loads_its_head_shard(tmp_path, fp8: bool) -> None:
    dump = _dump(3)
    artifact = calibrate_codebooks(dump, model_id="m", fp8=fp8)
    write_calibration_artifact(artifact, tmp_path)
    shard = _geometry(fp8, heads=1)

    rank1 = load_calibrated_codebooks(tmp_path, shard, head_start=1)

    assert rank1 == tier_codebooks(artifact.table, artifact.offset, fp8, slice(1, 2))
    raw = _raw(dump, fp8, slice(1, 2))
    encoded = encode_slot(raw, slot_id=0, geometry=shard, codebooks=rank1)
    assert (
        decode_slot(
            encoded.payload,
            mode=encoded.mode,
            slot_id=0,
            geometry=shard,
            codebooks=rank1,
        )
        == raw
    )
    with pytest.raises(ValueError, match="outside"):
        load_calibrated_codebooks(tmp_path, shard, head_start=2)


def test_load_rejects_mismatched_or_tampered_artifacts(tmp_path) -> None:
    artifact = calibrate_codebooks(_dump(4), model_id="m")
    write_calibration_artifact(artifact, tmp_path)
    with pytest.raises(ValueError, match="dtype"):
        load_calibrated_codebooks(tmp_path, _geometry(fp8=True))
    other = CompressedStoreGeometry(
        num_slots=1,
        slot_size=_LAYERS * 2 * _TOKENS * _HEADS * 64 * 2,
        block_tokens=_TOKENS,
        num_layers=_LAYERS,
        num_kv_heads=_HEADS,
        head_dim=64,
    )
    with pytest.raises(ValueError, match="head_dim"):
        load_calibrated_codebooks(tmp_path, other)

    np.savez(
        tmp_path / TABLES_FILE,
        table=artifact.table[::-1].copy(),
        offset=artifact.offset,
        fp8=False,
    )
    with pytest.raises(ValueError, match="digest"):
        load_calibrated_codebooks(tmp_path, _geometry(fp8=False))
