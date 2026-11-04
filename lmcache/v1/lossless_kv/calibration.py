# SPDX-License-Identifier: Apache-2.0
"""Offline calibration shared by both lossless KV record layouts.

One fit serves tier records (``codec``) and FragPack attention records
(``fragpack``): a table of the most frequent symbols for every (layer, K/V,
KV head) and, for FP8 E4M3, a per-channel exponent offset that lines up the
channels of a head before the table is counted (K fitted across RoPE
positions). The layouts differ only in how many table entries they use:
tier records take the first 14 (seven primary, seven secondary), FragPack
records all 22.

An artifact directory holds ``tables.npz`` (``table`` uint8 ``[L, 2, H,
22]``, ``offset`` uint8 ``[L, 2, H, head_dim]``, ``fp8``) and
``metadata.json``. FragPack's backend loads ``tables.npz`` directly; the
tier codec loads the directory with :func:`load_calibrated_codebooks`.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

# Third Party
from numpy.typing import NDArray
import numpy as np

# First Party
from lmcache.v1.lossless_kv.format import (
    CODEBOOK_ENTRIES,
    FP8_MANTISSA_BITS,
    CompressedStoreGeometry,
    digest_bytes,
)

TABLES_FILE = "tables.npz"
METADATA_FILE = "metadata.json"
_ARTIFACT_VERSION = 2


@dataclass(frozen=True)
class CalibrationArtifact:
    """Calibrated tables of one model and KV dtype.

    Attributes:
        model_id: Stable model identity supplied by the caller.
        table: uint8 ``[layers, 2, heads, 22]`` symbols per (layer, K/V,
            head), most frequent first: BF16 high bytes, or FP8 bytes with
            the mantissa masked off after the exponent shift.
        offset: uint8 ``[layers, 2, heads, head_dim]`` FP8 exponent offsets
            (zeros for BF16).
        fp8: The tables code FP8 E4M3 KV.
        sample_slots: Captured blocks the tables were fitted on.
        rope_theta: RoPE base FP8 K was re-rotated with, if any.
        max_positions: Longest context FP8 K was fitted over (0 if none).
        source_sha256: Optional digest of the input dump.

    Thread-safety:
        Immutable after construction; safe to share between threads.
    """

    model_id: str
    table: NDArray[np.uint8]
    offset: NDArray[np.uint8]
    fp8: bool
    sample_slots: int
    rope_theta: float | None = None
    max_positions: int = 0
    source_sha256: str | None = None

    @property
    def num_layers(self) -> int:
        """Return the calibrated KV layers."""
        return int(self.table.shape[0])

    @property
    def num_kv_heads(self) -> int:
        """Return the calibrated KV heads (every head of the model)."""
        return int(self.table.shape[2])

    @property
    def head_dim(self) -> int:
        """Return the scalars per head."""
        return int(self.offset.shape[3])

    def metadata(self) -> dict[str, object]:
        """Return reproducibility metadata for the tables.

        Returns:
            JSON-serializable model identity, geometry, fitting options and
            the digest of ``tables.npz``'s arrays.

        Thread-safety:
            Pure read-only serialization; safe to call concurrently.
        """
        return {
            "artifact_version": _ARTIFACT_VERSION,
            "model_id": self.model_id,
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "fp8": self.fp8,
            "table_entries": int(self.table.shape[3]),
            "sample_slots": self.sample_slots,
            "rope_theta": self.rope_theta,
            "max_positions": self.max_positions,
            "tables_sha256": _tables_digest(self.table, self.offset, self.fp8),
            "source_sha256": self.source_sha256,
        }


def calibrate_codebooks(
    dump: NDArray[np.uint16],
    *,
    model_id: str,
    fp8: bool = False,
    rope_theta: float | None = None,
    max_positions: int = 0,
    source_sha256: str | None = None,
) -> CalibrationArtifact:
    """Fit the tables of every (layer, K/V, head) from captured KV.

    Args:
        dump: uint16 ``[slots, layers, 2, tokens, heads, head_dim]`` BF16 bit
            patterns of captured blocks, every KV head of the model.
        model_id: Stable model identity bound into the metadata.
        fp8: Fit FP8 E4M3 tables for the unscaled FP8 quantization of the
            dump, as vLLM's ``--kv-cache-dtype fp8`` stores KV.
        rope_theta: With ``fp8``, the model's RoPE base: K is fitted on the
            captured K re-rotated across ``[0, max_positions)``, as RoPE
            rotates K channel pairs with position.
        max_positions: Longest context the tables serve (with
            ``rope_theta``).
        source_sha256: Optional digest of the dump file.

    Returns:
        The calibration artifact.

    Raises:
        ValueError: If ``model_id`` is empty, the dump is not six-dimensional
            with at least one slot, or ``rope_theta`` lacks ``max_positions``.

    Thread-safety:
        CPU-bound synchronous fitting over caller-owned input.
    """
    # First Party
    from lmcache.v1.lossless_kv.fragpack.attn_tables import calibrate

    if not model_id.strip():
        raise ValueError("model_id must not be empty")
    if dump.ndim != 6 or dump.shape[0] == 0:
        raise ValueError("dump must be [slots, layers, 2, tokens, heads, head_dim]")
    table, offset = calibrate(dump, fp8, rope_theta, max_positions)
    return CalibrationArtifact(
        model_id=model_id,
        table=table,
        offset=offset,
        fp8=fp8,
        sample_slots=int(dump.shape[0]),
        rope_theta=rope_theta if fp8 else None,
        max_positions=max_positions if fp8 and rope_theta is not None else 0,
        source_sha256=source_sha256,
    )


def tier_codebooks(
    table: NDArray[np.uint8],
    offset: NDArray[np.uint8],
    fp8: bool,
    heads: slice | None = None,
) -> bytes:
    """Return the tier codec's tables payload for a head shard.

    Args:
        table: uint8 ``[layers, 2, heads, entries]`` calibrated symbols.
        offset: uint8 ``[layers, 2, heads, head_dim]`` exponent offsets.
        fp8: The tables code FP8 E4M3 KV.
        heads: KV heads the codec serves (a TP rank's contiguous shard);
            every head when None.

    Returns:
        ``format.codec_tables_bytes`` payload: each row's 14 most frequent
        symbols (5-bit sign+exponent for FP8) plus a distinct sentinel, then
        the offsets.

    Raises:
        ValueError: If the shard is empty or a table holds fewer than 14
            entries.

    Thread-safety:
        Pure; safe to call from any thread.
    """
    heads = slice(None) if heads is None else heads
    table = np.asarray(table)[:, :, heads]
    offset = np.asarray(offset)[:, :, heads]
    if table.shape[2] == 0:
        raise ValueError(f"calibrated tables hold no head in {heads}")
    if table.shape[3] < CODEBOOK_ENTRIES - 1:
        raise ValueError("calibrated tables hold too few entries for tier records")
    rows = table.reshape(-1, table.shape[3])
    books = np.empty((len(rows), CODEBOOK_ENTRIES), dtype=np.uint8)
    for index, row in enumerate(rows):
        symbols = (row >> FP8_MANTISSA_BITS) if fp8 else row
        coded = [int(value) for value in symbols[: CODEBOOK_ENTRIES - 1]]
        # Entry 14 never codes a symbol; it only keeps each row distinct.
        sentinel = next(value for value in range(256) if value not in coded)
        books[index] = coded + [sentinel]
    offsets = np.ascontiguousarray(offset, dtype=np.uint8).reshape(len(rows), -1)
    return books.tobytes() + offsets.tobytes()


def load_calibrated_codebooks(
    artifact_dir: str | Path,
    geometry: CompressedStoreGeometry,
    head_start: int = 0,
) -> bytes:
    """Read the tier codec's tables of one serving geometry.

    Args:
        artifact_dir: Directory written by ``write_calibration_artifact``.
        geometry: KV geometry of the serving deployment; its
            ``num_kv_heads`` are the heads of this TP rank.
        head_start: First calibrated head this rank holds.

    Returns:
        The rank's ``format.codec_tables_bytes`` payload.

    Raises:
        ValueError: If the artifact was fitted for another layer count, head
            size or dtype, the rank's heads fall outside it, or its digest
            does not match.

    Thread-safety:
        Synchronous startup file read; safe to call from any thread.
    """
    table, offset, fp8 = load_artifact_tables(artifact_dir)
    if table.shape[0] != geometry.num_layers:
        raise ValueError("calibrated codebooks were fitted for another num_layers")
    if offset.shape[3] != geometry.head_dim:
        raise ValueError("calibrated codebooks were fitted for another head_dim")
    if fp8 != (geometry.dtype_bytes == 1):
        raise ValueError("calibrated codebooks were fitted for another dtype_bytes")
    head_end = head_start + geometry.num_kv_heads
    if head_start < 0 or head_end > table.shape[2]:
        raise ValueError(
            f"KV heads [{head_start}, {head_end}) are outside the "
            f"{table.shape[2]} calibrated heads"
        )
    return tier_codebooks(table, offset, fp8, slice(head_start, head_end))


def load_artifact_tables(
    artifact_dir: str | Path,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8], bool]:
    """Read and verify an artifact's tables.

    Args:
        artifact_dir: Directory written by ``write_calibration_artifact``.

    Returns:
        ``(table, offset, fp8)`` as stored in ``tables.npz``.

    Raises:
        ValueError: If the artifact version or digest does not match.

    Thread-safety:
        Synchronous file read; safe to call from any thread.
    """
    path = Path(artifact_dir)
    metadata = json.loads((path / METADATA_FILE).read_text("utf-8"))
    if metadata.get("artifact_version") != _ARTIFACT_VERSION:
        raise ValueError(
            f"calibration artifact {path} is not version {_ARTIFACT_VERSION}"
        )
    data = np.load(path / TABLES_FILE)
    table, offset, fp8 = data["table"], data["offset"], bool(data["fp8"])
    if metadata.get("tables_sha256") != _tables_digest(table, offset, fp8):
        raise ValueError("calibrated codebook digest mismatch")
    return table, offset, fp8


def write_calibration_artifact(
    artifact: CalibrationArtifact,
    output_dir: str | Path,
) -> None:
    """Write a calibration artifact without retaining activation bytes.

    Args:
        artifact: Calibrated tables and reproducibility metadata.
        output_dir: Directory receiving ``tables.npz`` and ``metadata.json``.

    Thread-safety:
        Synchronous filesystem writes; callers must serialize concurrent
        writes to the same output directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path / TABLES_FILE,
        table=artifact.table,
        offset=artifact.offset,
        fp8=artifact.fp8,
    )
    (output_path / METADATA_FILE).write_text(
        json.dumps(artifact.metadata(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: str | Path) -> str:
    """Return the hex SHA-256 of a file, read in 16 MiB pieces.

    Args:
        path: File to digest.

    Returns:
        Hex digest.

    Thread-safety:
        Synchronous file read; safe to call from any thread.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for piece in iter(lambda: source.read(1 << 24), b""):
            digest.update(piece)
    return digest.hexdigest()


def _tables_digest(
    table: NDArray[np.uint8], offset: NDArray[np.uint8], fp8: bool
) -> str:
    """Return the hex digest binding an artifact's arrays and dtype."""
    payload = (
        bytes([int(fp8)])
        + np.asarray(table.shape, dtype="<i8").tobytes()
        + np.ascontiguousarray(table, dtype=np.uint8).tobytes()
        + np.ascontiguousarray(offset, dtype=np.uint8).tobytes()
    )
    return digest_bytes(payload).hex()


__all__ = [
    "METADATA_FILE",
    "TABLES_FILE",
    "CalibrationArtifact",
    "calibrate_codebooks",
    "file_sha256",
    "load_artifact_tables",
    "load_calibrated_codebooks",
    "tier_codebooks",
    "write_calibration_artifact",
]
