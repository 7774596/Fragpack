# SPDX-License-Identifier: Apache-2.0
"""CPU checks for benchmark-only repeated and forced-fallback input construction."""

import numpy as np
import pytest

from benchmarks.native_codec_scale_bench import expand_capture
from fragpack.compression import attn_page as ap


@pytest.mark.parametrize(
    "pattern,fraction",
    [("capture-repeat", 0.0), ("fallback25", 0.25), ("fallback100", 1.0)],
)
def test_expanded_records_preserve_source_or_force_lossless_raw(
    pattern: str, fraction: float
) -> None:
    table = np.broadcast_to(np.arange(22, dtype=np.uint8), (1, 2, 4, 22)).copy()
    pool = np.full((2, 1, 2, 128, 4, 128), 0x123, dtype=np.uint16)
    pool[1] = 0x234
    original = pool.copy()
    expanded, metadata = expand_capture(pool, table, 3, pattern)
    assert np.array_equal(pool, original)
    assert metadata["forced_raw_fraction"] == fraction
    actual_raw = 0
    for block in range(3):
        for kv in range(2):
            for head in range(4):
                bits = expanded[block, 0, kv, :, head, :]
                encoded = ap.encode_record(bits, table[0, kv, head], kv)
                actual_raw += len(encoded) == ap.RAW_RECORD_BYTES
                assert np.array_equal(
                    ap.decode_record(encoded, table[0, kv, head], kv), bits
                )
                assert np.all((bits & 0x7F80) != 0x7F80)
                forced = pattern == "fallback100" or (
                    pattern == "fallback25" and (block + kv + head) % 4 == 0
                )
                if not forced:
                    assert np.array_equal(bits, pool[block % 2, 0, kv, :, head, :])
                else:
                    assert len(encoded) == ap.RAW_RECORD_BYTES
                    assert np.array_equal(
                        bits & 0xFF, pool[block % 2, 0, kv, :, head, :] & 0xFF
                    )
    assert actual_raw == metadata["forced_raw_records"]
