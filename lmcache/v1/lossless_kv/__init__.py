# SPDX-License-Identifier: Apache-2.0
"""Strict-lossless GPU KV compression for BF16 and FP8 E4M3 caches.

The record format (:mod:`.format`), the CPU reference codec (:mod:`.codec`)
and offline codebook calibration (:mod:`.calibration`) are shared by two
integrations:

- :mod:`.chunk_codec`: encodes the chunks LMCache stages on the GPU before
  they leave it and decodes them after they return, so PCIe, L1 and L2 see
  compressed bytes and vLLM is unchanged.
- :mod:`.fragpack`: keeps vLLM's paged KV cache compressed in HBM and decodes
  it inside attention, through a vLLM attention-backend plugin.
"""

# First Party
from lmcache.v1.lossless_kv.calibration import (
    CalibrationArtifact,
    calibrate_codebooks,
    load_calibrated_codebooks,
    write_calibration_artifact,
)
from lmcache.v1.lossless_kv.codec import (
    EncodedSlot,
    decode_slot,
    default_online_codebooks,
    encode_slot,
)
from lmcache.v1.lossless_kv.format import (
    CompressedStoreGeometry,
    SlotMode,
)

__all__ = [
    "CompressedStoreGeometry",
    "CalibrationArtifact",
    "EncodedSlot",
    "SlotMode",
    "default_online_codebooks",
    "calibrate_codebooks",
    "decode_slot",
    "encode_slot",
    "load_calibrated_codebooks",
    "write_calibration_artifact",
]
