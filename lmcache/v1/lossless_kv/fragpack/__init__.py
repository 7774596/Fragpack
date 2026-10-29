# SPDX-License-Identifier: Apache-2.0
"""FragPack: an HBM-resident compressed paged KV cache with fused decode.

:mod:`.attn_page` defines the per-head record format, :mod:`.attn_tables` the
calibrated lookup tables, :mod:`.hbm_kv` the record store over vLLM's cache,
:mod:`.fused_attn_decode` the decode-attention kernel, and :mod:`.backend` the
vLLM attention backend registered by :mod:`.plugin`.
"""
