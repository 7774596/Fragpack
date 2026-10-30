# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for optional FA3 replay geometry, source oracle and bit checks."""

from argparse import Namespace
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.native_codec_kernel_ab_bench import (
    HOST_EXPORTS,
    _assert_bf16_bits,
    _case_specs,
    _describe_sources,
    _hash_file,
    _native_source_layer,
    _pairs,
    _verify_sources,
)


def _options(**kwargs: object) -> Namespace:
    values = dict(blocks=None, page_tokens=None, case_pairs=None, read_queries=None)
    values.update(kwargs)
    return Namespace(**values)


def test_optional_reads_preserve_legacy_and_respect_explicit_geometry() -> None:
    legacy = _case_specs(_options())
    assert legacy == [(n, p) for n in [1, 4, 16, 64, 256] for p in [16, 128]]
    assert _case_specs(_options(read_queries=[1, 128])) == [
        (1, 128),
        (16, 128),
        (256, 128),
        (256, 16),
    ]
    assert _case_specs(_options(read_queries=[1], blocks=[3], page_tokens=[16])) == [
        (3, 16)
    ]
    pairs = _pairs("1:128,256:16")
    assert _case_specs(_options(read_queries=[128], case_pairs=pairs)) == pairs


@pytest.mark.parametrize(
    "options",
    [
        dict(case_pairs=[(1, 128)], blocks=[1]),
        dict(case_pairs=[(1, 128), (1, 128)]),
        dict(case_pairs=[(0, 128)]),
        dict(page_tokens=[17]),
        dict(read_queries=[129], case_pairs=[(1, 128)]),
        dict(read_queries=[1, 1]),
    ],
)
def test_invalid_or_ambiguous_replay_is_rejected(options: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _case_specs(_options(**options))


@pytest.mark.parametrize("page_tokens", [16, 128])
def test_original_source_oracle_keeps_subpage_and_token_order(page_tokens: int) -> None:
    source = np.arange(3 * 2 * 2 * 128 * 2, dtype=np.uint16).reshape(3, 2, 2, 128, 1, 2)
    original = source.copy()
    for layer in range(2):
        native = _native_source_layer(source, layer, page_tokens)
        ids = np.random.default_rng(314).permutation(np.arange(1, native.shape[1] + 1))
        physical = np.full(
            (2, len(ids) + 2, page_tokens, 1, 2), 0x7FC0, dtype=np.uint16
        )
        physical[:, ids] = native
        logical = physical[:, ids].reshape(2, 3, 128, 1, 2).transpose(1, 0, 2, 3, 4)
        np.testing.assert_array_equal(logical, source[:, layer])
        assert np.all(physical[:, [0, -1]] == 0x7FC0)
    np.testing.assert_array_equal(source, original)


def test_output_comparison_uses_bits_including_signed_zero_and_nan_payload() -> None:
    zeros = torch.tensor([0, -32768], dtype=torch.int16).view(torch.bfloat16)
    _assert_bf16_bits(zeros, zeros.clone())
    with pytest.raises(AssertionError):
        _assert_bf16_bits(zeros, torch.zeros_like(zeros))
    nan = torch.tensor([0x7FC1], dtype=torch.int16).view(torch.bfloat16)
    _assert_bf16_bits(nan, nan.clone())
    with pytest.raises(AssertionError):
        _assert_bf16_bits(
            nan, torch.tensor([0x7FC2], dtype=torch.int16).view(torch.bfloat16)
        )


def test_fa3_dependency_change_invalidates_source_guard(tmp_path: Path) -> None:
    dependency = tmp_path / "fa3.so"
    dependency.write_bytes(b"original dependency")
    source = {
        "modules": {},
        "driver_files_sha256": {},
        "attention": {"files_sha256": {str(dependency): _hash_file(dependency)}},
    }
    _verify_sources(source)
    dependency.write_bytes(b"changed dependency")
    with pytest.raises(RuntimeError, match="FA3 dependency changed"):
        _verify_sources(source)


def test_unchanged_generated_translation_unit_keeps_mtime(tmp_path: Path) -> None:
    cuda = tmp_path / "native_codec.cu"
    cuda.write_text("// Source identity only; no compilation in CPU tests.\n")
    cuda.with_name("fragpack_common.cuh").write_text("// common\n")
    cuda.with_name("native_codec_ext.cpp").write_text(
        "\n".join(f'bind("{name}", &{name});' for name in HOST_EXPORTS)
    )
    first = _describe_sources(cuda, "baseline", tmp_path / "build")
    generated = Path(first["generated_tu"])
    original = generated.read_bytes()
    os.utime(generated, ns=(123456789, 987654321))
    second = _describe_sources(cuda, "baseline", tmp_path / "build")
    assert generated.stat().st_mtime_ns == 987654321
    assert generated.read_bytes() == original
    assert second == first
    generated.write_bytes(b"mismatched existing translation unit")
    with pytest.raises(RuntimeError, match="source identity collision"):
        _describe_sources(cuda, "baseline", tmp_path / "build")
