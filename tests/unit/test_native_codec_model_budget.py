# SPDX-License-Identifier: Apache-2.0
"""Completed model-profile accounting is reconciled before choosing replay budgets."""

from types import SimpleNamespace

import pytest

from benchmarks.native_codec_model_budget import profile_accounting


def _worker() -> SimpleNamespace:
    spec = SimpleNamespace(block_size=128, page_size_bytes=16, dtype="bfloat16")
    cache = SimpleNamespace(
        num_blocks=7,
        kv_cache_tensors=[SimpleNamespace(size=112), SimpleNamespace(size=112)],
        kv_cache_groups=[SimpleNamespace(layer_names=["a", "b"], kv_cache_spec=spec)],
    )
    return SimpleNamespace(
        available_kv_cache_memory_bytes=250,
        requested_memory=900,
        total_consumed=600,
        peak_activation_memory=50,
        cudagraph_memory_estimate=0,
        model_runner=SimpleNamespace(model_memory_usage=500, kv_cache_config=cache),
        init_snapshot=SimpleNamespace(
            free_memory=1000,
            total_memory=1000,
            torch_allocated=0,
            torch_memory=0,
            non_torch_memory=0,
        ),
    )


def test_model_loading_is_not_subtracted_twice_and_native_padding_is_retained() -> None:
    result = profile_accounting(_worker())
    assert result["available_kv_cache_memory_bytes"] == 250
    assert result["allocated_native_kv_bytes"] == 224
    assert result["model_loading_memory_bytes"] == 500
    assert result["total_consumed_bytes"] == 600
    assert result["num_gpu_blocks"] == 7


def test_inconsistent_or_incomplete_profile_cannot_choose_a_budget() -> None:
    worker = _worker()
    worker.available_kv_cache_memory_bytes = 400
    with pytest.raises(ValueError, match="reconcile"):
        profile_accounting(worker)
    worker.available_kv_cache_memory_bytes = 0
    with pytest.raises(ValueError, match="invalid"):
        profile_accounting(worker)


def test_native_cache_larger_than_profile_is_rejected() -> None:
    worker = _worker()
    worker.model_runner.kv_cache_config.kv_cache_tensors[0].size = 200
    with pytest.raises(ValueError, match="exceeds"):
        profile_accounting(worker)
