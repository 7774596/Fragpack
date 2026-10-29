# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for version gating and the V2 runtime lifecycle adapter."""

from __future__ import annotations

import importlib
from importlib import metadata
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from fragpack.integrations.vllm import compat
from fragpack.integrations.vllm.lifecycle import OpenBlocks


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Provide a fresh adapter and installed-version metadata, without vLLM."""
    module = importlib.reload(compat)
    monkeypatch.setattr(module.metadata, "version", lambda name: "0.30.0+cu129")
    return module


def _runner_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Install a fake V2 runner at its public import path for CPU-only tests."""
    module = ModuleType("vllm.v1.worker.gpu.model_runner")

    class Runner:
        """Record model calls after the lifecycle adapter has run."""

        def execute_model(
            self, scheduler_output: Any, *args: Any, **kwargs: Any
        ) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
            """Return unchanged arguments to expose wrapper forwarding behavior."""
            return scheduler_output, args, kwargs

    module.GPUModelRunner = Runner
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def _step() -> SimpleNamespace:
    """Return one new writer whose second block is partially filled."""
    return SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=None,
        scheduled_new_reqs=[
            SimpleNamespace(req_id="a", block_ids=([10, 11],), num_computed_tokens=150)
        ],
        block_table_updates=None,
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], num_computed_tokens=[], new_block_ids=[], resumed_req_ids=set()
        ),
    )


@pytest.mark.parametrize("version", ["0.29.0", "0.30.0", "0.30.0+cu130", "0.31.0"])
def test_other_distributions_are_rejected_before_runner_mutation(
    adapter: ModuleType, monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    """A different release or CUDA wheel must not silently install the adapter."""
    runner_module = _runner_module(monkeypatch)
    original = runner_module.GPUModelRunner.execute_model
    monkeypatch.setattr(adapter.metadata, "version", lambda name: version)
    with pytest.raises(RuntimeError, match="release-specific"):
        adapter.install_tracker(OpenBlocks(128), use_v2_model_runner=True)
    assert runner_module.GPUModelRunner.execute_model is original


def test_missing_distribution_has_actionable_error(
    adapter: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absent engine is reported as a compatibility error, without imports."""

    def missing(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(adapter.metadata, "version", missing)
    with pytest.raises(RuntimeError, match="vLLM is not installed"):
        adapter.require_supported_vllm()


def test_non_v2_runner_is_rejected_before_mutation(
    adapter: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The supported wheel alone is insufficient when the V1 runner is chosen."""
    runner_module = _runner_module(monkeypatch)
    original = runner_module.GPUModelRunner.execute_model
    with pytest.raises(ValueError, match="V2 model runner"):
        adapter.install_tracker(OpenBlocks(128), use_v2_model_runner=False)
    assert runner_module.GPUModelRunner.execute_model is original


def test_real_step_updates_tracker_before_model_and_forwards_arguments(
    adapter: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wrapper publishes live blocks before invoking the original runner."""
    runner_module = _runner_module(monkeypatch)
    tracker = OpenBlocks(128)
    original = runner_module.GPUModelRunner.execute_model

    def observe(self: Any, so: Any, *args: Any, **kwargs: Any) -> Any:
        assert tracker.live_blocks().tolist() == [11]
        return original(self, so, *args, **kwargs)

    runner_module.GPUModelRunner.execute_model = observe
    adapter.install_tracker(tracker, use_v2_model_runner=True)
    step = _step()
    result = runner_module.GPUModelRunner().execute_model(step, "intermediate", flag=7)
    assert result == (step, ("intermediate",), {"flag": 7})


def test_dummy_step_bypasses_tracker_but_reaches_model(
    adapter: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dummy runs may omit scheduler metadata and must not change live blocks."""
    runner_module = _runner_module(monkeypatch)
    tracker = OpenBlocks(128)
    tracker.update(_step())
    adapter.install_tracker(tracker, use_v2_model_runner=True)
    assert runner_module.GPUModelRunner().execute_model(None, dummy_run=True) == (
        None,
        (),
        {"dummy_run": True},
    )
    assert tracker.live_blocks().tolist() == [11]


def test_installation_is_idempotent_and_rejects_a_second_tracker(
    adapter: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One worker has one tracker; repeated builders must not stack wrappers."""
    runner_module = _runner_module(monkeypatch)
    tracker = OpenBlocks(128)
    adapter.install_tracker(tracker, use_v2_model_runner=True)
    installed = runner_module.GPUModelRunner.execute_model
    adapter.install_tracker(tracker, use_v2_model_runner=True)
    assert runner_module.GPUModelRunner.execute_model is installed
    with pytest.raises(RuntimeError, match="already has a tracker"):
        adapter.install_tracker(OpenBlocks(128), use_v2_model_runner=True)
    runner_module.GPUModelRunner().execute_model(_step())
    assert tracker.live_blocks().tolist() == [11]


def test_plugin_registers_class_path_without_importing_backend(
    adapter: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plugin discovery registers lazily, leaving numerical code unloaded."""
    from fragpack.integrations.vllm import plugin

    module = ModuleType("vllm.v1.attention.backends.registry")
    module.AttentionBackendEnum = SimpleNamespace(CUSTOM="custom-slot")
    registrations: list[tuple[str, str]] = []

    def register_backend(slot: str, class_path: str) -> None:
        registrations.append((slot, class_path))

    module.register_backend = register_backend
    monkeypatch.setitem(sys.modules, module.__name__, module)
    backend_path = "fragpack.integrations.vllm.backend"
    # A None sentinel makes an attempted backend import fail, regardless of
    # whether a different test previously imported this optional integration.
    monkeypatch.setitem(sys.modules, backend_path, None)
    plugin.register()
    assert registrations == [
        ("custom-slot", "fragpack.integrations.vllm.backend.FragPackAttentionBackend")
    ]
    assert sys.modules[backend_path] is None


def test_plugin_rejects_version_before_registration(
    adapter: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An incompatible release cannot acquire the CUSTOM backend slot."""
    from fragpack.integrations.vllm import plugin

    module = ModuleType("vllm.v1.attention.backends.registry")
    module.AttentionBackendEnum = SimpleNamespace(CUSTOM="custom-slot")

    def register_backend(slot: str, class_path: str) -> None:
        pytest.fail("registration must not run on an unsupported engine")

    module.register_backend = register_backend
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(adapter.metadata, "version", lambda name: "0.31.0")
    with pytest.raises(RuntimeError, match="release-specific"):
        plugin.register()
