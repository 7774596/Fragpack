# SPDX-License-Identifier: Apache-2.0
"""Version-pinned runtime adaptation for vLLM's V2 model runner.

The execute_model wrapper consumes scheduler metadata not supplied to attention
backends. It is deliberately isolated here: this is a runtime monkey patch of a
specific vLLM release, not a stable public lifecycle extension point. Importing
this module and checking the distribution version do not import torch or vLLM.
"""

from __future__ import annotations

import functools
from importlib import metadata
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fragpack.integrations.vllm.lifecycle import OpenBlocks


SUPPORTED_VLLM_VERSION = "0.30.0+cu129"
_installed_tracker: OpenBlocks | None = None


def require_supported_vllm(*, use_v2_model_runner: bool | None = None) -> str:
    """Reject an unsupported vLLM distribution or an explicitly non-V2 runner.

    Args:
        use_v2_model_runner (bool | None): Resolved runner choice when available.
            None checks only the installed distribution during plugin discovery.

    Returns:
        str: Exact supported distribution version.

    Raises:
        RuntimeError: If vLLM is absent or its distribution version differs.
        ValueError: If the supplied runner choice is not V2.

    Async/thread-safety:
        Reads distribution metadata during synchronous worker initialization.
        Does not import inference modules, initialize CUDA, or modify the runner.
    """
    try:
        installed = metadata.version("vllm")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"FragPack requires vllm=={SUPPORTED_VLLM_VERSION}; vLLM is not installed"
        ) from exc
    if installed != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(
            f"FragPack requires vllm=={SUPPORTED_VLLM_VERSION}; found {installed}. "
            "Its V2 lifecycle adapter is release-specific."
        )
    if use_v2_model_runner is not None and not use_v2_model_runner:
        raise ValueError("FragPack attention needs the V2 model runner")
    return installed


def install_tracker(tracker: OpenBlocks, *, use_v2_model_runner: bool) -> None:
    """Wrap V2 execute_model once to track blocks before each real model step.

    Args:
        tracker (OpenBlocks): Worker-local block tracker shared with the backend.
        use_v2_model_runner (bool): Resolved vLLM runner choice; must be True.

    Returns:
        None. Repeated installation with the same tracker is a no-op.

    Raises:
        RuntimeError: For an unsupported distribution or a different tracker
            after installation, which would otherwise leave the backend stale.
        ValueError: If the selected runner is not V2.

    Async/thread-safety:
        Call during worker initialization, before concurrent model execution.
        The wrapper runs on the model-runner thread. This function imports the
        runner; plugin discovery must call require_supported_vllm instead.
    """
    global _installed_tracker
    require_supported_vllm(use_v2_model_runner=use_v2_model_runner)
    if _installed_tracker is tracker:
        return
    if _installed_tracker is not None:
        raise RuntimeError("FragPack V2 lifecycle adapter already has a tracker")

    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    original = GPUModelRunner.execute_model

    @functools.wraps(original)
    def execute_model(
        self: Any, scheduler_output: Any, *args: Any, **kwargs: Any
    ) -> Any:
        """Track real scheduler steps and forward the original runner contract.

        Args:
            self (Any): The V2 runner instance.
            scheduler_output (Any): Host metadata for this step.
            args (Any): Remaining positional runner arguments.
            kwargs (Any): Runner options, including the dummy_run flag.

        Returns:
            Any: The unmodified result of the original execute_model call.

        Async/thread-safety:
            Runs synchronously on the model-runner thread before attention.
        """
        if not kwargs.get("dummy_run", False):
            tracker.update(scheduler_output)
        return original(self, scheduler_output, *args, **kwargs)

    GPUModelRunner.execute_model = execute_model
    _installed_tracker = tracker
