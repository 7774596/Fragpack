# SPDX-License-Identifier: Apache-2.0
"""Lightweight ``fragpack`` entry point in ``vllm.general_plugins``."""

from __future__ import annotations

from fragpack.integrations.vllm.compat import require_supported_vllm


def register() -> None:
    """Validate the installed release and register FragPack as CUSTOM attention.

    Returns:
        None. Registration records the backend's import path without importing it.

    Raises:
        RuntimeError: If the installed vLLM distribution is unsupported.

    Async/thread-safety:
        Called synchronously by vLLM's plugin loader in each process at startup.
        Does not import FragPack kernels, create a tracker, or initialize CUDA.
    """
    require_supported_vllm()
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    register_backend(
        AttentionBackendEnum.CUSTOM,
        "fragpack.integrations.vllm.backend.FragPackAttentionBackend",
    )
