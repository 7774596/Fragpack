# SPDX-License-Identifier: Apache-2.0
"""Independent JIT extension for native-page external KV encoding."""

from __future__ import annotations

import functools
from pathlib import Path
from types import ModuleType

_DIR = Path(__file__).parent
_SOURCES = ("native_codec_ext.cpp", "native_codec.cu")


@functools.cache
def extension() -> ModuleType:
    """Return the codec-only extension, building it once per process.

    Initialization compiles under PyTorch's file lock. Calls exposed by the
    extension enqueue work on the current CUDA stream without synchronization.
    This extension does not compile or import HBM storage or attention kernels.
    """
    from torch.utils.cpp_extension import load

    return load(
        name="fragpack_native_codec_cuda",
        sources=[str(_DIR / name) for name in _SOURCES],
        extra_include_paths=[str(_DIR)],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        verbose=False,
    )


__all__ = ["extension"]
