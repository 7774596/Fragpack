# SPDX-License-Identifier: Apache-2.0
"""Shared CPU-safe test configuration; GPU suites opt in by explicit path.

The standalone tests intentionally do not import a serving engine, initialize
CUDA, or load DaseR fixtures during collection. Tests requiring CUDA live in
``tests/gpu`` and are not part of the default pytest testpaths.
"""
