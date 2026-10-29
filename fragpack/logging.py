# SPDX-License-Identifier: Apache-2.0
"""Small, thread-safe logger setup for the standalone package."""

from __future__ import annotations

import logging
import os
import threading
from typing import IO

_LOCK = threading.Lock()
_HANDLERS: dict[str, logging.Handler] = {}


def init_logger(
    name: str, level: str | None = None, stream: IO[str] | None = None
) -> logging.Logger:
    """Return a module logger with one package-owned stream handler.

    Args:
        name: Fully qualified module name.
        level: Explicit level, or FRAGPACK_LOG_LEVEL (default INFO).
        stream: Optional text destination, normally stderr.

    Returns:
        Configured logging.Logger; propagation is disabled to avoid duplicate
        messages when the serving engine also installs a root handler.

    Async/thread-safety:
        Setup uses a host lock; no CUDA operations or asynchronous work.
        Repeated calls reuse the handler unless a stream is supplied.
    """
    with _LOCK:
        logger = logging.getLogger(name)
        handler = _HANDLERS.get(name)
        if handler is not None and stream is not None:
            logger.removeHandler(handler)
            handler.close()
            handler = None
        if handler is None:
            handler = logging.StreamHandler(stream)
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)s FragPack %(name)s %(message)s"
                )
            )
            logger.addHandler(handler)
            _HANDLERS[name] = handler
        logger.setLevel((level or os.environ.get("FRAGPACK_LOG_LEVEL", "INFO")).upper())
        logger.propagate = False
        return logger
