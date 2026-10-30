# SPDX-License-Identifier: Apache-2.0
"""Optional FragPack integration for the fixed LMCache 0.5.6 interface.

Importing this package requires neither LMCache nor a serving engine. Server
and transport implementations are loaded explicitly by their callers.
"""

from fragpack.integrations.lmcache.identity import BlockKey, CacheIdentity

LMCACHE_VERSION = "0.5.6"
LMCACHE_SOURCE_COMMIT = "3e11e10812b448142c6cdb177a3058652bd826fa"

__all__ = ["BlockKey", "CacheIdentity", "LMCACHE_VERSION", "LMCACHE_SOURCE_COMMIT"]
