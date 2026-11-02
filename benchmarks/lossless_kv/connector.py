# SPDX-License-Identifier: Apache-2.0
"""Benchmark alias of LMCache's MP connector.

vLLM registers its own ``LMCacheMPConnector`` name, which would load the copy
bundled with vLLM; this alias makes vLLM import the connector of the LMCache
checkout on ``PYTHONPATH``.
"""

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector


class LosslessKVBenchMPConnector(LMCacheMPConnector):
    """Same as ``LMCacheMPConnector`` under a name vLLM does not register.

    Thread-safety:
        Same as ``LMCacheMPConnector``.
    """
