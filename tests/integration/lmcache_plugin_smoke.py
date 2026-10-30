# SPDX-License-Identifier: Apache-2.0
"""Opt-in CPU smoke of the unmodified LMCache 0.5.6 L1 and real ZMQ RPC.

Run in an environment with the exact optional LMCache dependency installed:
``CUDA_VISIBLE_DEVICES='' python
tests/integration/lmcache_plugin_smoke.py``. No monkeypatching or fake LMCache
objects are used. This is a functional storage check, not a serving benchmark.
"""

import argparse
from dataclasses import asdict
import json
import os
import socket
import struct
import sys


def main() -> None:
    """Exercise extension loading, real storage ownership and network framing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--serve-identity",
        type=json.loads,
        help="Serve this identity until stdin EOF, for a separate GPU client smoke",
    )
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Run with CUDA_VISIBLE_DEVICES='' for this CPU-only smoke")
    from lmcache.v1.distributed.config import (
        EvictionConfig,
        L1ManagerConfig,
        L1MemoryManagerConfig,
        StorageManagerConfig,
    )
    from lmcache.v1.multiprocess.config import CoordinatorConfig, MPServerConfig
    from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
    from lmcache.v1.multiprocess.ext_server_module import (
        ExtServerModuleSpec,
        build_server_module_router,
        load_server_module_components,
    )
    from lmcache.v1.multiprocess.protocols.server_module import ServerModuleCallRequest
    from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
    from lmcache.v1.multiprocess.transport.server_factory import create_request_server
    import torch

    from fragpack.compression import attn_page as ap
    from fragpack.integrations.lmcache.client import (
        RPC_METHOD,
        RemoteBlobClient,
        require_lmcache_056,
    )
    from fragpack.integrations.lmcache.identity import CacheIdentity
    from fragpack.integrations.lmcache.protocol import encode_message

    require_lmcache_056()
    identity = (
        CacheIdentity(**args.serve_identity)
        if args.serve_identity is not None
        else CacheIdentity("fragpack-cpu-integration-smoke", -123, 1, 1)
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp_config = MPServerConfig(host="127.0.0.1", port=port, max_cpu_workers=2)
    context = MPCacheServerContext(
        StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=8 << 20, use_lazy=False, shm_name=""
                )
            ),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
        ),
        chunk_size=128,
    )
    modules = []
    server = request_client = None
    try:
        components = load_server_module_components(
            [
                ExtServerModuleSpec(
                    "fragpack.integrations.lmcache.server",
                    config={"identity": asdict(identity)},
                )
            ],
            server_context=context,
            mp_config=mp_config,
            coordinator_config=CoordinatorConfig(),
            built_modules=[],
        )
        modules = list(components.modules)
        router = build_server_module_router(context, modules)
        assert router is not None
        server = create_request_server([router], mp_config)
        server.start()
        assert not torch.cuda.is_initialized()
        if args.serve_identity is not None:
            print(
                "FRAGPACK_SERVER_READY "
                + json.dumps({"endpoint": f"tcp://127.0.0.1:{port}"}),
                flush=True,
            )
            sys.stdin.read()
            return
        request_client = RequestClientFactory.create(f"tcp://127.0.0.1:{port}")
        client = RemoteBlobClient(request_client, identity)
        keys = identity.block_keys(list(range(128 * 3)))
        blob = bytearray(identity.max_blob_bytes)
        struct.pack_into("<4Iqq", blob, 0, ap.BLOB_MAGIC, 1, 1, 1, len(blob), -123)
        raw_var = ap.RAW_RECORD_BYTES - ap.DENSE_BYTES
        struct.pack_into("<2i", blob, ap.BLOB_FIXED_BYTES, raw_var, raw_var)
        blob[ap.blob_head_bytes(1, 1) :] = b"\x34" * (2 * ap.RAW_RECORD_BYTES)
        blob = bytes(blob)
        assert client.get(keys[0]) is None
        assert client.put(keys[0], blob)
        assert client.put(keys[0], blob)
        assert client.get(keys[0]) == blob
        assert client.put(keys[2], blob)
        assert client.lookup_prefix(keys) == 1
        assert client.put(keys[1], blob)
        assert client.lookup_prefix(keys) == 3
        salted = identity.block_keys(list(range(128)), cache_salt="other-tenant")
        assert client.get(salted[0]) is None

        # Exercise a different structural packed extent in the same real L1.
        # GPU codec semantics are checked by the separate transfer validation.
        declared = ap.blob_head_bytes(1, 1) + 2 * (ap.DENSE_BYTES + ap.HEADER_BYTES)
        packed_size = (declared + ap.BLOB_ALIGN - 1) // ap.BLOB_ALIGN * ap.BLOB_ALIGN
        packed = bytearray(packed_size)
        struct.pack_into("<4Iqq", packed, 0, ap.BLOB_MAGIC, 1, 1, 1, declared, -123)
        struct.pack_into("<2i", packed, ap.BLOB_FIXED_BYTES, *([ap.HEADER_BYTES] * 2))
        packed = bytes(packed)
        assert client.put(salted[0], packed)
        assert client.get(salted[0]) == packed
        assert len(packed) < len(blob)

        # Bypass the client's validation to exercise server rejection over ZMQ.
        bad = request_client.server_module_call(
            ServerModuleCallRequest(
                method=RPC_METHOD,
                payload=encode_message({"op": "put", "key": keys[0].to_dict()}, b"bad"),
            )
        ).result(timeout=10)
        assert not bad.success
        other = CacheIdentity("different-model", -123, 1, 1)
        foreign_key = other.block_keys(list(range(128)))[0]
        bad_namespace = request_client.server_module_call(
            ServerModuleCallRequest(
                method=RPC_METHOD,
                payload=encode_message({"op": "get", "key": foreign_key.to_dict()}),
            )
        ).result(timeout=10)
        assert not bad_namespace.success
        assert client.get(keys[0]) == blob
        assert not torch.cuda.is_initialized()
        print(
            json.dumps(
                {
                    "result": "PASS",
                    "lmcache": "0.5.6",
                    "transport": "real ZMQ loopback",
                    "storage": "real CPU L1, 8 MiB",
                    "blob_bytes": len(blob),
                    "packed_envelope_bytes": len(packed),
                    "prefix_blocks": 3,
                    "gpu_initialized": False,
                }
            )
        )
    finally:
        if request_client is not None:
            request_client.close()
        if server is not None:
            server.close()
        for module in modules:
            module.close()
        context.close()


if __name__ == "__main__":
    main()
