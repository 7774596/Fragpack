# LMCache 0.5.6 接入与验证

本页保留固定依赖、启动配置、显式调用示例及 2026-10-30 的验证记录。
历史验证段的版本、计数与范围保持当时含义，不代表当前所有 integration 的状态。
当前设计正文见 [LMCache 压缩记录接入](design/lmcache-integration.md)，
未实现的系统能力与原版接口边界见 [后续接入与优化](design/lmcache-roadmap.md)。
完整系统入口为 [FragPack 系统设计](design/README.md)。

本文的 HBM 显式路线通过官方 0.5.6 公开接口保存和恢复 CPU L1 中的变长 FPKB
记录，由 FragPack 管理有界传输池与后台 GPU 传输。这条路线尚未接入 vLLM
scheduler。独立的[原生 connector](design/native-codec-vllm-connector.md) 已通过
公开 hooks 接入原生 BF16 pages，并完成[2026-11-01 主模型对照](validation-native-codec-serving-20261101.md)。
其 FPKB 编解码和 FPNR Raw 对照使用[原生页传输](design/native-lmcache-transfer.md)，
不使用下文的 HBM export/import。两条路线均不启用 L2 adapter；安装 server module
本身不会自动配置模型 connector。原生路径配置见[开发说明](development.md)。

## Install and configure

Use a separate CPython 3.10 environment compatible with the existing CUDA 12.9
serving pins, then install the channel-specific, hash-pinned official wheel:

```bash
python -m pip install -r requirements/lmcache-cu129.txt
python -m pip install -e '.[dev]'
```

The optional dependency does not become a requirement of the representation or
GPU package. The runtime guard checks distribution version `0.5.6`; the wheel
URL/hash selects the official artifact. It cannot attest to local source edits
in a separately modified installation. This file is an artifact pin, not a full
transitive dependency lock.

The following stock 0.5.6 CLI starts a small CPU-L1-only server. Replace the
example identity with the client's exact `CacheIdentity`: the signed table
digest, layer count and local KV-head count come from its `HbmKvStore`.
Do not configure L2 adapters for this path. Bind a trusted local endpoint;
namespace separation is not network authentication.

```bash
CUDA_VISIBLE_DEVICES='' python -m lmcache.v1.multiprocess.server \
    --host 127.0.0.1 --port 5555 --chunk-size 128 \
    --l1-size-gb 0.0625 --no-l1-use-lazy --shm-name '' \
    --eviction-policy LRU \
    --server-module '{
      "module_path": "fragpack.integrations.lmcache.server",
      "config": {"identity": {
        "model_revision": "MODEL_COMMIT", "table_digest": 123,
        "num_layers": 2, "kv_heads": 1, "rank": 0, "world_size": 1
      }}
    }'
```

The flags above were parsed by the installed official 0.5.6 CLI. The functional
tests below instantiate the same extension factory/router with an 8 MiB L1.

## Explicit client operations

After constructing a compatible GPU `store`, create the client and bounded
transfer worker. `model_revision` must match the server configuration:

```python
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from fragpack.integrations.lmcache import CacheIdentity
from fragpack.integrations.lmcache.client import RemoteBlobClient
from fragpack.integrations.lmcache.connector import PrefixCacheConnector
from fragpack.integrations.lmcache.transfer import BlockTransfer

identity = CacheIdentity(
    model_revision, store.table_digest, store.num_layers, store.kv_heads
)
rpc = RequestClientFactory.create("tcp://127.0.0.1:5555")
client = RemoteBlobClient(rpc, identity)
transfer = BlockTransfer(store, client, identity, num_slots=2)
connector = PrefixCacheConnector(identity, client, transfer)
```

After **all layers** have encoded the source blocks, call
`connector.store_complete(tokens, physical_blocks, start_block=0, max_blocks=2)`.
Both lists describe the full prefix, including any final partial block; the
selected complete blocks alone are exported. Later batches keep the full lists
and advance `start_block`, preserving prefix hashes. `None` means no work was
admitted. A ticket separates `source_ready` (source blocks may be reused) from
`completion` (per-block external publication). Do not wait for RPC in attention.

`connector.lookup(tokens)` returns an advisory contiguous `PrefixMatch`.
Allocate exclusive non-open destination blocks, then call
`connector.restore(match, destinations)`. Limit each batch to available pool
slots; for a long match, pass `PrefixMatch(match.keys[start:stop])` and matching
destinations, stopping subsequent batches at the first failed restore. Keep
destinations alive and hidden from attention until `ticket.completion` resolves.
Only the contiguous successful prefix may skip recomputation. On an exception,
none of that ticket's destinations is reported reusable.

At shutdown, drain `transfer.close(timeout=...)` before closing `rpc`. A timeout
retains worker/buffer ownership; it does not cancel outstanding GPU work. This
API supplies the portable transfer mechanism. Binding it to request allocation,
all-layer completion and scheduler cache events remains a separate vLLM adapter
change.

## Validation — 2026-10-30

Implementation: `2d865c4` on `feat/lmcache-056-plugin`, based on `d8244d4`. Retained
artifact ID: `20261030-lmcache-plugin-056-rm7_yz2h`; the artifact manifest records
implementation file hashes and the logs listed below.

| Check | Observed result |
| --- | --- |
| Full CPU suite | 126 passed; namespace/prefix/length rules, admission, failed copies, RPC timeout and ownership cleanup |
| Stock CPU L1 + real ZMQ | Passed; exact 69,632 / 53,248-byte objects, contiguous lookup, tenant separation, bad-object/namespace rejection |
| GPU transfer tests | 2 passed; mixed packed/raw bit-exact reconstruction, pool refusal and native import overflow |
| GPU + real ZMQ + stock L1 + GPU | Passed; two 128-token blocks, 2 layers, 1 KV head; 98,304 / 135,168-byte blobs, bit-exact reconstruction |
| Static checks | Repository Ruff lint/format checks and `git diff --check` passed |

GPU work used only the explicitly selected, idle physical GPU 2 (H800 PCIe),
with Python 3.10.20, PyTorch 2.13.0+cu129 and CUDA toolkit 12.9. The CPU server
used 8 MiB L1, no shared-memory segment and no L2. Its CUDA initialization stayed
disabled. All owned servers exited; GPU memory returned to zero after the checks.

The official downloaded wheel's SHA256 matched the requirement pin. All 892
installed `lmcache/` files present in the wheel matched its contents. Validation
used an isolated LMCache installation with the existing pinned Torch environment
as a dependency source; this is not a fresh full-stack dependency-install test.
Logs: `cpu-tests-final.log`, `gpu-transfer-final.log`,
`lmcache-cpu-rpc-final.log`, `lmcache-gpu-rpc-final.log`, and
`official-wheel-verification.json`. Two rejected requests in the CPU RPC log are
intentional invalid-blob and wrong-namespace tests.

Review found and fixed lost prefetch cleanup handles on wait/query exceptions,
unbounded pending RPC accumulation after timeout, and insufficient explicit
retention during a failed producer submission. The earlier unit-test development
also caught an exception-propagation error. Initial environment setup exposed a
nested-venv dependency-path issue and an unavailable explicit CPU backend in the
CUDA wheel; validation used automatic CPU fallback with CUDA hidden, without
editing LMCache. No runtime failures remain in the final checks above.

Commands from the repository root, with scratch/JIT output on the approved data
filesystem and GPU selection set only after checking exclusive access:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python tests/integration/lmcache_plugin_smoke.py
python -m pytest -q tests/gpu/test_fragpack_lmcache_transfer.py
PYTHONPATH=. python tests/integration/lmcache_gpu_smoke.py
```

Use the pinned optional LMCache environment for the two integration scripts.
The GPU integration starts its CPU server with CUDA hidden and closes it on
exit. GPU transfer and serving performance are separate checks; no full-model
serving, CUDA Graph transfer, TP>1 coordination, L2 recovery or speedup is claimed.
