# LMCache integration target — 2026-10-30

## Decision and verification boundary

Select upstream **LMCache v0.5.6, multiprocess (MP) mode**, as the fixed target
for FragPack external prefix-cache integration. This is a source/API and
release-artifact assessment, not a completed integration or runtime validation.
No package installation, GPU execution or third-party modification was performed.

| Component | Selected value |
| --- | --- |
| Repository | `https://github.com/LMCache/LMCache` |
| Source tag | `v0.5.6` |
| Source commit | `3e11e10812b448142c6cdb177a3058652bd826fa` |
| CUDA release tag | `v0.5.6-cu129`, same source commit |
| Python / platform | CPython 3.10, Linux x86_64 |
| vLLM | Keep `0.30.0+cu129` |
| PyTorch | Keep `2.13.0+cu129` |
| CUDA toolkit | Keep 12.9 |
| Initial hardware / model boundary | H800, TP=1, BF16 KV, 128-token blocks, current validated Qwen3 MoE model |

The official stable release is dated October 14. Its build configuration pins
PyTorch 2.13.0 and allows Python 3.10; the CUDA 12.9 release includes the matching
CPython 3.10 wheel. This makes it preferable to introducing a newer RC or
changing the serving stack during the integration. These matches do not prove
native ABI loading, vLLM 0.30 behavior, or FragPack layout compatibility.
[Release](https://github.com/LMCache/LMCache/releases/tag/v0.5.6),
[pinned build configuration](https://github.com/LMCache/LMCache/blob/3e11e10812b448142c6cdb177a3058652bd826fa/pyproject.toml).

Pin this CUDA-channel-specific artifact, not just the package version:

```text
https://github.com/LMCache/LMCache/releases/download/v0.5.6-cu129/lmcache-0.5.6-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
SHA256: f577840e2de146b33bc84b96affe1ccee67b13f32592f59935a6307d73deb6ca
```

The hash was checked against the GitHub release asset metadata, not a locally
downloaded wheel. The wheel version is `0.5.6` without a `+cu129` suffix; the
release URL selects the CUDA channel. Dependency resolution belongs in a new
isolated environment constrained by the existing serving pins. The complete
transitive dependency lock and binary import checks are still outstanding.
[CUDA 12.9 assets](https://github.com/LMCache/LMCache/releases/expanded_assets/v0.5.6-cu129).

The existing local GentleCold LMCache checkout contains DaseR-specific changes.
It is useful historical context, but is not the upstream baseline selected here.
The subsequent [daser-codec reference review](lmcache-daser-codec-review-20261030.md)
identifies concrete variable-size prefetch and asynchronous-store mechanisms
to adapt, together with their persistence and failure-cleanup limitations.

## Intended behavior

The proposed architecture is available as an [editable SVG](figures/fragpack-vllm-lmcache.svg)
and [PDF](figures/fragpack-vllm-lmcache.pdf). The
[rendering source](figures/draw_architecture.py) accompanies both exports.

LMCache should own external cache residency and eviction. FragPack should own
GPU record interpretation, encoding, attention and import/export. Integration
code should live under `fragpack/integrations/lmcache/`; the representation and
GPU core must remain independent of LMCache.

```text
Save, after all layers of a block are encoded:
  FragPack arena / raw overflow
    -> export_blocks -> bounded packed GPU transfer buffers
    -> host transfer buffer / LMCache L1 compressed objects
    -> LMCache L2 NVMe compressed objects

Restore, after a contiguous prefix lookup:
  LMCache L1, or L2 -> L1
    -> bounded GPU blob buffers
    -> import_blocks -> destination arena / raw overflow
    -> successful import completion -> attention on the restored prefix
```

This is the proposed data path. It does not claim direct NVMe-to-GPU DMA or a
zero-copy import. Export/import move encoded records without first reconstructing
BF16 history. The attention reader remains selectable: gather + FA3 is the
default, and experimental fused attention can consume imported records later.
The direct-write option concerns newly generated complete blocks and is separate
from restoring encoded external records.

Initially use one external object per complete 128-token block, containing all
layers for one TP rank. Leave incomplete tails local. Align any LMCache token
chunking with that boundary; the proposed initial chunk size is 128, not an
assumption about upstream defaults. Keep the first validation at TP=1.

## Why stock tensor transfer is insufficient

FragPack's vLLM cache view is a packed arena exposed through the custom KV spec.
A stock layout detector can mistake that view for a normal dense KV layout;
successful registration would not establish correctness. Copying arena bytes
alone omits record lengths, relocatable record structure and raw-overflow data.
Absolute GPU record pointers cannot serve as a portable cache representation.
Use the existing block blob contract instead.

The standard MP engine-driven path registers fixed tensor layouts and uses
dense gather/scatter. Its SDK similarly assumes a KV tensor geometry. Selecting
an MP transfer mode does not add support for FragPack variable-length blobs.
[Pinned worker transfer](https://github.com/LMCache/LMCache/blob/3e11e10812b448142c6cdb177a3058652bd826fa/lmcache/v1/multiprocess/transfer_context/worker_transfer.py),
[SDK context](https://github.com/LMCache/LMCache/blob/3e11e10812b448142c6cdb177a3058652bd826fa/lmcache/sdk/context.py).

For the ordinary BF16 control arm, vLLM can explicitly load the LMCache-shipped
connector:

```json
{
  "kv_connector": "LMCacheMPConnector",
  "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
  "kv_role": "kv_both"
}
```

This is a control configuration, not a working FragPack configuration. The
FragPack arm needs its own connector module. vLLM supports external MP connector
selection in this version range; exact feature compatibility still requires
testing. [Official compatibility guidance](https://docs.lmcache.ai/getting_started/compatibility.html).

## Candidate extension boundary and unresolved design work

v0.5.6 provides out-of-tree server modules, a module factory receiving the
shared server context, namespaced opaque request/response handlers and a public
client call. This is the preferred extension boundary for a FragPack-owned
worker connector plus server plugin, before considering changes to LMCache.
The envelope is not itself an asynchronous bulk-transfer implementation.
[Pinned extension interface](https://github.com/LMCache/LMCache/blob/3e11e10812b448142c6cdb177a3058652bd826fa/lmcache/v1/multiprocess/ext_server_module.py),
[client interface](https://github.com/LMCache/LMCache/blob/3e11e10812b448142c6cdb177a3058652bd826fa/lmcache/v1/multiprocess/transport/base.py).

The design must resolve these contracts before implementation:

1. **Variable size and durable metadata.** `blob_stride` reserves the all-raw
   worst case; `export_blocks` returns actual aligned lengths on the GPU.
   Budget transfer buffers by bounded concurrency, and account separately for
   logical payload, allocation padding and reserved buffers. Persist format,
   length and codec identity so objects can be allocated correctly after L1
   eviction and, if required, server restart. Upstream `reserve_write` accepts
   a layout per reservation, but ordinary prefetch uses registered group layouts;
   the filesystem L2 adapter stores raw tensor bytes without a layout header.
   A size catalog/manifest or size-class design and a corresponding restore
   path are needed. An opaque byte RPC alone does not solve this.
   [Storage manager](https://github.com/LMCache/LMCache/blob/3e11e10812b448142c6cdb177a3058652bd826fa/lmcache/v1/distributed/storage_manager.py).
2. **Save ownership.** Export only after all layers are encoded and pin source
   blocks until export completes. vLLM layer save callbacks occur after each
   layer's attention, but FragPack's current export is all-layer; an early layer
   callback is therefore not a valid whole-block publication point. Prefer
   post-forward/all-layer completion initially. Publish a cache hit only when
   its payload and metadata are complete, with explicit write-failure cleanup.
3. **Restore ownership.** Import only into allocated, non-open destination
   blocks that attention cannot read yet. Order DMA, import and model streams
   with completion events. Check import status before reporting a successful
   load; corrupt records, table mismatches and overflow exhaustion must trigger
   invalidation/recomputation. Async status/length observation belongs outside
   the per-layer attention path. Import may update some layers before a later
   layer fails; failure must never be exposed as a partially usable block.
   Existing export/import allocate their lengths/status tensors; they should
   initially run outside model graph capture, rather than being described as
   allocation-free transfer APIs.
4. **Identity and partial reuse.** Cache keys must distinguish model revision,
   token-prefix chain, representation version, codec tables, KV geometry and
   rank/shard layout. Only contiguous valid prefix blocks may be skipped. Respect
   vLLM local APC ownership; do not overwrite an APC-shared block on restore.
5. **Budget and lifetime.** Use a bounded pool of transfer slots, retain owners
   until asynchronous work finishes, and apply backpressure. Preserve L1 read
   locks until restore has consumed the bytes. Do not allocate an additional
   full-history BF16 cache for compatibility with the default connector.

The plugin boundary is source-confirmed; the full compressed-object lifecycle
using unchanged upstream storage APIs remains a design candidate, not a proven
drop-in implementation.

## Initial acceptance sequence

1. In an isolated environment, validate the pinned native wheel and run an
   ordinary BF16 LMCache/vLLM control with the same model and serving settings.
2. First prove the variable-size storage protocol without a GPU: reserve actual
   payload bytes, publish, evict L1, restore from L2 using retained length
   metadata, and verify bytes and failure cleanup. Then implement and validate
   bounded all-layer blob save/restore against L1 only:
   bit-exact KV reconstruction, mixed compressed/raw records, eviction,
   cancellation, corrupted metadata and overflow failure.
3. Add L2 with persisted length metadata. Force an external hit by removing
   the GPU-local prefix; force an L2 hit by evicting its L1 copy. Verify actual
   L2 read, L1 promotion, successful GPU import and subsequent L1 reuse.
4. Measure matched BF16 versus FragPack runs: TTFT and throughput, transferred
   bytes, effective L1/L2 residency, import/export time, and all buffer overhead.
   Separate cold miss, L1 hit and L2 hit. Do not infer serving gains solely from
   codec ratio or byte-copy timing.

Full-model graphs, TP>1, cross-engine sharing, blending, direct storage DMA and
automatic selection of experimental write/prefill modes are subsequent work.
