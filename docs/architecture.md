# Architecture

FragPack couples a lossless KV representation with a GPU store and attention
implementation. The core operates on records and tensors; an inference-engine
adapter supplies block ownership and scheduling information.

## Representation

A record contains a 128-token by 128-dimension BF16 slice for one layer, K/V
component, and KV head. Low bytes are stored directly. High bytes use 3-bit
symbols, calibrated lookup tables, and escape data. Values absent from the
tables remain representable; records exceeding the encoding bound use raw
overflow storage.

Records share a block-layer page. Its configured size is
`2 * kv_heads * (22528 + 1280)` bytes. For four KV heads this is 190,464 bytes,
compared with 262,144 raw bytes. The resulting approximately 1.38 layout ratio
excludes auxiliary storage and is not a total-GPU-memory saving guarantee.

The table file contains `table`, a uint8 array shaped `[layers, 2, kv_heads, 22]`.
Calibration affects encoded size, while escape and raw paths preserve the
original BF16 bits. Exported records carry a table digest to reject incompatible
imports.

## GPU lifecycle

1. The engine adapter identifies newly claimed blocks and partially written
   blocks that remain live. The store prepares staging and ownership metadata.
2. Each layer writes new KV into staging. Attention reads encoded history and
   unfinished raw blocks through record address tables.
3. Blocks filled to 128 tokens are encoded into shared pages or raw overflow.
   A newly opened tail can inherit storage from a block closed in that step.
4. The final layer commits the step and releases unneeded staging.

The live-block set must be exact: retaining a block after reassignment can keep
stale staging alive and exhaust the pool. Overflow allocation and ownership
must remain valid across model and import streams. Failed imports must not
publish partially valid records or free records still owned by the destination.

Decode reconstructs operands inside the attention kernel and combines partial
attention results. Prefill gathers cached context into a bounded BF16 workspace
for FlashAttention. Within each workspace pass, a physical KV block is gathered
once and all requests referencing it share the decoded page. Passes count
unique blocks against the workspace budget and preserve request/token order;
a block can be gathered again in a later pass, layer, or model step. Shared
pages are read-only during attention, and the next gather reuses the workspace
only after the previous attention on the same stream. Open tails and closed
records still use the existing gather implementation. This reduces duplicate
materialization; it does not fuse decompression into prefill attention or
reduce the preallocated workspace size.

The host planner uses the scheduler's mirrored block lists and the V2 runner's
actual request-index mapping, never a GPU-to-CPU block-table readback. Dummy
runs use the original non-sharing behavior without consulting a previous real
batch. This planning change adds no hardware requirement; the existing SM90
store, gather, and attention requirements remain.

### Direct encoding of complete new blocks

The opt-in `FRAGPACK_WRITE=direct` path removes the kernel-order scratch
round trip for complete blocks generated in one step. It is supported with
`FRAGPACK_PREFILL=gather`; the experimental fused-prefill reader still uses
staged writes. `FRAGPACK_WRITE=staged` retains the comparison path.

During prepare, a GPU source-row map identifies fresh claims whose 128 input
slots are consecutive and cover the complete physical block. Existing open
blocks and incomplete new tails remain staged. Claims, scratch allocation,
overflow ownership and commit rules are unchanged: this first implementation
saves transfers, not the preallocated scratch capacity.

Coverage validation is cooperative within the existing prepare kernel. The
claim phase tentatively records source rows only for successfully allocated
fresh blocks that close in this step and have 128 available input rows. After
the existing claim barrier, one warp checks all 128 slots for each candidate
(four slots per lane); a failed comparison clears its source row. This accepts
arbitrary input-row offsets without trusting alignment or just the endpoints.
The map becomes consumable only after prepare finishes on the ordered stream.
Staged mode skips this validation, and the optimization adds no kernel launch,
device readback, buffer, or CTA barrier. Old tails, staging exhaustion, slot
inheritance and commit retain their existing behavior.

Prepare's final metadata pass visits `(layer, claim, K/V head)` order to match
the layer-major record tables. It still frees every old overflow record and
resets every claimed record's address and raw length, with one arena-fill reset
per block-layer. Each record is visited exactly once; an empty claim list
performs no indexing or division. Only traversal changes: direct blocks must
also detach their old overflow addresses so an interrupted or repeated prepare
cannot later free a slot already reassigned to another block. Locality gains
depend on the physical claim order and require measurement.

For each layer, write skips direct blocks. Gather copies their original BF16
K/V directly into the existing token-major attention workspace while other
blocks follow the record reader. FA3 therefore consumes new K/V without first
encoding and decoding them. After attention, the encoder loads those same
original tensors directly, rearranges values in registers, and uses the
unchanged codec, page allocation and raw-overflow paths. Existing tails still
encode from staging, including tail-slot inheritance.

The source row indexes the complete layer batch, not an individual gather
group. The caller must supply that layer's original K/V to every gather and
encode while the direct step is active, keep the tensors alive and immutable
through encoding, and order all operations on the same CUDA stream. Missing
inputs are an error; unfilled scratch must never be used as a substitute.
Encode rejects missing inputs on the host. Gather uses the GPU source map:
an uncommitted direct block without original inputs raises a CUDA assertion,
while committed records can be gathered normally. A Python active-step flag
cannot make this distinction because commit may execute inside graph replay.
The GPU map survives all layers of the step, is cleared by commit and reset
by prepare, so graph capture or dummy forwards cannot reuse stale input rows.
Ordinary record consumers and export are valid for direct blocks only after
their layer has been encoded; cross-stream consumers must wait for publication.
There is no record-format change or new synchronization/readback in this path.

### Direct fused prefill (experimental)

`FRAGPACK_PREFILL=fused` selects an experimental path at worker initialization;
`gather` retains the existing workspace-plus-FlashAttention path for comparison.
The fused path reads the original physical block table and record address tables.
It reconstructs K/V tiles inside a single attention kernel, reuses them across
multiple queries, and accumulates attention with an online softmax. Reconstructed
KV never goes through a global BF16 prefill workspace. Query positions use the
right-aligned causal boundary `context_length - query_length + query_index`,
including chunked prefill and partial final blocks.

In fused mode no BF16 prefill K/V workspace is allocated and its reported size
is zero. Compressed closed records, raw overflow records, and open blocks use
the same record interface. New tokens are still written to the store's open-block
staging before attention and encoded when full; this change removes the extra
prefill materialization, not the writable KV lifecycle. All reads, output writes,
and subsequent encode operations are ordered on the worker's current stream.
The record format and external export/import contracts are unchanged. The
experimental path retains the current SM90/BF16/128-dimension compatibility
boundary; correctness and per-layer timings are separate from serving speedups.

The short-suffix optimization splits each query tile's visible KV pages across
multiple CTAs. Each split writes FP32 unnormalized output, a base-2 softmax
maximum and its exponential sum; a second kernel merges those partials using
their common maximum. Empty splits write zero output/sum and negative infinity
as their maximum, and the combine ignores them. This workspace contains
attention partials, never reconstructed BF16 KV. A one-split launch writes the
final output directly and requires no partial workspace or combine launch.

Split selection uses the already available host query/context lengths and SM
count, without reading device metadata back. It targets at most one wave of
active 64-column CTAs, caps the split count at 32, and does not exceed the page
count of the largest active context. A one-page chunk is allowed: the split
count sweep found it faster than a two-page floor for the 1,024-token/16-query
case. Batches with enough query tiles keep one split.
The rectangular launch can also include padded CTAs that immediately return.
This bounds active partials by `num_sms * 64` query-head rows. The adapter owns
and reuses at most `num_sms * 64 * (128 + 2) * 4` bytes across layers, reported
separately as `prefill_partial_workspace`; the BF16 `prefill_workspace` remains
zero. This is a heuristic to validate, not an established optimal schedule.

The core API accepts an explicit split count and caller-owned FP32 output and
softmax buffers. They must be disjoint from inputs/outputs and other in-flight
calls. All split writes, combination and subsequent workspace reuse run on
the caller's current stream. Inactive splits are overwritten on every call so
graph replay and causal tails cannot consume stale partials. Records and their
owners remain immutable until combination completes; record formats, encode
ordering and external cache ownership do not change.

## Integration boundary

The vLLM adapter changes the KV page specification, provides metadata, and
registers a `CUSTOM` backend. It wraps the V2 model runner's execution and
batch-state gathering methods at runtime to observe scheduler block lists and
bind request indices to those lists. Workspace-based prefill metadata without
this binding is rejected rather than
silently using stale block ownership. No vLLM source files are edited,
but this wrapper is a version-sensitive integration point.

The initial compatibility target is SM90, BF16 KV, head size and block size 128,
GQA group 1/2/4/8, and no speculative decoding. A TP size must divide the model's
KV heads. These are implementation checks; successful standalone validation
for each combination must be recorded separately.

The core exposes block blob export/import for a future external adapter.
This extraction does not include a cache service, prefix index, SSD placement
policy, or LMCache integration. Tensor byte transport alone does not establish
correct cache lifecycle integration.

## Capacity and correctness

Report arena, unfinished-block staging, scratch, overflow, metadata, tables,
and attention workspace separately. Persistent tensor counts do not include
model weights, allocator reserve, CUDA Graph allocations, or all temporary
per-step tensors. Per-rank block counts must not be added together as request
capacity.

Losslessness concerns reconstructed KV bit patterns. Different attention
reduction orders can change floating-point outputs or generated tokens; compare
KV reconstruction exactly and attention outputs against an appropriate numeric
reference. Keep those contracts separate.
