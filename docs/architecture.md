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
for FlashAttention. This is not a claim that every operation avoids raw KV
materialization.

## Integration boundary

The vLLM adapter changes the KV page specification, provides metadata, and
registers a `CUSTOM` backend. It wraps the V2 model runner's execution method
at runtime to observe scheduler block lists. No vLLM source files are edited,
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
