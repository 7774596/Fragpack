# Direct native-layout Raw restore control

This benchmark-only control strengthens the raw read baseline. The existing
`raw_transfer="packed-scatter"` default keeps its block/head-major archive,
bounded GPU receive buffer, validated raw scatter and unchanged measurements.
An explicit `raw_transfer="native-direct"` case instead stores immutable pinned
BF16 archive bits as `[layer, K/V, logical block, token, head, dimension]`.
It copies contiguous target runs directly into `[K/V, physical block, token,
head, dimension]` native GPU pages using public contiguous same-dtype `copy_`.
It does not change the core, CUDA kernels, attention mechanism or third-party code.

Archive transpose and pinning happen before timing, like compressed archive
encoding. The measured restore includes complete CPU map validation, run planning,
Python slice/copy submission and one completion-event wait before stock attention.
Every target must be integral, unique and in range; all archive/destination layouts
are checked before the first write. Invalid metadata therefore writes no target.
CUDA copy errors abort the benchmark; this is not a transactional recovery API.
All queued work is drained on a submission exception before owners can be released.

The immutable pinned archive and registered target caches remain owned until
completion on the runtime's single ordered stream. Source and destination storage
must not alias. Each contiguous physical run requires two copies per layer;
fragmented maps honestly pay more Python and CUDA submissions. No restore mapping
upload, GPU status vector or compressed receive buffer is needed. Read metadata
for the unchanged attention remains allocated and transferred as before.
`restore_dispatch` and `restore_wait` still configure compressed arms; direct Raw
always uses eager public copies and one request completion. Its result identifies
this effective boundary and reports actual run/copy counts and bytes without
inventing successful GPU codec statuses.

The constructor branches before allocation: direct Raw never creates a codec or
full receive/map/offset/status buffers, including temporary dummy buffers. Zero-size
compatibility tensors own zero storage. Capacity fitting includes observed setup,
cold-request and timed tensor-allocation peaks, plus the usual attention graph
allocations. These per-arm minima can change LRU residency and miss counts under
the same total budget; no capacity matching or silent fallback hides that effect.
Host archive bytes, layout and hashes are retained. Offline layout preparation and
write costs are excluded, so results describe immutable read-only replay only.

CPU checks cover actual layout-copy bits, run mapping, global validation and no-write
errors. Separate GPU probes change payloads/maps, use partial final pages, poison
selected KV/output, preserve inactive pages and require all-layer exact restored
bits and stock attention outputs. Direct Raw checks invalid CPU metadata before
any DMA; the compressed/packed arms retain their GPU-error/status tests. Profiling
must separately confirm that the public contiguous copies use DMA and no hidden
staging. Until that observation, the implementation guarantees the requested tensor
layout and allocation accounting, not a measured CUDA transport implementation.
