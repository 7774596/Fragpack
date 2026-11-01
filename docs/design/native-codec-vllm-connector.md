# Native codec connector: serving boundary

This connector uses the public vLLM 0.30.0 `KVConnectorBase_V1` scheduler and
worker hooks. Attention, native BF16 KV writes and block allocation remain
unchanged. No runner wrapper or third-party source modification is involved.
The initial scope is TP=1, eager execution, one full-attention cache group,
head dimension 128, and text prompts without adapters or multimodal inputs.
The constructor rejects unsupported configurations; registration checks the
actual native tensor layout and codec-table identity.

## Representation and policy

The worker views vLLM's logical `[pages, heads, tokens, 256]` tensors as
`[2, pages, tokens, heads, 128]` without copying or changing attention layout.
The connector requests layer-compact `LBNHC` storage, so the registered layer
spans are disjoint. Its 128-token logical objects can map to several native
pages; local hits that end inside a logical object are left to native APC.

The cache is best effort. Both raw and compressed arms use the same finish-time
save rule: retain complete **prompt** blocks only after a normal request finish.
Generated tokens and incomplete tails are not newly admitted by this first
connector. Existing GPU APC is always consulted first by vLLM. Codec mode is
fixed at startup, and arms must use separate external namespaces or servers.
There is no new eviction policy, NVMe tier, or speculative prefetch mechanism.

## Scheduler and worker ownership

The scheduler computes chained keys on CPU and starts prefix lookup on a
bounded single-worker executor. While the result is pending,
`get_num_new_matched_tokens` returns `None`; no RPC waits inside attention or
the scheduler's lookup callback. Lookup is advisory. Remote hits are rounded to
complete 128-token objects and leave at least the final prompt token for normal
computation. After allocation, only the external suffix's exclusive pages are
sent to the worker. Restores are asynchronous and the request remains in
vLLM's remote-KV wait state until every requested object has successfully loaded
and decoded. A miss or invalid record reports a failed receive for recomputation;
it never turns a lookup into an unverified skip of prefill.

At request finish, `request_finished` returns delayed-free ownership before the
engine can recycle source pages. The number of delayed requests is bounded.
Metadata carries complete keys and physical page IDs, including their full
prefix identity. The transfer backend records the producer stream event on
submission and keeps its own map tensors alive. Transfers run in bounded
batches; later batches keep the same request source references. After the last
batch's `source_ready` resolves, the worker can report `finished_sending` and
release engine source pages, independently of external publication. Worker
metadata keeps the engine stepping until outstanding publications drain.

Store/restore submission and polling also run on empty model steps, using
`start_load_kv` and `get_transfer_results`; progress never depends on a future
token-producing request. Full-request completion is distinct from batch
completion. The scheduler attaches the admitted store IDs from that step's
`finished_req_ids` to the plans. The worker retains finish notifications only
for these owned store jobs; short and rejected requests cannot grow an
unbounded ID set, and a generic finish poll before plan submission cannot lose
the evidence needed to release a source. A missing scheduler finish event is
an explicit contract error. A source/readiness exception with unknown device completion is
fatal rather than permission to recycle memory. Ordinary publication failure
after known device completion is a lost cache opportunity, not KV corruption.

Request cancellation does not cancel DMA. vLLM retains cancelled asynchronous
receive destinations until receive completion. Cancelled requests are never
admitted as new stores. Finished source pages remain held until all submitted
source reads complete. Shutdown drains outstanding work before releasing
clients or buffers. The existing transfer backend owns CUDA cleanup; the
connector does not free a source or destination solely because a timeout fired.

## Validation and measurement

CPU tests cover lookup polling, local-prefix exclusion, incomplete tails,
bounded admission, failed-load retry, cancellation, empty-step progress,
batch completion and early source release. GPU and serving checks must prove
the native tensor view is a view, complete KV roundtrips are bit-exact, external
hits skip actual prefill only after successful restoration, and failure causes
recomputation. Equal-budget raw/compressed serving measurements include first
encoding, actual transfer bytes, publication, recall and background interference.
The [completed validation](../validation-native-codec-serving-20261101.md)
separately records GPU bitwise roundtrips, real CPU L1 integration, four
fixed-budget full-model offline runs, and matched cache-path output checks.
Its native FA3, TP=1, eager, sequential scope does not validate HTTP concurrency,
TP>1, full-model CUDA Graphs or task accuracy. Cold and cached computation can
produce different tokens even in the raw control; same-path raw/compressed
output comparisons and immutable-KV bitwise checks remain distinct evidence.
