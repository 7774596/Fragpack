# Request-level restore completion experiment

This benchmark-only candidate adds `restore_wait="request"` beside the existing
default `"chunk"`. The dispatch choices `eager`, `eager-capacity` and `graph`
remain independent controls. All four arms use the same wait policy. The
candidate uses one ordered stream and one bounded GPU receive buffer; it does
not introduce transfer/decode overlap, a second slot, or a production API.

## Ownership and ordering

Let `N` be the largest request's block count, `K` the receive chunk capacity,
and `C=ceil(N/K)`. Request mode adds fixed pinned buffers for `N` target IDs,
`N+C` rebased offsets, and `N` status values: `16*N+8*C` bytes. Each chunk owns
a distinct `count+1` offset segment and a distinct status destination. The
existing chunk-sized pinned buffers remain for graph initialization. The
request's slots are checked for count, range and uniqueness, then all metadata
is filled before enqueue. These CPU operations remain inside request wall time.
The LRU allocator also keeps the whole request disjoint from live sessions.

Every chunk enqueues metadata and actual-payload H2D, the same public restore,
and status D2H into its own pinned destination. The next chunk reuses the GPU
receive, mapping and offsets only after those operations on the same stream.
Its status reset cannot overtake the previous D2H. No pinned input segment is
modified until the final completion event. Immutable archive tensors stay
alive throughout. A single event wait at the end precedes checking every
request status; any failure raises before a successful restore return, and
the caller cannot enqueue dependent attention.

NativeCodec status and captured graph status are preallocated. The existing
eager `HbmKvStore.import_blocks` API allocates a status tensor on every call;
this experiment keeps that public API and reports its actual allocation peak.
It does not claim an allocation-free HBM eager path. There is no new request
GPU storage or hot pinned allocation. Actual GPU construction/cold/timed peaks
remain subject to the existing fixed tensor budget; pinned metadata is reported
separately. Per-chunk tensor views and CPU bookkeeping do not own new GPU data.

## Validation and interpretation

Validation first poisons all selected destinations using the existing stable
request-sized `read_ids`, queues restoration, waits once, and compares all KV
bits using those same stable IDs. It does not inspect the last chunk's reused
mapping as if it still described previous chunks. Full trace output validation
and inactive destination canaries remain in place.

A separate probe corrupts the first compressed block header, or uses an invalid
first raw mapping, followed by a valid later chunk. It must retain the first
failure and later zero statuses, preserve the poisoned invalid destination,
restore later valid KV bits, and leave an inactive canary unchanged. The probe
uses the same enqueue and completion helpers as request mode; it runs outside
timed replay. Single-chunk configurations cannot establish this property and
report that limitation instead of fabricating coverage.

Chunk mode keeps its original completion and fail-fast behavior. Request mode
may finish later valid chunks after an earlier failure, but publishes no
successful request and performs no attention on failure. An unexpected status
aborts the benchmark; recovery or transactional LRU rollback is not implemented.
Full request wall
time still includes policy, metadata preparation, all H2D, restoration, status
D2H/checking, attention and its completion. Any gain is reduced host waiting and
submission gaps on one stream; it is not evidence of transfer overlap or model
serving latency. The verified-archive receive-bound restriction of the graph
and eager-capacity modes still applies.
