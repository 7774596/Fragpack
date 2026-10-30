# Fixed-budget restore dispatch experiment

This benchmark-only candidate compares `restore_dispatch="eager"` (the existing
default), `"eager-capacity"` and `"graph"` for raw/native FA3, compressed/native
FA3, gather/FA3 and fused HBM. It changes submission of the existing public restore operation;
attention, external payloads, capacity fitting, per-chunk status transfer and
host completion checks keep their existing boundaries. It is not a transfer
pipeline or a new production API.

## Capture and ownership

Initialization derives the actual chunk counts used by the sessions, including
tails, and captures one graph per count. A chunk size larger than every request
does not cause capture or execution of nonexistent full chunks. Each graph
borrows the runtime's fixed receive buffer, offsets, mapping and destination
storage. Initialization uploads a valid archive with distinct legal target IDs,
warms the public operation, and establishes stream completion before capture.
Capture and warmup occur before timing or LRU admission; no live session owns
these initialization targets.

The captured calls are `NativeCodec.raw_scatter`, `NativeCodec.decode` and
`HbmKvStore.import_blocks`. All validation remains in the graph. Native calls
also capture their status reset. HBM import has no per-call Python ownership
mutation: its status allocation/zero and complete import kernel are captured,
while record installation, overflow allocation and previous-record release
remain GPU operations on every replay. The returned status tensor is retained
with its graph. Native status views are retained by count as well; replay never
uses `codec.status` whose Python active count reflects the last capture.

Compressed graphs borrow the full fixed receive capacity because payload size
varies between chunks with the same count. This experiment uses verified
immutable archives, copies only their actual payload bytes, and supplies their
current offsets. Bounds validation therefore refers to receive capacity, not an
independent dynamic transferred-length argument. Raw graphs use the exact
`count * RAW_BLOCK_BYTES` view required by the public raw API. Captured addresses
and tensor extents stay fixed; payloads, offsets and target IDs change on the
same ordered stream before replay.

The `eager-capacity` control submits those same full-capacity compressed views
through the ordinary eager public calls. It transfers only the actual payload,
as the other modes do; raw views retain their exact byte count. Default `eager`
keeps the original actual-payload extent. Thus `graph` versus `eager-capacity`
compares dispatch at the same validation bound, while `eager` versus
`eager-capacity` controls the change in that bound. Receive-capacity bounds can
accept truncations that actual-payload bounds reject; neither capacity mode is
a general API for validating the current transferred length. Both are scoped
to the benchmark's verified immutable archives.

## Budget and measurement

The existing fixed tensor-peak budget includes construction, all attention and
restore graph allocations, retained HBM status tensors and the largest cold
restore/read probe. Restore-graph allocation deltas and retained status sizes
are reported separately; the status tensors are a subset of that delta, not an
additional amount to sum twice. Native graph status views alias existing codec
storage. Allocator reservation, CUDA context and graph-driver objects remain
outside the stated tensor budget.

All three modes keep the same payload and metadata H2D copies, status D2H, per-chunk
wait/check and final attention completion wait. Complete request wall time still
starts before the LRU decision and includes all these operations. Graph setup
and validation are excluded from performance timing and their memory is counted.
Only the extent-matched graph versus eager-capacity comparison attributes a
difference to dispatch. The comparison does not establish transfer overlap, a
different codec algorithm, or serving latency.

## Validation

The separate validation trace poisons each missed chunk before restoration,
checks every restored KV bit, poisons attention output and compares the complete
output with the same reference in all three modes. A dispatch probe additionally
reuses every count with changing payloads and permutations of destination IDs,
including tail counts. It checks an inactive destination as a canary where one
exists. These probes execute before measured traces and do not alter the trace,
its session identities, admission policy or expected hit pattern.
