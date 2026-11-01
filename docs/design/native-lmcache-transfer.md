# Native-page LMCache transfer

This transport connects the independent codec to the existing CPU-L1 byte
service. It does not select eviction candidates or modify attention. A logical
128-token block maps to every native physical page it spans. The caller owns
and protects those pages; the transport owns bounded maps, codec workspaces,
GPU payload storage and pinned host buffers.

`NativeBlockTransfer` accepts `NativeKVPages`, tables, `BlobClient` and the
existing `CacheIdentity`. `compressed` uses existing FPKB v1 native encode/pack
and decode/scatter. The matched `raw` control uses native raw gather/scatter and
a distinct FPNR v1 envelope. Its records are token-major BF16
`[layer, K/V, head, token, dimension]`, whereas FPKB raw fallback records use
the codec's kernel order. FPNR retains the aligned header and length-table
structure but accepts only full raw records, without entropy coding or CPU
layout conversion. Its CPU framing and all transfer copies belong in measured
cost; it is not a claim about optimal stock LMCache raw transport.

`CacheIdentity.representation` defaults to `fpkb`, preserving every existing
FPKB namespace byte-for-byte. `native-raw` selects FPNR magic, format and
token-major layout in a distinct namespace. Construction rejects a transfer
mode/identity mismatch. Host blob validation rejects the opposite magic and
rejects any compressed-length entry in FPNR before H2D. No core codec importer
accepts FPNR; only the native raw scatter path consumes its unpacked records.
Canonicalization clears reserved envelope bytes without touching record bytes
or adding a layout conversion. The two representations cannot share objects.

Construction and all lease operations run on one background thread, preserving
the `NativeCodecPool` single-host-thread contract. Encode and decode each use
one fixed-capacity pool slot; raw uses one raw slot. At most `num_slots` batches
of at most `max_blocks` are admitted. Caller-thread submission only validates
metadata, owns a page-map copy, and records the producer event. No model-thread
device readback, blocking RPC or device synchronization is added. Initialization
may wait for background allocations; exceptional failed submission must drain
its producer stream before reporting that its map owners can be released.

For stores, the caller retains source-page references through the successful
`source_ready` Future. It is a Future rather than an unrecorded CUDA event: an
event not yet recorded by a background worker must never be interpreted as
completed. The Future resolves after the last kernel reading the source has
finished; packing, D2H and publication may still be pending. `completion`
reports confirmed publication in key order. Both Futures reject cancellation.
The initial explicit-transfer contract holds source references during the
bounded metadata probe as well; it does not claim an unpinned candidate queue.

Before encoding, `probe_many` checks each key without reading payload. Existing
objects skip all device work and upload. A local in-flight map merges duplicate
store keys in one transfer instance. This is advisory cache reuse, not a lease:
an object can be evicted immediately afterward, and later get must still handle
a miss. `precheck=False` retains the same transport but disables the existence
probe for an ablation. No persistent local `saved` flag is trusted.

Restores fetch only the available contiguous prefix, validate every blob, then
copy through owned pinned storage into the bounded receive slot. Decode status
must be observed before completion. All destination pages remain unavailable to
attention and other allocations until completion. Cache reads do not delete
external copies. The bridge exposes only the contiguous successful prefix.

RPC timeout is an uncertain publication outcome, not permission for automatic
retry or key deletion. The existing client failure semantics remain in force.
On any error the worker drains its owned stream and producer event before
releasing resources. If GPU completion cannot be established, the transport
fails closed and retains all affected owners until process recovery. A close
timeout leaves the worker and buffers alive; request cancellation does not
cancel submitted DMA.

CPU tests cover queue bounds, coalescing, precheck holes, partial publication,
misses, cancellation refusal and failure ownership. Explicit GPU tests cover
strided/disjoint native page maps, raw and compressed bitwise round trips,
source/publication separation and invalid destinations. Real LMCache and model
serving validation are separate from those component tests.

After canonicalization, CPU statistics read only the existing envelope length-table
view. Compressed mode counts all attempted records and the raw-fallback subset;
raw mode counts raw records without calling them fallbacks. K/V record-byte
totals include dense and variable regions but exclude the block envelope and
alignment padding. This work stays inside CPU packing timing and introduces no
GPU readback or second payload copy.

The controlled round-trip benchmark uses one immutable captured source per
logical key and distinct native source/destination pages. Its default source is
capture positions 8:32, cycled to 64 blocks; tables use positions 0:8. Raw with
precheck, compressed with precheck, and compressed without precheck each start
a fresh real LMCache CPU L1 and an isolated identity. Three sequential
store/restore rounds retain external objects. Bitwise checks and target resets
are outside the measured windows; bounded host polling, RPC and completion
draining are inside. Private-dictionary warmup precedes the real server and is
not part of the measured arms. This isolates transport and repeated-store work;
it is captured synthetic scale, not model serving or a cache-policy benchmark.
The [completed report](../validation-native-codec-serving-20261101.md) keeps
these measurements separate from the full-model runs.

Operation timing fields are host-observed spans. `get_seconds` includes the
complete byte-service call; decode/scatter includes submission, status handling
and synchronization. Queue wait and per-batch `source_hold_seconds` overlap other
work and must not be added to request wall time. The latter is not the lifetime
of all pages retained by one request. Connector scheduler lookup exceptions are
logged and fall back to recomputation; they do not increment worker transfer
failure counters, so a zero failure count must be checked against those logs.
