# Reference review: LMCache daser-codec — 2026-10-30

## Conclusion and inspected revision

The branch has substantial reference value for variable-sized external objects,
bounded transfer buffers and asynchronous publication. Its GPU data path must
be adapted, because it compresses ordinary BF16 KV when offloading and restores
ordinary BF16 KV on load. It does not preserve FragPack records end to end.

Inspected clean local checkout: `GentleCold/LMCache`, branch `daser-codec`,
commit `f3aea007a83ca200c630678f0c3dfe411d0258f3` (October 20). Its base is a
pristine LMCache 0.5.5 sdist import, `ef65258ddf021d0a09c3a2bc9b11d89e51643327`.
Four subsequent commits change 12 files, including tests. Remote branch-head
verification failed due to transport/TLS errors; this review does not assert
that the remote still has this exact head.

This is source inspection plus review of existing experiment artifacts. No
fresh tests, GPU work, dependency changes or third-party edits were performed.

## Actual data path

```text
Current reference branch:
  ordinary GPU BF16 KV -> raw gather pool -> DaseR codec encode
      -> actual-length L1/L2 record
      -> H2D -> DaseR codec decode -> scatter to ordinary GPU BF16 KV

Desired FragPack adaptation:
  GPU FragPack records -> export_blocks -> bounded encoded-blob pool
      -> actual-length L1/L2 object
      -> H2D -> import_blocks -> GPU FragPack records
```

`_create_daser_codec` imports `daser.ops.chunk_codec.KVMajorChunkCodec` and
supports BF16 KV with one kernel/object group, including ordinary separate and
fused K/V layouts. This is DaseR's compressed-online record path, not FragPack's
`FPKB` block-blob contract. The corresponding fused-K/V layout fix does not
make the default tensor copier safe for a FragPack arena.

## Reusable mechanisms

| Mechanism | Reference implementation | Value to FragPack |
| --- | --- | --- |
| Actual-length objects | `codec_byte_layout(length)`, per-record `reserve_write`, copy only `chunk[:length]` | Preserve external capacity and transfer savings instead of charging every object at raw capacity |
| L2 length discovery | `L2AdapterInterface.stored_object_sizes`, native adapter `_key_sizes` | Concrete example of the metadata contract needed before allocating a restore destination |
| Size-aware prefetch | Group flat uint8 objects by stored size and reserve matching L1 layouts | Addresses the fixed-layout limitation identified in the initial target assessment |
| Independent store worker | `CodecStoreWorker`, dedicated CUDA stream, pool, ready events, queue | Keep length observation and host allocation out of the shared GPU handler's normal store execution |
| Two completion points | Source gather completion versus later `finish_write` / `MP_STORE_END` | Separate permission to recycle engine blocks from external-hit visibility |
| Duplicate suppression | `l1_contains` precheck followed by authoritative `reserve_write` | Avoid unnecessary production of existing objects while tolerating a stale precheck |

The worker waits for the producer event, encodes, reserves actual-length host
objects, submits D2H, waits for its stream and then publishes with `finish_write`.
The original engine blocks can be released after their data has been copied to
the independently owned pool; the lookup must not treat that earlier event as
publication. For FragPack, export completion replaces the raw gather completion.

The pool defaults to two encode batches and is bounded. These are good ownership
and budgeting ideas; the DaseR encoder and raw pool geometry are not reusable
unchanged. A FragPack worker would manage exported blobs, actual lengths and
import status rather than re-encoding already compressed KV.

## Gaps to address before porting

1. **Lengths are process-local in the native L2 implementation.** `_key_sizes`
   starts empty and learns sizes from completed stores in that adapter instance.
   The new accessor exposes it but adds no durable recovery. Other adapters
   inherit an unknown-size result. The prefetch change falls back to the group
   bound when size is unknown. This does not establish exact-size recovery after
   restart, from another writer, or across every L2 backend. Persist a manifest,
   retrieve length from the backend, or explicitly restrict the supported scope.
   In particular, the Python filesystem adapter does not override the new size
   query and requires the read byte count to equal the destination length; a
   shorter variable record can therefore fail that load path. The native
   filesystem connector is a different path and can use the same-process size
   map. This distinction must appear in backend validation.
2. **Pool exhaustion still blocks the shared handler.** `acquire()` waits for
   free slots. Under sustained store pressure, retrieves can still queue behind
   a store waiting for pool space. Move admission/backpressure outside the shared
   retrieve handler or otherwise bound its effect; a worker thread alone does
   not guarantee store/retrieve isolation.
3. **Post-reservation failure needs stronger cleanup.** In `_store`, an error
   after some reservations/D2H submissions logs failure and releases GPU slots,
   without an explicit reserved-object rollback or guaranteed completion of
   already queued copies on that exception path. Safe slot reuse and L1 write
   ownership must be handled before porting. This is a source-level risk, not
   an observed corruption. The existing failure test throws during encode,
   before reservations, and does not exercise this case.
4. **The changes patch LMCache internals.** L1 manager, L2 interface, prefetch,
   storage manager, CUDA context and MP transfer code are modified. These are
   examples of necessary contracts, not functionality supplied by the unchanged
   v0.5.6 server-module API. Port the contracts deliberately to the selected
   v0.5.6 baseline, keeping an external plugin where feasible; do not copy the
   full older transfer implementation over the newer version.
5. **Retain FragPack-specific validation.** Keys need representation/table/shard
   identity, export needs all layers closed and source pinning, and receive
   completion needs successful import status as well as DMA completion. Raw
   overflow must survive export/import. A failed block cannot be published as
   a usable partially restored prefix.

## Existing evidence

The branch includes mock CPU tests for layout, ordering, batching and prefetch,
plus two CUDA-gated worker tests. The round-trip test uses the real DaseR codec
with a pinned-host test storage implementation, checks bit-exact reconstruction
and pool reuse, and skips refused keys. It is not a full MP/L2 integration test.

Historical R233 run `lmccib32r2-p16-c4-n300-l64-t0/lmcache-codec` completed 300
requests with zero failed requests and logs actual codec record stores. The
run uses vLLM 0.30 and native filesystem L2. Its status explicitly labels the
source `f3aea00+wt`: working-tree changes were present. GPU attention uses
ordinary BF16 KV, not FragPack. This demonstrates real integration exercise,
but is neither a pristine-commit acceptance result nor a FragPack performance
comparison. No old timing is adopted as a new performance promise.

## Effect on the selected integration plan

Keep the [official v0.5.6 target](lmcache-integration-target-20261030.md).
Use this branch as a concrete reference for actual-length allocation, prefetch
metadata, buffer ownership and separate publication. First validate the
variable-object storage/restore contract and post-reservation failure paths,
then connect FragPack export/import. The reference reduces design uncertainty;
it does not remove the need for the FragPack adapter or durable length metadata.

Source references at the inspected commit:

- [Store worker](https://github.com/GentleCold/LMCache/blob/f3aea007a83ca200c630678f0c3dfe411d0258f3/lmcache/v1/multiprocess/modules/daser_codec_store.py)
- [MP transfer](https://github.com/GentleCold/LMCache/blob/f3aea007a83ca200c630678f0c3dfe411d0258f3/lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py)
- [Prefetch controller](https://github.com/GentleCold/LMCache/blob/f3aea007a83ca200c630678f0c3dfe411d0258f3/lmcache/v1/distributed/storage_controllers/prefetch_controller.py)
- [Native L2 adapter](https://github.com/GentleCold/LMCache/blob/f3aea007a83ca200c630678f0c3dfe411d0258f3/lmcache/v1/distributed/l2_adapters/native_connector_l2_adapter.py)
- [Codec integration tests](https://github.com/GentleCold/LMCache/blob/f3aea007a83ca200c630678f0c3dfe411d0258f3/tests/v1/multiprocess/test_lmcache_driven_daser_codec.py)
