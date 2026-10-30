# Native codec restore kernel optimization

This work optimizes the independent native-page codec described in
[native-attention-external-codec.md](native-attention-external-codec.md). It
does not change FPKB v1, attention kernels, native page registration, or the
external-cache policy. The comparison baseline is commit `d12b780`.

## Coalesced native scatter

The decoder reconstructs BF16 words in the record's MMA layout. Writing each
word directly into native pages makes the V stores span different tokens and
repeats page-address calculation for each scalar. The candidate instead writes
the reconstructed words to a swizzled shared-memory tile, then cooperatively
scatters contiguous dimension groups into the registered native pages.

The shared tile uses the same token/dimension swizzle as the existing HBM-store
gather implementation, but its output is the caller's native page allocation.
No global BF16 staging allocation or attention-backend change is introduced.
Each CTA still owns exactly one `(block, layer, K/V, head)` record.

## Shared-memory lifetimes

The decoder reserves `32768 + 16976 = 49744` bytes of dynamic shared memory:

1. The first 32768 bytes initially hold a bounded, zero-padded compressed record.
   Raw fallback records are read directly from their input blob.
2. The last 16976 bytes hold the existing per-warp escape expansion. They never
   overlap the first region.
3. `decode_rec` loads the dense low bytes and symbols into registers and finishes
   reading the variable region before calling its first emission callback.
   A CTA barrier in that first callback ends the input-copy lifetime for every
   warp before any thread writes the first region as the decoded tile.
4. Subsequent reconstruction uses registers and the separate escape region.
   A final CTA barrier makes the complete decoded tile visible to the scatter.

The common record decoder remains unchanged. This overlay depends on its
documented preload-before-emission order; a future streaming implementation
must preserve that order or replace this overlay. The opt-in shared-memory
attribute is configured once per host thread and CUDA device before launching
the kernel; ordinary calls add no device synchronization or readback. The
49744-byte requirement matches the existing gather kernel's shared-memory
requirement and must be included in runtime validation.

## Destination and failure contracts

The header, full-record, and decode launches remain separate and ordered on the
caller's CUDA stream. All current malformed-record and destination-alias checks
remain intact. A rejected blob writes no native destination bytes, even when a
different record in that blob is valid.

The scatter writes eight adjacent BF16 values with a 128-bit store only when
the record's base address and page/token strides guarantee 16-byte alignment.
Other layouts use scalar 16-bit stores. Native page permutations, heterogeneous
layer strides, two-byte-aligned bases, holes, and untouched pages retain their
existing contracts. The tile stores and reloads integer bit patterns, preserving
all BF16 bits, including NaNs and signed zero.

## Evidence required

GPU correctness covers compressed and raw fallback records, each supported page
size, arbitrary supported strides, whole-allocation canaries, malformed blobs,
alias rejection, and repeated CUDA Graph restores from updated input buffers.
The comparison must retain the existing validation work and measure complete
restore plus native attention, as well as kernel-level diagnostics, on the
same exclusive GPU. Shared-memory occupancy and the extra local transpose may
offset the reduction in global-memory transactions; performance remains an
experimental result until that comparison completes.

Validation-launch changes, mapping-scan changes, and decode-only workspace
allocation are separate candidates and are not part of this kernel change.
