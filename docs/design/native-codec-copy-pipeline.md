# Two-slot copy/restore diagnostic

This benchmark-only prototype leaves the single-slot budget evaluator and core
unchanged. It allocates two fixed receive/map/offset slots and keeps native KV
or compressed arenas, a single ordered restore stream, and stock attention.
Three modes use identical storage. `original` directly binds the original
single-slot enqueue without additional ready/done events; it is the net-gain
control despite retaining the second slot in its allocation. `serial-double`
alternates slots with copies on the restore stream; `overlap` uses the same
slots/events and a separate H2D stream. Every mode retains the request status
gate. No public pool, connector or attention API changes.

Each slot's copy waits for its previous done event, copies distinct immutable
pinned metadata and actual payload, and records ready. The restore stream waits
for ready, replays the complete validated restore graph, copies status into that
chunk's disjoint pinned destination, and records done. Thus copying chunk i+1
can overlap restoring i, but copying i+2 cannot overwrite the slot used by i.
All restore kernels and status copies remain serialized. One final host event
wait and complete status check precede attention. All archives, pinned inputs,
targets, maps and slot storage remain owned until this boundary completes.
Teardown explicitly drains both streams before releasing tensor owners, also
on an exception between an H2D enqueue and its compute dependency. These waits
are outside measured request paths.

NativeCodec decoder scratch/status are shared because both slots restore on
one stream; per-count graph status views remain explicitly retained. HBM import
has captured status per slot/count and a shared store; its ownership updates
remain serialized. Graph source views use fixed receive capacity on the same
verified immutable archives as eager-capacity. The public pool cannot adopt the
existing first slot, so this prototype adds one bounded slot directly instead
of allocating a replacement pool and inflating the construction peak.

The additional K=4 receive payload is approximately 50.3 MB, plus a map, offsets
and HBM captured statuses. All actual construction/cold/timed tensor peaks are
budgeted. All three modes fit once against the same two-slot allocation and preserve
storage addresses and live allocation thereafter. This matched-allocation
diagnostic is separate from one-slot minimum deployment memory. In particular,
C4096 at 0.5 GB cannot fit 32 native blocks plus two K4 raw receive buffers
even before other costs; it remains an explicit infeasible result, with no
single-slot fallback.

Before timing, the alternating-slot modes check changing payloads/maps and
every count on both slots; the original mode checks its first slot. These probes
cover multi-chunk slot reuse, poisoned KV/output, and an early invalid block
followed by successful chunks. All host statuses must survive slot/status
reuse. The benchmark aborts an unexpected failure before attention; it does
not implement transactional LRU recovery. CPU scheduler tests check event
generation/order, status retention, and reuse, but do not replace GPU validation.

The paired driver rotates and mirrors A/B/C/C/B/A for twelve rounds per process,
reversing the initial order in a second process. Each round has two traces per mode; each
trace resets logical LRU. Ratios compare within-round mode means, with original
trace samples retained. Whole-trace wall includes policy, pinned preparation,
H2D, restore/status, stock attention and completion. Allocator/address/policy
guards and output writing are outside timing. No independent-process confidence
interval, dedicated CPU cores, whole-machine isolation, or guaranteed overlap
is claimed. The supervisor's existing NUMA binding remains recorded separately.
An optional profiler round enables CUDA profiler capture and per-mode NVTX
ranges only in that round. Such a process is labeled profiled evidence, may
use a single round, and is kept separate from normal timing results.

Extrapolating the resident K=4 decode measurements to 64 chunks gives about
4.8 ms for 32K tokens. Ideal overlap could hide about 4.7 ms per miss before
copy/compute contention, coordination and first/last-chunk costs. This estimate
is a ceiling to test; it is not a measured 32K pipeline latency or speedup.
