# Paired restore dispatch diagnostic

The benchmark-only `native_codec_dispatch_ab_bench.py` imports the unchanged
fixed-budget helper. For each arm it fits one final runtime with the superset
of graph and request buffers, then retains its capacity, storage addresses and
live allocations for four variants: eager-capacity/chunk, graph/chunk,
eager-capacity/request and graph/request. Only the two runtime mode attributes
change. NativeCodec active view shapes may change, but their underlying owned
storage must not. Captured statuses remain retained by count. Compressed input
extent is matched; raw always uses its exact byte extent.

The raw control transfers packed external pages into the bounded receive buffer
and validates/scatters them into native pages. It does not measure a different
archive layout or pitched H2D directly into final native pages. Comparisons
therefore describe this shared restore interface, not an optimal raw offload
implementation or a deployed LMCache service.

This is a matched-allocation diagnostic, not each mode's minimum deployment
memory. The existing fit includes graph construction and a largest cold request.
Every variant then performs a full non-validation trace with a peak budget
check. A transient eager HBM status that breaks that fit causes a recorded
failure; capacity is never silently changed between paired samples. Each
trace must retain the same physical addresses, live tensor allocation, policy
decisions and byte counts. Request-mode wait counts differ by design.

Before timing, every variant runs the helper's poisoned KV/output trace and
payload/map/canary checks. Request variants also run its early-error retention
and attention-gate check; chunk variants keep their original fail-fast policy.
One warmup trace per variant precedes twelve paired rounds. Each round executes
all four variants once; a cyclic Latin order balances positions over four
rounds. `--order reverse` reverses the initial order in a second process.
Every trace starts with the same empty logical LRU and immutable session trace.

The measured trace wall retains the helper's complete boundary: CPU policy and
metadata work, actual H2D, restore/status handling, attention, completion and
trace result collection. Mode switching, invariant checks, JSON writing and
source hashing are outside that window. The driver records five within-round
ratios and all original trace samples; it does not treat rounds as independent
processes or claim a confidence interval. Balanced order reduces particular
order effects; it does not guarantee immunity to host or GPU drift. The driver sets no CPU affinity. The measured campaign applies NUMA-node
binding externally; no dedicated CPU cores, whole-machine isolation or
transfer overlap is claimed.

The input file, selected case file, driver, helper and core source hashes are
recorded and rechecked at completion. The driver changes neither production
code nor the imported evaluator.
