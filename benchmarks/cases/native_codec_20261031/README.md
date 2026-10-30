# Native codec replay cases, 2026-10-31

These eight JSON files preserve the campaign case bytes exactly; `SHA256SUMS`
identifies them. They configure the existing
[budget replay driver](../../native_codec_budget_bench.py), without changing its
defaults. Every process uses **12 timed rounds, 1 warmup, seed 314**, calibration
blocks `0:8` and evaluation blocks `8:32`. Each timed trace starts with empty LRU.

## Process order and arms

`R=raw_fa3`, `N=native_fa3`, `G=gather_fa3`, `F=fused_hbm`.
The Raw archive layout is selected by each case's `raw_transfer` field.

| Experiment | Process order | Case file | Arm order |
|---|---:|---|---|
| GPU2 Raw controls | 1 | `raw_packed_forward.json` | R,N |
| GPU2 Raw controls | 2 | `raw_direct_forward.json` | R,N |
| GPU2 Raw controls | 3 | `raw_direct_reverse.json` | N,R |
| GPU2 Raw controls | 4 | `raw_packed_reverse.json` | N,R |
| GPU2 granularity | 1 | `granularity_forward.json` | R,N |
| GPU2 granularity | 2 | `granularity_reverse.json` | N,R |
| GPU4 validation | 1 | `gpu4_forward.json` | R,N,G,F |
| GPU4 validation | 2 | `gpu4_reverse.json` | F,G,N,R |

Reverse files also reverse case order. Main controls contain four configurations;
granularity contains six (.45/.5 GB × K1/K2/K4). GPU4 contains .45 GB C1024/W5/Q1
at K2 and K4, plus C32768/W3 at 8 GB/Q1 and 12 GB/Q128, both K4. GPU4 plans
384 timed traces and 8,448 requests if all cases complete; keep failures and
infeasible cases rather than inventing samples. A case file alone is not evidence
that its experiment completed.

## Environment and invocation

Use the [documented development environment](../../../docs/development.md).
The recorded environment is Python 3.10.20, PyTorch 2.13.0+cu129,
vLLM 0.30.0+cu129, NumPy 2.2.6 and SM90 H800 GPUs. Keep the capture, output,
JIT cache (`TORCH_EXTENSIONS_DIR`) and scratch on a data filesystem outside the
repository. Set `KV_DUMP`, `RESULT_DIR`, `GPU_UUID` and `NUMA_NODE`; export
`CUDA_HOME` (the matching CUDA toolkit) and `TORCH_EXTENSIONS_DIR`. Run this example
from the repository root after creating the result directory.

```bash
CUDA_VISIBLE_DEVICES="$GPU_UUID" CUDA_DEVICE_ORDER=PCI_BUS_ID \
TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4 NUMPY_MADVISE_HUGEPAGE=0 \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE" \
python -m benchmarks.native_codec_budget_bench \
  --dump "$KV_DUMP" \
  --cases benchmarks/cases/native_codec_20261031/raw_direct_forward.json \
  --out "$RESULT_DIR/raw_direct_forward.json" \
  --arms raw_fa3,native_fa3 \
  --rounds 12 --warmup 1 --seed 314 \
  --calib-pages 8 --eval-start 8 --eval-pages 24 \
  --input-description 'Historical BF16 capture; held-out blocks 8:32 rotated/repeated; read-only replay.'
```

Use the table's exact case/arm order and distinct output files for other processes.
Before **each** process, require exclusive access to its reserved GPU and at least
32 GB effective headroom: host `MemAvailable` and the remaining allowance of every
finite ancestor cgroup memory limit. This admission check is not a measured RAM
peak. Monitor exclusivity throughout and stop if another GPU user appears.
Apply a bounded runtime/deadline externally. NUMA binding does
not reserve CPU cores or isolate the machine. Preserve per-process source hashes,
GPU UUID, environment, complete output and failures.

- GPU2: `GPU-fb9880e5-6155-b080-837a-948d5fd08b4c`, H800 PCIe, NUMA node 0.
- GPU4: `GPU-910de65d-2e94-591b-557e-4b29047bf2fc`, H800, NUMA node 1.

Keep the two hardware groups separate; do not pool traces or interpret a
cross-GPU difference as a codec-only effect. Mirrored ordering does not eliminate
host drift, and 12 traces in one process are nested observations.

## Source and input identity

The implementation is committed as
`74ea52ea8b79e223d09fe9efb8386fcf48455cfc`. The measured isolated checkout recorded
base `eaa0006a601ae8f609d01f3bd8ae1105e167ca4a` plus the frozen changes later
committed unchanged. Relevant SHA256 identities are:

| Artifact | SHA256 |
|---|---|
| `benchmarks/native_codec_budget_bench.py` | `ace561513f5610b57cd9dcbbca482d70f6023b05ae311b88cbda052c3d659370` |
| `benchmarks/native_codec_raw_direct.py` | `ccb9f8adc5e2eac816b39e3fa6e4920812006a0c7d2d13f9d7458622e2a5b135` |
| `fragpack/ops/native_codec.cu` | `e024cbfe6b8a9e280c24a153f75b11b76d12df0e4610c25fcd4cd9ea18980175` |
| Capture `.npy` file | `1302f31ac929ea3f623db0edcae7fa5869e50449c4b572dcff4ac17e7ae39322` |
| Calibrated table array | `8cd2b8979b12101119104f1ed92652ce938752e49a4e2d1f5c69de47ff9d0a30` |
| Held-out capture pool array | `90d340536c304f6a0395b10f5c0abe9f73204ac2095b140b1c725485c175cb51` |

The capture is **not bundled**: it is uint16 BF16 bits with shape
`[32,48,2,128,4,128]`. Match the file hash to reproduce these inputs. Table/pool
hashes use the driver's array-hash convention, not `.npy` file hashes. Different
captures require separate provenance and do not reproduce the historical values.

## Measurement boundary

This is immutable-session LRU replay with actual pinned H2D and 48 attention
layers, including CPU policy, restore preparation, completion and dependent reads.
Raw layout preparation and compressed encoding are offline. It excludes outbound
KV encoding/D2H, projections/MLPs, serving/TTFT, network/SSD and new long-context
capture. The fixed tensor budget includes KV, receive/codec, metadata, Q/output
and graph allocations; preserve actual fit and next-block checks. RawDirect uses
CPU map validation and direct copies; compressed paths retain GPU validation.
Fused output uses the existing `.02` absolute/relative tolerance, while the FA3
arms require exact output bits; restored KV is checked bitwise for every arm.

See the [Raw/granularity report](../../../docs/benchmark-native-codec-raw-direct-20261031.md).
The [GPU4 report](../../../docs/benchmark-native-codec-gpu4-replication-20261031.md)
records its independent completed validation.
The earlier [pipeline report](../../../docs/benchmark-native-codec-copy-pipeline-20261031.md)
uses a separate paired driver and historical frozen helper; its O/S/P scheduling
protocol is not represented by this bundle.
