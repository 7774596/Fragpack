# SPDX-License-Identifier: Apache-2.0
"""Expanded four-arm, 48-layer KV restore and read-only attention comparison.

Cases describe ragged batches, shared physical prefixes, and repeated reads.
Long contexts repeat captured blocks and are explicitly synthetic scale tests.
Q=1 batches use the production FragPack decode kernel; other batches use its
current automatic split prefill kernel. No new KV is written during a sample.
Run only on one explicitly reserved exclusive GPU.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
import time
import traceback
from typing import Any

import numpy as np
import torch

from benchmarks.native_codec_prefill_bench import (
    ATOL,
    RTOL,
    _hash_array,
    _hash_file,
    _summary,
    _tensor_bytes,
    _write,
)
from benchmarks.native_codec_prefill_bench import (
    _Context as PreviousContext,
)
from benchmarks.native_codec_prefill_bench import (
    _source_identity as previous_source_identity,
)
from fragpack.compression.attn_tables import calibrate
from fragpack.ops.fused_attn_decode import (
    decode_attention,
    decode_units,
    workspace_elems,
)
from fragpack.ops.fused_attn_prefill import (
    prefill_attention,
    prefill_split_count,
    prefill_workspace_elems,
)

ARMS = ("raw_fa3", "native_fa3", "gather_fa3", "fused_hbm")
LAYERS, HEADS, QUERY_HEADS = 48, 4, 32
T, D = 128, 128
MODES = ("resident_graph", "cold_graph", "resident_eager", "cold_eager")


def _source_identity() -> dict[str, Any]:
    identity = previous_source_identity()
    identity["files_sha256"]["benchmarks/native_codec_comparison_bench.py"] = (
        _hash_file(Path(__file__))
    )
    identity["digest"] = hashlib.sha256(
        json.dumps(identity["files_sha256"], sort_keys=True).encode()
    ).hexdigest()
    return identity


def _orders(rounds: int, reverse: bool) -> list[tuple[str, ...]]:
    permutations = list(itertools.permutations(ARMS))
    return [
        tuple(reversed(permutations[index % 24]))
        if reverse
        else permutations[index % 24]
        for index in range(rounds)
    ]


def _cases(path: Path) -> list[dict[str, Any]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases JSON must contain a nonempty list")
    names = set()
    for case in cases:
        case.setdefault("shared_prefix", 0)
        case.setdefault("reads", 1)
        case.setdefault("fallback_fraction", 0.0)
        contexts, queries = case["contexts"], case["queries"]
        if (
            not isinstance(case["name"], str)
            or case["name"] in names
            or not contexts
            or len(contexts) != len(queries)
            or any(not isinstance(c, int) or c < 1 for c in contexts)
            or any(not isinstance(q, int) or q < 1 for q in queries)
            or any(q > c for q, c in zip(queries, contexts, strict=True))
        ):
            raise ValueError(f"invalid name or context/query lengths: {case}")
        shared = case["shared_prefix"]
        if (
            not isinstance(shared, int)
            or shared < 0
            or shared % T
            or any(shared > c - q for c, q in zip(contexts, queries, strict=True))
            or not isinstance(case["reads"], int)
            or case["reads"] < 1
            or not 0 <= case["fallback_fraction"] <= 1
        ):
            raise ValueError(f"invalid sharing, reads or fallback fraction: {case}")
        names.add(case["name"])
    return cases


def _page_table(case: dict[str, Any]) -> tuple[np.ndarray, int]:
    """Map shared-prefix and disjoint-suffix physical pages, zero for padding."""
    pages = [(c + T - 1) // T for c in case["contexts"]]
    shared = case["shared_prefix"] // T
    table = np.zeros((len(pages), max(pages)), dtype=np.int32)
    next_id = shared + 1
    for row, count in enumerate(pages):
        table[row, :shared] = np.arange(1, shared + 1)
        table[row, shared:count] = np.arange(next_id, next_id + count - shared)
        next_id += count - shared
    return table, next_id - 1


def _materialize(
    pool: np.ndarray, table: np.ndarray, blocks: int, fraction: float, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Repeat captured pages for scale; optionally force selected finite records raw."""
    indices = np.arange(blocks) % len(pool)
    content = np.array(pool[indices], copy=True)
    count = blocks * LAYERS * 2 * HEADS
    selected = int(round(count * fraction))
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(count, selected, replace=False))
    # These high bytes represent small finite magnitudes. At least ten of the
    # 32 candidates are absent from a 22-entry table; every scalar in a chosen
    # record then escapes, forcing the existing raw fallback without NaNs.
    candidates = np.array(list(range(0x30, 0x40)) + list(range(0xB0, 0xC0)))
    for flat in chosen:
        block, layer, kv, head = np.unravel_index(flat, (blocks, LAYERS, 2, HEADS))
        absent = candidates[~np.isin(candidates, table[layer, kv, head])]
        high = int(absent[0])
        low = rng.integers(0, 256, size=(T, D), dtype=np.uint16)
        content[block, layer, kv, :, head, :] = (high << 8) | low
    return content, {
        "available_distinct_capture_blocks": len(pool),
        "physical_blocks": blocks,
        "capture_block_indices_in_eval_pool": indices.tolist(),
        "repeated_capture_blocks": blocks > len(pool),
        "synthetic_scale": blocks > len(pool),
        "forced_fallback_records": selected,
        "forced_fallback_fraction": selected / count,
        "forced_record_indices_sha256": _hash_array(chosen),
        "interpretation": (
            "Physical copies of held-out captured blocks; replication preserves "
            "their measured distribution but is not a real longer-context model "
            "capture. Distinct physical pages are not deduplicated by content. "
            "Optional fallback replacements are synthetic finite BF16 records."
        ),
    }


class _Context(PreviousContext):
    """Reuse validated old setup and add a raw transfer arm over the same targets."""

    def __init__(self, table: np.ndarray, content: np.ndarray) -> None:
        super().__init__(len(content) * T, table, content)
        # The old immutable setup uses its three-arm labels internally.
        for values in (self.receive, self.offsets_receive, self.status_host):
            values["fused_hbm"] = values.pop("fused_prefill")
        self.raw_bytes = content.nbytes
        self.host_raw = torch.empty(self.raw_bytes, dtype=torch.uint8, pin_memory=True)
        self.receive["raw_fa3"] = torch.empty(
            self.raw_bytes, dtype=torch.uint8, device="cuda"
        )
        self.status_host["raw_fa3"] = torch.empty(
            self.blocks, dtype=torch.int32, pin_memory=True
        )
        # Archive construction is outside timing. The original encoded blobs
        # already live independently on the host, so reusing encode output here
        # cannot accidentally make the compressed restore read a raw payload.
        status = self.codec.raw_gather(self.source, self.mapping, self.codec.buffer)
        self.host_raw.copy_(self.codec.buffer[: self.raw_bytes], non_blocking=True)
        self.stream.synchronize()
        if bool(torch.any(status.cpu() != 0)):
            raise RuntimeError("raw archive gather failed")
        self.transfer("raw_fa3")
        status = self.restore("raw_fa3")
        self.status_host["raw_fa3"].copy_(status, non_blocking=True)
        self.ready.record(self.stream)
        self.ready.synchronize()
        self.check_status("raw_fa3")
        self.metadata["raw_restore_kv_validation"] = self.validate_kv()
        self.metadata["raw_payload_sha256"] = _hash_array(self.host_raw.numpy())
        memory = self.metadata["memory_bytes"]
        memory["raw_receive_payload"] = self.raw_bytes
        memory["raw_pinned_payload"] = self.raw_bytes
        memory["actual_torch_allocated_all_arms_setup"] = torch.cuda.memory_allocated()
        memory["actual_torch_reserved_all_arms_setup"] = torch.cuda.memory_reserved()
        memory["scope"] = (
            "All arms coexist. Raw and compressed-native arms reuse the same "
            "destination pages, with identical valid values and sequential "
            "samples. Gather/fused reuse the same compressed store. Arm receive "
            "buffers are distinct. Source pages and full codec encode workspaces "
            "remain allocated and are counted; not a per-arm deployment peak."
        )

    def transfer(self, arm: str) -> None:
        """Enqueue actual pinned H2D bytes; raw needs no serialized blob offsets."""
        if arm == "raw_fa3":
            self.receive[arm].copy_(self.host_raw, non_blocking=True)
        else:
            super().transfer(arm)

    def restore(self, arm: str) -> torch.Tensor:
        """Restore the arm's received payload into native or compressed pages."""
        if arm == "raw_fa3":
            return self.codec.raw_scatter(self.native, self.mapping, self.receive[arm])
        return super().restore(arm)


class _Attention:
    """Own batched queries and one full 48-layer captured read graph per arm."""

    def __init__(
        self, ctx: _Context, case: dict[str, Any], args: argparse.Namespace
    ) -> None:
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        self.ctx, self.case = ctx, case
        self.fa3 = flash_attn_varlen_func
        contexts, queries = case["contexts"], case["queries"]
        self.max_context, self.max_query = max(contexts), max(queries)
        self.total_queries = sum(queries)
        self.kernel = "decode" if all(q == 1 for q in queries) else "prefill"
        generator = torch.Generator().manual_seed(args.seed)
        host_q = torch.randn(
            LAYERS, self.total_queries, QUERY_HEADS, D, generator=generator
        ).to(torch.bfloat16)
        self.query_sha256 = _hash_array(host_q.view(torch.int16).numpy())
        self.q = host_q.to("cuda")
        self.q_layers = list(self.q.unbind(0))
        self.outputs = {arm: torch.empty_like(self.q) for arm in ARMS}
        self.output_layers = {
            arm: list(value.unbind(0)) for arm, value in self.outputs.items()
        }
        self.native_key = [cache[0] for cache in ctx.native.caches]
        self.native_value = [cache[1] for cache in ctx.native.caches]
        table, blocks = _page_table(case)
        if blocks != ctx.blocks:
            raise AssertionError("physical block count differs from context setup")
        self.table = torch.from_numpy(table).to("cuda")
        # Gather exactly the physical union once per layer, including sharing.
        # Padding maps to a valid slot but seq_lens excludes it from attention.
        gather_table = np.maximum(table - 1, 0)
        self.gather_table = torch.from_numpy(gather_table).to("cuda")
        self.cu_q = torch.tensor(
            [0, *np.cumsum(queries).tolist()], dtype=torch.int32, device="cuda"
        )
        self.lengths = torch.tensor(contexts, dtype=torch.int32, device="cuda")
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        self.splits = prefill_split_count(
            queries, contexts, HEADS, QUERY_HEADS // HEADS, sms
        )
        self.units = decode_units(len(contexts), HEADS, sms)
        self.fused_workspace: tuple[torch.Tensor, ...] | None
        if self.kernel == "decode":
            opart, ml, plan = workspace_elems(HEADS, self.units)
            self.fused_workspace = (
                torch.empty(opart, dtype=torch.float32, device="cuda"),
                torch.empty(ml, dtype=torch.float32, device="cuda"),
                torch.empty(plan, dtype=torch.int32, device="cuda"),
            )
        else:
            counts = prefill_workspace_elems(
                self.total_queries, QUERY_HEADS, self.splits
            )
            self.fused_workspace = (
                tuple(
                    torch.empty(n, dtype=torch.float32, device="cuda") for n in counts
                )
                if self.splits > 1
                else None
            )
        self.stream = torch.cuda.current_stream()
        self.done, self.status_ready = torch.cuda.Event(), torch.cuda.Event()
        self.graph_start = torch.cuda.Event(enable_timing=True)
        self.graph_end = torch.cuda.Event(enable_timing=True)
        self.phase_events = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
        self.graphs: dict[str, torch.cuda.CUDAGraph] = {}
        for arm in ARMS:
            for _ in range(args.warmup):
                self.read(arm)
        self.stream.synchronize()
        self.reference = self.outputs["native_fa3"].clone()
        self.validation = {"eager_before_capture": self.validate_outputs(graph=False)}
        before = torch.cuda.memory_allocated()
        for arm in ARMS:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.read(arm)
            self.graphs[arm] = graph
        self.validation["graph_after_capture"] = self.validate_outputs(graph=True)
        self.memory = {
            "queries": _tensor_bytes(self.q),
            "output_per_arm": _tensor_bytes(self.outputs[ARMS[0]]),
            "validation_reference": _tensor_bytes(self.reference),
            "fused_workspace_reused_across_layers": sum(
                _tensor_bytes(t) for t in self.fused_workspace or ()
            ),
            "attention_metadata": sum(
                _tensor_bytes(t)
                for t in (self.table, self.gather_table, self.cu_q, self.lengths)
            ),
            "torch_allocated_before_graphs": before,
            "torch_allocated_after_graphs": torch.cuda.memory_allocated(),
            "torch_reserved_after_graphs": torch.cuda.memory_reserved(),
        }

    def _fa3(
        self,
        layer: int,
        key: torch.Tensor,
        value: torch.Tensor,
        table: torch.Tensor,
        arm: str,
    ) -> None:
        self.fa3(
            q=self.q_layers[layer],
            k=key,
            v=value,
            out=self.output_layers[arm][layer],
            cu_seqlens_q=self.cu_q,
            max_seqlen_q=self.max_query,
            seqused_k=self.lengths,
            max_seqlen_k=self.max_context,
            softmax_scale=D**-0.5,
            causal=True,
            block_table=table,
            fa_version=3,
            num_splits=0,
        )

    def read(self, arm: str) -> None:
        """Submit 48 ordered layers; decode planning is paid once, at layer zero."""
        for layer in range(LAYERS):
            if arm in ("raw_fa3", "native_fa3"):
                self._fa3(
                    layer,
                    self.native_key[layer],
                    self.native_value[layer],
                    self.table,
                    arm,
                )
            elif arm == "gather_fa3":
                self.ctx.store.gather(
                    layer, self.ctx.ids, self.ctx.gather_k, self.ctx.gather_v
                )
                self._fa3(
                    layer, self.ctx.gather_k, self.ctx.gather_v, self.gather_table, arm
                )
            elif self.kernel == "decode":
                decode_attention(
                    self.q_layers[layer],
                    self.ctx.tables[layer],
                    self.table,
                    self.lengths,
                    D**-0.5,
                    self.units,
                    self.output_layers[arm][layer],
                    self.fused_workspace,
                    replan=layer == 0,
                )
            else:
                prefill_attention(
                    self.q_layers[layer],
                    self.ctx.tables[layer],
                    self.table,
                    self.cu_q,
                    self.lengths,
                    self.max_query,
                    D**-0.5,
                    self.output_layers[arm][layer],
                    num_splits=self.splits,
                    workspace=self.fused_workspace,
                )

    def validate_outputs(self, *, graph: bool) -> dict[str, Any]:
        """Poison outputs, run the measured reader, and validate layer by layer."""
        results = {}
        for arm in ARMS:
            self.outputs[arm].fill_(float("nan"))
            self.graphs[arm].replay() if graph else self.read(arm)
            maximum, exact = 0.0, True
            for layer in range(LAYERS):
                actual, expected = self.output_layers[arm][layer], self.reference[layer]
                torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
                exact &= torch.equal(
                    actual.view(torch.int16), expected.view(torch.int16)
                )
                maximum = max(
                    maximum, float((actual.float() - expected.float()).abs().max())
                )
            if arm != "fused_hbm" and not exact:
                raise AssertionError(f"{arm} output differs from native FA3 bits")
            results[arm] = {
                "bitwise_equal": exact,
                "max_abs": maximum,
                "atol": ATOL,
                "rtol": RTOL,
            }
        return results

    def sample(
        self, arm: str, mode: str, *, diagnostic: bool = False
    ) -> dict[str, Any]:
        """Time one restore (cold only) followed by R complete read-only replays."""
        cold, graph = mode.startswith("cold"), mode.endswith("graph")
        events = self.phase_events
        started = time.perf_counter_ns()
        if diagnostic:
            events[0].record(self.stream)
        if cold:
            self.ctx.transfer(arm)
            if diagnostic:
                events[1].record(self.stream)
            status = self.ctx.restore(arm)
            if diagnostic:
                events[2].record(self.stream)
            self.ctx.status_host[arm].copy_(status, non_blocking=True)
            if diagnostic:
                events[3].record(self.stream)
            self.status_ready.record(self.stream)
            waiting = time.perf_counter_ns()
            self.status_ready.synchronize()
            self.ctx.check_status(arm)
            ready = time.perf_counter_ns()
        else:
            waiting = ready = started
        if diagnostic:
            events[4].record(self.stream)
        if graph and not cold and not diagnostic:
            self.graph_start.record(self.stream)
        for _ in range(self.case["reads"]):
            self.graphs[arm].replay() if graph else self.read(arm)
        if diagnostic:
            events[5].record(self.stream)
        if graph and not cold and not diagnostic:
            self.graph_end.record(self.stream)
            self.graph_end.synchronize()
        else:
            self.done.record(self.stream)
            self.done.synchronize()
        finished = time.perf_counter_ns()
        result: dict[str, Any] = {
            "arm": arm,
            "mode": mode,
            "diagnostic": diagnostic,
            "reads": self.case["reads"],
            "wall_us": (finished - started) / 1000,
            "wall_us_per_read": (finished - started) / (1000 * self.case["reads"]),
        }
        if graph and not cold and not diagnostic:
            result["gpu_us"] = self.graph_start.elapsed_time(self.graph_end) * 1000
            result["gpu_us_per_read"] = result["gpu_us"] / self.case["reads"]
        if cold:
            result.update(
                {
                    "h2d_restore_ready_wall_us": (ready - started) / 1000,
                    "status_barrier_wait_check_us": (ready - waiting) / 1000,
                    "read_wall_us": (finished - ready) / 1000,
                    "payload_h2d_bytes": self.ctx.raw_bytes
                    if arm == "raw_fa3"
                    else self.ctx.payload_bytes,
                    "offsets_h2d_bytes": 0
                    if arm == "raw_fa3"
                    else _tensor_bytes(self.ctx.host_offsets),
                    "status_d2h_bytes": _tensor_bytes(self.ctx.status_host[arm]),
                }
            )
        if diagnostic:
            result["cuda_spans_us"] = {
                "read": events[4].elapsed_time(events[5]) * 1000,
                "full_stream_span": events[0].elapsed_time(events[5]) * 1000,
            }
            if cold:
                result["cuda_spans_us"].update(
                    {
                        "h2d_payload_offsets": events[0].elapsed_time(events[1]) * 1000,
                        "restore": events[1].elapsed_time(events[2]) * 1000,
                        "status_d2h": events[2].elapsed_time(events[3]) * 1000,
                    }
                )
        return result


def _aggregate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for arm in ARMS:
        rows = [sample for sample in samples if sample["arm"] == arm]
        output[arm] = {
            key: _summary([row[key] for row in rows])
            for key in rows[0]
            if key != "cuda_spans_us"
            and (key.endswith("_us") or key.endswith("_us_per_read"))
        }
        if "cuda_spans_us" in rows[0]:
            output[arm]["cuda_spans_us"] = {
                key: _summary([row["cuda_spans_us"][key] for row in rows])
                for key in rows[0]["cuda_spans_us"]
            }
    return output


def _run_case(
    args: argparse.Namespace,
    case: dict[str, Any],
    table: np.ndarray,
    pool: np.ndarray,
    row: dict[str, Any],
    report: dict[str, Any],
) -> None:
    _, blocks = _page_table(case)
    content, provenance = _materialize(
        pool, table, blocks, case["fallback_fraction"], args.seed
    )
    row["input"] = provenance
    ctx = _Context(table, content)
    row["setup"] = ctx.metadata
    attention = _Attention(ctx, case, args)
    row.update(
        {
            "query_sha256": attention.query_sha256,
            "fused_kernel": attention.kernel,
            "automatic_fused_splits": attention.splits
            if attention.kernel == "prefill"
            else None,
            "decode_units": attention.units if attention.kernel == "decode" else None,
            "memory_bytes": attention.memory,
            "validation": attention.validation,
            "measurements": {},
        }
    )
    _write(args.out, report)
    for mode in args.modes:
        measurement: dict[str, Any] = {
            "warmup_samples": [],
            "samples": [],
            "diagnostic_samples": [],
        }
        row["measurements"][mode] = measurement
        for order in _orders(args.warmup, args.order == "reverse"):
            for arm in order:
                measurement["warmup_samples"].append(attention.sample(arm, mode))
        for index, order in enumerate(_orders(args.rounds, args.order == "reverse")):
            for position, arm in enumerate(order):
                sample = attention.sample(arm, mode)
                sample.update({"round": index, "position": position, "order": order})
                measurement["samples"].append(sample)
        measurement["summary"] = _aggregate(measurement["samples"])
        if args.diagnostics:
            for arm in ARMS:
                attention.sample(arm, mode, diagnostic=True)
            for index, order in enumerate(
                _orders(args.diagnostics, args.order == "reverse")
            ):
                for position, arm in enumerate(order):
                    sample = attention.sample(arm, mode, diagnostic=True)
                    sample.update(
                        {"round": index, "position": position, "order": order}
                    )
                    measurement["diagnostic_samples"].append(sample)
            measurement["diagnostic_summary"] = _aggregate(
                measurement["diagnostic_samples"]
            )
        measurement["output_validation"] = attention.validate_outputs(
            graph=mode.endswith("graph")
        )
        if mode.startswith("cold"):
            # Validate each source format after an explicit restore; raw/native
            # share a destination, so checking just the final arm is insufficient.
            measurement["restore_validation"] = {}
            for arm in ("raw_fa3", "native_fa3", "gather_fa3", "fused_hbm"):
                if arm in ("raw_fa3", "native_fa3"):
                    for cache in ctx.native.caches:
                        cache[:, 1 : ctx.blocks + 1].view(torch.int16).fill_(0x6B6B)
                    poisoned = "selected native BF16 pages"
                else:
                    for arena in ctx.arenas:
                        arena[1 : ctx.blocks + 1].fill_(0x6B)
                    # Overflow holds record payload only. Leave allocation bits,
                    # record pointers/lengths, and unused-page canaries intact;
                    # import must replace every live byte before it is read.
                    ctx.store.overflow.fill_(0x6B)
                    poisoned = "selected arena pages and all overflow payload"
                ctx.transfer(arm)
                status = ctx.restore(arm)
                ctx.status_host[arm].copy_(status, non_blocking=True)
                ctx.ready.record(ctx.stream)
                ctx.ready.synchronize()
                ctx.check_status(arm)
                validation = ctx.validate_kv()
                validation["destination_poison_before_restore"] = poisoned
                measurement["restore_validation"][arm] = validation
        _write(args.out, report)
    row["status"] = "passed"
    print(
        json.dumps(
            {
                "name": case["name"],
                "fused_kernel": attention.kernel,
                "physical_blocks": blocks,
                "median_wall_us": {
                    mode: {
                        arm: row["measurements"][mode]["summary"][arm]["wall_us"][
                            "median"
                        ]
                        for arm in ARMS
                    }
                    for mode in args.modes
                },
            }
        ),
        flush=True,
    )


def main() -> None:
    """Run cases sequentially, writing completed evidence before the next case."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--calib-pages", type=int, default=8)
    parser.add_argument("--eval-start", type=int, default=8)
    parser.add_argument("--eval-pages", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--diagnostics", type=int, default=3)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--order", choices=("mixed", "reverse"), default="mixed")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--input-description", required=True)
    args = parser.parse_args()
    args.modes = args.modes.split(",")
    if (
        min(args.rounds, args.warmup, args.calib_pages, args.eval_pages) < 1
        or args.diagnostics < 0
        or args.eval_start < args.calib_pages
        or not args.modes
        or len(set(args.modes)) != len(args.modes)
        or any(mode not in MODES for mode in args.modes)
    ):
        raise ValueError("invalid sizes, disjoint positions, or timing modes")
    cases = _cases(args.cases)
    report: dict[str, Any] = {
        "schema": "fragpack.native-codec-expanded-comparison.v1",
        "status": "preparing",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "source": _source_identity(),
        "cases": [],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "vllm": importlib.metadata.version("vllm"),
        },
        "protocol": {
            "arms": {
                "raw_fa3": (
                    "raw pinned-host archive -> H2D -> native raw scatter -> FA3"
                ),
                "native_fa3": (
                    "FPKB pinned-host archive -> H2D -> decode into native BF16 pages "
                    "-> FA3"
                ),
                "gather_fa3": (
                    "FPKB -> H2D -> compressed import -> unique-page gather per layer "
                    "-> FA3"
                ),
                "fused_hbm": (
                    "FPKB -> H2D -> compressed import -> production decode for all-Q1, "
                    "otherwise automatic split prefill"
                ),
            },
            "work": (
                "48 ordered attention layers, per-layer independent synthetic Q, "
                "all KV already complete; no model forward or new KV write"
            ),
            "repeated_reads": (
                "R repetitions of the same immutable Q/KV after one restore; no "
                "growing autoregressive context, one captured 48-layer graph "
                "replayed R times"
            ),
            "resident": (
                "R complete read replays; resident_graph CUDA-event GPU span and "
                "complete host wall are separate, per-read divisions explicit"
            ),
            "cold": (
                "actual pinned H2D payload/offsets -> public restore -> status "
                "D2H/event wait/check -> R reads -> completion; complete host wall"
            ),
            "sharing": (
                "same physical shared-prefix page mapping for all arms; gather "
                "reconstructs each physical union page once per layer"
            ),
            "fa3": {
                "fa_version": 3,
                "num_splits": 0,
                "causal": True,
                "softmax_scale": D**-0.5,
            },
            "statistics": (
                "24-permutation arm schedule, raw samples retained, "
                "medians/p10/p90, no sample filtering; reverse run reverses every "
                "permutation"
            ),
            "diagnostics": (
                "separate phase-instrumented samples; CUDA spans can include host "
                "launch gaps, phase medians not additive"
            ),
            "restore_contracts": (
                "raw checks page mapping; native compressed additionally validates "
                "record payload; old import validates header/digest/record sizes; "
                "public interface differences retained"
            ),
            "excluded": (
                "archive construction/encoding/D2H, allocations/JIT/calibration, "
                "QKV projection/MLP/new KV writes, model dependencies, service "
                "scheduler/RPC/network/SSD, overlap and policy"
            ),
            "memory": (
                "all-arm allocator is not deployment peak; physical KV, codec "
                "workspace, receive/pinned, gather/fused workspaces, Q/output "
                "separately reported"
            ),
        },
    }
    _write(args.out, report)
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("select exactly one explicitly authorized exclusive GPU")
        props = torch.cuda.get_device_properties(0)
        report["environment"].update(
            {
                "gpu_name": props.name,
                "gpu_uuid": str(props.uuid),
                "sm_count": props.multi_processor_count,
                "compute_capability": [props.major, props.minor],
                "gpu_memory_bytes": props.total_memory,
            }
        )
        dump = np.load(args.dump, mmap_mode="r", allow_pickle=False)
        if dump.dtype != np.uint16 or dump.shape[1:] != (LAYERS, 2, T, HEADS, D):
            raise ValueError("dump must be uint16 [blocks,48,2,128,4,128]")
        if args.eval_start + args.eval_pages > len(dump):
            raise ValueError("evaluation pool exceeds available captured blocks")
        table = calibrate(np.asarray(dump[: args.calib_pages]))
        pool = np.array(dump[args.eval_start : args.eval_start + args.eval_pages])
        report["input"] = {
            "dump": str(args.dump.resolve()),
            "dump_sha256": _hash_file(args.dump),
            "shape": list(dump.shape),
            "description": args.input_description,
            "calibration_range": [0, args.calib_pages],
            "evaluation_pool_range": [
                args.eval_start,
                args.eval_start + args.eval_pages,
            ],
            "table_sha256": _hash_array(table),
            "pool_sha256": _hash_array(pool),
            "split_scope": (
                "disjoint positions of one capture, not independent documents"
            ),
        }
        report["status"] = "running"
        for case in cases:
            row = {"case": case, "status": "preparing"}
            report["cases"].append(row)
            _write(args.out, report)
            _run_case(args, case, table, pool, row, report)
            gc.collect()
            torch.cuda.empty_cache()
            _write(args.out, report)
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _write(args.out, report)


if __name__ == "__main__":
    main()
