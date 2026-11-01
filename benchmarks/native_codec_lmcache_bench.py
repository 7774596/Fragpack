# SPDX-License-Identifier: Apache-2.0
"""Measure immutable captured KV through a real LMCache CPU-L1 byte service.

The captured evaluation pool is repeated onto distinct physical GPU pages.
Timed operations include native gather/codec, pinned copies, RPC, storage waits,
host polling and transfer drain. This is a transport benchmark, not serving,
attention, NVMe performance or an independently captured long-context workload.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any, Sequence

import numpy as np
import torch

from benchmarks.native_codec_bench import Inputs
from benchmarks.native_codec_bench import _file_hash as file_hash
from benchmarks.native_codec_bench import _write as write_report
from benchmarks.native_codec_scale_bench import array_hash, load_inputs
from fragpack.compression import attn_page as ap
from fragpack.compression.attn_tables import device_tables
from fragpack.integrations.lmcache.identity import BlockKey, CacheIdentity
from fragpack.integrations.lmcache.native_transfer import (
    NativeBlockTransfer,
    TransportMode,
)
from fragpack.ops.native_codec import NativeKVPages

ARMS = ("raw-precheck", "compressed-precheck", "compressed-no-precheck")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse bounded experiment parameters without starting CUDA or services."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--server-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--input-description", required=True)
    parser.add_argument("--blocks", type=int, default=64)
    parser.add_argument("--calib-pages", type=int, default=8)
    parser.add_argument("--eval-start", type=int, default=8)
    parser.add_argument("--eval-count", type=int, default=24)
    parser.add_argument(
        "--page-tokens", type=int, choices=(16, 32, 64, 128), default=128
    )
    parser.add_argument("--chunk-blocks", type=int, default=4)
    parser.add_argument("--queue-slots", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--memory-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--poll-seconds", type=float, default=0.0005)
    parser.add_argument("--rpc-timeout", type=float, default=30.0)
    parser.add_argument("--operation-timeout", type=float, default=300.0)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    if min(args.blocks, args.chunk_blocks, args.queue_slots, args.eval_count) < 1:
        parser.error("blocks, chunks, queues and evaluation count must be positive")
    if args.chunk_blocks > args.blocks or args.rounds < 2:
        parser.error(
            "chunk must fit the workload and rounds must include repeat stores"
        )
    if not 0 < args.calib_pages <= args.eval_start:
        parser.error("calibration must precede the evaluation pool")
    if args.memory_bytes <= 0 or args.memory_bytes % ap.BLOB_ALIGN:
        parser.error("memory budget must be positive and 4096-byte aligned")
    if not 0 <= args.poll_seconds <= 0.1:
        parser.error("poll-seconds must be between 0 and 0.1")
    if (
        not all(
            math.isfinite(value)
            for value in (
                args.rpc_timeout,
                args.operation_timeout,
            )
        )
        or args.rpc_timeout <= 0
        or args.operation_timeout <= args.rpc_timeout
    ):
        parser.error("operation timeout must exceed the positive RPC timeout")
    if len(set(args.arms)) != len(args.arms):
        parser.error("arms must be distinct")
    return args


def counter_delta(
    before: dict[str, int | float], after: dict[str, int | float]
) -> dict[str, int | float]:
    """Retain every counter delta, rejecting counter resets within an arm."""
    delta = {
        key: after.get(key, 0) - before.get(key, 0)
        for key in before.keys() | after.keys()
    }
    if any(value < 0 for value in delta.values()):
        raise ValueError("transfer counters reset during a measured arm")
    return delta


def summarize_rounds(rounds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Separate first store from repeat stores, retaining all raw samples."""
    if len(rounds) < 2 or any(not row.get("bitwise_equal") for row in rounds):
        raise ValueError("at least two validated rounds are required")
    stores = [float(row["store"]["wall_seconds"]) for row in rounds]
    restores = [float(row["restore"]["wall_seconds"]) for row in rounds]
    return {
        "cold_store_seconds": stores[0],
        "repeat_store_seconds": stores[1:],
        "repeat_store_median_seconds": statistics.median(stores[1:]),
        "restore_seconds": restores,
        "restore_median_seconds": statistics.median(restores),
        "timed_transfer_sum_seconds": sum(stores) + sum(restores),
        "sum_scope": "Store plus restore wall windows; validation and setup excluded.",
        "sample_scope": (
            "Repeated rounds in one process and one retained L1, "
            "not independent replications."
        ),
    }


def source_identity() -> dict[str, Any]:
    """Hash the actual codec/integration/driver sources, including untracked edits."""
    root = Path(__file__).resolve().parents[1]
    selected = [
        path
        for path in (root / "fragpack").rglob("*")
        if path.suffix in {".py", ".cu", ".cuh", ".cpp", ".h"}
    ]
    selected.extend(
        root / path
        for path in (
            "benchmarks/native_codec_lmcache_bench.py",
            "benchmarks/native_codec_l1_server.py",
            "benchmarks/native_codec_scale_bench.py",
            "benchmarks/native_codec_bench.py",
        )
    )
    hashes = {str(path.relative_to(root)): file_hash(path) for path in sorted(selected)}
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "git_head": head.stdout.strip() if head.returncode == 0 else None,
        "files_sha256": hashes,
        "digest": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest(),
    }


class L1Service:
    """Own one CPU-only LMCache process and preserve its log/status artifacts."""

    def __init__(
        self, args: argparse.Namespace, identity: CacheIdentity, directory: Path
    ) -> None:
        directory.mkdir(parents=True, exist_ok=False)
        self.ready_file = directory / "server-status.json"
        identity_file = directory / "identity.json"
        write_report(identity_file, asdict(identity))
        self.log = (directory / "server.log").open("w", encoding="utf-8")
        root = Path(__file__).resolve().parents[1]
        self.process = subprocess.Popen(
            [
                str(args.server_python),
                "-m",
                "benchmarks.native_codec_l1_server",
                "--identity",
                str(identity_file),
                "--memory-bytes",
                str(args.memory_bytes),
                "--ready-file",
                str(self.ready_file),
                "--rpc-timeout",
                str(args.rpc_timeout),
            ],
            cwd=root,
            stdin=subprocess.PIPE,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            text=True,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONPATH": str(root) + os.pathsep + os.environ.get("PYTHONPATH", ""),
            },
        )

    def ready(self, timeout: float = 90) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    "LMCache exited before readiness; inspect server.log"
                )
            if self.ready_file.exists():
                report = json.loads(self.ready_file.read_text())
                if report["status"] == "ready":
                    return report
            time.sleep(0.05)
        raise TimeoutError("LMCache startup did not complete")

    def close(self) -> dict[str, Any]:
        """Stop only this owned process; retain its pre-shutdown storage snapshot."""
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)
            raise RuntimeError(
                "LMCache shutdown timed out; owned process killed"
            ) from None
        finally:
            self.log.close()
        report = (
            json.loads(self.ready_file.read_text()) if self.ready_file.exists() else {}
        )
        if self.process.returncode != 0 or not report.get("shutdown_complete"):
            raise RuntimeError(
                f"LMCache shutdown failed: return code {self.process.returncode}"
            )
        return report


def make_pages(
    values: np.ndarray, page_tokens: int
) -> tuple[NativeKVPages, torch.Tensor, torch.Tensor]:
    """Allocate distinct source/target native pages and populate immutable sources."""
    blocks, layers, _, _, heads, _ = values.shape
    pieces = ap.PAGE_TOKENS // page_tokens
    physical = blocks * pieces
    caches = []
    for layer in range(layers):
        cache = torch.zeros(
            (2, physical * 2, page_tokens, heads, 128),
            dtype=torch.bfloat16,
            device="cuda",
        )
        host = torch.from_numpy(np.ascontiguousarray(values[:, layer])).view(
            torch.bfloat16
        )
        host = host.reshape(blocks, 2, pieces, page_tokens, heads, 128)
        host = (
            host.permute(1, 0, 2, 3, 4, 5)
            .contiguous()
            .reshape(2, physical, page_tokens, heads, 128)
        )
        cache[:, :physical].copy_(host)
        caches.append(cache)
    pages = NativeKVPages(caches)
    mapping = torch.arange(physical * 2, dtype=torch.int32, device="cuda").reshape(
        2 * blocks, pieces
    )
    torch.cuda.synchronize()
    return pages, mapping[:blocks], mapping[blocks:]


def validate_pages(pages: NativeKVPages, values: np.ndarray, *, target: bool) -> None:
    """Compare every BF16 bit outside timing, including all layers and heads."""
    blocks = len(values)
    physical = blocks * ap.PAGE_TOKENS // pages.page_tokens
    for layer, cache in enumerate(pages.caches):
        selected = cache[:, physical:] if target else cache[:, :physical]
        host = selected.view(torch.int16).cpu().numpy().view(np.uint16)
        actual = host.reshape(2, blocks, ap.PAGE_TOKENS, pages.kv_heads, ap.HEAD_DIM)
        actual = actual.transpose(1, 0, 2, 3, 4)
        if not np.array_equal(actual, values[:, layer]):
            raise AssertionError(
                f"{'restored' if target else 'source'} KV differs at layer {layer}"
            )


class _WarmClient:
    """Unmeasured, owned-host warmup only; no result uses this as an LMCache arm."""

    def __init__(self) -> None:
        self.blobs: dict[BlockKey, bytes] = {}

    def probe_many(self, keys: Sequence[BlockKey]) -> tuple[bool, ...]:
        return tuple(key in self.blobs for key in keys)

    def put(self, key: BlockKey, blob: bytes) -> bool:
        self.blobs[key] = blob
        return True

    def get(self, key: BlockKey) -> bytes | None:
        return self.blobs.get(key)


def warmup(
    pages: NativeKVPages,
    table: np.ndarray,
    identity: CacheIdentity,
    source_map: torch.Tensor,
    target_map: torch.Tensor,
    args: argparse.Namespace,
    mode: TransportMode,
) -> None:
    """Warm codec kernels on actual pages before starting the cold real L1."""
    transfer = NativeBlockTransfer(
        pages, table, _WarmClient(), identity, max_blocks=args.chunk_blocks, mode=mode
    )
    keys = identity.block_keys(list(range(args.chunk_blocks * ap.PAGE_TOKENS)))
    try:
        expected = (True,) * len(keys)
        ticket = transfer.submit_store(keys, source_map[: len(keys)])
        if ticket.completion.result(args.operation_timeout) != expected:
            raise AssertionError("warmup store failed")
        ticket = transfer.submit_restore(keys, target_map[: len(keys)])
        if ticket.completion.result(args.operation_timeout) != expected:
            raise AssertionError("warmup restore failed")
    finally:
        transfer.close(args.operation_timeout)


def drive(
    transfer: NativeBlockTransfer,
    keys: Sequence[BlockKey],
    mapping: torch.Tensor,
    args: argparse.Namespace,
    *,
    restore: bool,
) -> dict[str, Any]:
    """Time bounded asynchronous submissions through final completion/drain."""
    started = time.perf_counter()
    pending = []
    records = []
    next_block = polls = refusals = 0
    submit = transfer.submit_restore if restore else transfer.submit_store
    while next_block < len(keys) or pending:
        if time.perf_counter() - started > args.operation_timeout:
            raise TimeoutError(
                "timed transfer did not complete; close will retain/drain owners"
            )
        while next_block < len(keys) and len(pending) < args.queue_slots:
            stop = min(next_block + args.chunk_blocks, len(keys))
            submitted = time.perf_counter()
            ticket = submit(keys[next_block:stop], mapping[next_block:stop])
            if ticket is None:
                refusals += 1
                if transfer.failed or not pending:
                    raise RuntimeError(
                        "transfer refused admission without pending work"
                    )
                break
            pending.append((next_block, stop, submitted, ticket))
            next_block = stop
        completed = []
        for index, (first, stop, submitted, ticket) in enumerate(pending):
            if not ticket.completion.done():
                continue
            result = ticket.completion.result()
            if result != (True,) * (stop - first):
                raise AssertionError(
                    f"transfer incomplete at logical block {first}: {result}"
                )
            if not restore:
                ticket.source_ready.result()
            records.append(
                {
                    "first_block": first,
                    "blocks": stop - first,
                    "elapsed_seconds": time.perf_counter() - submitted,
                    "completed": list(result),
                }
            )
            completed.append(index)
        for index in reversed(completed):
            del pending[index]
        if pending and not completed:
            polls += 1
            time.sleep(args.poll_seconds)
    return {
        "wall_seconds": time.perf_counter() - started,
        "polls": polls,
        "pool_refusals": refusals,
        "batches": sorted(records, key=lambda row: row["first_block"]),
    }


def run_arm(
    args: argparse.Namespace, inputs: Inputs, arm: str, report: dict[str, Any]
) -> None:
    """Run one independent namespace/server while keeping the original KV immutable."""
    from lmcache.v1.multiprocess.transport.factory import RequestClientFactory

    from fragpack.integrations.lmcache.client import RemoteBlobClient

    mode = "raw" if arm.startswith("raw") else "compressed"
    precheck = arm != "compressed-no-precheck"
    rank = device_tables(inputs.table, torch.device("cpu")).rank_of
    digest = int.from_bytes(
        hashlib.blake2b(rank.numpy().tobytes(), digest_size=8).digest(),
        "little",
        signed=True,
    )
    identity = CacheIdentity(
        "immutable-capture:" + inputs.provenance["dump_sha256"] + ":" + arm,
        digest,
        inputs.table.shape[0],
        inputs.table.shape[2],
        representation="native-raw" if mode == "raw" else "fpkb",
    )
    worst_archive = identity.max_blob_bytes * args.blocks
    if worst_archive >= int(args.memory_bytes * 0.7):
        raise ValueError(
            "L1 must retain the all-raw archive below 70% usage "
            "for this reuse experiment"
        )
    row = {
        "arm": arm,
        "mode": mode,
        "precheck": precheck,
        "identity": asdict(identity),
        "namespace": identity.namespace,
        "rounds": [],
        "status": "initializing",
    }
    report["arms"].append(row)
    write_report(args.out, report)
    pages, source_map, target_map = make_pages(inputs.evaluation, args.page_tokens)
    warmup(pages, inputs.table, identity, source_map, target_map, args, mode)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    service = L1Service(args, identity, args.work_dir / arm)
    request_client = transfer = None
    try:
        row["server_initial"] = service.ready()
        request_client = RequestClientFactory.create(row["server_initial"]["endpoint"])
        client = RemoteBlobClient(request_client, identity, timeout_s=args.rpc_timeout)
        transfer = NativeBlockTransfer(
            pages,
            inputs.table,
            client,
            identity,
            max_blocks=args.chunk_blocks,
            num_slots=args.queue_slots,
            mode=mode,
            precheck=precheck,
        )
        row["memory"] = {
            "native_source_and_target_bytes": int(inputs.evaluation.nbytes * 2),
            "native_registration": pages.memory_usage(),
            "transfer": transfer.memory_usage(),
            "torch_allocated_after_init": torch.cuda.memory_allocated(),
        }
        keys = identity.block_keys(list(range(args.blocks * ap.PAGE_TOKENS)))
        physical = args.blocks * ap.PAGE_TOKENS // args.page_tokens
        row["status"] = "running"
        for index in range(args.rounds):
            round_row = {
                "round": index,
                "store_kind": "cold" if index == 0 else "repeat",
                "status": "running",
            }
            row["rounds"].append(round_row)
            before = transfer.stats()
            stored = drive(transfer, keys, source_map, args, restore=False)
            after_store = transfer.stats()
            round_row.update(
                store=stored, store_stats=counter_delta(before, after_store)
            )
            write_report(args.out, report)
            for cache in pages.caches:
                cache[:, physical:].zero_()
            torch.cuda.synchronize()  # Untimed target reset, before the restore window.
            restored = drive(transfer, keys, target_map, args, restore=True)
            after_restore = transfer.stats()
            round_row.update(
                restore=restored,
                restore_stats=counter_delta(after_store, after_restore),
            )
            write_report(args.out, report)
            validation_start = time.perf_counter()
            validate_pages(pages, inputs.evaluation, target=True)
            round_row.update(
                bitwise_equal=True,
                status="passed",
                untimed_validation_seconds=time.perf_counter() - validation_start,
            )
            write_report(args.out, report)
        validate_pages(pages, inputs.evaluation, target=False)
        row["source_unchanged_bitwise"] = True
        row["summary"] = summarize_rounds(row["rounds"])
        row["stats"] = transfer.stats()
        row["memory"]["torch_peak_allocated"] = torch.cuda.max_memory_allocated()
        row["memory"]["torch_peak_reserved"] = torch.cuda.max_memory_reserved()
        expected_blocks = args.blocks * (args.rounds if not precheck else 1)
        counter = "raw_gathered_blocks" if mode == "raw" else "encoded_blocks"
        if row["stats"].get(counter) != expected_blocks:
            raise AssertionError(
                "encoding/gather count does not match the controlled reuse schedule"
            )
        row["status"] = "passed"
    except BaseException as error:
        row["status"] = "failed"
        row["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        unwinding = sys.exc_info()[0] is not None
        close_started = time.perf_counter()
        errors = []
        closers = []
        if transfer is not None:
            closers.append(
                (
                    "NativeBlockTransfer",
                    partial(
                        transfer.close,
                        args.operation_timeout,
                    ),
                )
            )
        if request_client is not None:
            closers.append(("RequestClient", request_client.close))
        for owner, close in closers:
            try:
                close()
            except BaseException as error:
                errors.append(f"{owner}: {error}")
        try:
            row["server_final"] = service.close()
        except BaseException as error:
            errors.append(f"L1Service: {error}")
        row["shutdown_wall_seconds"] = time.perf_counter() - close_started
        if errors:
            row["status"], row["shutdown_errors"] = "failed", errors
        write_report(args.out, report)
        if errors and not unwinding:
            raise RuntimeError("; ".join(errors))
    del transfer, pages, source_map, target_map
    gc.collect()
    torch.cuda.empty_cache()


def main(argv: Sequence[str] | None = None) -> None:
    """Prepare CPU artifacts or execute on the caller's explicitly reserved GPU."""
    args = parse_args(argv)
    if args.out.exists() or args.work_dir.exists():
        raise FileExistsError(
            "preserve earlier output/work directories; choose new paths"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    source = source_identity()
    report = {
        "schema": "fragpack.native-codec-lmcache-immutable.v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "preparing",
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "source_before": source,
        "arms": [],
        "scope": {
            "workload": (
                "Repeated captured immutable KV on distinct native GPU pages; "
                "synthetic scale."
            ),
            "storage": (
                "Fresh real LMCache CPU L1 per arm, "
                "retained throughout that arm's rounds."
            ),
            "timing": (
                "Store/restore wall includes host polling, RPC and owned-stream drain; "
                "excludes setup, warmup and bitwise checks."
            ),
            "warmup": (
                "Codec-only warmup uses an unmeasured private host dictionary "
                "before the real server starts."
            ),
            "excluded": (
                "No model forward, attention, serving, TTFT, NVMe, "
                "eviction-policy comparison or future-request prediction."
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    write_report(args.out, report)
    try:
        load_args = argparse.Namespace(**vars(args))
        load_args.blocks = [args.blocks]
        load_args.source_pattern = "capture-repeat"
        inputs = load_inputs(load_args)
        report["inputs"] = inputs.provenance
        report["input_payload_bytes"] = int(inputs.evaluation.nbytes)
        report["expanded_input_sha256"] = array_hash(inputs.evaluation)
        if args.prepare_only:
            report["status"] = "prepared_cpu_only"
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("an explicitly reserved CUDA GPU is required")
            report["environment"].update(
                lmcache=importlib.metadata.version("lmcache"),
                gpu_name=torch.cuda.get_device_name(),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            )
            args.work_dir.mkdir(parents=True, exist_ok=False)
            arms = args.arms if args.order == "forward" else list(reversed(args.arms))
            for arm in arms:
                run_arm(args, inputs, arm, report)
            report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        raise
    finally:
        report["source_after"] = source_identity()
        report["source_unchanged"] = (
            report["source_after"]["digest"] == source["digest"]
        )
        if not report["source_unchanged"] and report["status"] == "passed":
            report["status"] = "source_changed_during_run"
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_report(args.out, report)


if __name__ == "__main__":
    main()
