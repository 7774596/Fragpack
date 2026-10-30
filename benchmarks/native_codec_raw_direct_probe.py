# SPDX-License-Identifier: Apache-2.0
"""GPU-only bit/stream/storage probe; optional profiler data is not performance data."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from benchmarks import native_codec_raw_direct as direct


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("one explicitly selected exclusive GPU is required")
    paths = [Path(__file__), Path(direct.__file__)]
    identity = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report = {
        "status": "running",
        "scope": "functional/profile-only; no primary performance timings",
        "profiled": args.profile,
        "sources": identity,
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "probes": [],
    }

    def save() -> None:
        temp = args.out.with_suffix(".tmp")
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(args.out)

    save()
    active = False
    try:
        layers, blocks, capacity, tokens, heads, dim = 48, 5, 9, 128, 4, 128
        shape = (layers, 2, blocks, tokens, heads, dim)
        x = np.arange(np.prod(shape), dtype=np.uint32)
        bits = ((x * 40503) ^ (x >> 16) ^ 0x7F01).astype(np.uint16).reshape(shape)
        archives = []
        expected = []
        for phase in range(3):
            value = bits ^ np.uint16(phase * 0x0101)
            raw = torch.empty(value.nbytes, dtype=torch.uint8, pin_memory=True)
            raw.numpy()[:] = value.view(np.uint8).reshape(-1)
            archives.append(
                SimpleNamespace(raw=raw, raw_layout="native-direct", blocks=blocks)
            )
            expected.append(value)
        del x, bits, raw, value
        caches = [
            torch.empty(
                (2, capacity, tokens, heads, dim), dtype=torch.bfloat16, device="cuda"
            )
            for _ in range(layers)
        ]
        runtime = SimpleNamespace(
            pages=SimpleNamespace(caches=caches),
            stream=torch.cuda.Stream(),
            restored=torch.cuda.Event(),
        )
        caller = torch.cuda.Stream()
        mappings = [
            ("nonzero-contiguous", (2, 3, 4, 5, 6)),
            ("reverse-fragmented", (8, 6, 4, 2, 0)),
            ("fragmented", (0, 1, 4, 5, 8)),
        ]
        # Initialize stream/event machinery outside the profiled range.
        direct.restore_native_raw(runtime, archives[0], mappings[0][1], validate=False)
        torch.cuda.synchronize()
        if args.profile:
            torch.cuda.profiler.start()
            active = True
        for phase, (name, slots) in enumerate(mappings):
            for cache in caches:
                cache.view(torch.int16).fill_(0x6B6B)
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            with torch.cuda.stream(caller):
                with torch.cuda.nvtx.range("raw_direct/" + name):
                    result = direct.restore_native_raw(
                        runtime, archives[phase], slots, validate=False
                    )
                assert torch.cuda.current_stream() == caller
            peak = torch.cuda.max_memory_allocated()
            assert peak == before and torch.cuda.memory_allocated() == before
            for layer, cache in enumerate(caches):
                actual = cache.view(torch.int16).cpu().numpy().view(np.uint16)
                want = np.full(actual.shape, 0x6B6B, dtype=np.uint16)
                for source, target in enumerate(slots):
                    want[:, target] = expected[phase][layer, :, source]
                assert np.array_equal(actual, want), (name, layer)
            report["probes"].append(
                {
                    "name": name,
                    "slots": slots,
                    "all_48_layer_bits_exact": True,
                    "inactive_pages_intact": True,
                    "caller_stream_restored": True,
                    "runtime_stream_distinct_from_caller": runtime.stream.cuda_stream
                    != caller.cuda_stream,
                    "persistent_live_bytes": before,
                    "peak_bytes": peak,
                    "no_added_gpu_storage": True,
                    **result,
                }
            )
        if active:
            torch.cuda.profiler.stop()
            active = False
        # Invalid maps must not submit a copy, independent of the caller stream.
        for cache in caches:
            cache.view(torch.int16).fill_(0x6B6B)
        torch.cuda.synchronize()
        bad = [
            (0, 1, 2, 3, capacity),
            (0, 1, 2, 3, 0),
            (0, 1, 2, 3),
            (0, 1, 2, 3, True),
        ]
        for slots in bad:
            try:
                with torch.cuda.stream(caller):
                    direct.restore_native_raw(
                        runtime, archives[0], slots, validate=False
                    )
            except ValueError:
                pass
            else:
                raise AssertionError("invalid mapping accepted")
        for cache in caches:
            assert np.all(
                cache.view(torch.int16).cpu().numpy().view(np.uint16) == 0x6B6B
            )
        report["invalid_maps"] = {"count": len(bad), "all_pages_unchanged": True}
        del cache, caches, runtime
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        report["remaining_allocated_bytes"] = torch.cuda.memory_allocated()
        assert report["remaining_allocated_bytes"] == 0
        after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        assert identity == after
        report.update(status="passed", sources_after=after)
        save()
        print(
            json.dumps(
                {
                    "status": "passed",
                    "probes": len(report["probes"]),
                    "invalid_maps": len(bad),
                }
            ),
            flush=True,
        )
    except BaseException as error:
        report.update(
            status="failed", error={"type": type(error).__name__, "message": str(error)}
        )
        save()
        raise
    finally:
        if active:
            torch.cuda.profiler.stop()


if __name__ == "__main__":
    main()
