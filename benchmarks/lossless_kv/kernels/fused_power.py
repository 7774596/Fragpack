# SPDX-License-Identifier: Apache-2.0
"""Sustained time, energy, clock and power of fused decode vs FA3.

Per shape, in one process: FA3, fused over coded records, fused over raw
records, optionally FlashInfer's paged decode (``--flashinfer``, vLLM's
FLASHINFER setting), and the same kernels each followed by a fixed weight-read GEMM block
(a decode-step stand-in, so attention shares the step's clock). NVML energy,
SM clock, power, temperature and throttle reasons are read over each window.
The KV comes from one layer of a captured dump (``capture_kv.py``); the
first 8 slots calibrate, the rest fill the pages. Env: ``SMI_GPU`` (NVML
index of the measured GPU, default 0), ``HEADS`` (KV heads, default 4).

Usage::

    python -m benchmarks.lossless_kv.kernels.fused_power --dump KV.npy \
        OUT.json [--fp8] [--gemm-mb 600] [--flashinfer] [--no-gemm]
"""

# Standard
from functools import partial
from typing import Any, Callable
import argparse
import json
import math
import os
import statistics
import threading
import time

# Third Party
import numpy as np
import pynvml
import torch

# First Party
from benchmarks.lossless_kv.kernels.fi_decode import flashinfer_decode
from benchmarks.lossless_kv.kernels.fragpack_step_bench import (
    FILL_BLOCKS,
    T,
    fill_store,
)
from lmcache.v1.lossless_kv.fragpack.attn_tables import (
    calibrate,
    device_tables,
    fp8_bytes,
)
from lmcache.v1.lossless_kv.fragpack.fused_attn_decode import (
    decode_attention,
    decode_units,
    workspace_elems,
)
from lmcache.v1.lossless_kv.fragpack.hbm_kv import HbmKvStore, page_bytes

SHAPES = [
    [8000],
    [66000],
    [66000] * 4,
    [32000] * 8,
    [16000] * 16,
    [8000] * 32,
    [4000] * 64,
]


def sustained(
    fn: Callable[[], object], warm: float, window: float, handle: Any
) -> dict[str, Any]:
    """Run fn back to back; return (us/call, mJ/call, median SM MHz) over the window."""
    t0 = time.time()
    while time.time() - t0 < warm:
        for _ in range(50):
            fn()
        torch.cuda.synchronize()
    clocks, watts, temps, thr, stop = [], [], [], [0], [False]

    def poll():
        while not stop[0]:
            clocks.append(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
            watts.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1e3)
            temps.append(
                pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            )
            thr[0] |= pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
            time.sleep(0.05)

    th = threading.Thread(target=poll)
    th.start()
    e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    n, t0 = 0, time.time()
    while time.time() - t0 < window:
        for _ in range(50):
            fn()
        n += 50
        torch.cuda.synchronize()
    e.record()
    torch.cuda.synchronize()
    e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    stop[0] = True
    th.join()
    return dict(
        us=round(s.elapsed_time(e) / n * 1e3, 1),
        mJ=round((e1 - e0) / n, 2),
        MHz=statistics.median(clocks),
        W=round(statistics.median(watts)),
        C=statistics.median(temps),
        thr=hex(thr[0]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--dump", required=True, help="captured KV .npy")
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--fp8", action="store_true")
    ap.add_argument("--gemm-mb", type=int, default=600)
    ap.add_argument("--warm", type=float, default=4)
    ap.add_argument("--window", type=float, default=6)
    ap.add_argument("--shapes", default="")
    ap.add_argument("--flashinfer", action="store_true", help="add FlashInfer decode")
    ap.add_argument("--no-gemm", action="store_true", help="skip the +GEMM rows")
    a = ap.parse_args()
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    from vllm.vllm_flash_attn.flash_attn_interface import get_scheduler_metadata

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(os.environ.get("SMI_GPU", "0")))
    torch.manual_seed(0)
    heads, nb = int(os.environ.get("HEADS", "4")), 2112
    dump = np.load(a.dump, mmap_mode="r")
    layer = np.asarray(dump[:, a.layer : a.layer + 1, :, :, 0:heads])
    content = layer[8:, 0]
    kv = (
        torch.from_numpy(content[np.arange(nb) % content.shape[0]].view(np.int16))
        .cuda()
        .view(torch.bfloat16)
    )

    def make_store(raw):
        calib = layer[:8]
        if raw:
            calib = np.random.default_rng(0).integers(
                0, 1 << 16, calib.shape, np.uint16
            )
        table, offset = calibrate(calib, a.fp8)
        cache = torch.zeros(
            nb, page_bytes(heads, a.fp8), dtype=torch.uint8, device="cuda"
        )
        store = HbmKvStore(
            [cache],
            heads,
            device_tables(table, torch.device("cuda"), a.fp8, offset),
            FILL_BLOCKS * T,
            8,
            2 * heads * nb + 4096 if raw else 4096,
        )
        fill_store(store, kv)
        return store, cache

    store, cache = make_store(False)
    rstore, rcache = make_store(True)
    tables, rtables = store.layer_tables(0), rstore.layer_tables(0)
    if a.fp8:
        bits = kv.view(torch.int16).cpu().numpy().view(np.uint16)
        kv = torch.from_numpy(fp8_bytes(bits)).cuda().view(torch.float8_e4m3fn)
    kc, vc = kv[:, 0].contiguous(), kv[:, 1].contiguous()
    # Weight-read GEMM block: x[64, 4096] @ W_i, W_i 4096 x n, about gemm_mb MB
    # read per block.
    ncol = a.gemm_mb * 2**20 // (4 * 4096 * 2)
    gw = [
        torch.randn(4096, ncol, dtype=torch.bfloat16, device="cuda") for _ in range(4)
    ]
    gx = torch.randn(64, 4096, dtype=torch.bfloat16, device="cuda")
    gy = torch.empty(64, ncol, dtype=torch.bfloat16, device="cuda")

    def gemm():
        for w_ in gw:
            torch.matmul(gx, w_, out=gy)

    sms = torch.cuda.get_device_properties(0).multi_processor_count
    shapes = (
        SHAPES
        if not a.shapes
        else [[int(x) for x in s.split("/")] for s in a.shapes.split(",")]
    )
    rows = []
    for lens in shapes:
        b = len(lens)
        pages = [math.ceil(x / T) for x in lens]
        perm = torch.randperm(nb, device="cuda", dtype=torch.int32)
        bt = torch.zeros(b, max(pages), dtype=torch.int32, device="cuda")
        used = 0
        for i, p in enumerate(pages):
            bt[i, :p] = perm[used : used + p]
            used += p
        q = torch.randn(b, heads * 8, 128, dtype=torch.bfloat16, device="cuda")
        seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
        cu = torch.arange(b + 1, dtype=torch.int32, device="cuda")
        out, out2, out3 = torch.empty_like(q), torch.empty_like(q), torch.empty_like(q)
        u = decode_units(b, heads, sms)
        o, m, pl = workspace_elems(heads, u)
        w = (
            torch.empty(o, device="cuda"),
            torch.empty(m, device="cuda"),
            torch.empty(pl, dtype=torch.int32, device="cuda"),
        )
        decode_attention(q, tables, bt, seq, 128**-0.5, u, out, w, True)
        decode_attention(q, rtables, bt, seq, 128**-0.5, u, out, w, True)
        fa_q, quant = q, {}
        if a.fp8:
            fa_q = q.to(torch.float8_e4m3fn)
            one = torch.ones(b, heads, device="cuda")
            quant = dict(q_descale=one, k_descale=one, v_descale=one)
        meta = get_scheduler_metadata(
            batch_size=b,
            max_seqlen_q=1,
            max_seqlen_k=max(lens),
            num_heads_q=heads * 8,
            num_heads_kv=heads,
            headdim=128,
            cache_seqlens=seq,
            qkv_dtype=fa_q.dtype,
            cu_seqlens_q=cu,
            page_size=T,
            causal=True,
            num_splits=32,
        )
        fa3 = partial(
            flash_attn_varlen_func,
            q=fa_q,
            k=kc,
            v=vc,
            out=out2,
            cu_seqlens_q=cu,
            max_seqlen_q=1,
            seqused_k=seq,
            max_seqlen_k=max(lens),
            softmax_scale=128**-0.5,
            causal=True,
            block_table=bt,
            scheduler_metadata=meta,
            num_splits=32,
            fa_version=3,
            **quant,
        )
        fused = partial(
            decode_attention, q, tables, bt, seq, 128**-0.5, u, out, w, False
        )
        fraw = partial(
            decode_attention, q, rtables, bt, seq, 128**-0.5, u, out, w, False
        )
        row = {
            "lens": "/".join(str(x) for x in lens)
            if len(set(lens)) > 1
            else f"{b}x{lens[0]}"
        }
        kern = [("fa3", fa3), ("fused", fused), ("fused_raw", fraw)]
        if a.flashinfer:
            fi = flashinfer_decode(q, kc, vc, bt, lens, T, 128**-0.5, out3)
            fa3()
            fi()
            torch.cuda.synchronize()
            row["fi_maxdiff"] = float((out3.float() - out2.float()).abs().max())
            kern.append(("flashinfer", fi))
        extra = [] if a.no_gemm else (
            [(n + "+gemm", (lambda f: lambda: (f(), gemm()))(f)) for n, f in kern]
            + [("gemm", gemm)]
        )
        for name, fn in kern + extra:
            row[name] = sustained(fn, a.warm, a.window, handle)
            time.sleep(2)
        print(json.dumps(row), flush=True)
        rows.append(row)
    json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
