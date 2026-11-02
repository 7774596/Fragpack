# SPDX-License-Identifier: Apache-2.0
"""Decode-attention error vs an fp64 reference on captured KV.

BF16: fused FragPack vs FA3 over the BF16 cache. FP8: fused FP8 FragPack vs
FA3 FP8 (vLLM's path: query quantized to E4M3, unit scales) over the same
E4M3 cache. The reference uses the stored KV values and the BF16 query.
Usage::

    python -m benchmarks.lossless_kv.kernels.fused_error --dump KV.npy [--fp8]
"""

# Standard
import argparse
import math

# Third Party
import numpy as np
import torch

# First Party
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

SHAPES = [[131072], [66000] * 4, [32000] * 8, [4000] * 64]
GQA = 8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp8", action="store_true")
    ap.add_argument("--dump", required=True, help="captured KV .npy")
    ap.add_argument("--layer", type=int, default=24)
    a = ap.parse_args()
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    torch.manual_seed(0)
    heads, nb = 4, 2112
    dump = np.load(a.dump, mmap_mode="r")
    layer = np.asarray(dump[:, a.layer : a.layer + 1, :, :, 0:heads])
    table, offset = calibrate(layer[:8], a.fp8)
    content = layer[8:, 0]
    kv = (
        torch.from_numpy(content[np.arange(nb) % content.shape[0]].view(np.int16))
        .cuda()
        .view(torch.bfloat16)
    )
    cache = torch.zeros(nb, page_bytes(heads, a.fp8), dtype=torch.uint8, device="cuda")
    store = HbmKvStore(
        [cache],
        heads,
        device_tables(table, torch.device("cuda"), a.fp8, offset),
        FILL_BLOCKS * T,
        8,
        4096,
    )
    fill_store(store, kv)
    if a.fp8:
        b = kv.view(torch.int16).cpu().numpy().view(np.uint16)
        kv = torch.from_numpy(fp8_bytes(b)).cuda().view(torch.float8_e4m3fn)
    kc, vc = kv[:, 0].contiguous(), kv[:, 1].contiguous()  # [nb, T, H, D]
    tables = store.layer_tables(0)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    tot = {"fused": [], "fa3": []}
    for lens in SHAPES:
        bsz = len(lens)
        pages = [math.ceil(x / T) for x in lens]
        perm = torch.randperm(nb, device="cuda", dtype=torch.int32)
        bt = torch.zeros(bsz, max(pages), dtype=torch.int32, device="cuda")
        used = 0
        for i, p in enumerate(pages):
            bt[i, :p] = perm[used : used + p]
            used += p
        q = torch.randn(bsz, heads * GQA, 128, dtype=torch.bfloat16, device="cuda")
        seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
        cu = torch.arange(bsz + 1, dtype=torch.int32, device="cuda")
        out, out2 = torch.empty_like(q), torch.empty_like(q)
        u = decode_units(bsz, heads, sms)
        o, m, pl = workspace_elems(heads, u)
        w = (
            torch.empty(o, device="cuda"),
            torch.empty(m, device="cuda"),
            torch.empty(pl, dtype=torch.int32, device="cuda"),
        )
        decode_attention(q, tables, bt, seq, 128**-0.5, u, out, w, True)
        fa_q, quant = q, {}
        if a.fp8:
            fa_q = q.to(torch.float8_e4m3fn)
            one = torch.ones(bsz, heads, device="cuda")
            quant = dict(q_descale=one, k_descale=one, v_descale=one)
        flash_attn_varlen_func(
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
            fa_version=3,
            **quant,
        )
        torch.cuda.synchronize()
        errs = {"fused": [], "fa3": []}
        for i, n in enumerate(lens):
            blk = bt[i, : pages[i]].long()
            K = kc[blk].reshape(-1, heads, 128)[:n].double()
            V = vc[blk].reshape(-1, heads, 128)[:n].double()
            for h in range(heads):
                qh = q[i, h * GQA : (h + 1) * GQA].double()
                ref = torch.softmax(qh @ K[:, h].T * 128**-0.5, -1) @ V[:, h]
                for name, ob in (("fused", out), ("fa3", out2)):
                    errs[name].append(
                        (ob[i, h * GQA : (h + 1) * GQA].double() - ref).abs().flatten()
                    )
        row = f"{bsz}x{lens[0]:<6}"
        for name in ("fused", "fa3"):
            e = torch.cat(errs[name])
            tot[name].append(e)
            row += (
                f" | {name} mean {e.mean().item():.3e}"
                f" p99 {e.quantile(0.99).item():.3e} max {e.max().item():.3e}"
            )
        print(row, flush=True)
    for name in ("fused", "fa3"):
        e = torch.cat(tot[name])
        print(f"ALL {name}: mean {e.mean().item():.3e} max {e.max().item():.3e}")


if __name__ == "__main__":
    main()
