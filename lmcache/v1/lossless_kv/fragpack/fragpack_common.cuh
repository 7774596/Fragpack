// SPDX-License-Identifier: Apache-2.0
// FragPack record constants and device helpers shared by the decode-attention and
// HBM KV store kernels. Record format: attn_page.py.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace fragpack {

constexpr int kThreads = 256;
constexpr int kTile = 128 * 256;  // 128 tokens x 256 B (BF16)
constexpr int kSymBytes = 6144;
constexpr int kHeaderBytes = 48;
constexpr int kPrimaryDirect = 7;
constexpr int kTableEntries = 22;
constexpr int kMaxWarpRaw = 32;

// Record geometry of BF16 (F8 = false) or FP8 E4M3 (F8 = true) scalars. The
// dense part is the low stream (BF16 low bytes / FP8 mantissa words) then the
// symbol words; a raw record holds thread t's word st (8 scalars) at
// kWordBytes (256 st + t).
template <bool F8>
struct Fmt {
  static constexpr int kLowBytes = F8 ? kSymBytes : 16384;
  static constexpr int kDenseBytes = kLowBytes + kSymBytes;
  static constexpr int kRawRecord = F8 ? 16384 : 32768;
  static constexpr int kVarCap = F8 ? kRawRecord - kDenseBytes - 16 : 2048;
  static constexpr int kRawVarLen = kRawRecord - kDenseBytes;  // var_len of a raw record
  static constexpr int kWordBytes = kRawRecord / 2048;
  // Escape x of warp w is the 4-byte window (escape bytes x .. x + 3) at
  // 16 + 64 w + 4 x. Expand stores 8-window groups, raw patches reach 3 windows
  // back, and the 64 w shift keeps neighbouring warps' boundary groups disjoint.
  static constexpr int kEscCap = 16 + 64 * 8 + 4 * (2 * kVarCap + 16);
};

// 16-byte chunk j of row r. Low three bits are permuted so that both the
// fill (8 threads: 4 rows x 2 halves) and ldmatrix (8 consecutive rows at one
// chunk) touch eight distinct bank groups.
__device__ __forceinline__ int swz(int r, int j) { return j ^ ((r & 7) ^ ((j >> 1) & 4)); }

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ uint4 ldg16(const void* p) {
  return __ldg(reinterpret_cast<const uint4*>(p));
}

__device__ __forceinline__ uint32_t u4get(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

__device__ __forceinline__ uint32_t u2get(const uint2& v, int i) { return i == 0 ? v.x : v.y; }

__device__ __forceinline__ int warp_excl_scan(int v, int lane) {
  int incl = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    int n = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += n;
  }
  return incl - v;
}

__device__ __forceinline__ void group_words(const uint2 (&sy)[3], int gi, uint32_t (&w)[4]) {
  w[0] = u2get(sy[(3 * gi) >> 1], (3 * gi) & 1);
  w[1] = u2get(sy[(3 * gi + 1) >> 1], (3 * gi + 1) & 1);
  w[2] = u2get(sy[(3 * gi + 2) >> 1], (3 * gi + 2) & 1);
  w[3] = ((w[0] >> 3) & 0x11111111u) | ((w[1] >> 2) & 0x22222222u) | ((w[2] >> 1) & 0x44444444u);
}

// Raw PTX prmt: unlike __byte_perm the selector is used as is, so callers
// must keep bit 3 of each used nibble clear (it selects sign replication).
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
  return r;
}

__device__ __forceinline__ uint32_t ext_lookup(const uint4& sl, uint32_t n4) {
  const uint32_t nsel = n4 & 0x7777u;
  const uint32_t va = __byte_perm(sl.x, sl.y, nsel);
  const uint32_t vb = __byte_perm(sl.z, sl.w, nsel);
  return __byte_perm(va, vb, ((n4 >> 1) & 0x4444u) | 0x3210u);
}

// Four FP8 E4M3 bytes (scalars 0..3) to two f16x2 words, exactly.
__device__ __forceinline__ void fp8x4_to_f16(uint32_t v, uint32_t* out) {
  const __half2_raw lo = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(v & 0xffffu), __NV_E4M3);
  const __half2_raw hi = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(v >> 16), __NV_E4M3);
  out[0] = (uint32_t)lo.x | ((uint32_t)lo.y << 16);
  out[1] = (uint32_t)hi.x | ((uint32_t)hi.y << 16);
}

// Raises the exponent fields of four FP8 symbols by ``o`` (per byte, the
// offset << 3). Bytes cannot carry: 0x78 + 0x78 < 0x100.
__device__ __forceinline__ uint32_t raise_exp(uint32_t x, uint32_t o) {
  return (x & 0x80808080u) | (((x & 0x78787878u) + o) & 0x78787878u);
}

// Pre-shifted FP8 exponent offsets of one thread's scalars (attn_page kernel
// order): K word st sits in head dims 16 st + 2 c + {0, 1, 8, 9} (both halves
// take k[st]); V halves ha / hb sit in head dims 16 warp + g and 16 warp + 8 + g.
struct ExpOffsets {
  uint32_t k[8];
  uint32_t va, vb;
};

// Offsets from the 128 pre-shifted bytes of a K (``xk``) and a V (``xv``)
// record; either may be null when the caller decodes only the other kind.
__device__ __forceinline__ ExpOffsets load_exp_offsets(const uint8_t* xk, const uint8_t* xv,
                                                       int tid) {
  const int lane = tid & 31, warp = tid >> 5;
  ExpOffsets o{};
  if (xk != nullptr) {
#pragma unroll
    for (int st = 0; st < 8; ++st) {
      const uint8_t* d = xk + 16 * st + 2 * (lane & 3);
      o.k[st] = __byte_perm(*reinterpret_cast<const uint16_t*>(d),
                            *reinterpret_cast<const uint16_t*>(d + 8), 0x5410);
    }
  }
  if (xv != nullptr) {
    o.va = xv[16 * warp + (lane >> 2)] * 0x01010101u;
    o.vb = xv[16 * warp + 8 + (lane >> 2)] * 0x01010101u;
  }
  return o;
}

// Decodes this thread's 64 kernel-order scalars of one record and calls
// emit(st, b) for word st = 0..7, where b[0..1] / b[2..3] are the (b0, b1)
// fragments of n-blocks 0 / 1 for k-step st (layout in attn_page): BF16
// pairs, or f16 pairs for FP8 records (E4M3 widens to f16 exactly).
//
// Escapes: each warp expands its escape nibbles through the extended table into
// a sliding window per escape (bytes x .. x + 3) in its private ``esc`` range,
// patches raw scalars into the four windows holding them, then every word
// reads its next eight escape bytes as two windows and deposits them by
// exclusive rank with ``prmt``.
// Only __syncwarp is needed: no other warp touches this warp's range until the
// next __syncthreads.
//
// A raw record (``vlen`` past kVarCap, uniform across the CTA) already holds
// the words: thread t's word st at kWordBytes (256 st + t).
//
// FP8 symbols come back from the table with their exponents lowered by the
// head dim's offset; ``xo`` holds this thread's offsets of the record's (kv,
// head) and ``kv`` (CTA-uniform) picks the K or V ones (unused for BF16).
template <bool F8, typename Emit>
__device__ __forceinline__ void decode_rec(const uint8_t* rec, int vlen, uint32_t l0, uint32_t l1,
                                           const uint8_t* slp, const ExpOffsets& xo, int kv,
                                           uint8_t* esc, int tid, Emit&& emit) {
  using F = Fmt<F8>;
  if (vlen > F::kVarCap) {
#pragma unroll
    for (int st = 0; st < 8; ++st) {
      uint32_t bf[4];
      if constexpr (F8) {
        const uint2 v = reinterpret_cast<const uint2*>(rec)[st * kThreads + tid];
        fp8x4_to_f16(v.x, bf);
        fp8x4_to_f16(v.y, bf + 2);
      } else {
        const uint4 v = reinterpret_cast<const uint4*>(rec)[st * kThreads + tid];
        bf[0] = v.x;
        bf[1] = v.y;
        bf[2] = v.z;
        bf[3] = v.w;
      }
      emit(st, bf);
    }
    return;
  }
  const int lane = tid & 31, warp = tid >> 5;
  // Chunk-major dense regions: each warp reads 512 (or 256) contiguous bytes.
  // ``lo`` holds BF16 low bytes; FP8 mantissa words reuse the symbol layout.
  uint4 lo[F8 ? 1 : 4];
  uint2 mw[3];
  uint2 sy[3];
  if constexpr (F8) {
#pragma unroll
    for (int j = 0; j < 3; ++j) mw[j] = reinterpret_cast<const uint2*>(rec)[j * kThreads + tid];
  } else {
#pragma unroll
    for (int j = 0; j < 4; ++j) lo[j] = reinterpret_cast<const uint4*>(rec)[j * kThreads + tid];
  }
#pragma unroll
  for (int j = 0; j < 3; ++j)
    sy[j] = reinterpret_cast<const uint2*>(rec + F::kLowBytes)[j * kThreads + tid];
  const uint8_t* var = rec + F::kDenseBytes;
  const uint16_t* hdr = reinterpret_cast<const uint16_t*>(var);
  const uint32_t* hdr32 = reinterpret_cast<const uint32_t*>(var);
  esc += 16 + 64 * warp;
  const int eb = hdr[8 + warp];
  {
    const uint4 sl = *reinterpret_cast<const uint4*>(slp);
    const uint32_t* nw = reinterpret_cast<const uint32_t*>(var + 48);
    const int e_end = warp < 7 ? hdr[9 + warp] : (int)hdr32[0];
    __syncwarp();
    for (int k = (eb >> 3) + lane; 8 * k < e_end; k += 32) {
      const uint32_t w = nw[k];
      const uint32_t b0 = ext_lookup(sl, w), b1 = ext_lookup(sl, w >> 16);
      const uint32_t b2 = ext_lookup(sl, nw[k + 1]);
      uint4* dst = reinterpret_cast<uint4*>(esc + 32 * k);
      dst[0] = make_uint4(b0, __funnelshift_r(b0, b1, 8), __funnelshift_r(b0, b1, 16),
                          __funnelshift_r(b0, b1, 24));
      dst[1] = make_uint4(b1, __funnelshift_r(b1, b2, 8), __funnelshift_r(b1, b2, 16),
                          __funnelshift_r(b1, b2, 24));
    }
    // The encoder caps a warp at 32 raw scalars, so one predicated patch each.
    const int rb = hdr[16 + warp];
    const int re = warp < 7 ? hdr[17 + warp] : (int)hdr32[1];
    const uint32_t v =
        rb + lane < re ? reinterpret_cast<const uint32_t*>(var + hdr32[2])[rb + lane] : 0u;
    __syncwarp();
    if (rb + lane < re) {
      const int x = v & 0xffffu;
#pragma unroll
      for (int j = 0; j < 4; ++j) esc[4 * (x - j) + j] = (uint8_t)(v >> 16);
    }
  }
  uint32_t em[8];
  int cnt = 0;
#pragma unroll
  for (int gi = 0; gi < 2; ++gi) {
    uint32_t w[4];
    group_words(sy, gi, w);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      // Bit 3 of a nibble is set iff its primary code is 7 (escape).
      const uint32_t sk = w[k] & 0x77777777u;
      em[4 * gi + k] = (sk + 0x11111111u) & 0x88888888u;
      cnt += __popc(em[4 * gi + k]);
    }
  }
  int eo = eb + warp_excl_scan(cnt, lane);
  __syncwarp();
#pragma unroll
  for (int gi = 0; gi < 2; ++gi) {
    uint32_t w[4], m[4];
    group_words(sy, gi, w);
    if constexpr (F8) group_words(mw, gi, m);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const uint32_t sk = w[k] & 0x77777777u;
      const uint32_t e = em[4 * gi + k];
      uint32_t ha = prmt(l0, l1, sk);
      uint32_t hb = prmt(l0, l1, sk >> 16);
      // This word's escape bytes in order: v0/v1 = escapes eo .. eo + 8.
      const uint32_t* ew = reinterpret_cast<const uint32_t*>(esc) + eo;
      const uint32_t v0 = ew[0], v1 = ew[4];
      eo += __popc(e);
      // Exclusive escape rank per nibble: (e >> 3) * 0x11111110.
      const uint32_t rank = e * 0x02222222u;
      ha = prmt(ha, prmt(v0, v1, rank), (e >> 1) + 0x3210u);
      hb = prmt(hb, prmt(v0, v1, rank >> 16), (e >> 17) + 0x3210u);
      uint32_t bf[4];
      if constexpr (F8) {
        ha = raise_exp(ha, kv == 0 ? xo.k[4 * gi + k] : xo.va);
        hb = raise_exp(hb, kv == 0 ? xo.k[4 * gi + k] : xo.vb);
        // Mantissa nibble j becomes byte j via the identity table.
        const uint32_t mk = m[k] & 0x77777777u;
        fp8x4_to_f16(ha | prmt(0x03020100u, 0x07060504u, mk), bf);
        fp8x4_to_f16(hb | prmt(0x03020100u, 0x07060504u, mk >> 16), bf + 2);
      } else {
        const uint32_t la = u4get(lo[2 * gi + (k >> 1)], (2 * k) & 3);
        const uint32_t lb = u4get(lo[2 * gi + (k >> 1)], (2 * k + 1) & 3);
        bf[0] = __byte_perm(la, ha, 0x5140);
        bf[1] = __byte_perm(la, ha, 0x7362);
        bf[2] = __byte_perm(lb, hb, 0x5140);
        bf[3] = __byte_perm(lb, hb, 0x7362);
      }
      emit(4 * gi + k, bf);
    }
  }
}

}  // namespace fragpack
