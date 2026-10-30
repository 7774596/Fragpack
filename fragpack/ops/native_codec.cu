// SPDX-License-Identifier: Apache-2.0
// Native BF16 pages <-> FPKB v1. No HBM store or attention dependency.
// The record encoder mirrors hbm_kv.cu and attn_page.encode_record. Its input
// words are gathered into registers, never a global BF16 staging allocation.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <climits>
#include <cstdint>
#include <vector>

#include "fragpack_common.cuh"

namespace {
using namespace fragpack;

constexpr uint32_t kMagic = 0x424b5046u;
constexpr int kVersion = 1, kFixed = 64, kAlign = 4096;
constexpr int kDecodeSmem = kTile + kEscCap;
enum : int { kBadHeader = 1, kBadLengths = 2, kBadMapping = 4, kBadRecord = 8 };

__host__ __device__ int64_t head_bytes(int L, int H) {
  return (kFixed + 8LL * L * H + kAlign - 1) / kAlign * kAlign;
}
__host__ __device__ int64_t slot_bytes(int L, int H) {
  return head_bytes(L, H) + 2LL * L * H * kRawRecord;
}
__device__ bool valid_length(int v) {
  return v == kRawVarLen || (v >= kHeaderBytes && v <= kVarCap && !(v & 15));
}

// Descriptor memory belongs to NativeKVPages and remains immutable while any
// call is pending. Native page tensors, mappings, tables and buffers are held
// by the Python owner through completion on the caller's current CUDA stream.
struct Pages {
  const int64_t* desc;  // [L,6]: pointer, kv/page/token/head strides, page count
  const int32_t* map;   // [N,128/P]
  int N, L, H, P, logP;
};

__device__ uint16_t* element(const Pages& p, int i, int l, int kv, int h, int t, int d) {
  const int64_t* a = p.desc + 6 * l;
  // P is one of 16/32/64/128, validated on the host. All native paths use
  // these exact power-of-two operations, including the raw transfer control.
  const int page = p.map[i * (128 >> p.logP) + (t >> p.logP)];
  return reinterpret_cast<uint16_t*>(a[0]) + kv * a[1] + page * a[2] +
         (t & (p.P - 1)) * a[3] + h * a[4] + d;
}

__device__ int mapping_error(const Pages& p, int i, bool scatter) {
  const int M = 128 >> p.logP, tid = threadIdx.x;
  int bad = 0;
  for (int x = tid; x < p.L * M; x += blockDim.x) {
    const int page = p.map[i * M + x % M];
    bad |= page < 0 || page >= p.desc[6 * (x / M) + 5];
  }
  // Sources may alias; destinations must be distinct across the entire batch,
  // including within one 128-token logical block. Each duplicate gets rejected.
  if (scatter) {
    for (int x = tid; x < M; x += blockDim.x) {
      const int pos = i * M + x, page = p.map[pos];
      for (int y = 0; y < p.N * M; ++y)
        bad |= y != pos && p.map[y] == page;
    }
  }
  return __syncthreads_or(bad) ? kBadMapping : 0;
}

__global__ void validate_mapping(Pages p, int32_t* status, bool scatter) {
  const int bad = mapping_error(p, blockIdx.x, scatter);
  if (threadIdx.x == 0) status[blockIdx.x] = bad;
}

__global__ void __launch_bounds__(kThreads) encode_kernel(
    Pages p, const uint8_t* rank_of, uint8_t* records, int32_t* lens,
    const int32_t* status) {
  __shared__ uint8_t rank_s[256];
  __shared__ __align__(16) uint8_t var_s[kVarCap];
  __shared__ int wsum[2][8];
  const int i = blockIdx.x, r = blockIdx.y, tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5, g = lane >> 2, c = lane & 3;
  const int l = r / (2 * p.H), kvh = r % (2 * p.H), kv = kvh / p.H, h = kvh % p.H;
  const int64_t index = (int64_t)i * p.L * 2 * p.H + r;
  if (status[i]) {
    if (tid == 0) lens[index] = 0;
    return;
  }
  rank_s[tid] = rank_of[(int64_t)r * 256 + tid];
  uint4 wd[8];
#pragma unroll
  for (int st = 0; st < 8; ++st) {
    uint32_t words[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int a = 16 * warp + 8 * (q >> 1) + g;
      const int z = 16 * st + 8 * (q & 1) + 2 * c;
      const int tok = kv == 0 ? a : z, dim = kv == 0 ? z : a;
      words[q] = (uint32_t)*element(p, i, l, kv, h, tok, dim) |
                 ((uint32_t)*element(p, i, l, kv, h, tok + kv, dim + 1 - kv) << 16);
    }
    wd[st] = make_uint4(words[0], words[1], words[2], words[3]);
  }
  for (int x = tid; x < kVarCap / 16; x += kThreads)
    reinterpret_cast<uint4*>(var_s)[x] = make_uint4(0, 0, 0, 0);
  __syncthreads();

  uint32_t sym[6] = {0, 0, 0, 0, 0, 0};
  uint64_t emask = 0, rmask = 0;
#pragma unroll
  for (int st = 0; st < 8; ++st) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int sc = 8 * st + e, gi = sc >> 5, sg = sc & 31;
      const uint32_t val = (u4get(wd[st], e >> 1) >> (16 * (e & 1))) & 0xffffu;
      const int rank = rank_s[val >> 8];
      const uint32_t code = rank < kPrimaryDirect ? rank : kPrimaryDirect;
      if (sg < 24) {
        sym[3 * gi + (sg >> 3)] |= code << (4 * (sg & 7));
      } else {
#pragma unroll
        for (int k = 0; k < 3; ++k)
          sym[3 * gi + k] |= ((code >> k) & 1u) << (4 * (sg - 24) + 3);
      }
      if (rank >= kPrimaryDirect) emask |= 1ull << sc;
      if (rank >= kTableEntries) rmask |= 1ull << sc;
    }
  }
  const int ne = __popcll(emask), nr = __popcll(rmask);
  const int eex = warp_excl_scan(ne, lane), rex = warp_excl_scan(nr, lane);
  if (lane == 31) {
    wsum[0][warp] = eex + ne;
    wsum[1][warp] = rex + nr;
  }
  __syncthreads();
  int eo = eex, ro = rex, etot = 0, rtot = 0, rmax = 0;
#pragma unroll
  for (int w = 0; w < 8; ++w) {
    if (w < warp) {
      eo += wsum[0][w];
      ro += wsum[1][w];
    }
    etot += wsum[0][w];
    rtot += wsum[1][w];
    rmax = max(rmax, wsum[1][w]);
  }
  const int raw_offset = (kHeaderBytes + ((etot + 1) >> 1) + 3) & ~3;
  const int var_len = (raw_offset + 4 * rtot + 15) & ~15;
  uint8_t* dst = records + index * kRawRecord;
  if (var_len > kVarCap || rmax > kMaxWarpRaw) {
#pragma unroll
    for (int st = 0; st < 8; ++st)
      reinterpret_cast<uint4*>(dst)[st * kThreads + tid] = wd[st];
    if (tid == 0) lens[index] = kRawVarLen;
    return;
  }
  uint32_t* var32 = reinterpret_cast<uint32_t*>(var_s);
#pragma unroll
  for (int st = 0; st < 8; ++st) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int sc = 8 * st + e;
      if ((emask >> sc) & 1ull) {
        const uint32_t high = ((u4get(wd[st], e >> 1) >> (16 * (e & 1))) >> 8) & 0xffu;
        const int rank = rank_s[high];
        const uint32_t nib = rank < kTableEntries ? rank - kPrimaryDirect : 15u;
        const int bit = 8 * kHeaderBytes + 4 * eo;
        atomicOr(&var32[bit >> 5], nib << (bit & 31));
        if (rank >= kTableEntries) var32[(raw_offset >> 2) + ro++] = (uint32_t)eo | (high << 16);
        ++eo;
      }
    }
  }
  if (tid == 0) {
    var32[0] = etot;
    var32[1] = rtot;
    var32[2] = raw_offset;
    var32[3] = 0;
    uint16_t* h16 = reinterpret_cast<uint16_t*>(var_s);
    int ep = 0, rp = 0;
    for (int w = 0; w < 8; ++w) {
      h16[8 + w] = ep;
      h16[16 + w] = rp;
      ep += wsum[0][w];
      rp += wsum[1][w];
    }
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint4 a = wd[2 * j], b = wd[2 * j + 1];
    reinterpret_cast<uint4*>(dst)[j * kThreads + tid] =
        make_uint4(__byte_perm(a.x, a.y, 0x6420), __byte_perm(a.z, a.w, 0x6420),
                   __byte_perm(b.x, b.y, 0x6420), __byte_perm(b.z, b.w, 0x6420));
  }
#pragma unroll
  for (int m = 0; m < 3; ++m)
    reinterpret_cast<uint2*>(dst + kLowBytes)[m * kThreads + tid] =
        make_uint2(sym[2 * m], sym[2 * m + 1]);
  __syncthreads();
  for (int x = tid; x < var_len / 16; x += kThreads)
    reinterpret_cast<uint4*>(dst + kDenseBytes)[x] = reinterpret_cast<const uint4*>(var_s)[x];
  if (tid == 0) lens[index] = var_len;
}

// Exclusive byte prefix for a 256-record tile. Every lane returns the same
// total; the final barrier protects shared sums when the next tile starts.
// Both export planning and import validation use 64-bit arithmetic throughout.
__device__ int64_t block_exclusive_bytes(int64_t value, int64_t* warp_sums,
                                         int64_t& total) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int64_t inclusive = value;
#pragma unroll
  for (int delta = 1; delta < 32; delta <<= 1) {
    const int64_t preceding = __shfl_up_sync(0xffffffffu, inclusive, delta);
    if (lane >= delta) inclusive += preceding;
  }
  if (lane == 31) warp_sums[warp] = inclusive;
  __syncthreads();
  int64_t before = 0;
  total = 0;
#pragma unroll
  for (int w = 0; w < kThreads / 32; ++w) {
    if (w < warp) before += warp_sums[w];
    total += warp_sums[w];
  }
  __syncthreads();
  return before + inclusive - value;
}

// No host length observation occurs between encoding, planning, and packing.
__global__ void plan_kernel(const int32_t* lens, int32_t* status, int64_t* lengths,
                            int64_t* rec_offsets, int L, int H) {
  __shared__ int64_t warp_sums[kThreads / 32];
  const int i = blockIdx.x, tid = threadIdx.x, R = 2 * L * H;
  int64_t cursor = head_bytes(L, H);
  bool bad_length = false;
  for (int base = 0; base < R; base += kThreads) {
    const int r = base + tid;
    const int v = r < R ? lens[(int64_t)i * R + r] : 0;
    const bool valid = r >= R || valid_length(v);
    bad_length |= !valid;
    const int64_t bytes = r < R && valid ? kDenseBytes + v : 0;
    int64_t total;
    const int64_t before = block_exclusive_bytes(bytes, warp_sums, total);
    if (r < R) rec_offsets[(int64_t)i * R + r] = cursor + before;
    cursor += total;
  }
  const int length_status = __syncthreads_or(bad_length) ? kBadLengths : 0;
  if (tid == 0) {
    const int bad = status[i] | length_status;
    status[i] = bad;
    lengths[i] = bad ? 0 : ((cursor + kAlign - 1) / kAlign * kAlign);
  }
}

__global__ void batch_prefix(const int64_t* lengths, int64_t* offsets, int N) {
  if (threadIdx.x) return;
  int64_t off = 0;
  for (int i = 0; i < N; ++i) {
    offsets[i] = off;
    off += lengths[i];
  }
  offsets[N] = off;
}

__global__ void pack_kernel(const uint8_t* records, const int32_t* lens,
                            const int32_t* status, const int64_t* offsets,
                            const int64_t* lengths, const int64_t* rec_offsets,
                            uint8_t* out, int L, int H, int64_t digest) {
  const int i = blockIdx.x, r = blockIdx.y, tid = threadIdx.x, R = 2 * L * H;
  if (status[i]) return;
  const int64_t index = (int64_t)i * R + r;
  uint8_t* blob = out + offsets[i];
  const int bytes = kDenseBytes + lens[index];
  const uint4* src = reinterpret_cast<const uint4*>(records + index * kRawRecord);
  uint4* dst = reinterpret_cast<uint4*>(blob + rec_offsets[index]);
  for (int x = tid; x < bytes / 16; x += kThreads) dst[x] = src[x];
  if (r == 0) {
    const int64_t head = head_bytes(L, H);
    const int64_t last = (int64_t)i * R + R - 1;
    const int64_t declared = rec_offsets[last] + kDenseBytes + lens[last];
    // Whole header zeroing before assigning fields is local to this CTA;
    // payload writers begin after head and never race with this region.
    for (int x = tid; x < head / 16; x += kThreads)
      reinterpret_cast<uint4*>(blob)[x] = make_uint4(0, 0, 0, 0);
    for (int64_t x = declared + 16LL * tid; x < lengths[i]; x += 16LL * kThreads)
      *reinterpret_cast<uint4*>(blob + x) = make_uint4(0, 0, 0, 0);
    __syncthreads();
    if (tid == 0) {
      uint32_t* h32 = reinterpret_cast<uint32_t*>(blob);
      h32[0] = kMagic; h32[1] = kVersion; h32[2] = L; h32[3] = H;
      int64_t* h64 = reinterpret_cast<int64_t*>(blob + 16);
      h64[0] = declared; h64[1] = digest;
    }
    for (int x = tid; x < R; x += kThreads)
      reinterpret_cast<int32_t*>(blob + kFixed)[x] = lens[(int64_t)i * R + x];
  }
}

// Every blob's complete header and mapping are validated before any record
// reads or destination writes. Subtraction-based bounds avoid offset overflow.
__global__ void validate_headers(Pages p, const uint8_t* src, int64_t src_bytes,
                                 const int64_t* offsets, int64_t* rec_offsets,
                                 int32_t* status, int64_t digest) {
  __shared__ int header_status;
  __shared__ int64_t declared;
  __shared__ int64_t warp_sums[kThreads / 32];
  const int i = blockIdx.x, tid = threadIdx.x, R = 2 * p.L * p.H;
  const int map_bad = mapping_error(p, i, true);
  const int64_t off = offsets[i], head = head_bytes(p.L, p.H);
  if (tid == 0) {
    header_status = map_bad;
    if (off < 0 || (off & 15) || src_bytes < head || off > src_bytes - head) {
      header_status |= kBadHeader;
    } else {
      const uint8_t* blob = src + off;
      const uint32_t* h32 = reinterpret_cast<const uint32_t*>(blob);
      const int64_t* h64 = reinterpret_cast<const int64_t*>(blob + 16);
      declared = h64[0];
      if (h32[0] != kMagic || h32[1] != kVersion || h32[2] != p.L || h32[3] != p.H ||
          h64[1] != digest || declared < head || declared > src_bytes - off)
        header_status |= kBadHeader;
    }
  }
  __syncthreads();
  if (header_status & kBadHeader) {
    if (tid == 0) status[i] = header_status;
    return;
  }
  const int32_t* lens = reinterpret_cast<const int32_t*>(src + off + kFixed);
  int64_t cursor = head;
  bool bad_length = false;
  for (int base = 0; base < R; base += kThreads) {
    const int r = base + tid;
    const int v = r < R ? lens[r] : 0;
    const bool valid = r >= R || valid_length(v);
    bad_length |= !valid;
    const int64_t bytes = r < R && valid ? kDenseBytes + v : 0;
    int64_t total;
    const int64_t before = block_exclusive_bytes(bytes, warp_sums, total);
    if (r < R) rec_offsets[(int64_t)i * R + r] = cursor + before;
    cursor += total;
  }
  const int length_status =
      __syncthreads_or(bad_length || cursor != declared) ? kBadLengths : 0;
  if (tid == 0) status[i] = header_status | length_status;
}

// decode_rec assumes trusted variable-region metadata. Check symbol counts,
// warp prefixes, nibble/raw correspondence and raw ordinal bounds first.
// A separate validation launch prevents any layer from being written when
// another record in its blob is malformed. This is validation, not a checksum:
// changing valid low bytes to other valid bytes cannot be detected by FPKB v1.
__global__ void validate_records(const uint8_t* src, const int64_t* offsets,
                                 const int64_t* rec_offsets, int32_t* status, int R) {
  __shared__ int esc_counts[8];
  __shared__ int skip;
  const int i = blockIdx.x, r = blockIdx.y, tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  // Other record CTAs may reject this blob concurrently. One atomic snapshot
  // keeps the early return CTA-uniform even when that happens during launch.
  if (tid == 0) skip = atomicAdd(status + i, 0);
  __syncthreads();
  if (skip) return;
  const uint8_t* blob = src + offsets[i];
  const int vlen = reinterpret_cast<const int32_t*>(blob + kFixed)[r];
  if (vlen == kRawVarLen) return;
  const uint8_t* rec = blob + rec_offsets[(int64_t)i * R + r];
  const uint8_t* var = rec + kDenseBytes;
  const uint32_t* h32 = reinterpret_cast<const uint32_t*>(var);
  const uint16_t* h16 = reinterpret_cast<const uint16_t*>(var);
  const uint32_t ne = h32[0], nr = h32[1], raw_off = h32[2];
  const uint64_t expected_raw = (kHeaderBytes + ((uint64_t(ne) + 1) >> 1) + 3) & ~3ull;
  const uint64_t expected_len = (expected_raw + 4ull * nr + 15) & ~15ull;
  bool bad = ne > 2 * (kVarCap - kHeaderBytes) || nr > 8 * kMaxWarpRaw ||
             raw_off != expected_raw || expected_len != vlen || h32[3] != 0;
  if (__syncthreads_or(bad)) {
    if (tid == 0) atomicOr(status + i, kBadRecord);
    return;
  }
  uint2 sy[3];
#pragma unroll
  for (int j = 0; j < 3; ++j)
    sy[j] = reinterpret_cast<const uint2*>(rec + kLowBytes)[j * kThreads + tid];
  int count = 0;
#pragma unroll
  for (int gi = 0; gi < 2; ++gi) {
    uint32_t words[4];
    group_words(sy, gi, words);
#pragma unroll
    for (int k = 0; k < 4; ++k)
      count += __popc(((words[k] & 0x77777777u) + 0x11111111u) & 0x88888888u);
  }
  const int exclusive = warp_excl_scan(count, lane);
  if (lane == 31) esc_counts[warp] = exclusive + count;
  __syncthreads();
  int eb = 0, total = 0;
  for (int w = 0; w < 8; ++w) {
    if (w < warp) eb += esc_counts[w];
    total += esc_counts[w];
  }
  const int ee = eb + esc_counts[warp];
  const int rb = h16[16 + warp], re = warp < 7 ? h16[17 + warp] : nr;
  bad = total != ne || h16[8 + warp] != eb || rb > re || re > nr ||
        re - rb > kMaxWarpRaw || (warp == 0 && rb != 0);
  if (__syncthreads_or(bad)) {
    if (tid == 0) atomicOr(status + i, kBadRecord);
    return;
  }
  int raw_count = 0;
  for (int x = eb + lane; x < ee; x += 32)
    raw_count += ((var[kHeaderBytes + x / 2] >> (4 * (x & 1))) & 15) == 15;
  for (int delta = 16; delta; delta >>= 1)
    raw_count += __shfl_xor_sync(0xffffffffu, raw_count, delta);
  bad = raw_count != re - rb;
  const uint32_t* raw = reinterpret_cast<const uint32_t*>(var + raw_off);
  if (rb + lane < re) {
    const uint32_t word = raw[rb + lane], x = word & 0xffffu;
    bad |= x < eb || x >= ee || (word >> 24) != 0;
    if (rb + lane > rb) bad |= (raw[rb + lane - 1] & 0xffffu) >= x;
    if (x >= eb && x < ee)
      bad |= ((var[kHeaderBytes + x / 2] >> (4 * (x & 1))) & 15) != 15;
  }
  if (__syncthreads_or(bad) && tid == 0) atomicOr(status + i, kBadRecord);
}

__global__ void __launch_bounds__(kThreads) decode_kernel(
    Pages p, const int32_t* plut, const uint8_t* slut, const uint8_t* src,
    const int64_t* offsets, const int64_t* rec_offsets, const int32_t* status) {
  // The bounded, zero-padded encoded copy and decoded tile have disjoint
  // lifetimes. decode_rec finishes all record reads before its first emit;
  // the CTA barrier there lets every warp reuse this region as output.
  // Escape expansion stays separate because subsequent emits still read it.
  extern __shared__ __align__(16) uint8_t smem[];
  uint8_t* encoded = smem;
  uint8_t* tile = smem;
  uint8_t* esc = smem + kTile;
  const int i = blockIdx.x, r = blockIdx.y, tid = threadIdx.x;
  if (status[i]) return;
  const int l = r / (2 * p.H), kvh = r % (2 * p.H), kv = kvh / p.H, h = kvh % p.H;
  const int warp = tid >> 5, g = (tid & 31) >> 2, c = tid & 3;
  const uint8_t* blob = src + offsets[i];
  const int vlen = reinterpret_cast<const int32_t*>(blob + kFixed)[r];
  const uint8_t* rec = blob + rec_offsets[(int64_t)i * p.L * 2 * p.H + r];
  if (vlen != kRawVarLen) {
    for (int x = tid; x < (kDenseBytes + vlen) / 16; x += kThreads)
      reinterpret_cast<uint4*>(encoded)[x] = reinterpret_cast<const uint4*>(rec)[x];
    if (tid == 0)
      *reinterpret_cast<uint4*>(encoded + kDenseBytes + vlen) = make_uint4(0, 0, 0, 0);
    __syncthreads();
    rec = encoded;
  }
  const uint32_t l0 = plut[2 * r], l1 = plut[2 * r + 1];
  auto put = [&](int tok, int dim, uint32_t value, bool pair) {
    uint8_t* dst = tile + tok * 256 + (((dim >> 3) ^ (tok & 7)) << 4) + 2 * (dim & 7);
    if (pair)
      *reinterpret_cast<uint32_t*>(dst) = value;
    else
      *reinterpret_cast<uint16_t*>(dst) = (uint16_t)value;
  };
  decode_rec(rec, vlen, l0, l1, slut + 16 * r, esc, tid,
             [&](int st, const uint32_t (&bf)[4]) {
    // Raw records read their input directly; compressed records have already
    // loaded their dense words into registers and expanded their escapes.
    // No thread may overwrite encoded until all warps reach this point.
    if (st == 0) __syncthreads();
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int a = 16 * warp + 8 * (q >> 1) + g;
      const int z = 16 * st + 8 * (q & 1) + 2 * c;
      if (kv == 0) {
        put(a, z, bf[q], true);
      } else {
        put(z, a, bf[q], false);
        put(z + 1, a, bf[q] >> 16, false);
      }
    }
  });
  __syncthreads();
  const int64_t* a = p.desc + 6 * l;
  uint16_t* base = reinterpret_cast<uint16_t*>(a[0]) + kv * a[1] + h * a[4];
  const bool vectorized = (reinterpret_cast<uintptr_t>(base) & 15u) == 0 &&
                          (a[2] & 7) == 0 && (a[3] & 7) == 0;
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const int x = q * kThreads + tid, tok = x >> 4, chunk = x & 15;
    const uint4 value = *reinterpret_cast<const uint4*>(
        tile + tok * 256 + ((chunk ^ (tok & 7)) << 4));
    const int page = p.map[i * (128 >> p.logP) + (tok >> p.logP)];
    uint16_t* dst = base + page * a[2] + (tok & (p.P - 1)) * a[3] + 8 * chunk;
    if (vectorized) {
      *reinterpret_cast<uint4*>(dst) = value;
    } else {
      // Registration permits odd BF16 base offsets and positive strides.
      // Halfword stores preserve that contract and every padding canary.
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        const uint32_t word = u4get(value, w);
        dst[2 * w] = (uint16_t)word;
        dst[2 * w + 1] = (uint16_t)(word >> 16);
      }
    }
  }
}

void configure_decode(int device) {
  // CUDA function attributes are device-specific. Cache setup per calling
  // host thread/device; simultaneous identical setup calls are harmless.
  // No synchronization or device readback is added to the restore path.
  static thread_local std::vector<bool> configured;
  if ((int)configured.size() <= device) configured.resize(device + 1, false);
  if (!configured[device]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(decode_kernel,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, kDecodeSmem));
    configured[device] = true;
  }
}

// Fair raw control: coalesced token-major dimension pairs, identical mappings
// and status checks, without codec-specific layout conversion or FPKB padding.
template <bool Scatter>
__global__ void raw_kernel(Pages p, uint8_t* buffer, const int32_t* status) {
  const int i = blockIdx.x, r = blockIdx.y;
  if (status[i]) return;
  const int l = r / (2 * p.H), kvh = r % (2 * p.H), kv = kvh / p.H, h = kvh % p.H;
  uint32_t* raw = reinterpret_cast<uint32_t*>(buffer) +
                  ((int64_t)i * p.L * 2 * p.H + r) * (kRawRecord / 4);
  for (int x = threadIdx.x; x < 128 * 64; x += blockDim.x) {
    uint16_t* native = element(p, i, l, kv, h, x / 64, 2 * (x % 64));
    if constexpr (Scatter) {
      const uint32_t value = raw[x];
      native[0] = (uint16_t)value;
      native[1] = (uint16_t)(value >> 16);
    } else {
      raw[x] = (uint32_t)native[0] | ((uint32_t)native[1] << 16);
    }
  }
}

void check_tensor(const torch::Tensor& t, const torch::Device& dev,
                  torch::ScalarType dtype, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.device() == dev && t.scalar_type() == dtype && t.is_contiguous(),
              name, " must be contiguous CUDA tensor of required dtype on the descriptor device");
}

Pages check_pages(const torch::Tensor& desc, const torch::Tensor& map,
                   const torch::Tensor& status, int64_t P, int64_t H) {
  TORCH_CHECK(desc.is_cuda() && desc.scalar_type() == torch::kInt64 && desc.is_contiguous() &&
                  desc.dim() == 2 && desc.size(1) == 6 && desc.size(0) > 0,
              "desc must be CUDA int64 [layers,6]");
  TORCH_CHECK(P == 16 || P == 32 || P == 64 || P == 128, "page_tokens must be 16,32,64,128");
  TORCH_CHECK(H > 0 && H <= 128 && desc.size(0) * 2 * H <= 65535,
              "invalid head count or record grid size");
  check_tensor(map, desc.device(), torch::kInt32, "page_map");
  TORCH_CHECK(map.dim() == 2 && map.size(1) == 128 / P && map.size(0) <= INT_MAX,
              "page_map must have shape [N,128/page_tokens]");
  check_tensor(status, desc.device(), torch::kInt32, "status");
  TORCH_CHECK(status.dim() == 1 && status.numel() == map.size(0), "status must have shape [N]");
  const int logP = P == 16 ? 4 : P == 32 ? 5 : P == 64 ? 6 : 7;
  return {desc.data_ptr<int64_t>(), map.data_ptr<int32_t>(), (int)map.size(0),
          (int)desc.size(0), (int)H, (int)P, logP};
}

int check_lengths(const torch::Tensor& lens, const torch::Tensor& status,
                   int64_t L, int64_t H) {
  TORCH_CHECK(lens.is_cuda() && L > 0 && H > 0 && H <= 128 && L * 2 * H <= 65535,
              "invalid record geometry");
  check_tensor(lens, lens.device(), torch::kInt32, "rec_lens");
  TORCH_CHECK(lens.dim() == 4 && lens.size(1) == L && lens.size(2) == 2 &&
                  lens.size(3) == H && lens.size(0) <= INT_MAX,
              "rec_lens must be [N,L,2,H]");
  check_tensor(status, lens.device(), torch::kInt32, "status");
  TORCH_CHECK(status.dim() == 1 && status.numel() == lens.size(0), "status must be [N]");
  return lens.size(0);
}

void check_records(const torch::Tensor& records, const torch::Tensor& lens) {
  check_tensor(records, lens.device(), torch::kUInt8, "records");
  TORCH_CHECK(records.numel() >= lens.numel() * kRawRecord &&
                  reinterpret_cast<uintptr_t>(records.data_ptr()) % 16 == 0,
              "records needs aligned worst-case slots for every record");
}

void check_plan(const torch::Tensor& offsets, const torch::Tensor& lengths,
                 const torch::Tensor& rec_offsets, const torch::Tensor& lens, int N) {
  check_tensor(offsets, lens.device(), torch::kInt64, "offsets");
  check_tensor(lengths, lens.device(), torch::kInt64, "lengths");
  check_tensor(rec_offsets, lens.device(), torch::kInt64, "rec_offsets");
  TORCH_CHECK(offsets.dim() == 1 && offsets.numel() == N + 1 && lengths.dim() == 1 &&
                  lengths.numel() == N && rec_offsets.numel() == lens.numel(),
              "plan buffers must be offsets[N+1], lengths[N], rec_offsets[N,L,2,H]");
}

void check_bytes(const torch::Tensor& t, const torch::Device& dev, int64_t capacity,
                   const char* name) {
  check_tensor(t, dev, torch::kUInt8, name);
  TORCH_CHECK(t.numel() >= capacity && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
              name, " needs aligned byte storage with sufficient capacity");
}
}  // namespace

void encode_records(torch::Tensor desc, torch::Tensor page_map, torch::Tensor rank_of,
                    torch::Tensor records, torch::Tensor rec_lens, torch::Tensor status,
                    int64_t page_tokens, int64_t heads) {
  const Pages p = check_pages(desc, page_map, status, page_tokens, heads);
  TORCH_CHECK(check_lengths(rec_lens, status, p.L, p.H) == p.N, "record batch mismatch");
  check_records(records, rec_lens);
  check_tensor(rank_of, desc.device(), torch::kUInt8, "rank_of");
  TORCH_CHECK(rank_of.numel() == 2LL * p.L * p.H * 256, "rank table shape mismatch");
  const c10::cuda::CUDAGuard guard(desc.device());
  if (!p.N) return;
  const auto stream = at::cuda::getCurrentCUDAStream();
  validate_mapping<<<p.N, kThreads, 0, stream>>>(p, status.data_ptr<int32_t>(), false);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  encode_kernel<<<dim3(p.N, 2 * p.L * p.H), kThreads, 0, stream>>>(
      p, rank_of.data_ptr<uint8_t>(), records.data_ptr<uint8_t>(),
      rec_lens.data_ptr<int32_t>(), status.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void plan_blobs(torch::Tensor rec_lens, torch::Tensor status, torch::Tensor offsets,
                torch::Tensor lengths, torch::Tensor rec_offsets, int64_t layers, int64_t heads) {
  const int N = check_lengths(rec_lens, status, layers, heads);
  check_plan(offsets, lengths, rec_offsets, rec_lens, N);
  const c10::cuda::CUDAGuard guard(rec_lens.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (N) {
    plan_kernel<<<N, kThreads, 0, stream>>>(rec_lens.data_ptr<int32_t>(), status.data_ptr<int32_t>(),
        lengths.data_ptr<int64_t>(), rec_offsets.data_ptr<int64_t>(), layers, heads);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  batch_prefix<<<1, 32, 0, stream>>>(lengths.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(), N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void pack_blobs(torch::Tensor records, torch::Tensor rec_lens, torch::Tensor status,
                torch::Tensor offsets, torch::Tensor lengths, torch::Tensor rec_offsets,
                torch::Tensor out, int64_t layers, int64_t heads, int64_t digest) {
  const int N = check_lengths(rec_lens, status, layers, heads);
  check_records(records, rec_lens);
  check_plan(offsets, lengths, rec_offsets, rec_lens, N);
  check_bytes(out, rec_lens.device(), N * slot_bytes(layers, heads), "out");
  const c10::cuda::CUDAGuard guard(rec_lens.device());
  if (!N) return;
  pack_kernel<<<dim3(N, 2 * layers * heads), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      records.data_ptr<uint8_t>(), rec_lens.data_ptr<int32_t>(), status.data_ptr<int32_t>(),
      offsets.data_ptr<int64_t>(), lengths.data_ptr<int64_t>(), rec_offsets.data_ptr<int64_t>(),
      out.data_ptr<uint8_t>(), layers, heads, digest);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void decode_blobs(torch::Tensor desc, torch::Tensor page_map, torch::Tensor plut,
                  torch::Tensor slut, torch::Tensor src, torch::Tensor offsets,
                  torch::Tensor rec_offsets, torch::Tensor status, int64_t page_tokens,
                  int64_t heads, int64_t digest) {
  const Pages p = check_pages(desc, page_map, status, page_tokens, heads);
  check_tensor(plut, desc.device(), torch::kInt32, "plut");
  check_tensor(slut, desc.device(), torch::kUInt8, "slut");
  TORCH_CHECK(plut.numel() == 4LL * p.L * p.H && slut.numel() == 32LL * p.L * p.H &&
                  reinterpret_cast<uintptr_t>(slut.data_ptr()) % 16 == 0,
              "decoder table geometry or alignment mismatch");
  check_bytes(src, desc.device(), 0, "src");
  check_tensor(offsets, desc.device(), torch::kInt64, "offsets");
  check_tensor(rec_offsets, desc.device(), torch::kInt64, "rec_offsets");
  TORCH_CHECK(offsets.dim() == 1 && (offsets.numel() == p.N || offsets.numel() == p.N + 1) &&
                  rec_offsets.numel() == 2LL * p.N * p.L * p.H,
              "decode requires offsets[N or N+1], rec_offsets[N,L,2,H]");
  const c10::cuda::CUDAGuard guard(desc.device());
  if (!p.N) return;
  configure_decode(desc.get_device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  validate_headers<<<p.N, kThreads, 0, stream>>>(p, src.data_ptr<uint8_t>(), src.numel(),
      offsets.data_ptr<int64_t>(), rec_offsets.data_ptr<int64_t>(), status.data_ptr<int32_t>(), digest);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  validate_records<<<dim3(p.N, 2 * p.L * p.H), kThreads, 0, stream>>>(src.data_ptr<uint8_t>(),
      offsets.data_ptr<int64_t>(), rec_offsets.data_ptr<int64_t>(), status.data_ptr<int32_t>(), 2 * p.L * p.H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  decode_kernel<<<dim3(p.N, 2 * p.L * p.H), kThreads, kDecodeSmem, stream>>>(p,
      plut.data_ptr<int32_t>(), slut.data_ptr<uint8_t>(), src.data_ptr<uint8_t>(),
      offsets.data_ptr<int64_t>(), rec_offsets.data_ptr<int64_t>(), status.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void raw_gather(torch::Tensor desc, torch::Tensor page_map, torch::Tensor out,
                torch::Tensor status, int64_t page_tokens, int64_t heads) {
  const Pages p = check_pages(desc, page_map, status, page_tokens, heads);
  check_bytes(out, desc.device(), 2LL * p.N * p.L * p.H * kRawRecord, "out");
  const c10::cuda::CUDAGuard guard(desc.device());
  if (!p.N) return;
  const auto stream = at::cuda::getCurrentCUDAStream();
  validate_mapping<<<p.N, kThreads, 0, stream>>>(p, status.data_ptr<int32_t>(), false);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  raw_kernel<false><<<dim3(p.N, 2 * p.L * p.H), kThreads, 0, stream>>>(
      p, out.data_ptr<uint8_t>(), status.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void raw_scatter(torch::Tensor desc, torch::Tensor page_map, torch::Tensor src,
                 torch::Tensor status, int64_t page_tokens, int64_t heads) {
  const Pages p = check_pages(desc, page_map, status, page_tokens, heads);
  check_bytes(src, desc.device(), 2LL * p.N * p.L * p.H * kRawRecord, "src");
  const c10::cuda::CUDAGuard guard(desc.device());
  if (!p.N) return;
  const auto stream = at::cuda::getCurrentCUDAStream();
  validate_mapping<<<p.N, kThreads, 0, stream>>>(p, status.data_ptr<int32_t>(), true);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  raw_kernel<true><<<dim3(p.N, 2 * p.L * p.H), kThreads, 0, stream>>>(
      p, src.data_ptr<uint8_t>(), status.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
