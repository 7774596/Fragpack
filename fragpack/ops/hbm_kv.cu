// SPDX-License-Identifier: Apache-2.0
// HBM-resident FragPack KV store kernels (see hbm_kv.py for the state machine).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>
#include <optional>

#include "fragpack_common.cuh"

namespace {

using namespace fragpack;

// Counter slots (int32[kCounters]).
enum : int {
  kFreeTop = 0,      // staging free-stack depth
  kTrans = 1,        // transient slots handed out this step
  kClaims = 2,       // blocks claimed this step
  kCloses = 3,       // blocks closed this step
  kErrStaging = 4,   // dropped claims: staging exhausted
  kErrOverflow = 5,  // dropped encodes: overflow pool exhausted
  kFallbacks = 6,    // records stored raw in the overflow pool (cumulative)
  kEncoded = 7,      // records encoded (cumulative)
  kSpills = 8,       // coded records spilled raw: block-layer page full (cumulative)
  kHeirs = 9,        // blocks waiting for a closing block's slot this step
  kDonors = 10,      // closing blocks whose slot passes to an heir this step
};
// Per-block step stamps, plus the current batch row of a direct new block.
// kDirectRow is -1 for staged blocks; unlike the other rows it is not a stamp.
enum : int {
  kTagLive = 0, kTagClose = 1, kTagClaim = 2, kTagCloseDedupe = 3,
  kDirectRow = 4, kTags = 5
};

constexpr int kPrepThreads = 1024;

struct Store {
  int32_t* counters;
  int32_t* wslot;       // [NB] staging slot of an open block, -1 if none
  int32_t* tags;        // [kTags, NB]
  int32_t* free_stack;  // [S_p]
  uint32_t* ovf_bits;   // [ceil(O / 32)] overflow slots in use (tail bits preset)
  int32_t* claim_list;  // [2, cap]: claimed blocks, then heirs
  int32_t* close_list;  // [2, cap]: closing blocks, then the heir of each
  int64_t* rec_addr;    // [L, NB, 2, H]
  int32_t* rec_len;     // [L, NB, 2, H]
  int64_t staging;      // [S_p, L, 2, H, kRawRecord] persistent staging
  int64_t scratch;      // [S_t, 2, H, kRawRecord] one layer's transient staging
  int64_t ovf_base;     // [O, kRawRecord] raw records that do not code
  int64_t ovf_end;
  int32_t* fill;        // [L, NB] arena bytes used in each block-layer page
  int L, H, NB, S_p, S_t, ovf_words, cap;
};

Store make_store(torch::Tensor counters, torch::Tensor wslot, torch::Tensor tags,
                 torch::Tensor free_stack, torch::Tensor ovf_bits, torch::Tensor claim_list,
                 torch::Tensor close_list, torch::Tensor rec_addr, torch::Tensor rec_len,
                 torch::Tensor staging, torch::Tensor scratch, torch::Tensor overflow,
                 torch::Tensor fill) {
  TORCH_CHECK(tags.dim() == 2 && tags.size(0) == kTags &&
                  tags.size(1) == rec_addr.size(1) && tags.is_contiguous() &&
                  tags.scalar_type() == torch::kInt32 && tags.is_cuda() &&
                  tags.device() == rec_addr.device(),
              "store tags must include the direct source-row map");
  Store s;
  s.counters = counters.data_ptr<int32_t>();
  s.wslot = wslot.data_ptr<int32_t>();
  s.tags = tags.data_ptr<int32_t>();
  s.free_stack = free_stack.data_ptr<int32_t>();
  s.ovf_bits = reinterpret_cast<uint32_t*>(ovf_bits.data_ptr<int32_t>());
  s.ovf_words = ovf_bits.numel();
  s.claim_list = claim_list.data_ptr<int32_t>();
  s.close_list = close_list.data_ptr<int32_t>();
  s.rec_addr = rec_addr.data_ptr<int64_t>();
  s.rec_len = rec_len.data_ptr<int32_t>();
  s.staging = reinterpret_cast<int64_t>(staging.data_ptr());
  s.scratch = reinterpret_cast<int64_t>(scratch.data_ptr());
  s.ovf_base = reinterpret_cast<int64_t>(overflow.data_ptr());
  s.ovf_end = s.ovf_base + overflow.numel();
  s.fill = fill.data_ptr<int32_t>();
  s.L = rec_addr.size(0);
  s.NB = rec_addr.size(1);
  s.H = rec_addr.size(3);
  s.S_p = free_stack.numel();
  s.S_t = scratch.numel() / (2LL * s.H * kRawRecord);
  s.cap = claim_list.numel() / 2;
  return s;
}

// Borrowed for one layer only. The caller retains these original BF16 tensors
// through gather and encode, ordered on the same stream as their production.
struct RawInputs {
  const uint16_t* k = nullptr;
  const uint16_t* v = nullptr;
  int64_t ks0 = 0, ks1 = 0, vs0 = 0, vs1 = 0, rows = 0;
};

RawInputs raw_inputs(const torch::Tensor& key, const torch::Tensor& value,
                     int heads, const torch::Device& device) {
  TORCH_CHECK(key.is_cuda() && value.is_cuda() && key.device() == device &&
                  value.device() == device,
              "new K/V must be CUDA tensors on the store device");
  TORCH_CHECK(key.scalar_type() == torch::kBFloat16 && value.scalar_type() == torch::kBFloat16,
              "new K/V must be BF16");
  TORCH_CHECK(key.dim() == 3 && value.dim() == 3 && key.sizes() == value.sizes() &&
                  key.size(1) == heads && key.size(2) == 128,
              "new K/V must have matching [tokens, KV heads, 128] shapes");
  TORCH_CHECK(key.stride(2) == 1 && value.stride(2) == 1 &&
                  key.stride(0) > 0 && key.stride(1) > 0 &&
                  value.stride(0) > 0 && value.stride(1) > 0,
              "new K/V need contiguous head dimensions and positive row/head strides");
  return {reinterpret_cast<const uint16_t*>(key.data_ptr()),
          reinterpret_cast<const uint16_t*>(value.data_ptr()),
          key.stride(0), key.stride(1), value.stride(0), value.stride(1), key.size(0)};
}

// Staging record (kv * H + h) of the block in staging slot ``w`` at ``layer``.
__device__ __forceinline__ int64_t staging_addr(const Store& s, int w, int layer, int kvh) {
  return w < s.S_p ? s.staging + (((int64_t)w * s.L + layer) * 2 * s.H + kvh) * kRawRecord
                   : s.scratch + ((int64_t)(w - s.S_p) * 2 * s.H + kvh) * kRawRecord;
}

// Overflow slots are a bitmap so that the model stream (encode, claims) and a
// load stream (imports) may allocate and free concurrently. Returns the slot,
// or -1 when the pool is full. ``hint`` spreads concurrent callers.
__device__ int ovf_alloc(const Store& s, uint32_t hint) {
  for (int i = 0; i < s.ovf_words; ++i) {
    const int w = (int)((hint + i) % (uint32_t)s.ovf_words);
    uint32_t cur = *reinterpret_cast<volatile uint32_t*>(&s.ovf_bits[w]);
    while (cur != 0xffffffffu) {
      const int bit = __ffs(~cur) - 1;
      const uint32_t prev = atomicOr(&s.ovf_bits[w], 1u << bit);
      if (!(prev & (1u << bit))) return 32 * w + bit;
      cur = prev | (1u << bit);
    }
  }
  return -1;
}

// Frees the overflow slot holding ``addr``; other addresses are ignored.
__device__ __forceinline__ void ovf_free(const Store& s, int64_t addr) {
  if (addr < s.ovf_base || addr >= s.ovf_end) return;
  const int slot = (int)((addr - s.ovf_base) / kRawRecord);
  atomicAnd(&s.ovf_bits[slot >> 5], ~(1u << (slot & 31)));
}

// One CTA. Phases (each ends at a barrier):
// 1. GC: blocks not in ``live`` drop their persistent staging slot.
// 2. Mark blocks receiving their last token (offset 127) this step.
// 3. Claim: a block receiving offset 0 gets a staging slot (kept if it
//    still owns one; transient if it also closes this step; else popped).
//    With the free stack empty, a block that stays open becomes an *heir*:
//    it fills a transient slot and takes over the persistent slot of a block
//    closing this step (a request finishing its partial block and opening a
//    new tail in one chunk holds both slots until commit).
// 4. Cooperatively validate direct candidates, then build the close list:
//    blocks receiving offset 127 that own a slot; each closing persistent
//    slot is paired with the next waiting heir.
// 5. Point every claimed block's records (all layers) at its staging, release
//    overflow records they held, and empty its arena pages for the encodes.
// Direct blocks still reserve that slot, but write leaves it untouched: gather
// must borrow their original K/V until encode publishes valid stored records.
__global__ void __launch_bounds__(kPrepThreads) prepare_kernel(Store s, const int64_t* slots,
                                                               int n_tok, const int32_t* live,
                                                               int n_live, int32_t step,
                                                               bool direct_new_blocks) {
  const int tid = threadIdx.x;
  int32_t* live_tag = s.tags + kTagLive * s.NB;
  int32_t* close_tag = s.tags + kTagClose * s.NB;
  int32_t* claim_tag = s.tags + kTagClaim * s.NB;
  int32_t* dedupe_tag = s.tags + kTagCloseDedupe * s.NB;
  if (tid == 0) {
    s.counters[kTrans] = 0;
    s.counters[kClaims] = 0;
    s.counters[kCloses] = 0;
    s.counters[kHeirs] = 0;
    s.counters[kDonors] = 0;
  }
  for (int i = tid; i < n_live; i += kPrepThreads) live_tag[live[i]] = step;
  __syncthreads();
  for (int b = tid; b < s.NB; b += kPrepThreads) {
    s.tags[kDirectRow * s.NB + b] = -1;
    const int w = s.wslot[b];
    if (w >= s.S_p) {
      s.wslot[b] = -1;
    } else if (w >= 0 && live_tag[b] != step) {
      s.wslot[b] = -1;
      s.free_stack[atomicAdd(&s.counters[kFreeTop], 1)] = w;
    }
  }
  for (int i = tid; i < n_tok; i += kPrepThreads) {
    const int64_t sl = slots[i];
    if (sl >= 0 && (sl & 127) == 127) close_tag[sl >> 7] = step;
  }
  __syncthreads();
  for (int i = tid; i < n_tok; i += kPrepThreads) {
    const int64_t sl = slots[i];
    if (sl < 0 || (sl & 127) != 0) continue;
    const int b = sl >> 7;
    if (atomicExch(&claim_tag[b], step) == step) continue;
    int w = s.wslot[b];
    const bool fresh = w < 0;
    if (w < 0) {
      const bool closes = close_tag[b] == step;
      if (closes) {
        const int t = atomicAdd(&s.counters[kTrans], 1);
        if (t < s.S_t) w = s.S_p + t;
      }
      if (w < 0) {
        const int top = atomicSub(&s.counters[kFreeTop], 1) - 1;
        if (top >= 0) {
          w = s.free_stack[top];
        } else {
          atomicAdd(&s.counters[kFreeTop], 1);
          const int t = closes ? s.S_t : atomicAdd(&s.counters[kTrans], 1);
          if (t < s.S_t) {
            w = s.S_p + t;
            s.claim_list[s.cap + atomicAdd(&s.counters[kHeirs], 1)] = b;
          } else {
            atomicAdd(&s.counters[kErrStaging], 1);
          }
        }
      }
    }
    s.wslot[b] = w;
    if (w >= 0) {
      s.claim_list[atomicAdd(&s.counters[kClaims], 1)] = b;
      // Tentative candidate only: the warp pass below checks every slot
      // before a later kernel can consume this map. Existing tails remain
      // staged because their preceding values are not in the new input.
      if (direct_new_blocks && fresh && close_tag[b] == step && i <= n_tok - 128)
        s.tags[kDirectRow * s.NB + b] = i;
    }
  }
  __syncthreads();
  const int n_claim = s.counters[kClaims];
  if (direct_new_blocks) {
    const int lane = tid & 31, warp = tid >> 5;
    for (int c = warp; c < n_claim; c += kPrepThreads / 32) {
      const int b = s.claim_list[c];
      const int row = s.tags[kDirectRow * s.NB + b];
      if (row < 0) continue;  // Uniform across this warp.
      bool matches = true;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int off = 32 * q + lane;
        matches &= slots[row + off] == ((int64_t)b << 7) + off;
      }
      const bool complete = __all_sync(0xffffffffu, matches);
      if (lane == 0 && !complete) s.tags[kDirectRow * s.NB + b] = -1;
    }
  }
  // Close-list construction does not read the direct map; its existing
  // trailing CTA barrier also completes the cooperative coverage check.
  const int n_heirs = s.counters[kHeirs];
  for (int i = tid; i < n_tok; i += kPrepThreads) {
    const int64_t sl = slots[i];
    if (sl < 0 || (sl & 127) != 127) continue;
    const int b = sl >> 7;
    if (atomicExch(&dedupe_tag[b], step) == step) continue;
    const int w = s.wslot[b];
    if (w < 0) continue;
    const int c = atomicAdd(&s.counters[kCloses], 1);
    s.close_list[c] = b;
    int heir = -1;
    if (w < s.S_p && n_heirs) {
      const int d = atomicAdd(&s.counters[kDonors], 1);
      if (d < n_heirs) heir = s.claim_list[s.cap + d];
    }
    s.close_list[s.cap + c] = heir;
  }
  __syncthreads();
  // Every block open after a step belongs to a distinct running request, so
  // S_p >= max_num_seqs leaves no heir unpaired; one would lose its KV.
  if (tid == 0 && s.counters[kDonors] < n_heirs)
    atomicAdd(&s.counters[kErrStaging], n_heirs - s.counters[kDonors]);
  // Match the [layer, block, K/V head] tables while keeping all ownership
  // cleanup intact. A zero-claim step never enters the loop or its divisions.
  const int per = 2 * s.H, layer_records = n_claim * per;
  for (int x = tid; x < s.L * layer_records; x += kPrepThreads) {
    const int layer = x / layer_records, r = x % layer_records;
    const int b = s.claim_list[r / per], kvh = r % per;
    const int64_t idx = ((int64_t)layer * s.NB + b) * 2 * s.H + kvh;
    ovf_free(s, s.rec_addr[idx]);
    s.rec_addr[idx] = staging_addr(s, s.wslot[b], layer, kvh);
    s.rec_len[idx] = kRawVarLen;
    if (kvh == 0) s.fill[(int64_t)layer * s.NB + b] = 0;
  }
}

// One CTA per token; stores K / V in kernel order into the block's staging.
// K pair (dims 2 p, 2 p + 1) is one u32; a V pair is two tokens apart.
__global__ void write_kernel(Store s, const uint16_t* k, const uint16_t* v, int64_t ks0,
                             int64_t ks1, int64_t vs0, int64_t vs1, const int64_t* slots,
                             int layer) {
  const int64_t sl = slots[blockIdx.x];
  if (sl < 0) return;
  const int b = sl >> 7, off = sl & 127;
  const int w = s.wslot[b];
  if (w < 0 || s.tags[kDirectRow * s.NB + b] >= 0) return;
  for (int x = threadIdx.x; x < s.H * 64; x += blockDim.x) {
    const int h = x >> 6, p = x & 63;
    {
      const uint16_t* src = k + blockIdx.x * ks0 + h * ks1 + 2 * p;
      const uint32_t val = (uint32_t)src[0] | ((uint32_t)src[1] << 16);
      const int st = p >> 3, f = (p >> 2) & 1, c = p & 3;
      const int tw = off >> 4, j = (off >> 3) & 1, g = off & 7;
      const int t = 32 * tw + 4 * g + c;
      uint8_t* rec = reinterpret_cast<uint8_t*>(staging_addr(s, w, layer, h));
      *reinterpret_cast<uint32_t*>(rec + 16 * (256 * st + t) + 2 * (4 * j + 2 * f)) = val;
    }
    {
      const uint16_t* src = v + blockIdx.x * vs0 + h * vs1 + 2 * p;
      const int st = off >> 4, f = (off >> 3) & 1, c = (off >> 1) & 3, e = off & 1;
      uint8_t* rec = reinterpret_cast<uint8_t*>(staging_addr(s, w, layer, s.H + h));
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const int d = 2 * p + q;
        const int tw = d >> 4, j = (d >> 3) & 1, g = d & 7;
        const int t = 32 * tw + 4 * g + c;
        *reinterpret_cast<uint16_t*>(rec + 16 * (256 * st + t) + 2 * (4 * j + 2 * f + e)) = src[q];
      }
    }
  }
}

// grid (cap, 2 H): CTA (i, kvh) encodes record kvh of close_list[i] at
// ``layer`` from staging or original K/V into the block's arena page. Its 2 H
// records share: each takes the next 16-byte aligned ``kDenseBytes + var_len``
// bytes. A record that does not code, or no longer fits the page, is stored
// raw in the overflow pool. Matches attn_page.encode_record bit for bit.
template <bool Direct>
__global__ void __launch_bounds__(kThreads) encode_kernel(Store s, int layer, int64_t arena,
                                                          int64_t page_bytes,
                                                          const uint8_t* rank_of,
                                                          RawInputs inputs) {
  __shared__ uint8_t rank_s[256];
  __shared__ __align__(16) uint8_t var_s[kVarCap];
  __shared__ int wsum[2][8];
  __shared__ int s_off, s_ovf;
  if ((int)blockIdx.x >= s.counters[kCloses]) return;
  const int kvh = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int b = s.close_list[blockIdx.x];
  const uint8_t* src = reinterpret_cast<const uint8_t*>(staging_addr(s, s.wslot[b], layer, kvh));
  const int64_t idx = ((int64_t)layer * s.NB + b) * 2 * s.H + kvh;
  rank_s[tid] = rank_of[((int64_t)layer * 2 * s.H + kvh) * 256 + tid];
  uint4 wd[8];
  const int row = s.tags[kDirectRow * s.NB + b];
  if constexpr (Direct) {
    if (row >= 0) {
      CUDA_KERNEL_ASSERT((int64_t)row + 128 <= inputs.rows);
      const bool is_k = kvh < s.H;
      const int h = kvh % s.H, g = lane >> 2, c = lane & 3;
      const uint16_t* in = is_k ? inputs.k : inputs.v;
      const int64_t s0 = is_k ? inputs.ks0 : inputs.vs0;
      const int64_t s1 = is_k ? inputs.ks1 : inputs.vs1;
      // Recreate the exact kernel-order uint4 values in registers. Scalar
      // BF16 loads also support fused-QKV slices without extra alignment.
#pragma unroll
      for (int st = 0; st < 8; ++st) {
        uint32_t words[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int j = q >> 1, f = q & 1;
          const int a = 16 * warp + 8 * j + g;
          const int z = 16 * st + 8 * f + 2 * c;
          const int tok = is_k ? a : z, dim = is_k ? z : a;
          const uint16_t* p = in + ((int64_t)row + tok) * s0 + h * s1 + dim;
          words[q] = (uint32_t)p[0] | ((uint32_t)p[is_k ? 1 : s0] << 16);
        }
        wd[st] = make_uint4(words[0], words[1], words[2], words[3]);
      }
    } else {
#pragma unroll
      for (int st = 0; st < 8; ++st)
        wd[st] = reinterpret_cast<const uint4*>(src)[st * kThreads + tid];
    }
  } else {
    // Public Python methods reject missing inputs before launching. Keep a
    // native failure explicit as well, rather than coding unfilled scratch.
    CUDA_KERNEL_ASSERT(row < 0);
#pragma unroll
    for (int st = 0; st < 8; ++st)
      wd[st] = reinterpret_cast<const uint4*>(src)[st * kThreads + tid];
  }
  // Hand this layer's slot record to the heir: copy its transient record in
  // (each thread overwrites only the words it has just read) and point the
  // heir's table there; attention has read the transient one already.
  const int heir = s.close_list[s.cap + blockIdx.x];
  if (heir >= 0) {
    const uint4* from = reinterpret_cast<const uint4*>(staging_addr(s, s.wslot[heir], layer, kvh));
    uint4* to = reinterpret_cast<uint4*>(const_cast<uint8_t*>(src));
#pragma unroll
    for (int st = 0; st < 8; ++st) to[st * kThreads + tid] = from[st * kThreads + tid];
    if (tid == 0)
      s.rec_addr[((int64_t)layer * s.NB + heir) * 2 * s.H + kvh] = reinterpret_cast<int64_t>(src);
  }
  for (int i = tid; i < kVarCap / 16; i += kThreads)
    reinterpret_cast<uint4*>(var_s)[i] = make_uint4(0, 0, 0, 0);
  __syncthreads();

  uint32_t sym[6] = {0, 0, 0, 0, 0, 0};
  uint64_t emask = 0, rmask = 0;
#pragma unroll
  for (int st = 0; st < 8; ++st) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int sc = 8 * st + e, gi = sc >> 5, sg = sc & 31;
      const uint32_t val = (u4get(wd[st], e >> 1) >> (16 * (e & 1))) & 0xffffu;
      const int r = rank_s[val >> 8];
      const uint32_t code = r < kPrimaryDirect ? r : kPrimaryDirect;
      if (sg < 24) {
        sym[3 * gi + (sg >> 3)] |= code << (4 * (sg & 7));
      } else {
#pragma unroll
        for (int kk = 0; kk < 3; ++kk) sym[3 * gi + kk] |= ((code >> kk) & 1u) << (4 * (sg - 24) + 3);
      }
      if (r >= kPrimaryDirect) emask |= 1ull << sc;
      if (r >= kTableEntries) rmask |= 1ull << sc;
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

  const bool codes = var_len <= kVarCap && rmax <= kMaxWarpRaw;
  if (tid == 0) {
    // Claim page bytes with a CAS so a record that does not fit leaves the
    // fill unchanged for the block's smaller records.
    int off = -1;
    if (codes) {
      int32_t* fill = s.fill + (int64_t)layer * s.NB + b;
      const int need = kDenseBytes + var_len;
      int cur = *reinterpret_cast<volatile int32_t*>(fill);
      while (cur + need <= page_bytes) {
        const int prev = atomicCAS(fill, cur, cur + need);
        if (prev == cur) {
          off = cur;
          break;
        }
        cur = prev;
      }
      if (off < 0) atomicAdd(&s.counters[kSpills], 1);
    }
    s_off = off;
  }
  __syncthreads();
  if (s_off < 0) {
    if (tid == 0) {
      const int slot = ovf_alloc(s, (uint32_t)(b * 2 * s.H + kvh));
      if (slot < 0) atomicAdd(&s.counters[kErrOverflow], 1);
      atomicAdd(&s.counters[kFallbacks], 1);
      atomicAdd(&s.counters[kEncoded], 1);
      s_ovf = slot;
    }
    __syncthreads();
    if (s_ovf < 0) return;
    uint8_t* dst = reinterpret_cast<uint8_t*>(s.ovf_base) + (int64_t)s_ovf * kRawRecord;
#pragma unroll
    for (int st = 0; st < 8; ++st) reinterpret_cast<uint4*>(dst)[st * kThreads + tid] = wd[st];
    if (tid == 0) {
      s.rec_addr[idx] = reinterpret_cast<int64_t>(dst);
      s.rec_len[idx] = kRawVarLen;
    }
    return;
  }

  // Escape nibbles (from byte 48, two per byte, low nibble first) and raw
  // entries ``ordinal | high << 16`` in scalar order.
  uint32_t* var32 = reinterpret_cast<uint32_t*>(var_s);
#pragma unroll
  for (int st = 0; st < 8; ++st) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int sc = 8 * st + e;
      if ((emask >> sc) & 1ull) {
        const uint32_t high = ((u4get(wd[st], e >> 1) >> (16 * (e & 1))) >> 8) & 0xffu;
        const int r = rank_s[high];
        const uint32_t nib = r < kTableEntries ? r - kPrimaryDirect : 15u;
        const int bit = 8 * kHeaderBytes + 4 * eo;
        atomicOr(&var32[bit >> 5], nib << (bit & 31));
        if (r >= kTableEntries) var32[(raw_offset >> 2) + ro++] = (uint32_t)eo | (high << 16);
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
  uint8_t* dst = reinterpret_cast<uint8_t*>(arena) + (int64_t)b * page_bytes + s_off;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint4 a = wd[2 * j], c = wd[2 * j + 1];
    reinterpret_cast<uint4*>(dst)[j * kThreads + tid] =
        make_uint4(__byte_perm(a.x, a.y, 0x6420), __byte_perm(a.z, a.w, 0x6420),
                   __byte_perm(c.x, c.y, 0x6420), __byte_perm(c.z, c.w, 0x6420));
  }
#pragma unroll
  for (int m = 0; m < 3; ++m)
    reinterpret_cast<uint2*>(dst + kLowBytes)[m * kThreads + tid] = make_uint2(sym[2 * m], sym[2 * m + 1]);
  __syncthreads();
  for (int i = tid; i < var_len / 16; i += kThreads)
    reinterpret_cast<uint4*>(dst + kDenseBytes)[i] = reinterpret_cast<const uint4*>(var_s)[i];
  if (tid == 0) {
    s.rec_addr[idx] = reinterpret_cast<int64_t>(dst);
    s.rec_len[idx] = var_len;
    atomicAdd(&s.counters[kEncoded], 1);
  }
}

// Closed blocks release their staging slot, or pass it to their heir; the
// close list is then emptied so a forward without a fresh prepare (CUDA-graph
// capture) encodes nothing.
__global__ void commit_kernel(Store s) {
  const int n = s.counters[kCloses];
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int b = s.close_list[i];
    const int w = s.wslot[b];
    const int heir = s.close_list[s.cap + i];
    if (heir >= 0)
      s.wslot[heir] = w;
    else if (w >= 0 && w < s.S_p)
      s.free_stack[atomicAdd(&s.counters[kFreeTop], 1)] = w;
    s.wslot[b] = -1;
    s.tags[kDirectRow * s.NB + b] = -1;
  }
  __syncthreads();
  if (threadIdx.x == 0) s.counters[kCloses] = 0;
}

constexpr int kGatherSmem = kTile + kEscCap;

// grid (pages, 2 H): decodes record kvh of block src[i] at one layer into
// workspace page i, token-major [pages, 128, H, 128] BF16 (K or V). Direct
// new blocks copy original K/V instead of reading their unfilled scratch.
template <bool Direct>
__global__ void __launch_bounds__(kThreads) gather_kernel(const int32_t* src, const int64_t* rec_addr,
                                                          const int32_t* rec_len,
                                                          const uint32_t* plut, const uint8_t* slut,
                                                          uint8_t* wk, uint8_t* wv, int H,
                                                          const int32_t* direct_rows,
                                                          RawInputs inputs) {
  extern __shared__ __align__(16) uint8_t smem[];
  uint8_t* tile = smem;
  uint8_t* esc = smem + kTile;
  const int kvh = blockIdx.y, kv = kvh / H, h = kvh % H;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, c = lane & 3;
  if constexpr (Direct) {
    const int row = direct_rows[src[blockIdx.x]];
    if (row >= 0) {
      // A captured commit clears the map only when the graph executes.
      // Ordinary reads may therefore carry a map without source tensors;
      // an actually pending direct block must fail rather than read scratch.
      CUDA_KERNEL_ASSERT(inputs.k != nullptr && inputs.v != nullptr &&
                         (int64_t)row + 128 <= inputs.rows);
      const uint16_t* in = kv == 0 ? inputs.k : inputs.v;
      const int64_t s0 = kv == 0 ? inputs.ks0 : inputs.vs0;
      const int64_t s1 = kv == 0 ? inputs.ks1 : inputs.vs1;
      uint16_t* out = reinterpret_cast<uint16_t*>(kv == 0 ? wk : wv);
      // Typical fused-QKV slices preserve 16-byte alignment. Copy eight
      // BF16 values per instruction there, while odd-offset/stride views
      // keep the scalar path and the same public input contract.
      const bool vectorized =
          ((reinterpret_cast<uintptr_t>(in) | reinterpret_cast<uintptr_t>(out)) & 15u) == 0 &&
          (s0 & 7) == 0 && (s1 & 7) == 0;
      if (vectorized) {
        const uint16_t* src_head = in + (int64_t)row * s0 + h * s1;
        uint16_t* dst_head = out + ((int64_t)blockIdx.x * 128 * H + h) * 128;
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const int x = q * kThreads + tid, tok = x >> 4, chunk = x & 15;
          const uint4 val = *reinterpret_cast<const uint4*>(src_head + tok * s0 + 8 * chunk);
          *reinterpret_cast<uint4*>(dst_head + (int64_t)tok * H * 128 + 8 * chunk) = val;
        }
      } else {
        for (int x = tid; x < 128 * 128; x += kThreads) {
          const int tok = x >> 7, dim = x & 127;
          out[(((int64_t)blockIdx.x * 128 + tok) * H + h) * 128 + dim] =
              in[((int64_t)row + tok) * s0 + h * s1 + dim];
        }
      }
      return;
    }
  }
  const int64_t idx = (int64_t)src[blockIdx.x] * 2 * H + kvh;
  const uint8_t* rec = reinterpret_cast<const uint8_t*>(rec_addr[idx]);
  auto put = [&](int tok, int dim, uint32_t v, int bytes) {
    uint8_t* p = tile + tok * 256 + (((dim >> 3) ^ (tok & 7)) << 4) + 2 * (dim & 7);
    if (bytes == 4)
      *reinterpret_cast<uint32_t*>(p) = v;
    else
      *reinterpret_cast<uint16_t*>(p) = (uint16_t)v;
  };
  decode_rec(rec, rec_len[idx], plut[2 * kvh], plut[2 * kvh + 1], slut + 16 * kvh, esc, tid,
             [&](int st, const uint32_t (&bf)[4]) {
#pragma unroll
               for (int i = 0; i < 4; ++i) {
                 const int j = i >> 1, f = i & 1;
                 if (kv == 0) {
                   put(16 * warp + 8 * j + g, 16 * st + 8 * f + 2 * c, bf[i], 4);
                 } else {
                   const int tok = 16 * st + 8 * f + 2 * c, dim = 16 * warp + 8 * j + g;
                   put(tok, dim, bf[i] & 0xffffu, 2);
                   put(tok + 1, dim, bf[i] >> 16, 2);
                 }
               }
             });
  __syncthreads();
  uint8_t* out = kv == 0 ? wk : wv;
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const int x = q * kThreads + tid, r = x >> 4, jj = x & 15;
    const uint4 val = *reinterpret_cast<const uint4*>(tile + r * 256 + ((jj ^ (r & 7)) << 4));
    *reinterpret_cast<uint4*>(out + (((int64_t)blockIdx.x * 128 + r) * H + h) * 256 + jj * 16) = val;
  }
}

// Block blob: one closed block's records in an external tier. A fixed header
// (u32 magic, version, L, H; i64 bytes, digest) and the i32 rec_len table
// [L, 2, H] fill a kBlobAlign-padded head; the records follow in (layer, kv,
// head) order, each kDenseBytes + rec_len bytes (raw records 32768), so every
// record stays 16-byte aligned. ``bytes`` excludes the tail padding.
constexpr uint32_t kBlobMagic = 0x424B5046u;  // "FPKB"
constexpr uint32_t kBlobVersion = 1;
constexpr int kBlobFixed = 64;
constexpr int64_t kBlobAlign = 4096;
constexpr int kMaxHeads = 16;
// Import status bits, per destination block.
enum : int { kBlobBadHeader = 1, kBlobBadLengths = 2, kBlobNoOverflow = 4 };

__host__ __device__ __forceinline__ int64_t blob_head_bytes(int L, int H) {
  return (kBlobFixed + 8LL * L * H + kBlobAlign - 1) / kBlobAlign * kBlobAlign;
}

__device__ long long block_sum(long long v, long long* red) {
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  long long t = 0;
#pragma unroll
  for (int w = 0; w < kThreads / 32; ++w) t += red[w];
  return t;
}

// grid (n, L): CTA (i, l) copies layer l's records of blocks[i] into the blob
// at out + i * stride; layer 0 also writes the head and lengths[i] (the blob
// rounded up to kBlobAlign). Every CTA sums the block's lengths itself.
__global__ void __launch_bounds__(kThreads) blob_export_kernel(const int32_t* blocks,
                                                               const int64_t* rec_addr,
                                                               const int32_t* rec_len, uint8_t* out,
                                                               int64_t stride, int64_t* lengths,
                                                               int L, int H, int NB, int64_t digest) {
  __shared__ long long red[kThreads / 32];
  const int i = blockIdx.x, l = blockIdx.y, tid = threadIdx.x, per = 2 * H;
  const int b = blocks[i];
  long long before = 0, total = 0;
  for (int g = tid; g < L * per; g += kThreads) {
    const long long n = kDenseBytes + rec_len[((int64_t)(g / per) * NB + b) * per + g % per];
    total += n;
    if (g / per < l) before += n;
  }
  before = block_sum(before, red);
  total = block_sum(total, red);
  const int64_t head = blob_head_bytes(L, H);
  uint8_t* dst = out + (int64_t)i * stride;
  if (l == 0) {
    if (tid == 0) {
      uint32_t* h32 = reinterpret_cast<uint32_t*>(dst);
      h32[0] = kBlobMagic;
      h32[1] = kBlobVersion;
      h32[2] = L;
      h32[3] = H;
      int64_t* h64 = reinterpret_cast<int64_t*>(dst + 16);
      h64[0] = head + total;
      h64[1] = digest;
      lengths[i] = (head + total + kBlobAlign - 1) / kBlobAlign * kBlobAlign;
    }
    int32_t* table = reinterpret_cast<int32_t*>(dst + kBlobFixed);
    for (int g = tid; g < L * per; g += kThreads)
      table[g] = rec_len[((int64_t)(g / per) * NB + b) * per + g % per];
  }
  int64_t off = head + before;
  for (int kvh = 0; kvh < per; ++kvh) {
    const int64_t idx = ((int64_t)l * NB + b) * per + kvh;
    const int n = kDenseBytes + rec_len[idx];
    const uint4* src = reinterpret_cast<const uint4*>(rec_addr[idx]);
    uint4* d = reinterpret_cast<uint4*>(dst + off);
    for (int q = tid; q < n / 16; q += kThreads) d[q] = src[q];
    off += n;
  }
}

// grid (n, L): CTA (i, l) validates the blob at src + offsets[i] and installs
// its layer-l records for blocks[i]: coded records packed into the block's
// arena page, raw ones (and coded ones that no longer fit) in overflow slots.
// The block's old overflow records are freed. Every CTA validates the whole
// head, so all layers agree; a failure sets status[i] and leaves the layer's
// tables untouched (the caller must recompute the block).
__global__ void __launch_bounds__(kThreads) blob_import_kernel(Store s, const uint8_t* src,
                                                               int64_t src_bytes,
                                                               const int64_t* offsets,
                                                               const int32_t* blocks, int32_t* status,
                                                               const int64_t* arena_base,
                                                               const int64_t* arena_stride,
                                                               int64_t digest) {
  __shared__ long long red[kThreads / 32];
  __shared__ int64_t s_src[2 * kMaxHeads], s_dst[2 * kMaxHeads];
  __shared__ int s_fail;
  const int i = blockIdx.x, l = blockIdx.y, tid = threadIdx.x, L = s.L, per = 2 * s.H;
  const int b = blocks[i];
  const int64_t off = offsets[i], head = blob_head_bytes(L, s.H);
  if (off < 0 || off % 16 || off + head > src_bytes) {
    if (tid == 0) atomicOr(&status[i], kBlobBadHeader);
    return;
  }
  const uint8_t* blob = src + off;
  const uint32_t* h32 = reinterpret_cast<const uint32_t*>(blob);
  const int64_t* h64 = reinterpret_cast<const int64_t*>(blob + 16);
  const int64_t declared = h64[0];
  if (h32[0] != kBlobMagic || h32[1] != kBlobVersion || (int)h32[2] != L || (int)h32[3] != s.H ||
      h64[1] != digest || declared < head || off + declared > src_bytes) {
    if (tid == 0) atomicOr(&status[i], kBlobBadHeader);
    return;
  }
  const int32_t* table = reinterpret_cast<const int32_t*>(blob + kBlobFixed);
  long long before = 0, total = 0;
  int bad = 0;
  for (int g = tid; g < L * per; g += kThreads) {
    const int v = table[g];
    const bool ok = v == kRawVarLen || (v >= kHeaderBytes && v <= kVarCap && v % 16 == 0);
    bad |= !ok;
    const long long n = kDenseBytes + (ok ? v : 0);
    total += n;
    if (g / per < l) before += n;
  }
  bad = __syncthreads_or(bad);
  before = block_sum(before, red);
  total = block_sum(total, red);
  if (bad || head + total != declared) {
    if (tid == 0) atomicOr(&status[i], kBlobBadLengths);
    return;
  }
  const int* lens = table + l * per;
  if (tid == 0) {
    const int64_t page = arena_stride[l];
    const int64_t page_ptr = arena_base[l] + (int64_t)b * page;
    int64_t from = head + before;
    int fill = 0, placed = 0;
    for (; placed < per; ++placed) {
      const int n = kDenseBytes + lens[placed];
      int64_t d;
      if (lens[placed] != kRawVarLen && fill + n <= page) {
        d = page_ptr + fill;
        fill += n;
      } else {
        const int slot = ovf_alloc(s, (uint32_t)(b * per + placed));
        if (slot < 0) break;
        d = s.ovf_base + (int64_t)slot * kRawRecord;
      }
      s_src[placed] = from;
      s_dst[placed] = d;
      from += n;
    }
    s_fail = placed < per;
    if (s_fail) {
      for (int k = 0; k < placed; ++k) ovf_free(s, s_dst[k]);
      atomicOr(&status[i], kBlobNoOverflow);
    } else {
      s.fill[(int64_t)l * s.NB + b] = fill;
    }
  }
  __syncthreads();
  if (s_fail) return;
  for (int kvh = 0; kvh < per; ++kvh) {
    const uint4* from = reinterpret_cast<const uint4*>(blob + s_src[kvh]);
    uint4* to = reinterpret_cast<uint4*>(s_dst[kvh]);
    for (int q = tid; q < (kDenseBytes + lens[kvh]) / 16; q += kThreads) to[q] = from[q];
  }
  __syncthreads();
  if (tid < per) {
    const int64_t idx = ((int64_t)l * s.NB + b) * per + tid;
    ovf_free(s, s.rec_addr[idx]);
    s.rec_addr[idx] = s_dst[tid];
    s.rec_len[idx] = lens[tid];
  }
}

}  // namespace

void hbm_prepare(torch::Tensor counters, torch::Tensor wslot, torch::Tensor tags,
                 torch::Tensor free_stack, torch::Tensor ovf_bits, torch::Tensor claim_list,
                 torch::Tensor close_list, torch::Tensor rec_addr, torch::Tensor rec_len,
                 torch::Tensor staging, torch::Tensor scratch, torch::Tensor overflow,
                 torch::Tensor fill,
                 torch::Tensor slot_mapping, int64_t n_tok, torch::Tensor live, int64_t n_live,
                 int64_t step, bool direct_new_blocks) {
  TORCH_CHECK(slot_mapping.is_cuda() && slot_mapping.device() == rec_addr.device() &&
                  slot_mapping.scalar_type() == torch::kInt64 && slot_mapping.is_contiguous(),
              "slot_mapping must be contiguous CUDA int64 on the store device");
  TORCH_CHECK(n_tok >= 0 && n_tok <= slot_mapping.numel() && n_tok <= INT32_MAX &&
                  n_live >= 0 && n_live <= live.numel(),
              "invalid prepare token/live counts");
  TORCH_CHECK(live.is_cuda() && live.device() == rec_addr.device() &&
                  live.scalar_type() == torch::kInt32 && live.is_contiguous(),
              "live blocks must be contiguous CUDA int32 on the store device");
  // Claims, heirs and closes are each bounded by the token count.
  TORCH_CHECK(claim_list.numel() == close_list.numel() && 2 * n_tok <= claim_list.numel(),
              "block lists too small");
  Store s = make_store(counters, wslot, tags, free_stack, ovf_bits, claim_list, close_list,
                       rec_addr, rec_len, staging, scratch, overflow, fill);
  prepare_kernel<<<1, kPrepThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      s, slot_mapping.data_ptr<int64_t>(), n_tok, live.data_ptr<int32_t>(), n_live,
      static_cast<int32_t>(step), direct_new_blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hbm_write(torch::Tensor counters, torch::Tensor wslot, torch::Tensor tags,
               torch::Tensor free_stack, torch::Tensor ovf_bits, torch::Tensor claim_list,
               torch::Tensor close_list, torch::Tensor rec_addr, torch::Tensor rec_len,
               torch::Tensor staging, torch::Tensor scratch, torch::Tensor overflow,
               torch::Tensor fill,
               torch::Tensor key, torch::Tensor value, torch::Tensor slot_mapping,
               int64_t layer) {
  Store s = make_store(counters, wslot, tags, free_stack, ovf_bits, claim_list, close_list,
                       rec_addr, rec_len, staging, scratch, overflow, fill);
  const RawInputs inputs = raw_inputs(key, value, s.H, rec_addr.device());
  const int64_t n = inputs.rows;
  TORCH_CHECK(layer >= 0 && layer < s.L, "invalid write layer");
  TORCH_CHECK(slot_mapping.is_cuda() && slot_mapping.device() == rec_addr.device() &&
                  slot_mapping.scalar_type() == torch::kInt64 && slot_mapping.is_contiguous() &&
                  slot_mapping.numel() >= n, "write slot mapping must cover all input rows");
  if (n == 0) return;
  write_kernel<<<n, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      s, inputs.k, inputs.v, inputs.ks0, inputs.ks1, inputs.vs0, inputs.vs1,
      slot_mapping.data_ptr<int64_t>(), layer);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hbm_encode(torch::Tensor counters, torch::Tensor wslot, torch::Tensor tags,
                torch::Tensor free_stack, torch::Tensor ovf_bits, torch::Tensor claim_list,
                torch::Tensor close_list, torch::Tensor rec_addr, torch::Tensor rec_len,
                torch::Tensor staging, torch::Tensor scratch, torch::Tensor overflow,
                torch::Tensor fill,
                torch::Tensor rank_of, int64_t layer, int64_t arena, int64_t page_bytes,
                int64_t max_close, std::optional<torch::Tensor> key,
                std::optional<torch::Tensor> value) {
  TORCH_CHECK(key.has_value() == value.has_value(), "encode requires both new K and V or neither");
  Store s = make_store(counters, wslot, tags, free_stack, ovf_bits, claim_list, close_list,
                       rec_addr, rec_len, staging, scratch, overflow, fill);
  TORCH_CHECK(layer >= 0 && layer < s.L, "invalid encode layer");
  TORCH_CHECK(max_close >= 0 && 2 * max_close <= close_list.numel(), "close list too small");
  TORCH_CHECK(rank_of.numel() == (int64_t)s.L * 2 * s.H * 256, "rank_of must hold every layer");
  TORCH_CHECK(page_bytes > 0 && page_bytes % 16 == 0 && page_bytes < (1LL << 31),
              "bad arena page size");
  RawInputs inputs;
  if (key.has_value()) inputs = raw_inputs(*key, *value, s.H, rec_addr.device());
  if (max_close == 0) return;
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (key.has_value()) {
    encode_kernel<true><<<dim3(max_close, 2 * s.H), kThreads, 0, stream>>>(
        s, layer, arena, page_bytes, rank_of.data_ptr<uint8_t>(), inputs);
  } else {
    encode_kernel<false><<<dim3(max_close, 2 * s.H), kThreads, 0, stream>>>(
        s, layer, arena, page_bytes, rank_of.data_ptr<uint8_t>(), inputs);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hbm_commit(torch::Tensor counters, torch::Tensor wslot, torch::Tensor tags,
                torch::Tensor free_stack, torch::Tensor ovf_bits, torch::Tensor claim_list,
                torch::Tensor close_list, torch::Tensor rec_addr, torch::Tensor rec_len,
                torch::Tensor staging, torch::Tensor scratch, torch::Tensor overflow,
                torch::Tensor fill) {
  Store s = make_store(counters, wslot, tags, free_stack, ovf_bits, claim_list, close_list,
                       rec_addr, rec_len, staging, scratch, overflow, fill);
  commit_kernel<<<1, 256, 0, at::cuda::getCurrentCUDAStream()>>>(s);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hbm_gather(torch::Tensor src_blocks, torch::Tensor rec_addr, torch::Tensor rec_len,
                torch::Tensor plut, torch::Tensor slut, torch::Tensor wk, torch::Tensor wv,
                int64_t kv_heads, std::optional<torch::Tensor> direct_rows,
                std::optional<torch::Tensor> key, std::optional<torch::Tensor> value) {
  const bool direct = direct_rows.has_value();
  TORCH_CHECK(key.has_value() == value.has_value(),
              "gather requires both new K and V or neither");
  TORCH_CHECK(!key.has_value() || direct,
              "gather with new K/V requires direct source rows");
  TORCH_CHECK(kv_heads > 0 && rec_addr.dim() == 3 && rec_addr.size(1) == 2 &&
                  rec_addr.size(2) == kv_heads, "bad gather record-table shape");
  TORCH_CHECK(src_blocks.is_cuda() && src_blocks.device() == rec_addr.device() &&
                  src_blocks.scalar_type() == torch::kInt32 && src_blocks.is_contiguous(),
              "gather blocks must be contiguous CUDA int32 on the store device");
  TORCH_CHECK(wk.is_cuda() && wv.is_cuda() && wk.device() == rec_addr.device() &&
                  wv.device() == rec_addr.device() && wk.scalar_type() == torch::kBFloat16 &&
                  wv.scalar_type() == torch::kBFloat16 && wk.is_contiguous() && wv.is_contiguous(),
              "gather workspaces must be contiguous BF16 on the store device");
  const int64_t n = src_blocks.numel();
  RawInputs inputs;
  const int32_t* rows = nullptr;
  if (direct) {
    TORCH_CHECK(direct_rows->is_cuda() && direct_rows->device() == rec_addr.device() &&
                    direct_rows->scalar_type() == torch::kInt32 && direct_rows->dim() == 1 &&
                    direct_rows->numel() == rec_addr.size(0) && direct_rows->is_contiguous(),
                "direct source rows must be contiguous CUDA int32 [blocks]");
    if (key.has_value()) inputs = raw_inputs(*key, *value, kv_heads, rec_addr.device());
    rows = direct_rows->data_ptr<int32_t>();
  }
  if (n == 0) return;
  TORCH_CHECK(wk.numel() >= n * 128 * kv_heads * 128 && wv.numel() >= n * 128 * kv_heads * 128,
              "workspace too small");
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(gather_kernel<false>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, kGatherSmem));
    C10_CUDA_CHECK(cudaFuncSetAttribute(gather_kernel<true>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, kGatherSmem));
    attr = true;
  }
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (direct) {
    gather_kernel<true><<<dim3(n, 2 * kv_heads), kThreads, kGatherSmem, stream>>>(
        src_blocks.data_ptr<int32_t>(), rec_addr.data_ptr<int64_t>(), rec_len.data_ptr<int32_t>(),
        reinterpret_cast<const uint32_t*>(plut.data_ptr()), slut.data_ptr<uint8_t>(),
        reinterpret_cast<uint8_t*>(wk.data_ptr()), reinterpret_cast<uint8_t*>(wv.data_ptr()),
        kv_heads, rows, inputs);
  } else {
    gather_kernel<false><<<dim3(n, 2 * kv_heads), kThreads, kGatherSmem, stream>>>(
        src_blocks.data_ptr<int32_t>(), rec_addr.data_ptr<int64_t>(), rec_len.data_ptr<int32_t>(),
        reinterpret_cast<const uint32_t*>(plut.data_ptr()), slut.data_ptr<uint8_t>(),
        reinterpret_cast<uint8_t*>(wk.data_ptr()), reinterpret_cast<uint8_t*>(wv.data_ptr()),
        kv_heads, rows, inputs);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hbm_export(torch::Tensor blocks, torch::Tensor rec_addr, torch::Tensor rec_len,
                torch::Tensor out, int64_t stride, torch::Tensor lengths, int64_t digest) {
  const int64_t n = blocks.numel();
  if (n == 0) return;
  const int L = rec_addr.size(0), NB = rec_addr.size(1), H = rec_addr.size(3);
  TORCH_CHECK(blocks.scalar_type() == torch::kInt32 && lengths.scalar_type() == torch::kInt64 &&
                  lengths.numel() >= n,
              "export blocks must be int32 and lengths int64 [>= n]");
  TORCH_CHECK(out.scalar_type() == torch::kUInt8 && out.is_contiguous() &&
                  reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
              "export buffer must be contiguous 16-byte aligned uint8");
  TORCH_CHECK(stride % 16 == 0 &&
                  stride >= blob_head_bytes(L, H) + 2LL * L * H * kRawRecord &&
                  out.numel() >= n * stride,
              "export stride or buffer too small");
  blob_export_kernel<<<dim3(n, L), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      blocks.data_ptr<int32_t>(), rec_addr.data_ptr<int64_t>(), rec_len.data_ptr<int32_t>(),
      out.data_ptr<uint8_t>(), stride, lengths.data_ptr<int64_t>(), L, H, NB, digest);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hbm_import(torch::Tensor counters, torch::Tensor wslot, torch::Tensor tags,
                torch::Tensor free_stack, torch::Tensor ovf_bits, torch::Tensor claim_list,
                torch::Tensor close_list, torch::Tensor rec_addr, torch::Tensor rec_len,
                torch::Tensor staging, torch::Tensor scratch, torch::Tensor overflow,
                torch::Tensor fill,
                torch::Tensor src, torch::Tensor offsets, torch::Tensor blocks,
                torch::Tensor status, torch::Tensor arena_base, torch::Tensor arena_stride,
                int64_t digest) {
  const int64_t n = blocks.numel();
  if (n == 0) return;
  Store s = make_store(counters, wslot, tags, free_stack, ovf_bits, claim_list, close_list,
                       rec_addr, rec_len, staging, scratch, overflow, fill);
  TORCH_CHECK(s.H <= kMaxHeads, "too many KV heads");
  TORCH_CHECK(src.scalar_type() == torch::kUInt8 && src.is_contiguous() &&
                  reinterpret_cast<uintptr_t>(src.data_ptr()) % 16 == 0,
              "import buffer must be contiguous 16-byte aligned uint8");
  TORCH_CHECK(offsets.scalar_type() == torch::kInt64 && offsets.numel() == n &&
                  blocks.scalar_type() == torch::kInt32 && status.scalar_type() == torch::kInt32 &&
                  status.numel() >= n,
              "import metadata must be int64 offsets, int32 blocks / status");
  TORCH_CHECK(arena_base.numel() == s.L && arena_stride.numel() == s.L, "arena tables must hold every layer");
  blob_import_kernel<<<dim3(n, s.L), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      s, src.data_ptr<uint8_t>(), src.numel(), offsets.data_ptr<int64_t>(),
      blocks.data_ptr<int32_t>(), status.data_ptr<int32_t>(), arena_base.data_ptr<int64_t>(),
      arena_stride.data_ptr<int64_t>(), digest);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
