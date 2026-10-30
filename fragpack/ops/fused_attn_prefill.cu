// SPDX-License-Identifier: Apache-2.0
// Multi-query causal prefill directly over FragPack records. No decoded KV
// tensor is written to HBM: decode_rec emits MMA operands inside this CTA.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <optional>

#include "fragpack_common.cuh"

namespace {

using namespace fragpack;

// Columns flatten (query token, GQA head). One K/V decode serves all columns;
// GQA=8 therefore processes eight query tokens, rather than eight independent
// single-token decode launches. The transposed MMAs match the existing record
// layout: K Q^T followed by V^T P^T.
constexpr int kColumns = 64;
constexpr int kGroups = kColumns / 8;
constexpr int kQWords = 68;  // 128 BF16 + 8 BF16 padding avoids Q load bank conflicts
constexpr int kQBytes = kColumns * kQWords * 4;
constexpr int kPRel = kQBytes;
constexpr int kPBytes = kColumns * 256;
constexpr int kMaxRel = kPRel + kPBytes;
constexpr int kReductionBytes = kGroups * 8 * 8 * 4;
constexpr int kSumRel = kMaxRel + kReductionBytes;
constexpr int kEscRel = kSumRel + kReductionBytes;
constexpr int kSlotRel = (kEscRel + kEscCap + 15) & ~15;
constexpr int kBarRel = kSlotRel + 2 * kRawRecord;
constexpr int kSmem = kBarRel + 16;

struct Params {
  const uint32_t* q;
  const int64_t* rec_addr;
  const int32_t* rec_len;
  const uint32_t* plut;
  const uint8_t* slut;
  const int32_t* block_table;
  const int32_t* cu_q;
  const int32_t* seq_lens;
  __nv_bfloat16* out;
  float* opart;
  float* ml;
  int64_t q_stride, out_stride, bt_stride;
  int kv_heads, group, num_splits;
  float scale_log2;
};

__device__ __forceinline__ void mma(float (&acc)[4], const uint32_t (&a)[4],
                                    uint32_t b0, uint32_t b1) {
  // decode_rec's pair ordering differs from the transposed MMA A ordering.
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
      : "r"(a[0]), "r"(a[2]), "r"(a[1]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* ptr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_u32(ptr)));
}

__device__ __forceinline__ void init_bar(uint64_t* bar) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(smem_u32(bar)));
}

__device__ __forceinline__ void invalidate_bar(uint64_t* bar) {
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n" :: "r"(smem_u32(bar)) : "memory");
}

__device__ __forceinline__ void load_record(void* dst, const void* src, uint32_t bytes,
                                           uint64_t* bar) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
               :: "r"(smem_u32(bar)), "r"(bytes) : "memory");
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
      :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}

__device__ __forceinline__ void wait_record(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n.reg .pred ready;\n"
      "WAIT_RECORD:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 ready, [%0], %1;\n"
      "@!ready bra WAIT_RECORD;\n}\n"
      :: "r"(smem_u32(bar)), "r"(parity) : "memory");
}

template <bool kSplit>
__global__ void __launch_bounds__(kThreads, 1) prefill_attn(Params p) {
  const int request = blockIdx.y;
  int head = blockIdx.z, split = 0;
  if constexpr (kSplit) {
    head = blockIdx.z / p.num_splits;
    split = blockIdx.z % p.num_splits;
  }
  const int q_begin = p.cu_q[request], q_end = p.cu_q[request + 1];
  const int q_len = q_end - q_begin;
  const int tile_q = kColumns / p.group;
  const int q_first = blockIdx.x * tile_q;
  if (q_first >= q_len) return;  // Uniform CTA exit; no barrier has been initialized.
  const int seq = p.seq_lens[request];
  const int prefix = seq - q_len;
  // Later records cannot contribute to any query in this tile under causality.
  const int visible = min(seq, prefix + min(q_len, q_first + tile_q));
  const int pages = (max(visible, 0) + 127) / 128;
  // Split the visible pages of this query tile, not the full request's future
  // context. Every split owns a disjoint contiguous interval; early query rows
  // can still be fully masked in a nonempty interval. Those rows must publish
  // empty partials just like a split that receives no pages at all.
  const int first_page = kSplit ? pages * split / p.num_splits : 0;
  const int end_page = kSplit ? pages * (split + 1) / p.num_splits : pages;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, c = lane & 3;

  extern __shared__ __align__(16) uint8_t smem[];
  __shared__ __align__(16) uint8_t secondary[2][16];
  uint32_t* Qs = reinterpret_cast<uint32_t*>(smem);
  uint8_t* Ps = smem + kPRel;
  float* maxima = reinterpret_cast<float*>(smem + kMaxRel);
  float* sums = reinterpret_cast<float*>(smem + kSumRel);
  uint8_t* escape = smem + kEscRel;
  uint8_t* slots = smem + kSlotRel;
  uint64_t* barriers = reinterpret_cast<uint64_t*>(smem + kBarRel);

  // Q is reused for every history page. Padding only the shared row stride
  // avoids eight-way conflicts for the fragment-order loads below.
  for (int x = tid; x < kColumns * 64; x += kThreads) {
    const int column = x / 64, word = x % 64;
    const int query = q_first + column / p.group;
    const int q_head = head * p.group + column % p.group;
    Qs[column * kQWords + word] = query < q_len
        ? p.q[(int64_t)(q_begin + query) * p.q_stride + q_head * 64 + word]
        : 0u;
  }
  if (tid < 32) {
    const int kv = tid >> 4;
    secondary[kv][tid & 15] = p.slut[(kv * p.kv_heads + head) * 16 + (tid & 15)];
  }
  if (tid == 0) {
    init_bar(&barriers[0]);
    init_bar(&barriers[1]);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();

  const uint32_t kl0 = p.plut[head * 2], kl1 = p.plut[head * 2 + 1];
  const uint32_t vl0 = p.plut[(p.kv_heads + head) * 2];
  const uint32_t vl1 = p.plut[(p.kv_heads + head) * 2 + 1];
  const int32_t* blocks = p.block_table + (int64_t)request * p.bt_stride;
  auto index = [&](int page, int kv) {
    return ((int64_t)blocks[page] * 2 + kv) * p.kv_heads + head;
  };
  auto issue = [&](int page, int kv) {
    if (page < end_page) {
      const int64_t idx = index(page, kv);
      load_record(slots + kv * kRawRecord,
                  reinterpret_cast<const void*>(p.rec_addr[idx]),
                  kDenseBytes + p.rec_len[idx], &barriers[kv]);
    }
  };

  float output[kGroups][4] = {};
  float running_max[kGroups][2];
  float running_sum[kGroups][2] = {};
#pragma unroll
  for (int n = 0; n < kGroups; ++n) {
    running_max[n][0] = -INFINITY;
    running_max[n][1] = -INFINITY;
  }
  if (tid == 0 && first_page < end_page) {
    issue(first_page, 0);
    issue(first_page, 1);
  }
  for (int page = first_page; page < end_page; ++page) {
    // Each split initializes new barriers. Their phase counts local issues,
    // so a chunk starting at an odd physical page still waits for phase zero.
    const int iteration = page - first_page;
    // Every warp must finish the preceding PV before reusing P, escapes, V.
    __syncthreads();
    if (tid == 0 && iteration) issue(page, 1);
    wait_record(&barriers[0], iteration & 1);
    float score[kGroups][4] = {};
    decode_rec(slots, p.rec_len[index(page, 0)], kl0, kl1, secondary[0], escape, tid,
               [&](int step, const uint32_t (&key)[4]) {
#pragma unroll
                 for (int n = 0; n < kGroups; ++n) {
                   const uint32_t* query = Qs + (8 * n + g) * kQWords + 8 * step + c;
                   mma(score[n], key, query[0], query[4]);
                 }
               });

    // Each lane owns two query columns and two token positions. Warp reductions
    // cover token positions, followed by an eight-warp CTA reduction per column.
    const int token0 = page * 128 + 16 * warp + g;
#pragma unroll
    for (int n = 0; n < kGroups; ++n) {
      float mx[2];
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int query = q_first + (8 * n + 2 * c + e) / p.group;
        const int last_key = prefix + query;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int token = token0 + 8 * j;
          score[n][2 * j + e] = query < q_len && token < seq && token <= last_key
              ? score[n][2 * j + e] * p.scale_log2 : -INFINITY;
        }
        mx[e] = fmaxf(score[n][e], score[n][2 + e]);
#pragma unroll
        for (int x = 4; x < 32; x <<= 1)
          mx[e] = fmaxf(mx[e], __shfl_xor_sync(0xffffffffu, mx[e], x));
      }
      if (g == 0)
        *reinterpret_cast<float2*>(maxima + (n * 8 + warp) * 8 + 2 * c) =
            make_float2(mx[0], mx[1]);
    }
    __syncthreads();  // All K readers have retired; prefetch the next K page.
    if (tid == 0) issue(page + 1, 0);
#pragma unroll
    for (int n = 0; n < kGroups; ++n) {
      float2 mx = *reinterpret_cast<const float2*>(maxima + n * 64 + 2 * c);
#pragma unroll
      for (int w = 1; w < 8; ++w) {
        const float2 next = *reinterpret_cast<const float2*>(maxima + (n * 8 + w) * 8 + 2 * c);
        mx.x = fmaxf(mx.x, next.x);
        mx.y = fmaxf(mx.y, next.y);
      }
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const float m_new = fmaxf(running_max[n][e], e ? mx.y : mx.x);
        // Padded query columns are fully masked. Avoid -inf - -inf so they
        // cannot introduce NaNs into shared probabilities or MMA operands.
        const float alpha = running_max[n][e] == -INFINITY
            ? 0.f : exp2f(running_max[n][e] - m_new);
        running_max[n][e] = m_new;
        float sum = 0.f;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const float probability = score[n][2 * j + e] == -INFINITY
              ? 0.f : exp2f(score[n][2 * j + e] - m_new);
          sum += probability;
          const int column = 2 * c + e;
          *reinterpret_cast<__nv_bfloat16*>(
              Ps + (8 * n + column) * 256 + ((2 * warp + j) ^ column) * 16 + 2 * g) =
              __float2bfloat16_rn(probability);
          output[n][2 * j + e] *= alpha;
        }
        running_sum[n][e] = running_sum[n][e] * alpha + sum;
      }
    }
    __syncthreads();

    // Recover V directly into the MMA A fragments. Ignore uninitialized tail
    // tokens before multiplication: a zero attention probability alone does
    // not suppress NaN values (0 * NaN is still NaN).
    wait_record(&barriers[1], iteration & 1);
    uint32_t probabilities[kGroups][4];
    const int matrix = lane >> 3, probability_row = lane & 7;
    decode_rec(slots + kRawRecord, p.rec_len[index(page, 1)], vl0, vl1,
               secondary[1], escape, tid,
               [&](int step, const uint32_t (&value)[4]) {
                 uint32_t valid_value[4];
#pragma unroll
                 for (int j = 0; j < 4; ++j) {
                   const int token = page * 128 + 16 * step + 8 * (j & 1) + 2 * c;
                   valid_value[j] = token >= seq ? 0u
                       : token + 1 >= seq ? value[j] & 0xffffu : value[j];
                 }
#pragma unroll
                 for (int n = 0; n < kGroups; ++n) {
                   if (!(step & 1))
                     ldsm4(probabilities[n], Ps + (8 * n + probability_row) * 256 +
                           ((2 * step + matrix) ^ probability_row) * 16);
                   const int half = step & 1;
                   mma(output[n], valid_value, probabilities[n][2 * half],
                       probabilities[n][2 * half + 1]);
                 }
               });
  }

  // Reduce this split's exponential sums across all token-owning lanes.
  // S=1 normalizes directly into out. S>1 publishes an FP32 numerator and its
  // scale/sum so combine can merge chunks without reconstructing global KV.
#pragma unroll
  for (int n = 0; n < kGroups; ++n) {
    float total[2] = {running_sum[n][0], running_sum[n][1]};
#pragma unroll
    for (int e = 0; e < 2; ++e)
#pragma unroll
      for (int x = 4; x < 32; x <<= 1)
        total[e] += __shfl_xor_sync(0xffffffffu, total[e], x);
    if (g == 0)
      *reinterpret_cast<float2*>(sums + (n * 8 + warp) * 8 + 2 * c) =
          make_float2(total[0], total[1]);
  }
  __syncthreads();
#pragma unroll
  for (int n = 0; n < kGroups; ++n) {
    float2 total = make_float2(0.f, 0.f);
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      const float2 part = *reinterpret_cast<const float2*>(sums + (n * 8 + w) * 8 + 2 * c);
      total.x += part.x;
      total.y += part.y;
    }
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int column = 8 * n + 2 * c + e;
      const int query = q_first + column / p.group;
      const int q_head = head * p.group + column % p.group;
      const float denominator = e ? total.y : total.x;
      if (query < q_len) {
        if constexpr (kSplit) {
          const int64_t row =
              ((int64_t)(q_begin + query) * p.kv_heads * p.group + q_head) * p.num_splits + split;
#pragma unroll
          for (int j = 0; j < 2; ++j)
            p.opart[row * 128 + 16 * warp + 8 * j + g] =
                denominator > 0.f ? output[n][2 * j + e] : 0.f;
          // Every dimension writer sees the same global max and sum. Publish
          // each pair once; empty and fully masked chunks overwrite stale data.
          if (warp == 0 && g == 0) {
            p.ml[row * 2] = denominator > 0.f ? running_max[n][e] : -INFINITY;
            p.ml[row * 2 + 1] = denominator > 0.f ? denominator : 0.f;
          }
        } else {
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            const float result = denominator > 0.f ? output[n][2 * j + e] / denominator : 0.f;
            p.out[(int64_t)(q_begin + query) * p.out_stride + q_head * 128 +
                  16 * warp + 8 * j + g] = __float2bfloat16_rn(result);
          }
        }
      }
    }
  }
  // All waits and consumers are done before retiring the two shared barriers.
  __syncthreads();
  if (tid == 0) {
    invalidate_bar(&barriers[0]);
    invalidate_bar(&barriers[1]);
  }
}

// One CTA per (packed query row, query head). All 128 lanes independently
// combine the tiny (<=32) scale/sum list and coalesced numerator columns.
// Empty splits are skipped before accessing their numerator; 0 * NaN must
// never make a stale or poisoned empty buffer observable in the final output.
__global__ void combine_prefill(const float* opart, const float* ml,
                                __nv_bfloat16* out, int64_t out_stride,
                                int heads, int splits) {
  const int query = blockIdx.x, head = blockIdx.y, dim = threadIdx.x;
  const int64_t first = ((int64_t)query * heads + head) * splits;
  float common_max = -INFINITY;
  for (int s = 0; s < splits; ++s) {
    const int64_t row = first + s;
    if (ml[row * 2 + 1] > 0.f) common_max = fmaxf(common_max, ml[row * 2]);
  }
  float numerator = 0.f, denominator = 0.f;
  if (common_max != -INFINITY) {
    for (int s = 0; s < splits; ++s) {
      const int64_t row = first + s;
      const float sum = ml[row * 2 + 1];
      const float maximum = ml[row * 2];
      if (sum > 0.f && maximum != -INFINITY) {
        const float weight = exp2f(maximum - common_max);
        numerator += weight * opart[row * 128 + dim];
        denominator += weight * sum;
      }
    }
  }
  out[(int64_t)query * out_stride + head * 128 + dim] =
      __float2bfloat16_rn(denominator > 0.f ? numerator / denominator : 0.f);
}

}  // namespace

void fused_prefill(torch::Tensor q, torch::Tensor rec_addr, torch::Tensor rec_len,
                   torch::Tensor plut, torch::Tensor slut, torch::Tensor block_table,
                   torch::Tensor cu_seqlens_q, torch::Tensor seq_lens, torch::Tensor out,
                   std::optional<torch::Tensor> opart, std::optional<torch::Tensor> ml,
                   int64_t num_splits, int64_t max_seqlen_q, double scale) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16,
              "q must be a CUDA BF16 tensor");
  auto check = [&](const torch::Tensor& tensor, torch::ScalarType dtype, const char* name) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == q.device() && tensor.scalar_type() == dtype,
                name, " must have the expected dtype on q's CUDA device");
  };
  check(out, torch::kBFloat16, "out");
  check(rec_addr, torch::kInt64, "rec_addr");
  check(rec_len, torch::kInt32, "rec_len");
  check(plut, torch::kInt32, "plut");
  check(slut, torch::kUInt8, "slut");
  check(block_table, torch::kInt32, "block_table");
  check(cu_seqlens_q, torch::kInt32, "cu_seqlens_q");
  check(seq_lens, torch::kInt32, "seq_lens");
  TORCH_CHECK(q.dim() == 3 && q.size(2) == 128 && q.stride(2) == 1 &&
                  q.stride(1) == 128 && q.stride(0) >= q.size(1) * 128 && q.stride(0) % 2 == 0,
              "q must be [totalQ, heads, 128] with contiguous heads");
  // MMA input loading reads pairs of BF16 values. A contiguous-looking view
  // can still begin at an odd BF16 storage offset and violate u32 alignment.
  TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr()) % alignof(uint32_t) == 0,
              "q data must be 4-byte aligned");
  TORCH_CHECK(out.sizes() == q.sizes() && out.stride(2) == 1 && out.stride(1) == 128 &&
                  out.stride(0) >= out.size(1) * 128,
              "out must match q with contiguous heads");
  TORCH_CHECK(rec_addr.dim() == 3 && rec_addr.size(1) == 2 && rec_addr.size(2) > 0 &&
                  rec_addr.is_contiguous() && rec_len.is_contiguous() && rec_len.sizes() == rec_addr.sizes(),
              "record tables must be contiguous [blocks, 2, kv_heads]");
  const int heads = rec_addr.size(2);
  const int group = q.size(1) / heads;
  TORCH_CHECK(q.size(1) == heads * group && (group == 1 || group == 2 || group == 4 || group == 8),
              "prefill requires GQA group 1, 2, 4, or 8");
  TORCH_CHECK(plut.dim() == 3 && plut.size(0) == 2 && plut.size(1) == heads && plut.size(2) == 2 &&
                  plut.is_contiguous() && slut.dim() == 3 && slut.size(0) == 2 &&
                  slut.size(1) == heads && slut.size(2) == 16 && slut.is_contiguous(),
              "codec tables must be contiguous plut[2,H,2] and slut[2,H,16]");
  TORCH_CHECK(seq_lens.dim() == 1 && seq_lens.is_contiguous() &&
                  cu_seqlens_q.dim() == 1 && cu_seqlens_q.is_contiguous() &&
                  cu_seqlens_q.numel() == seq_lens.numel() + 1,
              "sequence metadata must be contiguous seq_lens[B] and cu_seqlens_q[B+1]");
  const int batch = seq_lens.numel();
  TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) == batch && block_table.stride(1) == 1,
              "block_table must have B rows with contiguous columns");
  TORCH_CHECK(std::isfinite(scale), "scale must be finite");
  TORCH_CHECK(num_splits >= 1 && num_splits <= 32,
              "num_splits must be between 1 and 32");
  if (num_splits > 1) {
    TORCH_CHECK(opart.has_value() && ml.has_value(),
                "split prefill requires caller-owned opart and ml buffers");
    check(*opart, torch::kFloat32, "opart");
    check(*ml, torch::kFloat32, "ml");
    TORCH_CHECK(opart->is_contiguous() && ml->is_contiguous(),
                "prefill partial buffers must be contiguous");
    const int64_t rows = q.size(0) * q.size(1) * num_splits;
    TORCH_CHECK(opart->numel() >= rows * 128 && ml->numel() >= rows * 2,
                "prefill partial buffers are too small for [Q,Hq,S,128] and [Q,Hq,S,2]");
  }
  if (q.size(0) == 0) return;
  TORCH_CHECK(batch > 0 && max_seqlen_q > 0 && max_seqlen_q <= q.size(0),
              "nonempty prefill requires a positive batch and valid max_seqlen_q");
  const c10::cuda::CUDAGuard device_guard(q.device());
  TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major == 9,
              "FragPack fused prefill currently requires SM90");
  Params p;
  p.q = reinterpret_cast<const uint32_t*>(q.data_ptr());
  p.rec_addr = rec_addr.data_ptr<int64_t>();
  p.rec_len = rec_len.data_ptr<int32_t>();
  p.plut = reinterpret_cast<const uint32_t*>(plut.data_ptr());
  p.slut = slut.data_ptr<uint8_t>();
  p.block_table = block_table.data_ptr<int32_t>();
  p.cu_q = cu_seqlens_q.data_ptr<int32_t>();
  p.seq_lens = seq_lens.data_ptr<int32_t>();
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  // S=1 neither allocates nor dereferences partial buffers and accepts None.
  p.opart = num_splits > 1 ? opart->data_ptr<float>() : nullptr;
  p.ml = num_splits > 1 ? ml->data_ptr<float>() : nullptr;
  p.num_splits = num_splits;
  p.q_stride = q.stride(0) / 2;
  p.out_stride = out.stride(0);
  p.bt_stride = block_table.stride(0);
  p.kv_heads = heads;
  p.group = group;
  p.scale_log2 = static_cast<float>(scale * 1.4426950408889634);
  const int tile_q = kColumns / group;
  const int query_tiles = (max_seqlen_q + tile_q - 1) / tile_q;
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (num_splits == 1) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(prefill_attn<false>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    prefill_attn<false><<<dim3(query_tiles, batch, heads), kThreads, kSmem, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    C10_CUDA_CHECK(cudaFuncSetAttribute(prefill_attn<true>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    prefill_attn<true><<<dim3(query_tiles, batch, heads * num_splits),
                         kThreads, kSmem, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    combine_prefill<<<dim3(q.size(0), q.size(1)), 128, 0, stream>>>(
        p.opart, p.ml, p.out, p.out_stride, q.size(1), num_splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}
