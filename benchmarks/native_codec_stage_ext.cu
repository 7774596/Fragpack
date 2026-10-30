// SPDX-License-Identifier: Apache-2.0
// Diagnostic-only extension. Include the actual production implementation so
// stage kernels, checks and compiler options do not drift into a proxy model.
// Rename exported host functions to avoid ELF symbol interposition with the
// separately loaded production extension; kernels have translation-local names.
#define encode_records profile_encode_records
#define plan_blobs profile_plan_blobs
#define pack_blobs profile_pack_blobs
#define decode_blobs profile_decode_blobs
#define raw_gather profile_raw_gather
#define raw_scatter profile_raw_scatter
#include "../fragpack/ops/native_codec.cu"
#undef encode_records
#undef plan_blobs
#undef pack_blobs
#undef decode_blobs
#undef raw_gather
#undef raw_scatter

namespace {
// Individual record/decode measurements require a successful complete restore
// first and immutable valid inputs thereafter. This interface is not a codec
// API: bypassing validation is only an explicitly labelled diagnostic ablation.
void profile_stage(int stage, torch::Tensor desc, torch::Tensor page_map,
                   torch::Tensor plut, torch::Tensor slut, torch::Tensor src,
                   torch::Tensor offsets, torch::Tensor rec_offsets,
                   torch::Tensor status, int64_t page_tokens, int64_t heads,
                   int64_t digest) {
  TORCH_CHECK(stage >= 0 && stage <= 4, "invalid diagnostic stage");
  const Pages p = check_pages(desc, page_map, status, page_tokens, heads);
  const c10::cuda::CUDAGuard guard(desc.device());
  if (!p.N) return;
  const auto stream = at::cuda::getCurrentCUDAStream();
  switch (stage) {
    case 0:
    case 1:
      validate_mapping<<<p.N, kThreads, 0, stream>>>(p, status.data_ptr<int32_t>(),
                                                   stage == 1);
      break;
    case 2:
      validate_headers<<<p.N, kThreads, 0, stream>>>(
          p, src.data_ptr<uint8_t>(), src.numel(), offsets.data_ptr<int64_t>(),
          rec_offsets.data_ptr<int64_t>(), status.data_ptr<int32_t>(), digest);
      break;
    case 3:
      validate_records<<<dim3(p.N, 2 * p.L * p.H), kThreads, 0, stream>>>(
          src.data_ptr<uint8_t>(), offsets.data_ptr<int64_t>(),
          rec_offsets.data_ptr<int64_t>(), status.data_ptr<int32_t>(), 2 * p.L * p.H);
      break;
    case 4:
      // This profiler is hash-locked to the baseline static-shared-memory
      // decoder. The Python entry point rejects other production CUDA sources.
      decode_kernel<<<dim3(p.N, 2 * p.L * p.H), kThreads, 0, stream>>>(
          p, plut.data_ptr<int32_t>(), slut.data_ptr<uint8_t>(), src.data_ptr<uint8_t>(),
          offsets.data_ptr<int64_t>(), rec_offsets.data_ptr<int64_t>(),
          status.data_ptr<int32_t>());
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("stage", &profile_stage);
  m.def("full_restore", &profile_decode_blobs);
}
