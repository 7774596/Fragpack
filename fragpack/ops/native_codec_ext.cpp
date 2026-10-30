// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>

void encode_records(torch::Tensor desc, torch::Tensor page_map, torch::Tensor rank_of,
                    torch::Tensor records, torch::Tensor rec_lens, torch::Tensor status,
                    int64_t page_tokens, int64_t heads);
void plan_blobs(torch::Tensor rec_lens, torch::Tensor status, torch::Tensor offsets,
                torch::Tensor lengths, torch::Tensor rec_offsets, int64_t layers, int64_t heads);
void pack_blobs(torch::Tensor records, torch::Tensor rec_lens, torch::Tensor status,
                torch::Tensor offsets, torch::Tensor lengths, torch::Tensor rec_offsets,
                torch::Tensor out, int64_t layers, int64_t heads, int64_t digest);
void decode_blobs(torch::Tensor desc, torch::Tensor page_map, torch::Tensor plut,
                  torch::Tensor slut, torch::Tensor src, torch::Tensor offsets,
                  torch::Tensor rec_offsets, torch::Tensor status, int64_t page_tokens,
                  int64_t heads, int64_t digest);
void raw_gather(torch::Tensor desc, torch::Tensor page_map, torch::Tensor out,
                torch::Tensor status, int64_t page_tokens, int64_t heads);
void raw_scatter(torch::Tensor desc, torch::Tensor page_map, torch::Tensor src,
                 torch::Tensor status, int64_t page_tokens, int64_t heads);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("encode_records", &encode_records);
  m.def("plan_blobs", &plan_blobs);
  m.def("pack_blobs", &pack_blobs);
  m.def("decode_blobs", &decode_blobs);
  m.def("raw_gather", &raw_gather);
  m.def("raw_scatter", &raw_scatter);
}
