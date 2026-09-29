#pragma once
#include <torch/extension.h>
void verification_metadata(
    torch::Tensor base, torch::Tensor positions,
    torch::Tensor append_indptr, torch::Tensor append_indices,
    torch::Tensor append_offsets, torch::Tensor causal_indptr,
    torch::Tensor causal_indices, torch::Tensor causal_offsets,
    int64_t batch, int64_t page_size, int64_t max_pages);
