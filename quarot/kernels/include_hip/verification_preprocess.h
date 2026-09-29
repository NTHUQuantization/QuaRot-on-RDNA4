#pragma once

#include <torch/extension.h>

torch::Tensor hadamard_h128(torch::Tensor input);

torch::Tensor chunk_q_norm_rope_hadamard(
    torch::Tensor query, torch::Tensor norm_weight, torch::Tensor cos,
    torch::Tensor sin, double eps);

void chunk_k_norm_rope_append_i4(
    torch::Tensor key, torch::Tensor value, torch::Tensor norm_weight,
    torch::Tensor cos, torch::Tensor sin, torch::Tensor kv_data,
    torch::Tensor kv_param, torch::Tensor kv_indptr, torch::Tensor kv_indices,
    torch::Tensor last_page_offset, int64_t layer_idx, double eps);
