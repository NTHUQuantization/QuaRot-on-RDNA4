#pragma once

#include <torch/extension.h>

torch::Tensor rms_norm_rows(
    torch::Tensor input, int64_t mean_dim, double eps);

std::vector<torch::Tensor> rms_norm_quant_i4_rows(
    torch::Tensor input, int64_t mean_dim, double eps);

std::vector<torch::Tensor> rms_norm_quant_i4_rows_clipped(
    torch::Tensor input, int64_t mean_dim, double eps, double clip_ratio);

std::vector<torch::Tensor> residual_rms_norm_quant_i4_rows(
    torch::Tensor input, torch::Tensor residual, int64_t mean_dim,
    double eps, double clip_ratio);

void fused_append_kv_i4(
    torch::Tensor kv_data, torch::Tensor kv_param, torch::Tensor kv_indptr,
    torch::Tensor kv_indices, torch::Tensor last_page_offset, torch::Tensor key,
    torch::Tensor value, int64_t num_layers, int64_t layer_idx,
    int64_t num_heads, int64_t page_size, int64_t batch_size);
torch::Tensor fused_rope_append_kv_i4(
    torch::Tensor query, torch::Tensor key, torch::Tensor value,
    torch::Tensor cos, torch::Tensor sin, torch::Tensor kv_data,
    torch::Tensor kv_param, torch::Tensor kv_indptr, torch::Tensor kv_indices,
    torch::Tensor last_page_offset, int64_t num_layers, int64_t layer_idx,
    int64_t page_size);

std::vector<torch::Tensor> fused_attention_hadamard_quant(
    torch::Tensor attention, int64_t num_heads);

std::vector<torch::Tensor> fused_rmsnorm_quant_i4(
    torch::Tensor input, double eps, double clip_ratio);

std::vector<torch::Tensor> fused_attention_hadamard_quant_general(
    torch::Tensor attention, int64_t num_heads, torch::Tensor matrix);

std::vector<torch::Tensor> fused_ffn_silu_hadamard_quant_grouped256(
    torch::Tensor gate, torch::Tensor up);


std::vector<torch::Tensor> fused_ffn_silu_hadamard_quant(
    torch::Tensor gate, torch::Tensor up);

std::vector<torch::Tensor> fused_ffn_silu_hadamard_quant_single_fp16lds(
    torch::Tensor gate, torch::Tensor up, torch::Tensor matrix);

std::vector<torch::Tensor> fused_ffn_silu_hadamard_quant_general(
    torch::Tensor gate, torch::Tensor up, torch::Tensor hadamard);
