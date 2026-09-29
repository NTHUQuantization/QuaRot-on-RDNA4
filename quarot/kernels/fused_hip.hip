#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <vector>
#include <type_traits>

#include <torch/extension.h>

#include <fused.h>
#include <hadamard_wmma.cuh>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxFusedWidth = 8192;
constexpr int kMaxGeneralFfnWidth = 14336;
// Qwen3-32B uses a 25 * 1024 transform. A float row needs 100 KiB,
// so widths above the single-block LDS limit use a tiled pipeline.
constexpr int kMaxLargeGeneralFfnWidth = 28672;

__device__ __forceinline__ uint8_t pack_s4_pair(int lo, int hi) {
  // QuaRot GEMM consumes two's-complement signed nibbles, low element first.
  return static_cast<uint8_t>((lo & 0x0f) | ((hi & 0x0f) << 4));
}

__device__ __forceinline__ uint8_t pack_u4_pair(int lo, int hi) {
  return static_cast<uint8_t>((lo & 0x0f) | ((hi & 0x0f) << 4));
}

__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float reduce_max(float value, float* scratch) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
    __syncthreads();
  }
  return scratch[0];
}

__global__ void rms_norm_rows_kernel(
    const half* input, half* output, int rows, int width, float denominator,
    float eps) {
  __shared__ float scratch[kThreads];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  float local_sum = 0.0f;
  for (int column = tid; column < width; column += blockDim.x) {
    const float value = __half2float(input[row * width + column]);
    local_sum += value * value;
  }
  scratch[tid] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) scratch[tid] += scratch[tid + stride];
    __syncthreads();
  }
  const float inverse = rsqrtf(scratch[0] / denominator + eps);
  for (int column = tid; column < width; column += blockDim.x) {
    output[row * width + column] =
        __float2half(__half2float(input[row * width + column]) * inverse);
  }
}
template <bool ClipScale, bool AddResidual>
__global__ void rms_norm_quant_i4_rows_kernel(
    const half* input, const half* residual, half* residual_output,
    uint8_t* packed, half* scales, int rows, int width,
    float denominator, float eps, float clip_ratio) {
  __shared__ float reduce_scratch[kThreads];
  extern __shared__ half normalized[];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;

  float local_sum = 0.0f;
  for (int column = tid; column < width; column += blockDim.x) {
    half input_h = input[row * width + column];
    if (AddResidual) {
      // torch.add(FP16, FP16) materializes a rounded FP16 residual before
      // RMSNorm. Preserve that boundary even though the launches are fused.
      input_h = __float2half(__half2float(input_h) +
          __half2float(residual[row * width + column]));
      residual_output[row * width + column] = input_h;
      normalized[column] = input_h;
    }
    const float value = __half2float(input_h);
    local_sum += value * value;
  }
  reduce_scratch[tid] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) reduce_scratch[tid] += reduce_scratch[tid + stride];
    __syncthreads();
  }

  const float inverse = rsqrtf(reduce_scratch[0] / denominator + eps);
  for (int column = tid; column < width; column += blockDim.x) {
    normalized[column] = __float2half(
        __half2float(AddResidual ? normalized[column]
                               : input[row * width + column]) * inverse);
  }
  __syncthreads();

  float local_max = 0.0f;
  for (int column = tid; column < width; column += blockDim.x) {
    local_max = fmaxf(local_max, fabsf(__half2float(normalized[column])));
  }
  reduce_scratch[tid] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce_scratch[tid] =
          fmaxf(reduce_scratch[tid], reduce_scratch[tid + stride]);
    }
    __syncthreads();
  }
  half scale_h = __hdiv(__float2half(reduce_scratch[0]),
                       __float2half(7.0f));
  if (ClipScale) {
    // Quantizer computes FP16 max/7, then scalar clipping, then FP16's tiny
    // clamp. Applying clipping in FP32 before the division's FP16 round is
    // observably different at INT4 thresholds.
    scale_h = __float2half(__half2float(scale_h) * clip_ratio);
    scale_h = __float2half(fmaxf(__half2float(scale_h), 0x1.0p-14f));
  }
  if (tid == 0) scales[row] = scale_h;

  for (int pair = tid; pair < width / 2; pair += blockDim.x) {
    const half low_h = __hdiv(normalized[pair * 2], scale_h);
    const half high_h = __hdiv(normalized[pair * 2 + 1], scale_h);
    const int low = max(-8, min(7, __float2int_rn(__half2float(low_h))));
    const int high = max(-8, min(7, __float2int_rn(__half2float(high_h))));
    packed[row * (width / 2) + pair] = pack_s4_pair(low, high);
  }
}


__global__ void fused_rmsnorm_quant_i4_kernel(
    const half* input, uint8_t* packed, half* scales, int rows, int cols,
    float eps, float clip_ratio) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  __shared__ float sum_scratch[kThreads];
  __shared__ float max_scratch[kThreads];
  float sum = 0.0f;
  float max_abs = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = __half2float(input[row * cols + col]);
    sum = fmaf(value, value, sum);
    max_abs = fmaxf(max_abs, fabsf(value));
  }
  sum_scratch[threadIdx.x] = sum;
  max_scratch[threadIdx.x] = max_abs;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sum_scratch[threadIdx.x] += sum_scratch[threadIdx.x + stride];
      max_scratch[threadIdx.x] = fmaxf(
          max_scratch[threadIdx.x], max_scratch[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float inv_rms = rsqrtf(sum_scratch[0] / cols + eps);
  const float scale = fmaxf(
      max_scratch[0] * inv_rms * clip_ratio / 7.0f, 0x1.0p-14f);
  if (threadIdx.x == 0) scales[row] = __float2half(scale);
  for (int pair = threadIdx.x; pair < cols / 2; pair += blockDim.x) {
    const int col = pair * 2;
    int lo = __float2int_rn(
        __half2float(input[row * cols + col]) * inv_rms / scale);
    int hi = __float2int_rn(
        __half2float(input[row * cols + col + 1]) * inv_rms / scale);
    lo = max(-8, min(7, lo));
    hi = max(-8, min(7, hi));
    packed[row * (cols / 2) + pair] = pack_s4_pair(lo, hi);
  }
}

// Adapted from third-party/hadacore's wave-register path. gfx12 executes
// wave32: H32 is computed with shuffles, while H64/H128/H256 keep 2/4/8
// values per lane and finish the remaining butterfly stages in registers.
__device__ __forceinline__ float hadacore_shfl_xor_f32(float value, int mask) {
  return __shfl_xor_sync(0xffffffffULL, value, mask, 32);
}

template <int Regs>
__device__ __forceinline__ void hadacore_warp_hadamard_regs(
    float* values, int lane) {
  for (int mask = 1; mask < 32; mask <<= 1) {
#pragma unroll
    for (int r = 0; r < Regs; ++r) {
      const float a = values[r];
      const float b = hadacore_shfl_xor_f32(a, mask);
      values[r] = (lane & mask) == 0 ? a + b : b - a;
    }
  }

  // The wave shuffle computes H32 for every register column. Complete H64,
  // H128, or H256 by applying the remaining butterfly stages locally across
  // the 2, 4, or 8 values owned by each lane.
#pragma unroll
  for (int stride = 1; stride < Regs; stride <<= 1) {
#pragma unroll
    for (int base = 0; base < Regs; base += stride << 1) {
#pragma unroll
      for (int offset = 0; offset < stride; ++offset) {
        const float a = values[base + offset];
        const float b = values[base + offset + stride];
        values[base + offset] = a + b;
        values[base + offset + stride] = a - b;
      }
    }
  }
}

__device__ __forceinline__ float hadacore_warp_reduce_max(float value) {
  for (int mask = 16; mask > 0; mask >>= 1) {
    value = fmaxf(value, hadacore_shfl_xor_f32(value, mask));
  }
  return value;
}

__device__ __forceinline__ float hadacore_warp_reduce_min(float value) {
  for (int mask = 16; mask > 0; mask >>= 1) {
    value = fminf(value, hadacore_shfl_xor_f32(value, mask));
  }
  return value;
}

template <int Inner>
__device__ __forceinline__ void hadacore_warp_hadamard_group(
    float* values, int group, int lane) {
  static_assert(Inner >= 32 && Inner <= 256 && (Inner & (Inner - 1)) == 0,
                "warp Hadamard supports H32 through H256");
  constexpr int kRegs = Inner / 32;
  float registers[kRegs];
#pragma unroll
  for (int r = 0; r < kRegs; ++r) {
    registers[r] = values[group * Inner + r * 32 + lane];
  }
  hadacore_warp_hadamard_regs<kRegs>(registers, lane);
#pragma unroll
  for (int r = 0; r < kRegs; ++r) {
    values[group * Inner + r * 32 + lane] = registers[r];
  }
}

__device__ __forceinline__ void hadamard(float* values, int width) {
  for (int stride = 1; stride < width; stride <<= 1) {
    for (int pair = threadIdx.x; pair < width / 2; pair += blockDim.x) {
      int low = ((pair / stride) * (stride << 1)) + (pair & (stride - 1));
      float a = values[low];
      float b = values[low + stride];
      values[low] = a + b;
      values[low + stride] = a - b;
    }
    __syncthreads();
  }
}

template <int Width>
__device__ __forceinline__ void hadamard_fixed(float* values) {
#pragma unroll
  for (int stride = 1; stride < Width; stride <<= 1) {
    for (int pair = threadIdx.x; pair < Width / 2; pair += blockDim.x) {
      const int low = ((pair / stride) * (stride << 1)) +
                      (pair & (stride - 1));
      const float a = values[low];
      const float b = values[low + stride];
      values[low] = a + b;
      values[low + stride] = a - b;
    }
    __syncthreads();
  }
}

template <int Inner>
__device__ __forceinline__ void hadamard_grouped_fixed(
    float* values, int width) {
#pragma unroll
  for (int stride = 1; stride < Inner; stride <<= 1) {
    constexpr int pairs_per_group = Inner / 2;
    for (int pair = threadIdx.x; pair < width / 2; pair += blockDim.x) {
      const int group = pair / pairs_per_group;
      const int local_pair = pair - group * pairs_per_group;
      const int low = group * Inner +
          (local_pair / stride) * (stride << 1) +
          (local_pair & (stride - 1));
      const float a = values[low];
      const float b = values[low + stride];
      values[low] = a + b;
      values[low + stride] = a - b;
    }
    __syncthreads();
  }
}

template <int HeadDim>
__global__ void fused_append_kv_i4_kernel(
    const half* key, const half* value, uint8_t* kv_data, half2* kv_param,
    const int32_t* kv_indptr, const int32_t* kv_indices,
    const int32_t* last_page_offset, int num_layers, int layer_idx,
    int num_heads, int page_size, int batch_size, int added_length) {
  static_assert(HeadDim == 64 || HeadDim == 128,
                "fused KV append supports head dimensions 64 and 128");
  constexpr int kRegs = HeadDim / 32;
  const int lane = threadIdx.x;
  const int row = blockIdx.x;
  const int batch = row / added_length;
  const int token = row - batch * added_length;
  const int head = blockIdx.y;
  if (batch >= batch_size || head >= num_heads) return;

  const int seq_len = (kv_indptr[batch + 1] - kv_indptr[batch] - 1) * page_size + last_page_offset[batch];
  const int position = seq_len - added_length + token;
  if (position < 0) return;
  const int page =
      kv_indices[kv_indptr[batch] + position / page_size];
  const int slot = position % page_size;
  const int input_base = (row * num_heads + head) * HeadDim;
  float k_values[kRegs];
  float v_values[kRegs];
#pragma unroll
  for (int r = 0; r < kRegs; ++r) {
    const int i = r * 32 + lane;
    k_values[r] = __half2float(key[input_base + i]);
    v_values[r] = __half2float(value[input_base + i]);
  }

  hadacore_warp_hadamard_regs<kRegs>(k_values, lane);
  const float norm = rsqrtf(float(HeadDim));
  float kmax = -INFINITY;
  float kmin = INFINITY;
#pragma unroll
  for (int r = 0; r < kRegs; ++r) {
    k_values[r] *= norm;
    kmax = fmaxf(kmax, k_values[r]);
    kmin = fminf(kmin, k_values[r]);
  }
  kmax = hadacore_warp_reduce_max(kmax);
  kmin = hadacore_warp_reduce_min(kmin);
  half kscale_h = __float2half(fmaxf((kmax - kmin) / 15.0f, 1.0e-5f));
  half kzero_h = __float2half(-kmin);
  float kscale = __half2float(kscale_h);
  float kzero = __half2float(kzero_h);
  const size_t k_data_base =
      (((static_cast<size_t>(page) * num_layers + layer_idx) * 2 *
        num_heads + head) * page_size + slot) * (HeadDim / 2);
#pragma unroll
  for (int r = 0; r < kRegs; ++r) {
    const float neighbor = hadacore_shfl_xor_f32(k_values[r], 1);
    if ((lane & 1) == 0) {
      const int q0 = max(0, min(15, static_cast<int>(
          nearbyintf((k_values[r] + kzero) / kscale))));
      const int q1 = max(0, min(15, static_cast<int>(
          nearbyintf((neighbor + kzero) / kscale))));
      kv_data[k_data_base + ((r * 32 + lane) >> 1)] =
          pack_u4_pair(q0, q1);
    }
  }
  if (lane == 0) {
    size_t index = ((static_cast<size_t>(page) * num_layers + layer_idx) * 2 * num_heads + head) * page_size + slot;
    kv_param[index] = __halves2half2(kscale_h, kzero_h);
  }

  // QuaRot rotates Q/K inside each attention head, but V remains in its
  // original basis. This must match the prefill/chunk asym-quant path.
  float vmax = -INFINITY;
  float vmin = INFINITY;
#pragma unroll
  for (int r = 0; r < kRegs; ++r) {
    vmax = fmaxf(vmax, v_values[r]);
    vmin = fminf(vmin, v_values[r]);
  }
  vmax = hadacore_warp_reduce_max(vmax);
  vmin = hadacore_warp_reduce_min(vmin);
  half vscale_h = __float2half(fmaxf((vmax - vmin) / 15.0f, 1.0e-5f));
  half vzero_h = __float2half(-vmin);
  float vscale = __half2float(vscale_h);
  float vzero = __half2float(vzero_h);
  const size_t v_data_base =
      ((((static_cast<size_t>(page) * num_layers + layer_idx) * 2 + 1) *
        num_heads + head) * page_size + slot) * (HeadDim / 2);
#pragma unroll
  for (int r = 0; r < kRegs; ++r) {
    const float neighbor = hadacore_shfl_xor_f32(v_values[r], 1);
    if ((lane & 1) == 0) {
      const int q0 = max(0, min(15, static_cast<int>(
          nearbyintf((v_values[r] + vzero) / vscale))));
      const int q1 = max(0, min(15, static_cast<int>(
          nearbyintf((neighbor + vzero) / vscale))));
      kv_data[v_data_base + ((r * 32 + lane) >> 1)] =
          pack_u4_pair(q0, q1);
    }
  }
  if (lane == 0) {
    size_t index = (((static_cast<size_t>(page) * num_layers + layer_idx) * 2 + 1) * num_heads + head) * page_size + slot;
    kv_param[index] = __halves2half2(vscale_h, vzero_h);
  }
}

template <int HeadDim>
__global__ void fused_rope_append_kv_i4_kernel(
    const half* query, const half* key, const half* value,
    const half* cos, const half* sin, half* query_out,
    uint8_t* kv_data, half2* kv_param,
    const int32_t* kv_indptr, const int32_t* kv_indices,
    const int32_t* last_page_offset, int num_layers, int layer_idx,
    int num_query_heads, int num_kv_heads, int page_size, int batch_size) {
  static_assert(HeadDim == 64 || HeadDim == 128,
                "fused RoPE/KV append supports head dimensions 64 and 128");
  constexpr int kRegs = HeadDim / 32;
  const int lane = threadIdx.x;
  const int batch = blockIdx.x;
  const int head = blockIdx.y;
  if (batch >= batch_size || head >= num_query_heads) return;

  const int half_dim = HeadDim / 2;
  const int q_base = (batch * num_query_heads + head) * HeadDim;
#pragma unroll
  for (int r = 0; r < kRegs; ++r) {
    const int i = r * 32 + lane;
    const int partner = i < half_dim ? i + half_dim : i - half_dim;
    const int trig = batch * HeadDim + i;
    const half q = query[q_base + i];
    const half qr = i < half_dim ? __hneg(query[q_base + partner])
                                 : query[q_base + partner];
    // Plain __hmul/__hadd may contract to a half FMA under -O3. Match
    // eager RoPE and chunk preprocessing: round both products, then the sum.
    query_out[q_base + i] = __hadd_rn(
        __hmul_rn(q, cos[trig]), __hmul_rn(qr, sin[trig]));
  }

  // Only the first num_kv_heads blocks own K/V cache entries. This branch is
  // uniform across the block, so blocks for the remaining GQA query heads can
  // finish after writing rotated Q without entering any barrier.
  if (head >= num_kv_heads) return;

  const int seq_len =
      (kv_indptr[batch + 1] - kv_indptr[batch] - 1) * page_size +
      last_page_offset[batch];
  if (seq_len <= 0) return;
  const int page =
      kv_indices[kv_indptr[batch] + (seq_len - 1) / page_size];
  const int slot = (seq_len - 1) % page_size;
  const int kv_base = (batch * num_kv_heads + head) * HeadDim;
  float k_values[kRegs];
  float v_values[kRegs];
#pragma unroll
  for (int r = 0; r < kRegs; ++r) {
    const int i = r * 32 + lane;
    const int partner = i < half_dim ? i + half_dim : i - half_dim;
    const int trig = batch * HeadDim + i;
    const half k = key[kv_base + i];
    const half kr = i < half_dim ? __hneg(key[kv_base + partner])
                                 : key[kv_base + partner];
    // Preserve the old two-kernel numerical boundary: round RoPE output to
    // FP16 before converting to FP32 for Hadamard and cache quantization.
    const half rotated_k = __hadd_rn(
        __hmul_rn(k, cos[trig]), __hmul_rn(kr, sin[trig]));
    k_values[r] = __half2float(rotated_k);
    v_values[r] = __half2float(value[kv_base + i]);
  }

  hadacore_warp_hadamard_regs<kRegs>(k_values, lane);
  const float norm = rsqrtf(float(HeadDim));
  float kmax = -INFINITY;
  float kmin = INFINITY;
#pragma unroll
  for (int r = 0; r < kRegs; ++r) {
    k_values[r] *= norm;
    kmax = fmaxf(kmax, k_values[r]);
    kmin = fminf(kmin, k_values[r]);
  }
  kmax = hadacore_warp_reduce_max(kmax);
  kmin = hadacore_warp_reduce_min(kmin);
  const half kscale_h =
      __float2half(fmaxf((kmax - kmin) / 15.0f, 1.0e-5f));
  const half kzero_h = __float2half(-kmin);
  const float kscale = __half2float(kscale_h);
  const float kzero = __half2float(kzero_h);
  const size_t k_data_base =
      (((static_cast<size_t>(page) * num_layers + layer_idx) * 2 *
        num_kv_heads + head) * page_size + slot) * (HeadDim / 2);
#pragma unroll
  for (int r = 0; r < kRegs; ++r) {
    const float neighbor = hadacore_shfl_xor_f32(k_values[r], 1);
    if ((lane & 1) == 0) {
      const int q0 = max(0, min(15, static_cast<int>(
          nearbyintf((k_values[r] + kzero) / kscale))));
      const int q1 = max(0, min(15, static_cast<int>(
          nearbyintf((neighbor + kzero) / kscale))));
      kv_data[k_data_base + ((r * 32 + lane) >> 1)] =
          pack_u4_pair(q0, q1);
    }
  }
  if (lane == 0) {
    const size_t index =
        ((static_cast<size_t>(page) * num_layers + layer_idx) * 2 *
         num_kv_heads + head) * page_size + slot;
    kv_param[index] = __halves2half2(kscale_h, kzero_h);
  }

  float vmax = -INFINITY;
  float vmin = INFINITY;
#pragma unroll
  for (int r = 0; r < kRegs; ++r) {
    vmax = fmaxf(vmax, v_values[r]);
    vmin = fminf(vmin, v_values[r]);
  }
  vmax = hadacore_warp_reduce_max(vmax);
  vmin = hadacore_warp_reduce_min(vmin);
  const half vscale_h =
      __float2half(fmaxf((vmax - vmin) / 15.0f, 1.0e-5f));
  const half vzero_h = __float2half(-vmin);
  const float vscale = __half2float(vscale_h);
  const float vzero = __half2float(vzero_h);
  const size_t v_data_base =
      ((((static_cast<size_t>(page) * num_layers + layer_idx) * 2 + 1) *
        num_kv_heads + head) * page_size + slot) * (HeadDim / 2);
#pragma unroll
  for (int r = 0; r < kRegs; ++r) {
    const float neighbor = hadacore_shfl_xor_f32(v_values[r], 1);
    if ((lane & 1) == 0) {
      const int q0 = max(0, min(15, static_cast<int>(
          nearbyintf((v_values[r] + vzero) / vscale))));
      const int q1 = max(0, min(15, static_cast<int>(
          nearbyintf((neighbor + vzero) / vscale))));
      kv_data[v_data_base + ((r * 32 + lane) >> 1)] =
          pack_u4_pair(q0, q1);
    }
  }
  if (lane == 0) {
    const size_t index =
        (((static_cast<size_t>(page) * num_layers + layer_idx) * 2 + 1) *
         num_kv_heads + head) * page_size + slot;
    kv_param[index] = __halves2half2(vscale_h, vzero_h);
  }
}

__global__ void fused_attention_hadamard_quant_kernel(
    const half* attention, uint8_t* packed, half* scales, int rows,
    int num_heads, int head_dim) {
  __shared__ float values[kMaxFusedWidth];
  __shared__ float scratch[kThreads];
  const int row = blockIdx.x;
  const int width = num_heads * head_dim;
  if (row >= rows) return;
  if (num_heads == 32 || num_heads == 64 ||
      num_heads == 128 || num_heads == 256) {
    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int waves = blockDim.x >> 5;
    const int regs = num_heads >> 5;
    for (int d = wave; d < head_dim; d += waves) {
      float registers[8] = {0.0f, 0.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        if (r < regs) {
          const int h = r * 32 + lane;
          registers[r] = __half2float(
              attention[(row * num_heads + h) * head_dim + d]);
        }
      }
      if (num_heads == 32) {
        hadacore_warp_hadamard_regs<1>(registers, lane);
      } else if (num_heads == 64) {
        hadacore_warp_hadamard_regs<2>(registers, lane);
      } else if (num_heads == 128) {
        hadacore_warp_hadamard_regs<4>(registers, lane);
      } else {
        hadacore_warp_hadamard_regs<8>(registers, lane);
      }
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        if (r < regs) values[d * num_heads + r * 32 + lane] = registers[r];
      }
    }
    __syncthreads();
  } else {
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
      const int d = i / num_heads;
      const int h = i % num_heads;
      values[i] = __half2float(attention[(row * num_heads + h) * head_dim + d]);
    }
    __syncthreads();
    for (int d = 0; d < head_dim; ++d) {
      float* group = values + d * num_heads;
      for (int stride = 1; stride < num_heads; stride <<= 1) {
        for (int pair = threadIdx.x; pair < num_heads / 2; pair += blockDim.x) {
          int low = ((pair / stride) * (stride << 1)) + (pair & (stride - 1));
          float a = group[low];
          float b = group[low + stride];
          group[low] = a + b;
          group[low + stride] = a - b;
        }
        __syncthreads();
      }
    }
  }
  const float norm = rsqrtf(float(num_heads));
  float local_max = 0.0f;
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    values[i] *= norm;
    local_max = fmaxf(local_max, fabsf(values[i]));
  }
  half scale_h = __float2half(fmaxf(reduce_max(local_max, scratch) / 7.0f, 6.103515625e-5f));
  float scale = __half2float(scale_h);
  if (threadIdx.x == 0) scales[row] = scale_h;
  for (int pair = threadIdx.x; pair < width / 2; pair += blockDim.x) {
    int out0 = pair << 1;
    int out1 = out0 + 1;
    int h0 = out0 / head_dim, d0 = out0 % head_dim;
    int h1 = out1 / head_dim, d1 = out1 % head_dim;
    int q0 = max(-8, min(7, static_cast<int>(nearbyintf(values[d0 * num_heads + h0] / scale))));
    int q1 = max(-8, min(7, static_cast<int>(nearbyintf(values[d1 * num_heads + h1] / scale))));
    packed[row * (width / 2) + pair] = pack_s4_pair(q0, q1);
  }
}

__global__ void fused_attention_hadamard_quant_general_kernel(
    const half* attention, const half* matrix, uint8_t* packed, half* scales,
    int rows, int num_heads, int head_dim, int remainder, int inner) {
  extern __shared__ float values[];
  __shared__ float scratch[kThreads];
  const int row = blockIdx.x;
  const int width = num_heads * head_dim;
  if (row >= rows) return;
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    const int d = i / num_heads;
    const int h = i - d * num_heads;
    values[i] = __half2float(attention[(row * num_heads + h) * head_dim + d]);
  }
  __syncthreads();
  for (int stride = 1; stride < inner; stride <<= 1) {
    const int pairs_per_group = inner >> 1;
    const int groups = head_dim * remainder;
    for (int pair = threadIdx.x; pair < groups * pairs_per_group;
         pair += blockDim.x) {
      const int group = pair / pairs_per_group;
      const int local = pair - group * pairs_per_group;
      const int low = group * inner + (local / stride) * (stride << 1)
                      + (local & (stride - 1));
      const float a = values[low], b = values[low + stride];
      values[low] = a + b;
      values[low + stride] = a - b;
    }
    __syncthreads();
  }
  constexpr int kOwned = (kMaxFusedWidth / 2 + kThreads - 1) / kThreads;
  float transformed[kOwned * 2];
  float local_max = 0.0f;
  const float norm = rsqrtf(float(num_heads));
  int owned = 0;
  for (int pair = threadIdx.x; pair < width / 2;
       pair += blockDim.x, ++owned) {
#pragma unroll
    for (int which = 0; which < 2; ++which) {
      const int flat = pair * 2 + which;
      const int h = flat / head_dim;
      const int d = flat - h * head_dim;
      const int out_k = h / inner;
      const int p = h - out_k * inner;
      float value = 0.0f;
      for (int in_k = 0; in_k < remainder; ++in_k) {
        value += __half2float(matrix[out_k * remainder + in_k])
                 * values[d * num_heads + in_k * inner + p];
      }
      value *= norm;
      transformed[owned * 2 + which] = value;
      local_max = fmaxf(local_max, fabsf(value));
    }
  }
  half scale_h = __float2half(fmaxf(
      reduce_max(local_max, scratch) / 7.0f, 6.103515625e-5f));
  const float scale = __half2float(scale_h);
  if (threadIdx.x == 0) scales[row] = scale_h;
  owned = 0;
  for (int pair = threadIdx.x; pair < width / 2;
       pair += blockDim.x, ++owned) {
    const int q0 = max(-8, min(7, static_cast<int>(
        nearbyintf(transformed[owned * 2] / scale))));
    const int q1 = max(-8, min(7, static_cast<int>(
        nearbyintf(transformed[owned * 2 + 1] / scale))));
    packed[row * (width / 2) + pair] = pack_s4_pair(q0, q1);
  }
}

template <int GroupSize>
__global__ void fused_ffn_silu_hadamard_quant_grouped_kernel(
    const half* gate, const half* up, uint8_t* packed, half* scales,
    int rows, int width) {
  __shared__ float values[GroupSize];
  __shared__ float scratch[GroupSize];
  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  const int group = blockIdx.y;
  const int base = row * width + group * GroupSize;

  values[tid] = silu(__half2float(gate[base + tid])) *
                __half2float(up[base + tid]);
  __syncthreads();
  hadamard_fixed<GroupSize>(values);
  const float rotated = values[tid] * rsqrtf(static_cast<float>(GroupSize));
  values[tid] = rotated;
  const float scale = fmaxf(
      reduce_max(fabsf(rotated), scratch) / 7.0f, 6.103515625e-5f);
  const half scale_h = __float2half(scale);
  if (tid == 0) scales[row * gridDim.y + group] = scale_h;
  if ((tid & 1) == 0) {
    const int q0 = max(-8, min(7, static_cast<int>(
        nearbyintf(values[tid] / scale))));
    const int q1 = max(-8, min(7, static_cast<int>(
        nearbyintf(values[tid + 1] / scale))));
    packed[(base + tid) >> 1] = pack_s4_pair(q0, q1);
  }
}

__global__ void fused_ffn_silu_hadamard_quant_hadacore256_kernel(
    const half* gate, const half* up, uint8_t* packed, half* scales,
    int rows, int width) {
  __shared__ uint16_t input[256];
  __shared__ uint16_t temporary[256];
  __shared__ uint16_t output[256];
  __shared__ float scratch[32];
  const int lane = threadIdx.x;
  const int row = blockIdx.x;
  const int group = blockIdx.y;
  const int base = row * width + group * 256;
  for (int i = lane; i < 256; i += 32)
    input[i] = quarot_hadamard_wmma::f32_to_bits(
        silu(__half2float(gate[base + i])) * __half2float(up[base + i]));
  __syncthreads();
  quarot_hadamard_wmma::h256(input, temporary, output, lane);
  float local_max = 0.0f;
  for (int i = lane; i < 256; i += 32)
    local_max = fmaxf(local_max, fabsf(quarot_hadamard_wmma::bits_to_f32(output[i])));
  scratch[lane] = local_max;
  __syncthreads();
  for (int stride = 16; stride > 0; stride >>= 1) {
    if (lane < stride) scratch[lane] = fmaxf(scratch[lane], scratch[lane + stride]);
    __syncthreads();
  }
  const float scale = fmaxf(scratch[0] / 7.0f, 6.103515625e-5f);
  const half scale_h = __float2half(scale);
  if (lane == 0) scales[row * gridDim.y + group] = scale_h;
  for (int pair = lane; pair < 128; pair += 32) {
    const float x0 = quarot_hadamard_wmma::bits_to_f32(output[pair * 2]);
    const float x1 = quarot_hadamard_wmma::bits_to_f32(output[pair * 2 + 1]);
    const int q0 = max(-8, min(7, static_cast<int>(nearbyintf(x0 / scale))));
    const int q1 = max(-8, min(7, static_cast<int>(nearbyintf(x1 / scale))));
    packed[(base + pair * 2) >> 1] = pack_s4_pair(q0, q1);
  }
}

// One wave owns one H256 group. Each lane keeps eight strided values in
// registers: wave shuffles compute the H32 component and local butterflies
// finish H8 across the registers. No LDS transpose or workgroup barrier is
// needed for the transform or scale reduction.
__global__ void fused_ffn_silu_hadamard_quant_warp256_kernel(
    const half* gate, const half* up, uint8_t* packed, half* scales,
    int rows, int width) {
  constexpr int kRegs = 8;
  const int lane = threadIdx.x;
  const int row = blockIdx.x;
  const int group = blockIdx.y;
  const int base = row * width + group * 256;
  float values[kRegs];
#pragma unroll
  for (int r = 0; r < kRegs; ++r) {
    const int i = r * 32 + lane;
    values[r] = silu(__half2float(gate[base + i])) *
                __half2float(up[base + i]);
  }
  hadacore_warp_hadamard_regs<kRegs>(values, lane);

  constexpr float kNorm = 0.0625f;
  float local_max = 0.0f;
#pragma unroll
  for (int r = 0; r < kRegs; ++r) {
    values[r] *= kNorm;
    local_max = fmaxf(local_max, fabsf(values[r]));
  }
  const float scale = fmaxf(
      hadacore_warp_reduce_max(local_max) / 7.0f, 6.103515625e-5f);
  if (lane == 0) {
    scales[row * gridDim.y + group] = __float2half(scale);
  }
#pragma unroll
  for (int r = 0; r < kRegs; ++r) {
    const float neighbor = hadacore_shfl_xor_f32(values[r], 1);
    if ((lane & 1) == 0) {
      const int q0 = max(-8, min(7, static_cast<int>(
          nearbyintf(values[r] / scale))));
      const int q1 = max(-8, min(7, static_cast<int>(
          nearbyintf(neighbor / scale))));
      packed[(base + r * 32 + lane) >> 1] = pack_s4_pair(q0, q1);
    }
  }
}

template <int Width>
__global__ void fused_ffn_silu_hadamard_quant_warp_kernel(
    const half* gate, const half* up, uint8_t* packed, half* scales,
    int rows) {
  static_assert(Width >= 32 && Width <= 256 && (Width & (Width - 1)) == 0,
                "warp FFN kernel supports H32 through H256");
  constexpr int kRegs = Width / 32;
  const int lane = threadIdx.x;
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int base = row * Width;
  float values[kRegs];
#pragma unroll
  for (int r = 0; r < kRegs; ++r) {
    const int i = r * 32 + lane;
    values[r] = silu(__half2float(gate[base + i])) *
                __half2float(up[base + i]);
  }
  hadacore_warp_hadamard_regs<kRegs>(values, lane);

  const float norm = rsqrtf(float(Width));
  float local_max = 0.0f;
#pragma unroll
  for (int r = 0; r < kRegs; ++r) {
    values[r] *= norm;
    local_max = fmaxf(local_max, fabsf(values[r]));
  }
  const half scale_h = __float2half(fmaxf(
      hadacore_warp_reduce_max(local_max) / 7.0f, 6.103515625e-5f));
  const float scale = __half2float(scale_h);
  if (lane == 0) scales[row] = scale_h;
#pragma unroll
  for (int r = 0; r < kRegs; ++r) {
    const float neighbor = hadacore_shfl_xor_f32(values[r], 1);
    if ((lane & 1) == 0) {
      const int q0 = max(-8, min(7, static_cast<int>(
          nearbyintf(values[r] / scale))));
      const int q1 = max(-8, min(7, static_cast<int>(
          nearbyintf(neighbor / scale))));
      packed[(base + r * 32 + lane) >> 1] = pack_s4_pair(q0, q1);
    }
  }
}

__global__ void fused_ffn_silu_hadamard_quant_kernel(
    const half* gate, const half* up, uint8_t* packed, half* scales,
    int rows, int width) {
  __shared__ float values[kMaxFusedWidth];
  __shared__ float scratch[kThreads];
  const int row = blockIdx.x;
  if (row >= rows) return;
  for (int i = threadIdx.x; i < width; i += blockDim.x) values[i] = silu(__half2float(gate[row * width + i])) * __half2float(up[row * width + i]);
  __syncthreads();
  hadamard(values, width);
  float local_max = 0.0f;
  const float norm = rsqrtf(float(width));
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    values[i] *= norm;
    local_max = fmaxf(local_max, fabsf(values[i]));
  }
  half scale_h = __float2half(fmaxf(reduce_max(local_max, scratch) / 7.0f, 6.103515625e-5f));
  float scale = __half2float(scale_h);
  if (threadIdx.x == 0) scales[row] = scale_h;
  for (int pair = threadIdx.x; pair < width / 2; pair += blockDim.x) {
    int q0 = max(-8, min(7, static_cast<int>(nearbyintf(values[pair * 2] / scale))));
    int q1 = max(-8, min(7, static_cast<int>(nearbyintf(values[pair * 2 + 1] / scale))));
    packed[row * (width / 2) + pair] = pack_s4_pair(q0, q1);
  }
}

// For width=K*P, QuaRot applies a power-of-two H_P followed by its
// fixed H_K remainder matrix. Recompute H_K for quantization to avoid a second
// width-sized LDS buffer (which exceeds 64 KiB for Llama-2 7B's width 11008).
__global__ void fused_ffn_silu_hadamard_quant_general_kernel(
    const half* gate, const half* up, const half* hadamard, uint8_t* packed,
    half* scales, int rows, int width, int remainder, int inner) {
  extern __shared__ float values[];
  __shared__ float scratch[kThreads];
  const int row = blockIdx.x;
  if (row >= rows) return;
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    values[i] = silu(__half2float(gate[row * width + i])) *
                __half2float(up[row * width + i]);
  }
  __syncthreads();

  if (inner == 32 || inner == 64 || inner == 128 || inner == 256) {
    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int waves = blockDim.x >> 5;
    for (int group = wave; group < remainder; group += waves) {
      if (inner == 32) {
        hadacore_warp_hadamard_group<32>(values, group, lane);
      } else if (inner == 64) {
        hadacore_warp_hadamard_group<64>(values, group, lane);
      } else if (inner == 128) {
        hadacore_warp_hadamard_group<128>(values, group, lane);
      } else {
        hadacore_warp_hadamard_group<256>(values, group, lane);
      }
    }
    __syncthreads();
  } else if (inner == 512) {
    hadamard_grouped_fixed<512>(values, width);
  } else if (inner == 1024) {
    hadamard_grouped_fixed<1024>(values, width);
  } else {
    for (int stride = 1; stride < inner; stride <<= 1) {
      const int pairs_per_group = inner >> 1;
      for (int pair = threadIdx.x; pair < width / 2; pair += blockDim.x) {
        const int group = pair / pairs_per_group;
        const int local_pair = pair - group * pairs_per_group;
        const int low = group * inner +
            (local_pair / stride) * (stride << 1) +
            (local_pair & (stride - 1));
        const float a = values[low];
        const float b = values[low + stride];
        values[low] = a + b;
        values[low + stride] = a - b;
      }
      __syncthreads();
    }
  }

  constexpr int kMaxPairsPerThread =
      (kMaxGeneralFfnWidth / 2 + kThreads - 1) / kThreads;
  float transformed[kMaxPairsPerThread * 2];
  const float norm = rsqrtf(float(width));
  float local_max = 0.0f;
  int owned_pairs = 0;
  for (int pair = threadIdx.x; pair < width / 2;
       pair += blockDim.x, ++owned_pairs) {
#pragma unroll
    for (int which = 0; which < 2; ++which) {
      const int out = pair * 2 + which;
      const int out_k = out / inner;
      const int p = out - out_k * inner;
      float value = 0.0f;
      for (int in_k = 0; in_k < remainder; ++in_k) {
        value += __half2float(hadamard[out_k * remainder + in_k]) *
                 values[in_k * inner + p];
      }
      value *= norm;
      transformed[owned_pairs * 2 + which] = value;
      local_max = fmaxf(local_max, fabsf(value));
    }
  }
  half scale_h = __float2half(fmaxf(reduce_max(local_max, scratch) / 7.0f,
                                    6.103515625e-5f));
  const float scale = __half2float(scale_h);
  if (threadIdx.x == 0) scales[row] = scale_h;

  owned_pairs = 0;
  for (int pair = threadIdx.x; pair < width / 2;
       pair += blockDim.x, ++owned_pairs) {
    const int q0 = max(-8, min(7, static_cast<int>(
        nearbyintf(transformed[owned_pairs * 2] / scale))));
    const int q1 = max(-8, min(7, static_cast<int>(
        nearbyintf(transformed[owned_pairs * 2 + 1] / scale))));
    packed[row * (width / 2) + pair] = pack_s4_pair(q0, q1);
  }
}

// Split a large tensor-product transform at its natural boundary. Each block
// fuses SiLU*up with one inner power-of-two transform and uses only
// inner*sizeof(float) LDS. The finish kernel applies the remainder transform,
// computes the row scale, and directly packs INT4.
template <int Inner>
__global__ void fused_ffn_general_inner_fixed_kernel(
    const half* gate, const half* up, float* workspace, int rows, int width,
    int remainder) {
  __shared__ float values[Inner];
  constexpr int inner = Inner;
  const int row = blockIdx.x;
  const int group = blockIdx.y;
  if (row >= rows || group >= remainder) return;
  const int base = row * width + group * inner;
  for (int i = threadIdx.x; i < inner; i += blockDim.x) {
    values[i] = silu(__half2float(gate[base + i])) *
                __half2float(up[base + i]);
  }
  __syncthreads();
  hadamard_fixed<Inner>(values);
  for (int i = threadIdx.x; i < inner; i += blockDim.x) {
    workspace[base + i] = values[i];
  }
}

template <int Inner>
__global__ void fused_ffn_general_inner_warp_kernel(
    const half* gate, const half* up, float* workspace, int rows, int width,
    int remainder) {
  static_assert(Inner >= 32 && Inner <= 256 && (Inner & (Inner - 1)) == 0,
                "warp inner kernel supports H32 through H256");
  constexpr int kRegs = Inner / 32;
  const int lane = threadIdx.x;
  const int row = blockIdx.x;
  const int group = blockIdx.y;
  if (row >= rows || group >= remainder) return;
  const int base = row * width + group * Inner;
  float values[kRegs];
#pragma unroll
  for (int r = 0; r < kRegs; ++r) {
    const int i = r * 32 + lane;
    values[r] = silu(__half2float(gate[base + i])) *
                __half2float(up[base + i]);
  }
  hadacore_warp_hadamard_regs<kRegs>(values, lane);
#pragma unroll
  for (int r = 0; r < kRegs; ++r) {
    workspace[base + r * 32 + lane] = values[r];
  }
}

__global__ void fused_ffn_general_large_finish_kernel(
    const float* values, const half* hadamard, uint8_t* packed, half* scales,
    int rows, int width, int remainder, int inner) {
  __shared__ float scratch[kThreads];
  const int row = blockIdx.x;
  if (row >= rows) return;
  constexpr int kMaxPairsPerThread =
      (kMaxLargeGeneralFfnWidth / 2 + kThreads - 1) / kThreads;
  float transformed[kMaxPairsPerThread * 2];
  const float norm = rsqrtf(float(width));
  float local_max = 0.0f;
  int owned_pairs = 0;
  for (int pair = threadIdx.x; pair < width / 2;
       pair += blockDim.x, ++owned_pairs) {
#pragma unroll
    for (int which = 0; which < 2; ++which) {
      const int out = pair * 2 + which;
      const int out_k = out / inner;
      const int p_index = out - out_k * inner;
      float value = 0.0f;
      for (int in_k = 0; in_k < remainder; ++in_k) {
        value += __half2float(hadamard[out_k * remainder + in_k]) *
                 values[row * width + in_k * inner + p_index];
      }
      value *= norm;
      transformed[owned_pairs * 2 + which] = value;
      local_max = fmaxf(local_max, fabsf(value));
    }
  }
  const half scale_h = __float2half(fmaxf(
      reduce_max(local_max, scratch) / 7.0f, 6.103515625e-5f));
  const float scale = __half2float(scale_h);
  if (threadIdx.x == 0) scales[row] = scale_h;
  owned_pairs = 0;
  for (int pair = threadIdx.x; pair < width / 2;
       pair += blockDim.x, ++owned_pairs) {
    const int q0 = max(-8, min(7, static_cast<int>(
        nearbyintf(transformed[owned_pairs * 2] / scale))));
    const int q1 = max(-8, min(7, static_cast<int>(
        nearbyintf(transformed[owned_pairs * 2 + 1] / scale))));
    packed[row * (width / 2) + pair] = pack_s4_pair(q0, q1);
  }
}


// gfx12 FP16 WMMA fragments used by the single-kernel remainder GEMM.
typedef _Float16 quarot_f16_frag __attribute__((ext_vector_type(8)));
typedef float quarot_f32_frag __attribute__((ext_vector_type(8)));

__device__ __forceinline__ quarot_f32_frag zero_quarot_f32_frag() {
  quarot_f32_frag result;
#pragma unroll
  for (int i = 0; i < 8; ++i) result[i] = 0.0f;
  return result;
}

__device__ __forceinline__ quarot_f32_frag quarot_wmma_f16(
    quarot_f16_frag a, quarot_f16_frag b, quarot_f32_frag c) {
#if defined(__HIP_DEVICE_COMPILE__) && \
    (defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx1250__))
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
#else
  return c;
#endif
}

__device__ __forceinline__ quarot_f16_frag load_remainder_a(
    const half* matrix, int remainder, int out_base, int in_base, int lane) {
  quarot_f16_frag frag;
  const int out = out_base + (lane & 15);
  const int k_base = in_base + (lane >> 4) * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int k = k_base + i;
    frag[i] = (out < remainder && k < remainder)
        ? static_cast<_Float16>(matrix[out * remainder + k])
        : static_cast<_Float16>(0.0f);
  }
  return frag;
}

__device__ __forceinline__ quarot_f16_frag load_remainder_b(
    const half* values, int remainder, int inner, int in_base, int p_base,
    int lane) {
  quarot_f16_frag frag;
  const int p = p_base + (lane & 15);
  const int k_base = in_base + (lane >> 4) * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int k = k_base + i;
    frag[i] = (k < remainder && p < inner)
        ? static_cast<_Float16>(values[k * inner + p])
        : static_cast<_Float16>(0.0f);
  }
  return frag;
}

// True single-kernel large FFN fusion for small generalized remainders.
// Storage stays FP16 in LDS; FHT and matrix accumulators use FP32 registers.
// One workgroup owns a row, so scale reduction and INT4 packing require no
// grid-wide synchronization or global intermediate.
__global__ void fused_ffn_silu_hadamard_quant_single_fp16lds_kernel(
    const half* gate, const half* up, const half* matrix, uint8_t* packed,
    half* scales, int rows, int width, int remainder, int inner) {
  __shared__ half values[kMaxLargeGeneralFfnWidth];       // <= 56 KiB
  __shared__ float inner_values[1024];                    // 4 KiB
  __shared__ half remainder_tile[48 * 16];                // 1.5 KiB
  __shared__ float reduce_scratch[kThreads];              // 1 KiB
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= rows) return;

  for (int i = tid; i < width; i += blockDim.x) {
    const float g = __half2float(gate[row * width + i]);
    const float u = __half2float(up[row * width + i]);
    values[i] = __float2half(silu(g) * u);
  }
  __syncthreads();

  // Each group is independent. FP32 butterfly scratch prevents per-stage FP16
  // rounding while the complete row remains compact in FP16 LDS.
  for (int group = 0; group < remainder; ++group) {
    if (inner == 32 || inner == 64 || inner == 128 || inner == 256) {
      if (tid < 32) {
        float registers[8];
        const int regs = inner / 32;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
          if (r < regs) {
            registers[r] = __half2float(
                values[group * inner + r * 32 + tid]);
          }
        }
        if (inner == 32) {
          hadacore_warp_hadamard_regs<1>(registers, tid);
        } else if (inner == 64) {
          hadacore_warp_hadamard_regs<2>(registers, tid);
        } else if (inner == 128) {
          hadacore_warp_hadamard_regs<4>(registers, tid);
        } else {
          hadacore_warp_hadamard_regs<8>(registers, tid);
        }
#pragma unroll
        for (int r = 0; r < 8; ++r) {
          if (r < regs) {
            values[group * inner + r * 32 + tid] =
                __float2half(registers[r]);
          }
        }
      }
      __syncthreads();
    } else {
      for (int p = tid; p < inner; p += blockDim.x) {
        inner_values[p] = __half2float(values[group * inner + p]);
      }
      __syncthreads();
      hadamard(inner_values, inner);
      for (int p = tid; p < inner; p += blockDim.x) {
        values[group * inner + p] = __float2half(inner_values[p]);
      }
      __syncthreads();
    }
  }

  const int lane = tid & 31;
  const int wave = tid >> 5;
  const int out_tiles = (remainder + 15) >> 4;
  const int k_tiles = (remainder + 15) >> 4;
  const float norm = rsqrtf(static_cast<float>(width));
  for (int p_base = 0; p_base < inner; p_base += 16) {
    if (wave < out_tiles) {
      quarot_f32_frag accum = zero_quarot_f32_frag();
      const int out_base = wave * 16;
      for (int kt = 0; kt < k_tiles; ++kt) {
        const int in_base = kt * 16;
        const quarot_f16_frag a = load_remainder_a(
            matrix, remainder, out_base, in_base, lane);
        const quarot_f16_frag b = load_remainder_b(
            values, remainder, inner, in_base, p_base, lane);
        accum = quarot_wmma_f16(a, b, accum);
      }
      const int p_col = lane & 15;
      const int out_row_base = (lane >> 4) * 8;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int out_k = out_base + out_row_base + i;
        if (out_k < remainder && p_base + p_col < inner) {
          remainder_tile[out_k * 16 + p_col] =
              __float2half(accum[i] * norm);
        }
      }
    }
    __syncthreads();
    for (int index = tid; index < remainder * 16; index += blockDim.x) {
      const int out_k = index >> 4;
      const int p_col = index & 15;
      if (p_base + p_col < inner) {
        values[out_k * inner + p_base + p_col] = remainder_tile[index];
      }
    }
    __syncthreads();
  }

  float local_max = 0.0f;
  for (int i = tid; i < width; i += blockDim.x) {
    local_max = fmaxf(local_max, fabsf(__half2float(values[i])));
  }
  const half scale_h = __float2half(fmaxf(
      reduce_max(local_max, reduce_scratch) / 7.0f, 6.103515625e-5f));
  const float scale = __half2float(scale_h);
  if (tid == 0) scales[row] = scale_h;
  for (int pair = tid; pair < width / 2; pair += blockDim.x) {
    const int q0 = max(-8, min(7, static_cast<int>(nearbyintf(
        __half2float(values[pair * 2]) / scale))));
    const int q1 = max(-8, min(7, static_cast<int>(nearbyintf(
        __half2float(values[pair * 2 + 1]) / scale))));
    packed[row * (width / 2) + pair] = pack_s4_pair(q0, q1);
  }
}

void check_cuda_half_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA/HIP tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16, name, " must be float16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

bool power_of_two(int64_t x) { return x > 0 && (x & (x - 1)) == 0; }

}  // namespace

torch::Tensor rms_norm_rows(
    torch::Tensor input, int64_t mean_dim, double eps) {
  check_cuda_half_contiguous(input, "input");
  TORCH_CHECK(input.dim() >= 1 && input.size(-1) == mean_dim,
              "RMSNorm mean_dim must equal the input width");
  const int64_t width = input.size(-1);
  const int64_t rows = input.numel() / width;
  auto output = torch::empty_like(input);
  at::hip::HIPGuard guard{static_cast<char>(input.get_device())};
  hipLaunchKernelGGL(rms_norm_rows_kernel, dim3(rows), dim3(kThreads), 0,
      at::hip::getCurrentHIPStream().stream(),
      reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      static_cast<int>(rows), static_cast<int>(width),
      static_cast<float>(mean_dim), static_cast<float>(eps));
  TORCH_CHECK(hipGetLastError() == hipSuccess,
              "row-independent RMSNorm launch failed");
  return output;
}

std::vector<torch::Tensor> rms_norm_quant_i4_rows(
    torch::Tensor input, int64_t mean_dim, double eps) {
  check_cuda_half_contiguous(input, "input");
  TORCH_CHECK(input.dim() >= 1 && input.size(-1) == mean_dim,
              "RMSNorm mean_dim must equal the input width");
  const int64_t width = input.size(-1);
  TORCH_CHECK(width > 0 && width <= kMaxFusedWidth && width % 2 == 0,
              "fused RMSNorm+INT4 supports positive even widths up to ",
              kMaxFusedWidth);
  const int64_t rows = input.numel() / width;
  TORCH_CHECK(rows > 0, "fused RMSNorm+INT4 does not support empty tensors");
  auto packed_shape = input.sizes().vec(); packed_shape.back() = width / 2;
  auto scale_shape = input.sizes().vec(); scale_shape.back() = 1;
  auto packed = torch::empty(packed_shape, input.options().dtype(torch::kUInt8));
  auto scales = torch::empty(scale_shape, input.options());
  at::hip::HIPGuard guard{static_cast<char>(input.get_device())};
  hipLaunchKernelGGL((rms_norm_quant_i4_rows_kernel<false, false>), dim3(rows),
      dim3(kThreads), width * sizeof(half),
      at::hip::getCurrentHIPStream().stream(),
      reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
      nullptr, nullptr,
      packed.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(scales.data_ptr<at::Half>()),
      static_cast<int>(rows), static_cast<int>(width),
      static_cast<float>(mean_dim), static_cast<float>(eps), 1.0f);
  TORCH_CHECK(hipGetLastError() == hipSuccess,
              "fused RMSNorm+INT4 launch failed");
  return {packed, scales};
}

namespace {

template <bool AddResidual>
std::vector<torch::Tensor> rms_norm_quant_i4_rows_clipped_impl(
    torch::Tensor input, torch::Tensor residual, int64_t mean_dim,
    double eps, double clip_ratio) {
  check_cuda_half_contiguous(input, "input");
  TORCH_CHECK(input.dim() >= 1 && input.size(-1) == mean_dim,
              "RMSNorm mean_dim must equal the input width");
  TORCH_CHECK(mean_dim > 0 && mean_dim <= kMaxFusedWidth && mean_dim % 2 == 0,
              "fused RMSNorm+INT4 supports positive even widths up to ",
              kMaxFusedWidth);
  TORCH_CHECK(std::isfinite(eps) && eps > 0,
              "RMSNorm eps must be finite and positive");
  TORCH_CHECK(std::isfinite(clip_ratio) && clip_ratio > 0 && clip_ratio <= 1,
              "input_clip_ratio must be in (0, 1]");
  const int64_t rows = input.numel() / mean_dim;
  TORCH_CHECK(rows > 0, "fused RMSNorm+INT4 does not support empty tensors");
  if (AddResidual) {
    check_cuda_half_contiguous(residual, "residual");
    TORCH_CHECK(residual.sizes() == input.sizes(),
                "residual must have the input shape");
    TORCH_CHECK(residual.device() == input.device(),
                "residual must be on the input device");
  }
  at::hip::HIPGuard guard{static_cast<char>(input.get_device())};
  auto packed_shape = input.sizes().vec(); packed_shape.back() = mean_dim / 2;
  auto scale_shape = input.sizes().vec(); scale_shape.back() = 1;
  auto packed = torch::empty(packed_shape, input.options().dtype(torch::kUInt8));
  auto scales = torch::empty(scale_shape, input.options());
  auto residual_output = AddResidual ? torch::empty_like(input) : torch::Tensor();
  hipLaunchKernelGGL((rms_norm_quant_i4_rows_kernel<true, AddResidual>),
      dim3(rows), dim3(kThreads), mean_dim * sizeof(half),
      at::hip::getCurrentHIPStream().stream(),
      reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
      AddResidual ? reinterpret_cast<const half*>(residual.data_ptr<at::Half>())
                  : nullptr,
      AddResidual ? reinterpret_cast<half*>(residual_output.data_ptr<at::Half>())
                  : nullptr,
      packed.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(scales.data_ptr<at::Half>()),
      static_cast<int>(rows), static_cast<int>(mean_dim),
      static_cast<float>(mean_dim), static_cast<float>(eps),
      static_cast<float>(clip_ratio));
  TORCH_CHECK(hipGetLastError() == hipSuccess,
              "clipped row-independent RMSNorm+INT4 launch failed");
  if (AddResidual) return {packed, scales, residual_output};
  return {packed, scales};
}

}  // namespace

std::vector<torch::Tensor> rms_norm_quant_i4_rows_clipped(
    torch::Tensor input, int64_t mean_dim, double eps, double clip_ratio) {
  return rms_norm_quant_i4_rows_clipped_impl<false>(
      input, torch::Tensor(), mean_dim, eps, clip_ratio);
}

std::vector<torch::Tensor> residual_rms_norm_quant_i4_rows(
    torch::Tensor input, torch::Tensor residual, int64_t mean_dim,
    double eps, double clip_ratio) {
  return rms_norm_quant_i4_rows_clipped_impl<true>(
      input, residual, mean_dim, eps, clip_ratio);
}

void fused_append_kv_i4(
    torch::Tensor kv_data, torch::Tensor kv_param, torch::Tensor kv_indptr,
    torch::Tensor kv_indices, torch::Tensor last_page_offset, torch::Tensor key,
    torch::Tensor value, int64_t num_layers, int64_t layer_idx,
    int64_t num_heads, int64_t page_size, int64_t batch_size) {
  check_cuda_half_contiguous(key, "key");
  check_cuda_half_contiguous(value, "value");
  TORCH_CHECK(kv_data.is_cuda() && kv_data.is_contiguous() && kv_data.scalar_type() == torch::kUInt8, "kv_data must be a contiguous uint8 CUDA/HIP tensor");
  TORCH_CHECK(kv_param.is_cuda() && kv_param.is_contiguous() && kv_param.scalar_type() == torch::kFloat16, "kv_param must be a contiguous float16 CUDA/HIP tensor");
  TORCH_CHECK(kv_indptr.is_cuda() && kv_indices.is_cuda() && last_page_offset.is_cuda(), "paged-cache metadata must be CUDA/HIP tensors");
  TORCH_CHECK(kv_indptr.is_contiguous() && kv_indices.is_contiguous() && last_page_offset.is_contiguous(), "paged-cache metadata must be contiguous");
  TORCH_CHECK(kv_indptr.scalar_type() == torch::kInt32 && kv_indices.scalar_type() == torch::kInt32 && last_page_offset.scalar_type() == torch::kInt32, "paged-cache metadata must be int32");
  TORCH_CHECK(kv_data.get_device() == key.get_device() && kv_param.get_device() == key.get_device() &&
              kv_indptr.get_device() == key.get_device() && kv_indices.get_device() == key.get_device() &&
              last_page_offset.get_device() == key.get_device() && value.get_device() == key.get_device(),
              "all fused KV append tensors must be on the same device");
  TORCH_CHECK(kv_indptr.numel() == batch_size + 1 && last_page_offset.numel() == batch_size,
              "paged-cache metadata lengths do not match batch_size");
  TORCH_CHECK(num_layers > 0 && num_heads > 0 && page_size > 0 && batch_size > 0,
              "cache dimensions must be positive");
  TORCH_CHECK(key.sizes() == value.sizes() &&
              (key.dim() == 3 || key.dim() == 4),
              "key/value must be rank 3 or 4");
  const int64_t added_length = key.dim() == 4 ? key.size(1) : 1;
  const int64_t head_axis = key.dim() - 2;
  const int64_t head_dim = key.size(-1);
  TORCH_CHECK(key.size(0) == batch_size &&
              key.size(head_axis) == num_heads &&
              (head_dim == 64 || head_dim == 128),
              "key/value must have shape [batch, (tokens), heads, 64 or 128]");
  TORCH_CHECK(added_length >= 1,
              "fused KV append requires at least one token");
  TORCH_CHECK(layer_idx >= 0 && layer_idx < num_layers, "layer_idx is outside the cache");
  at::hip::HIPGuard guard{static_cast<char>(key.get_device())};
#define LAUNCH_FUSED_APPEND(HEAD_DIM) hipLaunchKernelGGL(\
      (fused_append_kv_i4_kernel<HEAD_DIM>), dim3(batch_size * added_length, num_heads), \
      dim3(32), 0, at::hip::getCurrentHIPStream().stream(), \
      reinterpret_cast<const half*>(key.data_ptr<at::Half>()), \
      reinterpret_cast<const half*>(value.data_ptr<at::Half>()), \
      kv_data.data_ptr<uint8_t>(), reinterpret_cast<half2*>(kv_param.data_ptr<at::Half>()), \
      kv_indptr.data_ptr<int32_t>(), kv_indices.data_ptr<int32_t>(), \
      last_page_offset.data_ptr<int32_t>(), num_layers, layer_idx, num_heads, \
      page_size, batch_size, added_length)
  if (head_dim == 64) { LAUNCH_FUSED_APPEND(64); }
  else { LAUNCH_FUSED_APPEND(128); }
#undef LAUNCH_FUSED_APPEND
  TORCH_CHECK(hipGetLastError() == hipSuccess, "fused_append_kv_i4 launch failed");
}

torch::Tensor fused_rope_append_kv_i4(
    torch::Tensor query, torch::Tensor key, torch::Tensor value,
    torch::Tensor cos, torch::Tensor sin, torch::Tensor kv_data,
    torch::Tensor kv_param, torch::Tensor kv_indptr, torch::Tensor kv_indices,
    torch::Tensor last_page_offset, int64_t num_layers, int64_t layer_idx,
    int64_t page_size) {
  check_cuda_half_contiguous(query, "query");
  check_cuda_half_contiguous(key, "key");
  check_cuda_half_contiguous(value, "value");
  check_cuda_half_contiguous(cos, "cos");
  check_cuda_half_contiguous(sin, "sin");
  TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4 &&
                  query.size(1) == 1 && key.size(1) == 1 &&
                  value.sizes() == key.sizes() &&
                  query.size(0) == key.size(0) &&
                  query.size(3) == key.size(3),
              "K1 Q/K/V must be [batch, 1, heads, head_dim]");
  const int batch = query.size(0);
  const int query_heads = query.size(2);
  const int kv_heads = key.size(2);
  const int head_dim = query.size(3);
  TORCH_CHECK(query_heads % kv_heads == 0,
              "query heads must be divisible by KV heads");
  TORCH_CHECK((head_dim == 64 || head_dim == 128) &&
                  cos.numel() == batch * head_dim &&
                  sin.numel() == batch * head_dim,
              "invalid K1 RoPE shape");
  TORCH_CHECK(kv_data.is_cuda() && kv_data.is_contiguous() &&
                  kv_data.scalar_type() == torch::kUInt8,
              "kv_data must be contiguous uint8 CUDA/HIP");
  TORCH_CHECK(kv_param.is_cuda() && kv_param.is_contiguous() &&
                  kv_param.scalar_type() == torch::kFloat16,
              "kv_param must be contiguous float16 CUDA/HIP");
  TORCH_CHECK(kv_indptr.is_cuda() && kv_indices.is_cuda() &&
                  last_page_offset.is_cuda() &&
                  kv_indptr.is_contiguous() && kv_indices.is_contiguous() &&
                  last_page_offset.is_contiguous() &&
                  kv_indptr.scalar_type() == torch::kInt32 &&
                  kv_indices.scalar_type() == torch::kInt32 &&
                  last_page_offset.scalar_type() == torch::kInt32,
              "K1 paged-cache metadata must be contiguous int32 CUDA/HIP");
  TORCH_CHECK(kv_data.size(1) == num_layers && kv_data.size(3) == kv_heads &&
                  kv_data.size(4) == page_size &&
                  kv_data.size(5) * 2 == head_dim,
              "K1 cache dimensions do not match Q/K/V");
  TORCH_CHECK(kv_indptr.numel() == batch + 1 &&
                  last_page_offset.numel() == batch &&
                  layer_idx >= 0 && layer_idx < num_layers,
              "invalid K1 metadata or layer index");
  at::hip::HIPGuard guard{static_cast<char>(query.get_device())};
  auto query_out = torch::empty_like(query);
#define LAUNCH_FUSED_K1(HEAD_DIM) hipLaunchKernelGGL(\
      (fused_rope_append_kv_i4_kernel<HEAD_DIM>), \
      dim3(batch, query_heads), dim3(32), 0, \
      at::hip::getCurrentHIPStream().stream(), \
      reinterpret_cast<const half*>(query.data_ptr<at::Half>()), \
      reinterpret_cast<const half*>(key.data_ptr<at::Half>()), \
      reinterpret_cast<const half*>(value.data_ptr<at::Half>()), \
      reinterpret_cast<const half*>(cos.data_ptr<at::Half>()), \
      reinterpret_cast<const half*>(sin.data_ptr<at::Half>()), \
      reinterpret_cast<half*>(query_out.data_ptr<at::Half>()), \
      kv_data.data_ptr<uint8_t>(), \
      reinterpret_cast<half2*>(kv_param.data_ptr<at::Half>()), \
      kv_indptr.data_ptr<int32_t>(), kv_indices.data_ptr<int32_t>(), \
      last_page_offset.data_ptr<int32_t>(), num_layers, layer_idx, \
      query_heads, kv_heads, page_size, batch)
  if (head_dim == 64) { LAUNCH_FUSED_K1(64); }
  else { LAUNCH_FUSED_K1(128); }
#undef LAUNCH_FUSED_K1
  TORCH_CHECK(hipGetLastError() == hipSuccess,
              "fused_rope_append_kv_i4 launch failed");
  return query_out;
}

std::vector<torch::Tensor> fused_rmsnorm_quant_i4(
    torch::Tensor input, double eps, double clip_ratio) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
              "input must be a contiguous CUDA/HIP tensor");
  TORCH_CHECK(input.scalar_type() == torch::kFloat16,
              "input must be FP16");
  TORCH_CHECK(input.dim() >= 2 && input.size(-1) % 2 == 0,
              "last dimension must be even");
  const int cols = input.size(-1);
  const int rows = input.numel() / cols;
  auto packed_sizes = input.sizes().vec();
  packed_sizes.back() = cols / 2;
  auto scale_sizes = input.sizes().vec();
  scale_sizes.back() = 1;
  auto packed = torch::empty(
      packed_sizes, input.options().dtype(torch::kUInt8));
  auto scales = torch::empty(scale_sizes, input.options());
  hipLaunchKernelGGL(fused_rmsnorm_quant_i4_kernel, dim3(rows),
                     dim3(kThreads), 0, at::hip::getCurrentHIPStream(),
                     reinterpret_cast<const half*>(input.data_ptr()),
                     packed.data_ptr<uint8_t>(),
                     reinterpret_cast<half*>(scales.data_ptr()), rows, cols,
                     static_cast<float>(eps), static_cast<float>(clip_ratio));
  TORCH_CHECK(hipGetLastError() == hipSuccess,
              "fused_rmsnorm_quant_i4 launch failed");
  return {packed, scales};
}

std::vector<torch::Tensor> fused_attention_hadamard_quant(torch::Tensor attention, int64_t num_heads) {
  check_cuda_half_contiguous(attention, "attention");
  TORCH_CHECK(attention.dim() == 4, "attention must have shape [batch, sequence, heads, head_dim]");
  TORCH_CHECK(attention.size(2) == num_heads && power_of_two(num_heads), "num_heads must be a positive power of two");
  const int64_t head_dim = attention.size(3);
  const int64_t width = num_heads * head_dim;
  TORCH_CHECK(width <= kMaxFusedWidth && width % 2 == 0, "fused attention supports even hidden sizes up to ", kMaxFusedWidth);
  const int64_t rows = attention.numel() / width;
  TORCH_CHECK(rows > 0, "fused attention does not support empty tensors");
  auto packed = torch::empty({attention.size(0), attention.size(1), width / 2}, attention.options().dtype(torch::kUInt8));
  auto scales = torch::empty({attention.size(0), attention.size(1), 1}, attention.options());
  at::hip::HIPGuard guard{static_cast<char>(attention.get_device())};
  hipLaunchKernelGGL(fused_attention_hadamard_quant_kernel, dim3(rows), dim3(kThreads), 0,
      at::hip::getCurrentHIPStream().stream(), reinterpret_cast<const half*>(attention.data_ptr<at::Half>()),
      packed.data_ptr<uint8_t>(), reinterpret_cast<half*>(scales.data_ptr<at::Half>()), rows, num_heads, head_dim);
  TORCH_CHECK(hipGetLastError() == hipSuccess, "fused_attention_hadamard_quant launch failed");
  return {packed, scales};
}

std::vector<torch::Tensor> fused_attention_hadamard_quant_general(
    torch::Tensor attention, int64_t num_heads, torch::Tensor matrix) {
  check_cuda_half_contiguous(attention, "attention");
  check_cuda_half_contiguous(matrix, "matrix");
  TORCH_CHECK(attention.dim() == 4 && attention.size(2) == num_heads,
              "attention must have shape [batch, sequence, heads, head_dim]");
  TORCH_CHECK(matrix.dim() == 2 && matrix.size(0) == matrix.size(1),
              "matrix must be square");
  TORCH_CHECK(attention.get_device() == matrix.get_device(),
              "attention and matrix must be on the same device");
  const int64_t remainder = matrix.size(0);
  TORCH_CHECK(num_heads % remainder == 0,
              "num_heads must be divisible by the remainder order");
  const int64_t inner = num_heads / remainder;
  TORCH_CHECK(power_of_two(inner), "heads / remainder must be a power of two");
  const int64_t head_dim = attention.size(3);
  const int64_t width = num_heads * head_dim;
  TORCH_CHECK(width <= kMaxFusedWidth && width % 2 == 0,
              "general fused attention supports even widths up to ",
              kMaxFusedWidth);
  const int64_t rows = attention.numel() / width;
  auto packed = torch::empty(
      {attention.size(0), attention.size(1), width / 2},
      attention.options().dtype(torch::kUInt8));
  auto scales = torch::empty(
      {attention.size(0), attention.size(1), 1}, attention.options());
  at::hip::HIPGuard guard{static_cast<char>(attention.get_device())};
  hipLaunchKernelGGL(fused_attention_hadamard_quant_general_kernel,
      dim3(rows), dim3(kThreads), width * sizeof(float),
      at::hip::getCurrentHIPStream().stream(),
      reinterpret_cast<const half*>(attention.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(matrix.data_ptr<at::Half>()),
      packed.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(scales.data_ptr<at::Half>()), rows,
      num_heads, head_dim, remainder, inner);
  TORCH_CHECK(hipGetLastError() == hipSuccess,
              "fused_attention_hadamard_quant_general launch failed");
  return {packed, scales};
}

std::vector<torch::Tensor> fused_ffn_silu_hadamard_quant(torch::Tensor gate, torch::Tensor up) {
  check_cuda_half_contiguous(gate, "gate");
  check_cuda_half_contiguous(up, "up");
  TORCH_CHECK(gate.sizes() == up.sizes() && gate.dim() >= 1, "gate and up must have identical non-empty shapes");
  const int64_t width = gate.size(-1);
  TORCH_CHECK(power_of_two(width) && width <= kMaxFusedWidth && width % 2 == 0, "fused FFN supports power-of-two widths up to ", kMaxFusedWidth);
  const int64_t rows = gate.numel() / width;
  TORCH_CHECK(rows > 0, "fused FFN does not support empty tensors");
  auto packed_shape = gate.sizes().vec(); packed_shape.back() = width / 2;
  auto scale_shape = gate.sizes().vec(); scale_shape.back() = 1;
  auto packed = torch::empty(packed_shape, gate.options().dtype(torch::kUInt8));
  auto scales = torch::empty(scale_shape, gate.options());
  at::hip::HIPGuard guard{static_cast<char>(gate.get_device())};
  auto stream = at::hip::getCurrentHIPStream().stream();
#define LAUNCH_FFN_WARP(WIDTH) hipLaunchKernelGGL(\
      (fused_ffn_silu_hadamard_quant_warp_kernel<WIDTH>), \
      dim3(rows), dim3(32), 0, stream, \
      reinterpret_cast<const half*>(gate.data_ptr<at::Half>()), \
      reinterpret_cast<const half*>(up.data_ptr<at::Half>()), \
      packed.data_ptr<uint8_t>(), \
      reinterpret_cast<half*>(scales.data_ptr<at::Half>()), rows)
  if (width == 32) { LAUNCH_FFN_WARP(32); }
  else if (width == 64) { LAUNCH_FFN_WARP(64); }
  else if (width == 128) { LAUNCH_FFN_WARP(128); }
  else if (width == 256) { LAUNCH_FFN_WARP(256); }
  else {
    hipLaunchKernelGGL(fused_ffn_silu_hadamard_quant_kernel,
        dim3(rows), dim3(kThreads), 0, stream,
        reinterpret_cast<const half*>(gate.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(up.data_ptr<at::Half>()),
        packed.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(scales.data_ptr<at::Half>()), rows, width);
  }
#undef LAUNCH_FFN_WARP
  TORCH_CHECK(hipGetLastError() == hipSuccess, "fused_ffn_silu_hadamard_quant launch failed");
  return {packed, scales};
}


std::vector<torch::Tensor> fused_ffn_silu_hadamard_quant_single_fp16lds(
    torch::Tensor gate, torch::Tensor up, torch::Tensor matrix) {
  check_cuda_half_contiguous(gate, "gate");
  check_cuda_half_contiguous(up, "up");
  check_cuda_half_contiguous(matrix, "matrix");
  TORCH_CHECK(gate.sizes() == up.sizes() && gate.dim() >= 1,
              "gate and up must have identical non-empty shapes");
  TORCH_CHECK(matrix.dim() == 2 && matrix.size(0) == matrix.size(1),
              "matrix must be square");
  TORCH_CHECK(gate.get_device() == matrix.get_device(),
              "gate, up, and matrix must be on the same device");
  const int64_t width = gate.size(-1);
  const int64_t remainder = matrix.size(0);
  TORCH_CHECK(width > 0 && width <= kMaxLargeGeneralFfnWidth &&
                  width % remainder == 0 && width % 2 == 0,
              "unsupported single-kernel FFN width/remainder");
  const int64_t inner = width / remainder;
  TORCH_CHECK(power_of_two(inner) && inner <= 1024,
              "single-kernel FFN inner transform must be power-of-two <=1024");
  TORCH_CHECK(remainder <= 48,
              "single-kernel WMMA remainder supports orders up to 48");
  const int64_t rows = gate.numel() / width;
  TORCH_CHECK(rows > 0, "single-kernel FFN does not support empty tensors");
  auto packed_shape = gate.sizes().vec(); packed_shape.back() = width / 2;
  auto scale_shape = gate.sizes().vec(); scale_shape.back() = 1;
  auto packed = torch::empty(packed_shape, gate.options().dtype(torch::kUInt8));
  auto scales = torch::empty(scale_shape, gate.options());
  at::hip::HIPGuard guard{static_cast<char>(gate.get_device())};
  hipLaunchKernelGGL(fused_ffn_silu_hadamard_quant_single_fp16lds_kernel,
      dim3(rows), dim3(kThreads), 0,
      at::hip::getCurrentHIPStream().stream(),
      reinterpret_cast<const half*>(gate.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(up.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(matrix.data_ptr<at::Half>()),
      packed.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(scales.data_ptr<at::Half>()), rows, width,
      remainder, inner);
  TORCH_CHECK(hipGetLastError() == hipSuccess,
              "single-kernel FP16-LDS fused FFN launch failed");
  return {packed, scales};
}

std::vector<torch::Tensor> fused_ffn_silu_hadamard_quant_general(
    torch::Tensor gate, torch::Tensor up, torch::Tensor hadamard) {
  check_cuda_half_contiguous(gate, "gate");
  check_cuda_half_contiguous(up, "up");
  check_cuda_half_contiguous(hadamard, "hadamard");
  TORCH_CHECK(gate.sizes() == up.sizes() && gate.dim() >= 1,
              "gate and up must have identical non-empty shapes");
  TORCH_CHECK(hadamard.dim() == 2 && hadamard.size(0) == hadamard.size(1),
              "hadamard must be a square matrix");
  TORCH_CHECK(gate.get_device() == hadamard.get_device(),
              "gate, up, and hadamard must be on the same device");
  const int64_t width = gate.size(-1);
  const int64_t remainder = hadamard.size(0);
  TORCH_CHECK(width > 0 && width <= kMaxLargeGeneralFfnWidth &&
                  width % remainder == 0 && width % 2 == 0,
              "width must be a positive even multiple of the Hadamard order and at most ",
              kMaxLargeGeneralFfnWidth);
  const int64_t inner = width / remainder;
  TORCH_CHECK(power_of_two(inner),
              "width / Hadamard order must be a power of two");
  const int64_t rows = gate.numel() / width;
  TORCH_CHECK(rows > 0, "fused FFN does not support empty tensors");
  auto packed_shape = gate.sizes().vec(); packed_shape.back() = width / 2;
  auto scale_shape = gate.sizes().vec(); scale_shape.back() = 1;
  auto packed = torch::empty(packed_shape, gate.options().dtype(torch::kUInt8));
  auto scales = torch::empty(scale_shape, gate.options());
  at::hip::HIPGuard guard{static_cast<char>(gate.get_device())};
  auto stream = at::hip::getCurrentHIPStream().stream();
  if (width <= kMaxGeneralFfnWidth) {
    hipLaunchKernelGGL(fused_ffn_silu_hadamard_quant_general_kernel,
        dim3(rows), dim3(kThreads), width * sizeof(float), stream,
        reinterpret_cast<const half*>(gate.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(up.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(hadamard.data_ptr<at::Half>()),
        packed.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(scales.data_ptr<at::Half>()), rows, width,
        remainder, inner);
  } else {
    auto workspace = torch::empty(
        gate.sizes(), gate.options().dtype(torch::kFloat32));
#define LAUNCH_LARGE_INNER_WARP(INNER) hipLaunchKernelGGL(\
        (fused_ffn_general_inner_warp_kernel<INNER>), \
        dim3(rows, remainder), dim3(32), 0, stream, \
        reinterpret_cast<const half*>(gate.data_ptr<at::Half>()), \
        reinterpret_cast<const half*>(up.data_ptr<at::Half>()), \
        workspace.data_ptr<float>(), rows, width, remainder)
#define LAUNCH_LARGE_INNER_LDS(INNER) hipLaunchKernelGGL(\
        (fused_ffn_general_inner_fixed_kernel<INNER>), \
        dim3(rows, remainder), dim3(kThreads), 0, stream, \
        reinterpret_cast<const half*>(gate.data_ptr<at::Half>()), \
        reinterpret_cast<const half*>(up.data_ptr<at::Half>()), \
        workspace.data_ptr<float>(), rows, width, remainder)
    if (inner == 128) { LAUNCH_LARGE_INNER_WARP(128); }
    else if (inner == 256) { LAUNCH_LARGE_INNER_WARP(256); }
    else if (inner == 512) { LAUNCH_LARGE_INNER_LDS(512); }
    else if (inner == 1024) { LAUNCH_LARGE_INNER_LDS(1024); }
    else {
      TORCH_CHECK(false,
                  "large fused FFN supports inner H128/H256/H512/H1024; got H",
                  inner);
    }
#undef LAUNCH_LARGE_INNER_WARP
#undef LAUNCH_LARGE_INNER_LDS
    TORCH_CHECK(hipGetLastError() == hipSuccess,
                "large fused FFN inner-transform launch failed");
    hipLaunchKernelGGL(fused_ffn_general_large_finish_kernel,
        dim3(rows), dim3(kThreads), 0, stream, workspace.data_ptr<float>(),
        reinterpret_cast<const half*>(hadamard.data_ptr<at::Half>()),
        packed.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(scales.data_ptr<at::Half>()), rows, width,
        remainder, inner);
  }
  TORCH_CHECK(hipGetLastError() == hipSuccess,
              "fused_ffn_silu_hadamard_quant_general launch failed");
  return {packed, scales};
}

std::vector<torch::Tensor> fused_ffn_silu_hadamard_quant_grouped256(
    torch::Tensor gate, torch::Tensor up) {
  check_cuda_half_contiguous(gate, "gate");
  check_cuda_half_contiguous(up, "up");
  TORCH_CHECK(gate.sizes() == up.sizes() && gate.dim() >= 1,
              "gate and up must have identical non-empty shapes");
  const int64_t width = gate.size(-1);
  TORCH_CHECK(width > 0 && width % 256 == 0,
              "grouped H256 FFN requires a width divisible by 256");
  const int64_t rows = gate.numel() / width;
  auto packed_shape = gate.sizes().vec(); packed_shape.back() = width / 2;
  auto scale_shape = gate.sizes().vec(); scale_shape.back() = width / 256;
  auto packed = torch::empty(packed_shape,
                             gate.options().dtype(torch::kUInt8));
  auto scales = torch::empty(scale_shape, gate.options());
  at::hip::HIPGuard guard{static_cast<char>(gate.get_device())};
  auto stream = at::hip::getCurrentHIPStream().stream();
  hipLaunchKernelGGL(fused_ffn_silu_hadamard_quant_warp256_kernel,
    dim3(rows, width / 256), dim3(32), 0, stream,
    reinterpret_cast<const half*>(gate.data_ptr<at::Half>()),
    reinterpret_cast<const half*>(up.data_ptr<at::Half>()),
    packed.data_ptr<uint8_t>(),
    reinterpret_cast<half*>(scales.data_ptr<at::Half>()), rows, width);
  TORCH_CHECK(hipGetLastError() == hipSuccess,
              "grouped H256 FFN launch failed");
  return {packed, scales};
}
