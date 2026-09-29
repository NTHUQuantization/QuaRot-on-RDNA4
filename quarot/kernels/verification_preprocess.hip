#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <verification_preprocess.h>

namespace {

using F16Fragment = _Float16 __attribute__((ext_vector_type(8)));
using F32Fragment = float __attribute__((ext_vector_type(8)));
constexpr int kWidth = 128;
constexpr int kSafeRows = 8;

// This reproduces the existing <=8-row Hadacore path, including its FP16
// intermediate and FP16 H8 coefficient. The unsafe backend's 16-row tile is
// deliberately NOT enabled: each workgroup owns at most eight input rows,
// supplies zero fragments for rows 8..15, and guards every global store.
// All workgroups are dispatched together, so there is no padding/fill/cat.
__device__ __forceinline__ F16Fragment h16_fragment(int lane, float scale) {
  F16Fragment h;
  const int column = lane & 15;
  const int base = (lane >> 4) * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    h[i] = static_cast<_Float16>(
        (__builtin_popcount(static_cast<unsigned>((base + i) & column)) & 1)
        ? -scale : scale);
  }
  return h;
}

__device__ __forceinline__ void h128_safe_tile(
    const half* input, half* output, half* temporary, int valid_rows) {
  const int lane = threadIdx.x;
  if (lane < 32) {
    const F16Fragment h = h16_fragment(lane, 0.25f);
    const int row = lane & 15;
    const int base = (lane >> 4) * 8;
    const int column = lane & 15;
    const int output_row = (lane >> 4) * 8;
#pragma unroll
    for (int group = 0; group < 8; ++group) {
      F16Fragment a;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        a[i] = row < valid_rows
            ? static_cast<_Float16>(__half2float(
                  input[row * kWidth + group * 16 + base + i]))
            : static_cast<_Float16>(0.0f);
      }
      const F32Fragment zero = {0, 0, 0, 0, 0, 0, 0, 0};
      const F32Fragment d =
          __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, h, zero);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        temporary[(output_row + i) * kWidth + group * 16 + column] =
            __float2half(d[i]);
      }
    }
  }
  __syncthreads();
  if (lane < 32) {
    const F16Fragment h = h16_fragment(lane, rsqrtf(8.0f));
    const int row = lane & 15;
    const int base = (lane >> 4) * 8;
    const int column = lane & 15;
    const int output_row = (lane >> 4) * 8;
#pragma unroll
    for (int low = 0; low < 16; ++low) {
      F16Fragment a;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        a[i] = base + i < 8
            ? static_cast<_Float16>(__half2float(
                  temporary[row * kWidth + (base + i) * 16 + low]))
            : static_cast<_Float16>(0.0f);
      }
      const F32Fragment zero = {0, 0, 0, 0, 0, 0, 0, 0};
      const F32Fragment d =
          __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, h, zero);
      if (column < 8) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          if (output_row + i < valid_rows) {
            output[(output_row + i) * kWidth + column * 16 + low] =
                __float2half(d[i]);
          }
        }
      }
    }
  }
}

__global__ void batched_h128_safe_rows_kernel(
    const half* input, half* output, int rows) {
  __shared__ half temporary[16 * kWidth];
  const int row = blockIdx.x * kSafeRows;
  h128_safe_tile(input + row * kWidth, output + row * kWidth, temporary,
                min(kSafeRows, rows - row));
}

struct Strides4 {
  int64_t batch, token, head;
};

__device__ __forceinline__ float fp32_boundary(float value) {
  // Eager norm materializes FP32 intermediate tensors. An opaque register
  // boundary prevents reassociation across rsqrt/multiply/conversion when
  // reproducing that sequence inside one kernel (no memory traffic).
  asm volatile("" : "+v"(value));
  return value;
}

__device__ __forceinline__ float norm_inverse(float sum, float eps) {
  const float mean = fp32_boundary(sum * (1.0f / kWidth));
  const float denominator = fp32_boundary(mean + eps);
  // PyTorch's HIP rsqrt_wrapper calls ::rsqrt(float). In this HIP release
  // that overload promotes to double, refines in FP64, then casts to FP32.
  // rsqrtf is a different 1-ULP approximation on some inputs. Match the
  // existing operator explicitly, rather than relaxing INT4 parity tests.
  return fp32_boundary(static_cast<float>(::rsqrt(static_cast<double>(denominator))));
}

__device__ __forceinline__ int64_t head_offset(
    Strides4 stride, int batch, int token, int head) {
  return batch * stride.batch + token * stride.token + head * stride.head;
}

// Qwen3's per-head norm: accumulate FP32 variance, normalize in FP32,
// round to the input dtype, then multiply by the FP16 checkpoint weight.
// The separate FP16 multiplication and FP16 RoPE products/add preserve
// materialization boundaries of the existing eager model.
template <int Threads>
__device__ __forceinline__ void norm_rope_h128(
    const half* input, const half* weight, const half* cosine,
    const half* sine, float eps, float* scratch, half* normalized,
    half* rotated) {
  const int tid = threadIdx.x;
  for (int column = tid; column < kWidth; column += Threads) {
    const float x = __half2float(input[column]);
    scratch[column] = x * x;
  }
  __syncthreads();
  // Mirror ATen's ROCm ReduceOp<float, MeanOps<...>, ..., 4, 4>.
  // Width 128 vectorizes into four consecutive squared values per lane;
  // each lane combines these left-to-right, followed by shuffle-down in
  // ascending offsets. A conventional descending 128-thread reduction has
  // a different FP32 association and can change downstream INT4 codes.
  if (tid < 32) {
    float sum = scratch[tid * 4] + scratch[tid * 4 + 1];
    sum += scratch[tid * 4 + 2];
    sum += scratch[tid * 4 + 3];
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1)
      sum += __shfl_down(sum, offset, 32);
    if (tid == 0) scratch[0] = sum;
  }
  __syncthreads();
  if (tid == 0) scratch[1] = norm_inverse(scratch[0], eps);
  __syncthreads();
  const float inv = scratch[1];
  for (int column = tid; column < kWidth; column += Threads) {
    const float x = __half2float(input[column]);
    normalized[column] = __hmul_rn(
        __float2half(fp32_boundary(x * inv)), weight[column]);
  }
  __syncthreads();
  for (int column = tid; column < kWidth; column += Threads) {
    const int partner = column ^ 64;
    const half other = column < 64 ? __hneg(normalized[partner])
                                   : normalized[partner];
    // *_rn explicitly disables FP contraction in the HIP header; plain
    // __hadd/__hmul can become one half FMA and skip an eager rounding step.
    rotated[column] = __hadd_rn(__hmul_rn(normalized[column], cosine[column]),
                                __hmul_rn(other, sine[column]));
  }
  __syncthreads();
}

__global__ void chunk_q_norm_rope_h128_kernel(
    const half* query, const half* weight, const half* cosine,
    const half* sine, half* output, Strides4 strides, int tokens, int heads,
    int rows, float eps) {
  __shared__ float scratch[kWidth];
  __shared__ half normalized[kWidth];
  __shared__ half rotated[kSafeRows * kWidth];
  __shared__ half temporary[16 * kWidth];
  const int first_row = blockIdx.x * kSafeRows;
  const int valid_rows = min(kSafeRows, rows - first_row);
  for (int local_row = 0; local_row < valid_rows; ++local_row) {
    const int row = first_row + local_row;
    const int head = row % heads;
    const int token_row = row / heads;
    const int token = token_row % tokens;
    const int batch = token_row / tokens;
    norm_rope_h128<32>(query + head_offset(strides, batch, token, head), weight,
                      cosine + token_row * kWidth, sine + token_row * kWidth,
                      eps, scratch, normalized, rotated + local_row * kWidth);
  }
  h128_safe_tile(rotated, output + first_row * kWidth, temporary, valid_rows);
}

__device__ __forceinline__ float block_max(float value, float* scratch) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (int step = kWidth / 2; step > 0; step >>= 1) {
    if (threadIdx.x < step)
      scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x],
                                  scratch[threadIdx.x + step]);
    __syncthreads();
  }
  return scratch[0];
}

__device__ __forceinline__ void quant_append(
    float* values, float* scratch, uint8_t* data, half2* parameters,
    size_t entry) {
  const int tid = threadIdx.x;
  const float x = values[tid];
  const float maximum = block_max(x, scratch);
  __syncthreads();
  const float minimum = -block_max(-x, scratch);
  // Deliberately identical to fused_append_kv_i4: clamp after dividing.
  // The separate Python fallback has different FP16 arithmetic boundaries.
  const half scale_h = __float2half(fmaxf((maximum - minimum) / 15.0f, 1e-5f));
  const half zero_h = __float2half(-minimum);
  const float scale = __half2float(scale_h), zero = __half2float(zero_h);
  if ((tid & 1) == 0) {
    const int low = max(0, min(15, static_cast<int>(
        nearbyintf((values[tid] + zero) / scale))));
    const int high = max(0, min(15, static_cast<int>(
        nearbyintf((values[tid + 1] + zero) / scale))));
    data[entry * (kWidth / 2) + tid / 2] =
        static_cast<uint8_t>(low | (high << 4));
  }
  if (tid == 0) parameters[entry] = __halves2half2(scale_h, zero_h);
  __syncthreads();
}

__global__ void chunk_k_norm_rope_append_kernel(
    const half* key, const half* value, const half* weight,
    const half* cosine, const half* sine, uint8_t* data, half2* parameters,
    const int32_t* indptr, const int32_t* indices, const int32_t* last,
    Strides4 k_strides, Strides4 v_strides, int tokens, int heads,
    int layers, int layer, int page_size, float eps) {
  __shared__ float scratch[kWidth];
  __shared__ float values[kWidth];
  __shared__ half normalized[kWidth];
  __shared__ half rotated[kWidth];
  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  const int head = row % heads;
  const int token_row = row / heads;
  const int token = token_row % tokens;
  const int batch = token_row / tokens;
  const int length = (indptr[batch + 1] - indptr[batch] - 1) * page_size
                     + last[batch];
  const int position = length - tokens + token;
  if (position < 0) return;
  const int page = indices[indptr[batch] + position / page_size];
  const int slot = position % page_size;
  norm_rope_h128<128>(key + head_offset(k_strides, batch, token, head), weight,
                cosine + token_row * kWidth, sine + token_row * kWidth,
                eps, scratch, normalized, rotated);
  values[tid] = __half2float(rotated[tid]);
  __syncthreads();
  // Keep the FP32 butterfly/order used by the pre-existing fused KV append.
#pragma unroll
  for (int step = 1; step < kWidth; step <<= 1) {
    if (tid < kWidth / 2) {
      const int low = (tid / step) * (step << 1) + (tid & (step - 1));
      const float a = values[low], b = values[low + step];
      values[low] = a + b;
      values[low + step] = a - b;
    }
    __syncthreads();
  }
  values[tid] *= rsqrtf(float(kWidth));
  __syncthreads();
  const size_t key_entry =
      ((static_cast<size_t>(page) * layers + layer) * 2 * heads + head)
      * page_size + slot;
  quant_append(values, scratch, data, parameters, key_entry);
  values[tid] = __half2float(
      value[head_offset(v_strides, batch, token, head) + tid]);
  __syncthreads();
  quant_append(values, scratch, data, parameters,
               key_entry + heads * page_size);
}

void check_half(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf,
              name, " must be a HIP FP16 tensor");
}

void check_head_input(const torch::Tensor& x, const torch::Tensor& weight,
                      const torch::Tensor& cosine, const torch::Tensor& sine) {
  check_half(x, "input");
  check_half(weight, "norm_weight");
  check_half(cosine, "cos");
  check_half(sine, "sin");
  TORCH_CHECK(x.dim() == 4 && x.size(3) == kWidth && x.stride(3) == 1,
              "input must be [B, M, heads, 128] with contiguous head values");
  TORCH_CHECK(x.numel() > 0, "input must not be empty");
  TORCH_CHECK(x.numel() <= INT32_MAX,
              "chunk input exceeds int32 indexing capacity");
  TORCH_CHECK(weight.numel() == kWidth && weight.is_contiguous(),
              "norm_weight must contain 128 contiguous elements");
  TORCH_CHECK(cosine.numel() == x.size(0) * x.size(1) * kWidth &&
              sine.sizes() == cosine.sizes() && cosine.is_contiguous() &&
              sine.is_contiguous(), "cos/sin must contain B*M*128 values");
  TORCH_CHECK(x.device() == weight.device() && x.device() == cosine.device()
              && x.device() == sine.device(), "inputs must share a device");
}

Strides4 strides(const torch::Tensor& x) {
  return {x.stride(0), x.stride(1), x.stride(2)};
}

const half* half_ptr(const torch::Tensor& x) {
  return reinterpret_cast<const half*>(x.data_ptr<at::Half>());
}

void check_launch() {
  TORCH_CHECK(hipGetLastError() == hipSuccess,
              "verification preprocessing HIP launch failed");
}

}  // namespace

torch::Tensor hadamard_h128(torch::Tensor input) {
  check_half(input, "input");
  TORCH_CHECK(input.numel() <= INT32_MAX,
              "H128 input exceeds int32 indexing capacity");
  TORCH_CHECK(input.dim() >= 1 && input.size(-1) == kWidth &&
              input.is_contiguous(), "H128 input must be contiguous");
  at::hip::HIPGuard guard{static_cast<char>(input.get_device())};
  auto output = torch::empty_like(input);
  const int64_t rows = input.numel() / kWidth;
  if (rows == 0) return output;
  hipLaunchKernelGGL(batched_h128_safe_rows_kernel,
      dim3((rows + kSafeRows - 1) / kSafeRows), dim3(32), 0,
      at::hip::getCurrentHIPStream().stream(), half_ptr(input),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()), int(rows));
  check_launch();
  return output;
}

torch::Tensor chunk_q_norm_rope_hadamard(
    torch::Tensor query, torch::Tensor norm_weight, torch::Tensor cos,
    torch::Tensor sin, double eps) {
  check_head_input(query, norm_weight, cos, sin);
  TORCH_CHECK(eps >= 0 && std::isfinite(eps), "eps must be finite/nonnegative");
  at::hip::HIPGuard guard{static_cast<char>(query.get_device())};
  auto output = torch::empty(query.sizes(), query.options());
  hipLaunchKernelGGL(chunk_q_norm_rope_h128_kernel,
      dim3((query.numel() / kWidth + kSafeRows - 1) / kSafeRows), dim3(32), 0,
      at::hip::getCurrentHIPStream().stream(), half_ptr(query),
      half_ptr(norm_weight), half_ptr(cos), half_ptr(sin),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()), strides(query),
      int(query.size(1)), int(query.size(2)), int(query.numel() / kWidth), float(eps));
  check_launch();
  return output;
}

void chunk_k_norm_rope_append_i4(
    torch::Tensor key, torch::Tensor value, torch::Tensor norm_weight,
    torch::Tensor cos, torch::Tensor sin, torch::Tensor kv_data,
    torch::Tensor kv_param, torch::Tensor kv_indptr, torch::Tensor kv_indices,
    torch::Tensor last_page_offset, int64_t layer_idx, double eps) {
  check_head_input(key, norm_weight, cos, sin);
  check_half(value, "value");
  check_half(kv_param, "kv_param");
  TORCH_CHECK(value.sizes() == key.sizes() && value.stride(3) == 1,
              "key/value shapes must match");
  TORCH_CHECK(kv_data.dim() == 6 && kv_data.scalar_type() == at::kByte &&
              kv_data.is_contiguous() && kv_data.is_cuda() &&
              kv_data.size(2) == 2 && kv_data.size(3) == key.size(2) &&
              kv_data.size(5) == kWidth / 2, "invalid paged INT4 cache");
  auto param_shape = kv_data.sizes().vec(); param_shape.back() = 2;
  TORCH_CHECK(kv_param.sizes().vec() == param_shape && kv_param.is_contiguous(),
              "invalid paged KV scale/zero shape");
  for (const auto& metadata : {kv_indptr, kv_indices, last_page_offset}) {
    TORCH_CHECK(metadata.is_cuda() && metadata.is_contiguous() &&
                metadata.dim() == 1 && metadata.scalar_type() == at::kInt,
                "KV metadata must be contiguous HIP int32 vectors");
  }
  TORCH_CHECK(kv_indptr.numel() == key.size(0) + 1 &&
              last_page_offset.numel() == key.size(0),
              "append metadata must describe the real batch");
  TORCH_CHECK(layer_idx >= 0 && layer_idx < kv_data.size(1) &&
              kv_data.size(4) > 0, "invalid layer/page size");
  TORCH_CHECK(eps >= 0 && std::isfinite(eps), "eps must be finite/nonnegative");
  for (const auto& tensor : {value, kv_data, kv_param, kv_indptr,
                             kv_indices, last_page_offset}) {
    TORCH_CHECK(tensor.device() == key.device(), "inputs must share a device");
  }
  at::hip::HIPGuard guard{static_cast<char>(key.get_device())};
  hipLaunchKernelGGL(chunk_k_norm_rope_append_kernel,
      dim3(key.numel() / kWidth), dim3(kWidth), 0,
      at::hip::getCurrentHIPStream().stream(), half_ptr(key), half_ptr(value),
      half_ptr(norm_weight), half_ptr(cos), half_ptr(sin),
      kv_data.data_ptr<uint8_t>(), reinterpret_cast<half2*>(kv_param.data_ptr()),
      kv_indptr.data_ptr<int32_t>(), kv_indices.data_ptr<int32_t>(),
      last_page_offset.data_ptr<int32_t>(), strides(key), strides(value),
      int(key.size(1)), int(key.size(2)), int(kv_data.size(1)), int(layer_idx),
      int(kv_data.size(4)), float(eps));
  check_launch();
}
