// Selected Wave FHT kernels from QuaRot_Version, Hadacore commit
// 6c30374901038138ac5094e6eaed35a5ca35db2f. See source_manifest.json.
#include <hip/hip_runtime.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <torch/extension.h>
#include <cmath>
#include <climits>

__device__ __forceinline__ uint16_t f32_to_bf16_bits(float x)
{
    uint32_t bits = __float_as_uint(x);
    const uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

__device__ __forceinline__ float bf16_bits_to_f32(uint16_t x)
{
    return __uint_as_float(static_cast<uint32_t>(x) << 16);
}

__device__ __forceinline__ _Float16 f16_bits_to_f16(uint16_t x)
{
    union {
        uint16_t u;
        _Float16 h;
    } value;
    value.u = x;
    return value.h;
}

__device__ __forceinline__ uint16_t f16_to_bits(_Float16 x)
{
    union {
        _Float16 h;
        uint16_t u;
    } value;
    value.h = x;
    return value.u;
}

__device__ __forceinline__ uint16_t f32_to_f16_bits(float x)
{
    return f16_to_bits(static_cast<_Float16>(x));
}

__device__ __forceinline__ float shfl_xor_f32(float value, int mask)
{
    return __shfl_xor_sync(0xffffffffULL, value, mask, 32);
}

template <bool IsBf16>
__device__ __forceinline__ float load_f16_or_bf16(uint16_t bits)
{
    return IsBf16
        ? bf16_bits_to_f32(bits)
        : static_cast<float>(f16_bits_to_f16(bits));
}

template <bool IsBf16>
__device__ __forceinline__ uint16_t store_f16_or_bf16(float value)
{
    return IsBf16 ? f32_to_bf16_bits(value) : f32_to_f16_bits(value);
}

// N=32/64. The low five index bits cross lanes through shuffles. For N=64,
// the sixth bit is the final local butterfly between the two registers.
template <bool IsBf16>
__device__ __forceinline__ void hadacore_small_warp_body(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    float scale)
{
    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int waves_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * waves_per_block + wave;
    if (row >= rows) {
        return;
    }

    const int registers = n >> 5;
    const uint16_t* in_row = in + static_cast<size_t>(row) * n;
    uint16_t* out_row = out + static_cast<size_t>(row) * n;
    float values[2] = {0.0f, 0.0f};

#pragma unroll
    for (int r = 0; r < 2; ++r) {
        if (r < registers) {
            values[r] = load_f16_or_bf16<IsBf16>(in_row[r * 32 + lane]);
        }
    }

#pragma unroll
    for (int mask = 1; mask < 32; mask <<= 1) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            if (r < registers) {
                const float a = values[r];
                const float b = shfl_xor_f32(a, mask);
                values[r] = (lane & mask) == 0 ? a + b : b - a;
            }
        }
    }

    if (registers == 2) {
        const float a = values[0];
        const float b = values[1];
        values[0] = a + b;
        values[1] = a - b;
    }

#pragma unroll
    for (int r = 0; r < 2; ++r) {
        if (r < registers) {
            out_row[r * 32 + lane] =
                store_f16_or_bf16<IsBf16>(values[r] * scale);
        }
    }
}

#define DEFINE_SMALL_WARP_KERNEL(NAME, IS_BF16)                          \
    extern "C" __global__ void NAME(                                    \
        const uint16_t* __restrict__ in,                                  \
        uint16_t* __restrict__ out,                                       \
        int rows,                                                         \
        int n,                                                            \
        float scale)                                                      \
    {                                                                     \
        hadacore_small_warp_body<IS_BF16>(in, out, rows, n, scale);       \
    }

DEFINE_SMALL_WARP_KERNEL(hadacore_f16_warp_small_kernel, false)
DEFINE_SMALL_WARP_KERNEL(hadacore_bf16_warp_small_kernel, true)

#undef DEFINE_SMALL_WARP_KERNEL

// N=128..2048. A wave owns a complete row distributed as
// value[r * 32 + lane]. H32 is performed across lanes, while H(N/32) is
// performed locally on each lane's register array. No LDS round-trip or
// workgroup barrier is required.
template <int N, bool IsBf16>
__device__ __forceinline__ void hadacore_warp_regs_body(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows)
{
    constexpr int kRegisters = N / 32;
    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int waves_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * waves_per_block + wave;
    if (row >= rows) {
        return;
    }

    const uint16_t* in_row = in + static_cast<size_t>(row) * N;
    uint16_t* out_row = out + static_cast<size_t>(row) * N;
    float values[kRegisters];

#pragma unroll
    for (int r = 0; r < kRegisters; ++r) {
        values[r] = load_f16_or_bf16<IsBf16>(in_row[r * 32 + lane]);
    }

#pragma unroll
    for (int mask = 1; mask < 32; mask <<= 1) {
#pragma unroll
        for (int r = 0; r < kRegisters; ++r) {
            const float a = values[r];
            const float b = shfl_xor_f32(a, mask);
            values[r] = (lane & mask) == 0 ? a + b : b - a;
        }
    }

#pragma unroll
    for (int stride = 1; stride < kRegisters; stride <<= 1) {
#pragma unroll
        for (int base = 0; base < kRegisters; base += stride << 1) {
#pragma unroll
            for (int offset = 0; offset < stride; ++offset) {
                const float a = values[base + offset];
                const float b = values[base + stride + offset];
                values[base + offset] = a + b;
                values[base + stride + offset] = a - b;
            }
        }
    }

    const float scale = rsqrtf(static_cast<float>(N));
#pragma unroll
    for (int r = 0; r < kRegisters; ++r) {
        out_row[r * 32 + lane] =
            store_f16_or_bf16<IsBf16>(values[r] * scale);
    }
}

#define DEFINE_WARP_REGS_KERNEL(NAME, N, IS_BF16)                        \
    extern "C" __global__ void NAME(                                    \
        const uint16_t* __restrict__ in,                                  \
        uint16_t* __restrict__ out,                                       \
        int rows)                                                         \
    {                                                                     \
        hadacore_warp_regs_body<N, IS_BF16>(in, out, rows);              \
    }

DEFINE_WARP_REGS_KERNEL(hadacore_f16_warp_regs_128_kernel, 128, false)
DEFINE_WARP_REGS_KERNEL(hadacore_f16_warp_regs_256_kernel, 256, false)
DEFINE_WARP_REGS_KERNEL(hadacore_bf16_warp_regs_128_kernel, 128, true)
DEFINE_WARP_REGS_KERNEL(hadacore_bf16_warp_regs_256_kernel, 256, true)

#undef DEFINE_WARP_REGS_KERNEL

namespace {
constexpr int kWavesPerBlock = 8;

void check_hip(hipError_t status)
{
    TORCH_CHECK(status == hipSuccess, hipGetErrorString(status));
}

dim3 row_grid(int64_t rows)
{
    return dim3(static_cast<unsigned int>(
        (rows + kWavesPerBlock - 1) / kWavesPerBlock));
}

torch::Tensor launch_small_warp(
    torch::Tensor x,
    int n,
    hipStream_t stream)
{
    auto out = torch::empty_like(x);
    const int rows = static_cast<int>(x.size(0));
    const dim3 block(32 * kWavesPerBlock);
    const float scale = 1.0f / std::sqrt(static_cast<float>(n));
    const auto* input = reinterpret_cast<const uint16_t*>(x.data_ptr());
    auto* output = reinterpret_cast<uint16_t*>(out.data_ptr());

    if (x.scalar_type() == torch::ScalarType::BFloat16) {
        hipLaunchKernelGGL(
            hadacore_bf16_warp_small_kernel,
            row_grid(rows),
            block,
            0,
            stream,
            input,
            output,
            rows,
            n,
            scale);
    } else {
        hipLaunchKernelGGL(
            hadacore_f16_warp_small_kernel,
            row_grid(rows),
            block,
            0,
            stream,
            input,
            output,
            rows,
            n,
            scale);
    }
    check_hip(hipGetLastError());
    return out;
}

torch::Tensor launch_warp_regs(
    torch::Tensor x,
    int n,
    hipStream_t stream)
{
    auto out = torch::empty_like(x);
    const int rows = static_cast<int>(x.size(0));
    const dim3 grid = row_grid(rows);
    const dim3 block(32 * kWavesPerBlock);
    const auto* input = reinterpret_cast<const uint16_t*>(x.data_ptr());
    auto* output = reinterpret_cast<uint16_t*>(out.data_ptr());
    const bool bf16 = x.scalar_type() == torch::ScalarType::BFloat16;

#define LAUNCH_WARP_REGS(N)                                               \
    do {                                                                  \
        if (bf16) {                                                       \
            hipLaunchKernelGGL(                                          \
                hadacore_bf16_warp_regs_##N##_kernel,                    \
                grid, block, 0, stream, input, output, rows);            \
        } else {                                                          \
            hipLaunchKernelGGL(                                          \
                hadacore_f16_warp_regs_##N##_kernel,                     \
                grid, block, 0, stream, input, output, rows);            \
        }                                                                 \
    } while (false)

    switch (n) {
    case 128:
        LAUNCH_WARP_REGS(128);
        break;
    case 256:
        LAUNCH_WARP_REGS(256);
        break;
    default:
        TORCH_CHECK(false, "warp_regs does not support N=", n);
    }

#undef LAUNCH_WARP_REGS

    check_hip(hipGetLastError());
    return out;
}

} // namespace

// Standalone binding for the four report-selected widths. The operation and
// scaling order match the original dispatcher for these widths and dtypes.
torch::Tensor wave_hadamard(torch::Tensor input, float scale)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA/HIP tensor");
    TORCH_CHECK(input.dim() >= 1, "input must have at least one dimension");
    TORCH_CHECK(input.scalar_type() == torch::kHalf ||
                input.scalar_type() == torch::kBFloat16,
                "Wave FHT supports float16 and bfloat16");
    const int n = static_cast<int>(input.size(-1));
    TORCH_CHECK(n == 32 || n == 64 || n == 128 || n == 256,
                "Wave FHT supports widths 32, 64, 128 and 256");
    const auto shape = input.sizes().vec();
    auto x = input.reshape({-1, n});
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    if (x.numel() == 0) {
        return torch::empty_like(input);
    }
    TORCH_CHECK(x.size(0) <= INT32_MAX, "too many rows for the Hadamard grid");
    at::hip::HIPGuard device_guard{static_cast<char>(x.get_device())};
    const hipStream_t stream = at::hip::getCurrentHIPStream().stream();
    auto out = (n == 32 || n == 64)
        ? launch_small_warp(x, n, stream)
        : launch_warp_regs(x, n, stream);
    out = out.reshape(shape);
    const float normalized_scale = static_cast<float>(
        1.0 / std::sqrt(static_cast<double>(n))
    );
    const float scale_ratio = scale / normalized_scale;
    return std::abs(scale_ratio - 1.0f) <= 1e-6f
        ? out
        : out * scale_ratio;
}
