#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_runtime.h>

#include <assert.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <mutex>
#include <set>
#include <sstream>
#include <stdint.h>
#include <string>

using torch::indexing::Slice;

#define HIP_ASSERT(x) (assert((x) == hipSuccess))

// Automatic dispatch:
//   bf16 winners from tuning:
//     N=32/64: small warp-shuffle path.
//     N=256: transpose le256 WMMA.
//     N=128/512/1024/2048: direct WMMA + scalar outer.
//     N=4096: row-centric CUDA-style kernel.
//   fp16 winners from tuning:
//     N=32/64: small warp-shuffle path.
//     N=128: direct-layout le256 WMMA (Llama-2 13B/CodeLlama 34B inner FHT).
//     N=256: transpose-layout le256 WMMA (Qwen2.5-32B inner FHT).
//     N=512: direct dual-H256 butterfly with fused outer H2.
//     N=1024: single-store transpose WMMA (Qwen3/Qwen2.5/Llama-2-70B inner FHT).
//     N=4096: row-centric CUDA-style kernel.
//     N=2048: fp32 LDS fallback.
//   fp32, or unsupported fp16/bf16 sizes: fp32 LDS fallback.
//   fallback requires power-of-two N <= 16384 due to LDS size.

extern "C" __global__ void hadacore_wmma_status_kernel(int* out);

extern "C" __global__ void hadacore_bf16_wmma16_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows
);

extern "C" __global__ void hadacore_bf16_wmma256_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows
);

extern "C" __global__ void hadacore_bf16_wmma_le256_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int transpose_layout
);

extern "C" __global__ void hadacore_bf16_wmma_h256_stage1_direct_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ temp_out,
    int rows,
    int n
);

extern "C" __global__ void hadacore_bf16_wmma_h256_stage1_transpose_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ temp_t,
    int rows,
    int n
);

extern "C" __global__ void hadacore_bf16_wmma_chunk_stage2_direct_kernel(
    const uint16_t* __restrict__ temp_in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int chunks
);

extern "C" __global__ void hadacore_bf16_wmma_chunk_stage2_transpose_kernel(
    const uint16_t* __restrict__ temp_t,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int chunks
);

extern "C" __global__ void hadacore_bf16_wmma_chunk_stage2_direct_smem_kernel(
    const uint16_t* __restrict__ temp_in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int chunks
);

extern "C" __global__ void hadacore_bf16_wmma_chunk_stage2_transpose_smem_kernel(
    const uint16_t* __restrict__ temp_t,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int chunks
);

extern "C" __global__ void hadacore_bf16_wmma_single_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int chunks
);

extern "C" __global__ void hadacore_bf16_wmma_single_store_t_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int chunks
);

extern "C" __global__ void hadacore_bf16_wmma_single_store_t_tile_flush_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int chunks
);

extern "C" __global__ void hadacore_bf16_wmma_single_store_t_uint4_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int chunks
);

extern "C" __global__ void hadacore_bf16_wmma_single_store_t_uint4_tiled_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int chunks
);

extern "C" __global__ void hadacore_bf16_wmma_cuda_style_4096_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows
);

extern "C" __global__ void hadacore_f16_wmma_cuda_style_4096_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows
);

extern "C" __global__ void hadacore_f16_wmma_le256_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int transpose_layout
);

extern "C" __global__ void hadacore_bf16_warp_small_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    float scale
);

extern "C" __global__ void hadacore_f16_warp_small_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    float scale
);

extern "C" __global__ void hadacore_f16_h512_direct_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows
);

extern "C" __global__ void hadacore_f16_wmma_single_store_t_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int chunks
);

extern "C" __global__ void hadacore_f16_wmma_single_store_t_uint4_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int chunks
);

extern "C" __global__ void hadacore_bf16_wmma_single_store_t_4row_flush_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int chunks
);

extern "C" __global__ void hadacore_bf16_wmma_single_store_t_4row_direct_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int chunks
);

extern "C" __global__ void hadacore_bf16_wmma_single_store_t_4row_inner_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int chunks
);

extern "C" __global__ void hadacore_bf16_wmma_single_inner_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int chunks
);

extern "C" __global__ void hadacore_bf16_chunk_stage2_scalar_kernel(
    const uint16_t* __restrict__ temp_in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int chunks
);

extern "C" __global__ void hadacore_bf16_chunk_stage2_scalar_transpose_kernel(
    const uint16_t* __restrict__ temp_t,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int chunks
);

extern "C" __global__ void hadacore_fallback_fp32_kernel(
    float* __restrict__ data,
    int n,
    float scale,
    int logn
);

constexpr bool is_power_of_two(uint32_t x) {
    return x && !(x & (x - 1));
}

static int log2_exact(int n) {
    int logn = 0;
    while ((1 << logn) < n) {
        ++logn;
    }
    return logn;
}

static dim3 choose_fallback_block(int n) {
    if (n <= 512) {
        return dim3(128);
    }
    return dim3(256);
}

static const char* dtype_name(torch::ScalarType dtype) {
    if (dtype == torch::ScalarType::Float) {
        return "float32";
    }
    if (dtype == torch::ScalarType::Half) {
        return "float16";
    }
    if (dtype == torch::ScalarType::BFloat16) {
        return "bfloat16";
    }
    return "unknown";
}

static std::string shape_string(c10::IntArrayRef sizes) {
    std::ostringstream out;
    out << "(";
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << sizes[i];
    }
    if (sizes.size() == 1) {
        out << ",";
    }
    out << ")";
    return out.str();
}

static void print_fallback_once(c10::IntArrayRef shape, torch::ScalarType dtype) {
    static std::mutex mutex;
    static std::set<std::string> seen;

    std::string shape_text = shape_string(shape);
    std::string dtype_text = dtype_name(dtype);
    std::string key = shape_text + "|" + dtype_text;

    std::lock_guard<std::mutex> lock(mutex);
    if (seen.insert(key).second) {
        TORCH_WARN(
            "[hadacore] 此shape是",
            shape_text,
            "，精度是",
            dtype_text,
            "，使用fallback"
        );
    }
}

enum class WmmaMode {
    Auto,
    Fallback,
    Direct,
    DirectWmma,
    Transpose,
    TransposeWmma,
    SingleWmma,
    SingleStoreT,
    SingleStoreTTileFlush,
    SingleStoreTUint4,
    CudaStyle4096,
    SmallWarp,
    SingleStoreT4RowFlush,
    SingleStoreT4RowDirect,
    SingleStoreT4RowInner,
    SingleInner,
    Inner,
    OuterScalar,
};

static WmmaMode parse_wmma_mode(const std::string& mode) {
    if (mode == "auto") {
        return WmmaMode::Auto;
    }
    if (mode == "fallback") {
        return WmmaMode::Fallback;
    }
    if (mode == "transpose") {
        return WmmaMode::Transpose;
    }
    if (mode == "direct") {
        return WmmaMode::Direct;
    }
    if (mode == "direct_wmma") {
        return WmmaMode::DirectWmma;
    }
    if (mode == "inner") {
        return WmmaMode::Inner;
    }
    if (mode == "outer_scalar") {
        return WmmaMode::OuterScalar;
    }
    if (mode == "transpose_wmma") {
        return WmmaMode::TransposeWmma;
    }
    if (mode == "single_wmma") {
        return WmmaMode::SingleWmma;
    }
    if (mode == "single_store_t") {
        return WmmaMode::SingleStoreT;
    }
    if (mode == "single_store_t_tile_flush") {
        return WmmaMode::SingleStoreTTileFlush;
    }
    if (mode == "single_store_t_uint4") {
        return WmmaMode::SingleStoreTUint4;
    }
    if (mode == "cuda_style" || mode == "row_wmma" ||
        mode == "cuda_style_4096" || mode == "row_wmma_4096") {
        return WmmaMode::CudaStyle4096;
    }
    if (mode == "small_warp") {
        return WmmaMode::SmallWarp;
    }
    if (mode == "single_store_t_4row_flush") {
        return WmmaMode::SingleStoreT4RowFlush;
    }
    if (mode == "single_store_t_4row_direct") {
        return WmmaMode::SingleStoreT4RowDirect;
    }
    if (mode == "single_store_t_4row_inner") {
        return WmmaMode::SingleStoreT4RowInner;
    }
    if (mode == "single_inner") {
        return WmmaMode::SingleInner;
    }
    TORCH_CHECK(
        false,
        "Unsupported hadacore WMMA mode: ",
        mode,
        ". Use auto, fallback, small_warp, direct, direct_wmma, transpose, transpose_wmma, single_wmma, single_store_t, single_store_t_tile_flush, single_store_t_uint4, cuda_style, row_wmma, cuda_style_4096, row_wmma_4096, single_store_t_4row_flush, single_store_t_4row_direct, single_store_t_4row_inner, single_inner, inner or outer_scalar."
    );
    return WmmaMode::Transpose;
}

static WmmaMode resolve_auto_wmma_mode(int had_size, WmmaMode mode) {
    if (mode != WmmaMode::Auto) {
        return mode;
    }
    if (had_size == 32 || had_size == 64) {
        return WmmaMode::SmallWarp;
    }
    if (had_size == 256) {
        return WmmaMode::Transpose;
    }
    if (had_size == 128 || had_size == 512 || had_size == 1024 || had_size == 2048) {
        return WmmaMode::Direct;
    }
    if (had_size == 4096) {
        return WmmaMode::CudaStyle4096;
    }
    return WmmaMode::Transpose;
}

static WmmaMode resolve_auto_f16_wmma_mode(int had_size, WmmaMode mode) {
    if (mode != WmmaMode::Auto) {
        return mode;
    }
    if (had_size == 32 || had_size == 64) {
        return WmmaMode::SmallWarp;
    }
    if (had_size == 128) {
        return WmmaMode::Direct;
    }
    if (had_size == 256) {
        return WmmaMode::Transpose;
    }
    if (had_size == 512) {
        return WmmaMode::Direct;
    }
    if (had_size == 1024) {
        return WmmaMode::SingleStoreT;
    }
    if (had_size == 4096) {
        return WmmaMode::CudaStyle4096;
    }
    return WmmaMode::Fallback;
}

static bool cuda_style_supported_size(int had_size) {
    return had_size == 4096;
}

static bool small_warp_supported_size(int had_size) {
    return had_size == 32 || had_size == 64;
}

static bool bf16_wmma_enabled_for_size(int had_size) {
#ifdef HADACORE_ENABLE_EXPERIMENTAL_WMMA
    return had_size >= 16 && had_size <= 4096;
#else
    (void)had_size;
    return false;
#endif
}

static bool f16_wmma_enabled_for_mode(int had_size, WmmaMode mode) {
#ifdef HADACORE_ENABLE_EXPERIMENTAL_WMMA
    WmmaMode resolved = resolve_auto_f16_wmma_mode(had_size, mode);
    if (small_warp_supported_size(had_size)) {
        return resolved == WmmaMode::SmallWarp;
    }
    if (had_size == 128) {
        return resolved == WmmaMode::Direct;
    }
    if (had_size == 256) {
        return resolved == WmmaMode::Transpose;
    }
    if (had_size == 512) {
        return resolved == WmmaMode::Direct;
    }
    if (had_size == 1024) {
        return resolved == WmmaMode::SingleStoreT;
    }
    return cuda_style_supported_size(had_size) &&
           resolved == WmmaMode::CudaStyle4096;
#else
    (void)had_size;
    (void)mode;
    return false;
#endif
}

torch::Tensor wmma_status() {
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);
    torch::Tensor out = torch::empty({1}, options);

    at::hip::HIPGuard device_guard{static_cast<char>(out.get_device())};
    hipStream_t stream = at::hip::getCurrentHIPStream().stream();

    hipLaunchKernelGGL(
        hadacore_wmma_status_kernel,
        dim3(1),
        dim3(1),
        0,
        stream,
        out.data_ptr<int>()
    );
    HIP_ASSERT(hipGetLastError());
    return out;
}

static void launch_fp32_fallback(torch::Tensor work, int had_size, hipStream_t stream) {
    int logn = log2_exact(had_size);
    float scale = 1.0f / std::sqrt(static_cast<float>(had_size));
    size_t smem = static_cast<size_t>(had_size) * sizeof(float);

    TORCH_CHECK(
        smem <= 64 * 1024,
        "fp32 LDS fallback currently supports Hadamard sizes up to 16384 on a 64KB LDS budget; got ",
        had_size,
        ". Extend the hadacore WMMA path for larger sizes instead."
    );

    dim3 grid(static_cast<unsigned int>(work.size(0)));
    dim3 block = choose_fallback_block(had_size);
    float* ptr = work.data_ptr<float>();

    hipLaunchKernelGGL(
        hadacore_fallback_fp32_kernel,
        grid,
        block,
        smem,
        stream,
        ptr,
        had_size,
        scale,
        logn
    );
    HIP_ASSERT(hipGetLastError());
}

static torch::Tensor run_bf16_wmma(
    torch::Tensor x,
    int had_size,
    bool inplace,
    hipStream_t stream,
    WmmaMode mode
) {
    mode = resolve_auto_wmma_mode(had_size, mode);

    int64_t rows = x.size(0);
    int64_t pad_rows = (16 - (rows & 15)) & 15;

    if (pad_rows != 0) {
        x = torch::nn::functional::pad(
            x,
            torch::nn::functional::PadFuncOptions({0, 0, 0, pad_rows})
        );
    }

    bool force_out_of_place =
        (mode == WmmaMode::SingleWmma ||
         mode == WmmaMode::SingleStoreT ||
         mode == WmmaMode::SingleStoreTTileFlush ||
         mode == WmmaMode::SingleStoreTUint4 ||
         mode == WmmaMode::SingleStoreT4RowFlush ||
         mode == WmmaMode::SingleStoreT4RowDirect ||
         mode == WmmaMode::SingleStoreT4RowInner ||
         mode == WmmaMode::SingleInner) &&
        had_size > 256;
    torch::Tensor out = (inplace && pad_rows == 0 && !force_out_of_place) ? x : torch::empty_like(x);
    int padded_rows = static_cast<int>(x.size(0));

    const uint16_t* in_ptr = reinterpret_cast<const uint16_t*>(x.data_ptr());
    uint16_t* out_ptr = reinterpret_cast<uint16_t*>(out.data_ptr());

    unsigned int row_groups = static_cast<unsigned int>(padded_rows / 16);
    dim3 grid(row_groups);
    dim3 block(32);
    size_t h256_smem = 16 * 256 * sizeof(float);

    if (mode == WmmaMode::SmallWarp) {
        TORCH_CHECK(small_warp_supported_size(had_size), "small_warp supports only N=32/64");
        constexpr int warps_per_block = 8;
        dim3 small_block(32 * warps_per_block);
        dim3 small_grid(static_cast<unsigned int>((padded_rows + warps_per_block - 1) / warps_per_block));
        hipLaunchKernelGGL(
            hadacore_bf16_warp_small_kernel,
            small_grid,
            small_block,
            0,
            stream,
            in_ptr,
            out_ptr,
            padded_rows,
            had_size,
            1.0f / std::sqrt(static_cast<float>(had_size))
        );
        HIP_ASSERT(hipGetLastError());
    } else if (had_size <= 256) {
        hipLaunchKernelGGL(
            hadacore_bf16_wmma_le256_kernel,
            grid,
            block,
            h256_smem,
            stream,
            in_ptr,
            out_ptr,
            padded_rows,
            had_size,
            mode == WmmaMode::Transpose ? 1 : 0
        );
        HIP_ASSERT(hipGetLastError());
    } else if (mode == WmmaMode::Inner) {
        torch::Tensor temp = torch::empty_like(x);
        uint16_t* temp_ptr = reinterpret_cast<uint16_t*>(temp.data_ptr());
        int chunks = had_size / 256;

        hipLaunchKernelGGL(
            hadacore_bf16_wmma_h256_stage1_direct_kernel,
            dim3(row_groups, static_cast<unsigned int>(chunks)),
            block,
            h256_smem,
            stream,
            in_ptr,
            temp_ptr,
            padded_rows,
            had_size
        );
        HIP_ASSERT(hipGetLastError());

        out = temp;
    } else if (mode == WmmaMode::SingleWmma) {
        int chunks = had_size / 256;
        size_t smem = 2 * 16 * 16 * 16 * sizeof(uint16_t);

        hipLaunchKernelGGL(
            hadacore_bf16_wmma_single_kernel,
            grid,
            block,
            smem,
            stream,
            in_ptr,
            out_ptr,
            padded_rows,
            had_size,
            chunks
        );
        HIP_ASSERT(hipGetLastError());
    } else if (mode == WmmaMode::SingleStoreT) {
        int chunks = had_size / 256;
        size_t smem = 2 * 16 * 16 * 16 * sizeof(uint16_t);

        hipLaunchKernelGGL(
            hadacore_bf16_wmma_single_store_t_kernel,
            grid,
            block,
            smem,
            stream,
            in_ptr,
            out_ptr,
            padded_rows,
            had_size,
            chunks
        );
        HIP_ASSERT(hipGetLastError());
    } else if (mode == WmmaMode::SingleStoreTTileFlush) {
        int chunks = had_size / 256;
        if (had_size <= 1024) {
            size_t smem =
                2 * 16 * 16 * 16 * sizeof(uint16_t) +
                16 * static_cast<size_t>(had_size) * sizeof(uint16_t);

            hipLaunchKernelGGL(
                hadacore_bf16_wmma_single_store_t_tile_flush_kernel,
                grid,
                block,
                smem,
                stream,
                in_ptr,
                out_ptr,
                padded_rows,
                had_size,
                chunks
            );
            HIP_ASSERT(hipGetLastError());
        } else {
            size_t smem = 2 * 16 * 16 * 16 * sizeof(uint16_t);

            hipLaunchKernelGGL(
                hadacore_bf16_wmma_single_store_t_kernel,
                grid,
                block,
                smem,
                stream,
                in_ptr,
                out_ptr,
                padded_rows,
                had_size,
                chunks
            );
            HIP_ASSERT(hipGetLastError());
        }
    } else if (mode == WmmaMode::SingleStoreTUint4) {
        int chunks = had_size / 256;
        if (had_size <= 1024) {
            size_t smem =
                16 * 256 * sizeof(uint16_t) +
                16 * static_cast<size_t>(had_size) * sizeof(uint16_t);

            hipLaunchKernelGGL(
                hadacore_bf16_wmma_single_store_t_uint4_kernel,
                grid,
                block,
                smem,
                stream,
                in_ptr,
                out_ptr,
                padded_rows,
                had_size,
                chunks
            );
            HIP_ASSERT(hipGetLastError());
        } else {
            size_t smem =
                16 * 256 * sizeof(uint16_t) +
                8 * 8 * 16 * 16 * sizeof(uint16_t);

            hipLaunchKernelGGL(
                hadacore_bf16_wmma_single_store_t_uint4_tiled_kernel,
                grid,
                block,
                smem,
                stream,
                in_ptr,
                out_ptr,
                padded_rows,
                had_size,
                chunks
            );
            HIP_ASSERT(hipGetLastError());
        }
    } else if (mode == WmmaMode::CudaStyle4096) {
        int chunks = had_size / 256;
        if (had_size == 4096) {
            size_t smem = (8 * 256 + 16 * 256) * sizeof(uint16_t);

            hipLaunchKernelGGL(
                hadacore_bf16_wmma_cuda_style_4096_kernel,
                dim3(static_cast<unsigned int>(padded_rows)),
                dim3(8 * 32),
                smem,
                stream,
                in_ptr,
                out_ptr,
                padded_rows
            );
            HIP_ASSERT(hipGetLastError());
        } else if (had_size <= 1024) {
            size_t smem =
                16 * 256 * sizeof(uint16_t) +
                16 * static_cast<size_t>(had_size) * sizeof(uint16_t);

            hipLaunchKernelGGL(
                hadacore_bf16_wmma_single_store_t_uint4_kernel,
                grid,
                block,
                smem,
                stream,
                in_ptr,
                out_ptr,
                padded_rows,
                had_size,
                chunks
            );
            HIP_ASSERT(hipGetLastError());
        } else {
            size_t smem =
                16 * 256 * sizeof(uint16_t) +
                8 * 8 * 16 * 16 * sizeof(uint16_t);

            hipLaunchKernelGGL(
                hadacore_bf16_wmma_single_store_t_uint4_tiled_kernel,
                grid,
                block,
                smem,
                stream,
                in_ptr,
                out_ptr,
                padded_rows,
                had_size,
                chunks
            );
            HIP_ASSERT(hipGetLastError());
        }
    } else if (mode == WmmaMode::SingleStoreT4RowFlush) {
        int chunks = had_size / 256;
        constexpr int rows_per_wave = 4;
        size_t smem =
            2 * 16 * 16 * 16 * sizeof(uint16_t) +
            rows_per_wave * static_cast<size_t>(had_size) * sizeof(uint16_t);

        hipLaunchKernelGGL(
            hadacore_bf16_wmma_single_store_t_4row_flush_kernel,
            dim3(static_cast<unsigned int>((padded_rows + rows_per_wave - 1) / rows_per_wave)),
            block,
            smem,
            stream,
            in_ptr,
            out_ptr,
            padded_rows,
            had_size,
            chunks
        );
        HIP_ASSERT(hipGetLastError());
    } else if (mode == WmmaMode::SingleStoreT4RowDirect) {
        int chunks = had_size / 256;
        constexpr int rows_per_wave = 4;
        size_t smem = 2 * 16 * 16 * 16 * sizeof(uint16_t);

        hipLaunchKernelGGL(
            hadacore_bf16_wmma_single_store_t_4row_direct_kernel,
            dim3(static_cast<unsigned int>((padded_rows + rows_per_wave - 1) / rows_per_wave)),
            block,
            smem,
            stream,
            in_ptr,
            out_ptr,
            padded_rows,
            had_size,
            chunks
        );
        HIP_ASSERT(hipGetLastError());
    } else if (mode == WmmaMode::SingleStoreT4RowInner) {
        int chunks = had_size / 256;
        constexpr int rows_per_wave = 4;
        size_t smem = 16 * 16 * 16 * sizeof(uint16_t);

        hipLaunchKernelGGL(
            hadacore_bf16_wmma_single_store_t_4row_inner_kernel,
            dim3(static_cast<unsigned int>((padded_rows + rows_per_wave - 1) / rows_per_wave)),
            block,
            smem,
            stream,
            in_ptr,
            out_ptr,
            padded_rows,
            had_size,
            chunks
        );
        HIP_ASSERT(hipGetLastError());
    } else if (mode == WmmaMode::SingleInner) {
        int chunks = had_size / 256;
        size_t smem = 16 * 16 * 16 * sizeof(uint16_t);

        hipLaunchKernelGGL(
            hadacore_bf16_wmma_single_inner_kernel,
            grid,
            block,
            smem,
            stream,
            in_ptr,
            out_ptr,
            padded_rows,
            had_size,
            chunks
        );
        HIP_ASSERT(hipGetLastError());
    } else if (mode == WmmaMode::DirectWmma) {
        torch::Tensor temp = torch::empty_like(x);
        uint16_t* temp_ptr = reinterpret_cast<uint16_t*>(temp.data_ptr());
        int chunks = had_size / 256;

        hipLaunchKernelGGL(
            hadacore_bf16_wmma_h256_stage1_direct_kernel,
            dim3(row_groups, static_cast<unsigned int>(chunks)),
            block,
            h256_smem,
            stream,
            in_ptr,
            temp_ptr,
            padded_rows,
            had_size
        );
        HIP_ASSERT(hipGetLastError());

        hipLaunchKernelGGL(
            hadacore_bf16_wmma_chunk_stage2_direct_smem_kernel,
            dim3(row_groups, 256),
            block,
            16 * 16 * sizeof(uint16_t),
            stream,
            temp_ptr,
            out_ptr,
            padded_rows,
            had_size,
            chunks
        );
        HIP_ASSERT(hipGetLastError());
    } else if (mode == WmmaMode::Direct || mode == WmmaMode::OuterScalar) {
        torch::Tensor temp = torch::empty_like(x);
        uint16_t* temp_ptr = reinterpret_cast<uint16_t*>(temp.data_ptr());
        int chunks = had_size / 256;

        hipLaunchKernelGGL(
            hadacore_bf16_wmma_h256_stage1_direct_kernel,
            dim3(row_groups, static_cast<unsigned int>(chunks)),
            block,
            h256_smem,
            stream,
            in_ptr,
            temp_ptr,
            padded_rows,
            had_size
        );
        HIP_ASSERT(hipGetLastError());

        hipLaunchKernelGGL(
            hadacore_bf16_chunk_stage2_scalar_kernel,
            dim3(static_cast<unsigned int>(padded_rows)),
            dim3(256),
            0,
            stream,
            temp_ptr,
            out_ptr,
            padded_rows,
            had_size,
            chunks
        );
        HIP_ASSERT(hipGetLastError());
    } else {
        int chunks = had_size / 256;
        torch::Tensor temp_t = torch::empty({padded_rows, static_cast<int64_t>(4096)}, x.options());
        uint16_t* temp_t_ptr = reinterpret_cast<uint16_t*>(temp_t.data_ptr());

        hipLaunchKernelGGL(
            hadacore_bf16_wmma_h256_stage1_transpose_kernel,
            dim3(row_groups, static_cast<unsigned int>(chunks)),
            block,
            h256_smem,
            stream,
            in_ptr,
            temp_t_ptr,
            padded_rows,
            had_size
        );
        HIP_ASSERT(hipGetLastError());

        if (mode == WmmaMode::TransposeWmma) {
            hipLaunchKernelGGL(
                hadacore_bf16_wmma_chunk_stage2_transpose_smem_kernel,
                dim3(row_groups, 256),
                block,
                16 * 16 * sizeof(uint16_t),
                stream,
                temp_t_ptr,
                out_ptr,
                padded_rows,
                had_size,
                chunks
            );
            HIP_ASSERT(hipGetLastError());
        } else {
            hipLaunchKernelGGL(
                hadacore_bf16_chunk_stage2_scalar_transpose_kernel,
                dim3(static_cast<unsigned int>(padded_rows)),
                dim3(256),
                0,
                stream,
                temp_t_ptr,
                out_ptr,
                padded_rows,
                had_size,
                chunks
            );
            HIP_ASSERT(hipGetLastError());
        }
    }

    if (pad_rows != 0) {
        out = out.index({Slice(0, rows)});
    }
    return out;
}

static torch::Tensor run_f16_wmma(
    torch::Tensor x,
    int had_size,
    bool inplace,
    hipStream_t stream,
    WmmaMode mode
) {
    mode = resolve_auto_f16_wmma_mode(had_size, mode);
    int64_t rows = x.size(0);

    if (had_size == 512 && mode == WmmaMode::Direct) {
        torch::Tensor out = inplace ? x : torch::empty_like(x);
        hipLaunchKernelGGL(
            hadacore_f16_h512_direct_kernel,
            dim3(static_cast<unsigned int>(rows)),
            dim3(256),
            0,
            stream,
            reinterpret_cast<const uint16_t*>(x.data_ptr()),
            reinterpret_cast<uint16_t*>(out.data_ptr()),
            static_cast<int>(rows)
        );
        HIP_ASSERT(hipGetLastError());
        return out;
    }

    if (mode == WmmaMode::CudaStyle4096) {
        TORCH_CHECK(cuda_style_supported_size(had_size), "fp16 cuda_style currently supports only N=4096");
        torch::Tensor out = inplace ? x : torch::empty_like(x);

        const uint16_t* in_ptr = reinterpret_cast<const uint16_t*>(x.data_ptr());
        uint16_t* out_ptr = reinterpret_cast<uint16_t*>(out.data_ptr());
        size_t smem = (8 * 256 + 16 * 256) * sizeof(uint16_t);

        hipLaunchKernelGGL(
            hadacore_f16_wmma_cuda_style_4096_kernel,
            dim3(static_cast<unsigned int>(rows)),
            dim3(8 * 32),
            smem,
            stream,
            in_ptr,
            out_ptr,
            static_cast<int>(rows)
        );
        HIP_ASSERT(hipGetLastError());
        return out;
    }

    int64_t pad_rows = (16 - (rows & 15)) & 15;
    if (pad_rows != 0) {
        x = torch::nn::functional::pad(
            x,
            torch::nn::functional::PadFuncOptions({0, 0, 0, pad_rows})
        );
    }

    bool force_out_of_place =
        (mode == WmmaMode::SingleStoreT ||
         mode == WmmaMode::SingleStoreTUint4) &&
        had_size > 256;
    torch::Tensor out = (inplace && pad_rows == 0 && !force_out_of_place) ? x : torch::empty_like(x);
    int padded_rows = static_cast<int>(x.size(0));

    const uint16_t* in_ptr = reinterpret_cast<const uint16_t*>(x.data_ptr());
    uint16_t* out_ptr = reinterpret_cast<uint16_t*>(out.data_ptr());
    unsigned int row_groups = static_cast<unsigned int>(padded_rows / 16);
    dim3 grid(row_groups);
    dim3 block(32);

    if (mode == WmmaMode::SmallWarp) {
        TORCH_CHECK(small_warp_supported_size(had_size), "small_warp supports only N=32/64");
        constexpr int warps_per_block = 8;
        dim3 small_block(32 * warps_per_block);
        dim3 small_grid(static_cast<unsigned int>((padded_rows + warps_per_block - 1) / warps_per_block));
        hipLaunchKernelGGL(
            hadacore_f16_warp_small_kernel,
            small_grid,
            small_block,
            0,
            stream,
            in_ptr,
            out_ptr,
            padded_rows,
            had_size,
            1.0f / std::sqrt(static_cast<float>(had_size))
        );
        HIP_ASSERT(hipGetLastError());
    } else if (had_size <= 256) {
        size_t smem = 16 * 256 * sizeof(float);
        hipLaunchKernelGGL(
            hadacore_f16_wmma_le256_kernel,
            grid,
            block,
            smem,
            stream,
            in_ptr,
            out_ptr,
            padded_rows,
            had_size,
            mode == WmmaMode::Transpose ? 1 : 0
        );
        HIP_ASSERT(hipGetLastError());
    } else if (mode == WmmaMode::SingleStoreTUint4) {
        int chunks = had_size / 256;
        size_t smem =
            16 * 256 * sizeof(uint16_t) +
            16 * static_cast<size_t>(had_size) * sizeof(uint16_t);

        hipLaunchKernelGGL(
            hadacore_f16_wmma_single_store_t_uint4_kernel,
            grid,
            block,
            smem,
            stream,
            in_ptr,
            out_ptr,
            padded_rows,
            had_size,
            chunks
        );
        HIP_ASSERT(hipGetLastError());
    } else if (mode == WmmaMode::SingleStoreT) {
        int chunks = had_size / 256;
        size_t smem = 2 * 16 * 16 * 16 * sizeof(uint16_t);

        hipLaunchKernelGGL(
            hadacore_f16_wmma_single_store_t_kernel,
            grid,
            block,
            smem,
            stream,
            in_ptr,
            out_ptr,
            padded_rows,
            had_size,
            chunks
        );
        HIP_ASSERT(hipGetLastError());
    } else {
        TORCH_CHECK(false, "unsupported fp16 hadacore WMMA mode for this N");
    }

    if (pad_rows != 0) {
        out = out.index({Slice(0, rows)});
    }
    return out;
}

torch::Tensor hadamard_transform(at::Tensor& in, bool inplace, const std::string& mode) {
    TORCH_CHECK(in.dim() >= 1, "input must have at least one dimension");
    TORCH_CHECK(in.is_cuda(), "input must be a CUDA/HIP tensor");

    auto dtype = in.scalar_type();
    TORCH_CHECK(
        dtype == torch::ScalarType::Float ||
        dtype == torch::ScalarType::Half ||
        dtype == torch::ScalarType::BFloat16,
        "Only fp32, fp16 and bf16 are supported currently"
    );

    int had_size = static_cast<int>(in.size(-1));
    TORCH_CHECK(
        is_power_of_two(static_cast<uint32_t>(had_size)) && had_size <= (1 << 14),
        "Only power-of-two Hadamard sizes up to 16384 are supported, got ",
        had_size
    );

    const auto result_shape = in.sizes();
    torch::Tensor x = in.reshape({-1, had_size});
    if (x.stride(-1) != 1) {
        x = x.contiguous();
    }

    at::hip::HIPGuard device_guard{static_cast<char>(x.get_device())};
    hipStream_t stream = at::hip::getCurrentHIPStream().stream();

    torch::Tensor out;
    WmmaMode parsed_mode = parse_wmma_mode(mode);
    bool force_fallback = parsed_mode == WmmaMode::Fallback;
    bool use_bf16_wmma =
        !force_fallback &&
        dtype == torch::ScalarType::BFloat16 &&
        bf16_wmma_enabled_for_size(had_size);
    bool use_f16_wmma =
        !force_fallback &&
        dtype == torch::ScalarType::Half &&
        f16_wmma_enabled_for_mode(had_size, parsed_mode);

    if (use_bf16_wmma) {
        out = run_bf16_wmma(x, had_size, inplace, stream, parsed_mode);
    } else if (use_f16_wmma) {
        out = run_f16_wmma(x, had_size, inplace, stream, parsed_mode);
    } else {
        print_fallback_once(result_shape, dtype);

        torch::Tensor work;
        if (dtype == torch::ScalarType::Float) {
            work = inplace ? x : x.clone();
        } else {
            work = x.to(torch::kFloat32);
        }

        launch_fp32_fallback(work, had_size, stream);
        out = (dtype == torch::ScalarType::Float) ? work : work.to(dtype);
    }

    if (inplace && out.data_ptr() != in.data_ptr()) {
        in.copy_(out.reshape(result_shape));
        return in;
    }

    return out.reshape(result_shape);
}

torch::Tensor fast_hadamard(torch::Tensor in) {
    return hadamard_transform(in, false, "auto");
}

torch::Tensor hadacore(at::Tensor& in, double scale) {
    torch::Tensor out = hadamard_transform(in, false, "auto");
    double default_scale = 1.0 / std::sqrt(static_cast<double>(in.size(-1)));
    double ratio = scale / default_scale;
    if (std::abs(ratio - 1.0) < 1e-12) {
        return out;
    }
    return out * ratio;
}


namespace py = pybind11;
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hadamard_transform", &hadacore, py::arg("x"), py::arg("scale"));
}
