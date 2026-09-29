#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

// Hadacore-style HIP kernels for gfx12 WMMA plus a fp32 LDS fallback.
// Public dispatch lives in binding.hip and selects the fastest tuned path by
// dtype and Hadamard size.

typedef short bf16_frag __attribute__((ext_vector_type(8)));
typedef _Float16 f16_frag __attribute__((ext_vector_type(8)));
typedef float f32_frag __attribute__((ext_vector_type(8)));

#if defined(HADACORE_ENABLE_EXPERIMENTAL_WMMA) && defined(HADACORE_FORCE_GFX12_WMMA)
#define HADACORE_GFX12_WMMA 1
#elif defined(HADACORE_ENABLE_EXPERIMENTAL_WMMA) && defined(__HIP_DEVICE_COMPILE__) && \
    (defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx1250__))
#define HADACORE_GFX12_WMMA 1
#else
#define HADACORE_GFX12_WMMA 0
#endif

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

#if defined(HADACORE_ENABLE_EXPERIMENTAL_WMMA) && defined(HADACORE_FORCE_GFX12_WMMA)
#define HADACORE_HAS_F32_BF16_WMMA 1
#elif HADACORE_GFX12_WMMA && \
    __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12)
#define HADACORE_HAS_F32_BF16_WMMA 1
#else
#define HADACORE_HAS_F32_BF16_WMMA 0
#endif

#if defined(HADACORE_ENABLE_EXPERIMENTAL_WMMA) && defined(HADACORE_FORCE_GFX12_WMMA)
#define HADACORE_HAS_F32_F16_WMMA 1
#elif HADACORE_GFX12_WMMA && \
    __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12)
#define HADACORE_HAS_F32_F16_WMMA 1
#else
#define HADACORE_HAS_F32_F16_WMMA 0
#endif

extern "C" __global__ void hadacore_wmma_status_kernel(int* out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = HADACORE_HAS_F32_BF16_WMMA;
    }
}

__device__ __forceinline__ uint16_t f32_to_bf16_bits(float x) {
    uint32_t bits = __float_as_uint(x);
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

__device__ __forceinline__ float bf16_bits_to_f32(uint16_t x) {
    return __uint_as_float(static_cast<uint32_t>(x) << 16);
}

__device__ __forceinline__ _Float16 f16_bits_to_f16(uint16_t x) {
    union {
        uint16_t u;
        _Float16 h;
    } v;
    v.u = x;
    return v.h;
}

__device__ __forceinline__ uint16_t f16_to_bits(_Float16 x) {
    union {
        _Float16 h;
        uint16_t u;
    } v;
    v.h = x;
    return v.u;
}

__device__ __forceinline__ uint16_t f32_to_f16_bits(float x) {
    return f16_to_bits(static_cast<_Float16>(x));
}

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
    return static_cast<uint32_t>(f32_to_bf16_bits(lo)) |
           (static_cast<uint32_t>(f32_to_bf16_bits(hi)) << 16);
}

__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
    return static_cast<uint32_t>(f32_to_f16_bits(lo)) |
           (static_cast<uint32_t>(f32_to_f16_bits(hi)) << 16);
}

__device__ __forceinline__ float shfl_xor_f32(float value, int mask) {
    return __shfl_xor_sync(0xffffffffULL, value, mask, 32);
}

__device__ __forceinline__ void warp_hadamard_regs(float vals[2], int regs, int lane) {
    for (int mask = 1; mask < 32; mask <<= 1) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            if (r < regs) {
                float a = vals[r];
                float b = shfl_xor_f32(a, mask);
                vals[r] = (lane & mask) == 0 ? a + b : b - a;
            }
        }
    }

    if (regs == 2) {
        float a = vals[0];
        float b = vals[1];
        vals[0] = a + b;
        vals[1] = a - b;
    }
}

extern "C" __global__ void hadacore_bf16_warp_small_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    float scale
) {
    int lane = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp_in_block;
    if (row >= rows) {
        return;
    }

    int regs = n >> 5;
    const uint16_t* in_row = in + row * n;
    uint16_t* out_row = out + row * n;
    float vals[2];

#pragma unroll
    for (int r = 0; r < 2; ++r) {
        vals[r] = (r < regs) ? bf16_bits_to_f32(in_row[r * 32 + lane]) : 0.0f;
    }

    warp_hadamard_regs(vals, regs, lane);

#pragma unroll
    for (int r = 0; r < 2; ++r) {
        if (r < regs) {
            out_row[r * 32 + lane] = f32_to_bf16_bits(vals[r] * scale);
        }
    }
}

extern "C" __global__ void hadacore_f16_warp_small_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    float scale
) {
    int lane = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp_in_block;
    if (row >= rows) {
        return;
    }

    int regs = n >> 5;
    const uint16_t* in_row = in + row * n;
    uint16_t* out_row = out + row * n;
    float vals[2];

#pragma unroll
    for (int r = 0; r < 2; ++r) {
        vals[r] = (r < regs) ? static_cast<float>(f16_bits_to_f16(in_row[r * 32 + lane])) : 0.0f;
    }

    warp_hadamard_regs(vals, regs, lane);

#pragma unroll
    for (int r = 0; r < 2; ++r) {
        if (r < regs) {
            out_row[r * 32 + lane] = f32_to_f16_bits(vals[r] * scale);
        }
    }
}

__device__ __forceinline__ float shfl_f32(float value, int src_lane) {
    return __shfl(value, src_lane, 32);
}

__device__ __forceinline__ float shfl_frag_elem(f32_frag frag, int elem, int src_lane) {
    float v0 = shfl_f32(frag[0], src_lane);
    float v1 = shfl_f32(frag[1], src_lane);
    float v2 = shfl_f32(frag[2], src_lane);
    float v3 = shfl_f32(frag[3], src_lane);
    float v4 = shfl_f32(frag[4], src_lane);
    float v5 = shfl_f32(frag[5], src_lane);
    float v6 = shfl_f32(frag[6], src_lane);
    float v7 = shfl_f32(frag[7], src_lane);

    switch (elem) {
        case 0:
            return v0;
        case 1:
            return v1;
        case 2:
            return v2;
        case 3:
            return v3;
        case 4:
            return v4;
        case 5:
            return v5;
        case 6:
            return v6;
        default:
            return v7;
    }
}

__device__ __forceinline__ f32_frag zero_f32_frag() {
    f32_frag frag;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        frag[i] = 0.0f;
    }
    return frag;
}

__device__ __forceinline__ bf16_frag load_a_rowmajor_16x16(
    const uint16_t* base,
    int stride,
    int lane
) {
    bf16_frag frag;
    int row = lane & 15;
    int k_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        frag[i] = static_cast<short>(base[row * stride + k_base + i]);
    }
    return frag;
}

__device__ __forceinline__ bf16_frag load_a_rowmajor_rows_padded_16x16(
    const uint16_t* base,
    int stride,
    int valid_rows,
    int lane
) {
    bf16_frag frag;
    int row = lane & 15;
    int k_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (row < valid_rows) {
            frag[i] = static_cast<short>(base[row * stride + k_base + i]);
        } else {
            frag[i] = static_cast<short>(0);
        }
    }
    return frag;
}

__device__ __forceinline__ bf16_frag load_a_high_axis_16x16(
    const uint16_t* base,
    int low_col,
    int lane
) {
    bf16_frag frag;
    int row = lane & 15;
    int k_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int high = k_base + i;
        frag[i] = static_cast<short>(base[row * 256 + high * 16 + low_col]);
    }
    return frag;
}

__device__ __forceinline__ void store_d_rowmajor_16x16(
    uint16_t* base,
    int stride,
    int lane,
    f32_frag frag
) {
    int col = lane & 15;
    int row_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        base[(row_base + i) * stride + col] = f32_to_bf16_bits(frag[i]);
    }
}

__device__ __forceinline__ bf16_frag load_a_transposed_square_16x16(
    const uint16_t* base,
    int lane
) {
    bf16_frag frag;
    int row = lane & 15;
    int k_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        frag[i] = static_cast<short>(base[(k_base + i) * 16 + row]);
    }
    return frag;
}

__device__ __forceinline__ void store_d_transposed_square_16x16(
    uint16_t* base,
    int lane,
    f32_frag frag
) {
    int col = lane & 15;
    int row_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        base[col * 16 + row_base + i] = f32_to_bf16_bits(frag[i]);
    }
}

__device__ __forceinline__ void store_d_high_axis_16x16(
    uint16_t* base,
    int low_col,
    int lane,
    f32_frag frag
) {
    int high_out = lane & 15;
    int row_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        base[(row_base + i) * 256 + high_out * 16 + low_col] = f32_to_bf16_bits(frag[i]);
    }
}

__device__ __forceinline__ bf16_frag load_a_contiguous_k_padded_16x16(
    const uint16_t* base,
    int row_stride,
    int valid_k,
    int lane
) {
    bf16_frag frag;
    int row = lane & 15;
    int k_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int k = k_base + i;
        if (k < valid_k) {
            frag[i] = static_cast<short>(base[row * row_stride + k]);
        } else {
            frag[i] = static_cast<short>(0);
        }
    }
    return frag;
}

__device__ __forceinline__ bf16_frag load_a_strided_axis_padded_16x16(
    const uint16_t* base,
    int row_stride,
    int axis_stride,
    int valid_k,
    int low,
    int lane
) {
    bf16_frag frag;
    int row = lane & 15;
    int k_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int k = k_base + i;
        if (k < valid_k) {
            frag[i] = static_cast<short>(base[row * row_stride + k * axis_stride + low]);
        } else {
            frag[i] = static_cast<short>(0);
        }
    }
    return frag;
}

__device__ __forceinline__ void store_d_strided_axis_16x16(
    uint16_t* base,
    int row_stride,
    int axis_stride,
    int low,
    int valid_cols,
    int lane,
    f32_frag frag
) {
    int col = lane & 15;
    int row_base = (lane >> 4) * 8;
    if (col < valid_cols) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            base[(row_base + i) * row_stride + col * axis_stride + low] =
                f32_to_bf16_bits(frag[i]);
        }
    }
}

__device__ __forceinline__ void store_d_strided_axis_rows_16x16(
    uint16_t* base,
    int row_stride,
    int axis_stride,
    int low,
    int valid_cols,
    int valid_rows,
    int lane,
    f32_frag frag
) {
    int col = lane & 15;
    int row_base = (lane >> 4) * 8;
    if (col < valid_cols) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int row = row_base + i;
            if (row < valid_rows) {
                base[row * row_stride + col * axis_stride + low] =
                    f32_to_bf16_bits(frag[i]);
            }
        }
    }
}

__device__ __forceinline__ void store_d_low8_uint4_strided_axis_16x16(
    uint16_t* base,
    int row_stride,
    int axis_stride,
    int low_base,
    int valid_cols,
    int lane,
    f32_frag d0,
    f32_frag d1,
    f32_frag d2,
    f32_frag d3,
    f32_frag d4,
    f32_frag d5,
    f32_frag d6,
    f32_frag d7
) {
    int row = lane & 15;
    int row_elem = row & 7;
    int src_lane_hi = (row >= 8) ? 16 : 0;

    for (int col_base = 0; col_base < valid_cols; col_base += 2) {
        int col = col_base + (lane >> 4);
        if (col < valid_cols) {
            int src_lane = src_lane_hi + col;
            uint4 packed;
            packed.x = pack_bf16x2(
                shfl_frag_elem(d0, row_elem, src_lane),
                shfl_frag_elem(d1, row_elem, src_lane)
            );
            packed.y = pack_bf16x2(
                shfl_frag_elem(d2, row_elem, src_lane),
                shfl_frag_elem(d3, row_elem, src_lane)
            );
            packed.z = pack_bf16x2(
                shfl_frag_elem(d4, row_elem, src_lane),
                shfl_frag_elem(d5, row_elem, src_lane)
            );
            packed.w = pack_bf16x2(
                shfl_frag_elem(d6, row_elem, src_lane),
                shfl_frag_elem(d7, row_elem, src_lane)
            );
            uint4* dst = reinterpret_cast<uint4*>(
                base + row * row_stride + col * axis_stride + low_base
            );
            *dst = packed;
        }
    }
}

__device__ __forceinline__ void store_d_transposed_group_16x16(
    uint16_t* base,
    int row_stride,
    int group,
    int lane,
    f32_frag frag
) {
    int low = lane & 15;
    int row_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        base[(row_base + i) * row_stride + low * 16 + group] =
            f32_to_bf16_bits(frag[i]);
    }
}

__device__ __forceinline__ void store_d_h256_transposed_chunk_16x16(
    uint16_t* base,
    int chunk,
    int low4,
    int lane,
    f32_frag frag
) {
    int high_out = lane & 15;
    int row_base = (lane >> 4) * 8;
    int low256 = high_out * 16 + low4;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        base[(row_base + i) * 4096 + low256 * 16 + chunk] =
            f32_to_bf16_bits(frag[i]);
    }
}

__device__ __forceinline__ bf16_frag make_h16_b_fragment(int lane, float scale) {
    bf16_frag frag;
    int col = lane & 15;
    int k_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int k = k_base + i;
        bool positive = (__builtin_popcount(static_cast<unsigned>(k & col)) & 1) == 0;
        float value = positive ? scale : -scale;
        frag[i] = static_cast<short>(f32_to_bf16_bits(value));
    }
    return frag;
}

__device__ __forceinline__ f32_frag wmma_h16_bf16_f32(bf16_frag a, bf16_frag h) {
    f32_frag c = zero_f32_frag();
#if HADACORE_HAS_F32_BF16_WMMA
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, h, c);
#else
    return c;
#endif
}

__device__ void scalar_bf16_fht_tile(
    const uint16_t* in,
    uint16_t* out,
    int rows,
    int n,
    int logn,
    float scale,
    float* tile
) {
    int tid = threadIdx.x;
    int row_base = blockIdx.x * 16;
    int remaining_rows = rows - row_base;
    int valid_rows = remaining_rows < 16 ? remaining_rows : 16;
    if (valid_rows <= 0) {
        return;
    }

    for (int idx = tid; idx < valid_rows * n; idx += blockDim.x) {
        int r = idx / n;
        int c = idx - r * n;
        tile[idx] = bf16_bits_to_f32(in[(row_base + r) * n + c]);
    }
    __syncthreads();

    for (int step = 0; step < logn; ++step) {
        int stride = 1 << step;
        int pairs_per_row = n >> 1;
        for (int p = tid; p < valid_rows * pairs_per_row; p += blockDim.x) {
            int r = p / pairs_per_row;
            int q = p - r * pairs_per_row;
            int idx = (q / stride) * (2 * stride) + (q & (stride - 1));
            int a_idx = r * n + idx;
            int b_idx = a_idx + stride;
            float a = tile[a_idx];
            float b = tile[b_idx];
            tile[a_idx] = a + b;
            tile[b_idx] = a - b;
        }
        __syncthreads();
    }

    for (int idx = tid; idx < valid_rows * n; idx += blockDim.x) {
        int r = idx / n;
        int c = idx - r * n;
        out[(row_base + r) * n + c] = f32_to_bf16_bits(tile[idx] * scale);
    }
}

extern "C" __global__ void hadacore_bf16_wmma16_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows
) {
    int lane = threadIdx.x;
    if (lane >= 32) {
        return;
    }

#if HADACORE_HAS_F32_BF16_WMMA
    int row_base = blockIdx.x * 16;
    if (row_base >= rows) {
        return;
    }

    const uint16_t* in_tile = in + row_base * 16;
    uint16_t* out_tile = out + row_base * 16;

    bf16_frag a = load_a_rowmajor_16x16(in_tile, 16, lane);
    bf16_frag h = make_h16_b_fragment(lane, 0.25f);
    f32_frag d = wmma_h16_bf16_f32(a, h);
    store_d_rowmajor_16x16(out_tile, 16, lane, d);
#else
    extern __shared__ uint32_t smem16_u32[];
    float* tile = reinterpret_cast<float*>(smem16_u32);
    scalar_bf16_fht_tile(in, out, rows, 16, 4, 0.25f, tile);
#endif
}

extern "C" __global__ void hadacore_bf16_wmma256_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows
) {
    int lane = threadIdx.x;
    if (lane >= 32) {
        return;
    }

#if HADACORE_HAS_F32_BF16_WMMA
    extern __shared__ uint32_t smem_u32[];
    uint16_t* temp = reinterpret_cast<uint16_t*>(smem_u32);

    int row_base = blockIdx.x * 16;
    if (row_base >= rows) {
        return;
    }

    const uint16_t* in_tile = in + row_base * 256;
    uint16_t* out_tile = out + row_base * 256;
    bf16_frag h = make_h16_b_fragment(lane, 0.25f);

    // Phase 1: transform the low four index bits within each 16-column tile.
#pragma unroll
    for (int high = 0; high < 16; ++high) {
        bf16_frag a = load_a_rowmajor_16x16(in_tile + high * 16, 256, lane);
        f32_frag d = wmma_h16_bf16_f32(a, h);
        store_d_rowmajor_16x16(temp + high * 16, 256, lane, d);
    }
    __syncthreads();

    // Phase 2: transform the high four index bits for every low-bit column.
#pragma unroll
    for (int low = 0; low < 16; ++low) {
        bf16_frag a = load_a_high_axis_16x16(temp, low, lane);
        f32_frag d = wmma_h16_bf16_f32(a, h);
        store_d_high_axis_16x16(out_tile, low, lane, d);
    }
#else
    extern __shared__ uint32_t smem256_u32[];
    float* tile = reinterpret_cast<float*>(smem256_u32);
    scalar_bf16_fht_tile(in, out, rows, 256, 8, 0.0625f, tile);
#endif
}

extern "C" __global__ void hadacore_bf16_wmma_le256_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int transpose_layout
) {
    int lane = threadIdx.x;
    if (lane >= 32) {
        return;
    }

#if HADACORE_HAS_F32_BF16_WMMA
    extern __shared__ uint32_t smem_le256_u32[];
    uint16_t* temp = reinterpret_cast<uint16_t*>(smem_le256_u32);

    int row_base = blockIdx.x * 16;
    if (row_base >= rows) {
        return;
    }

    int groups = n >> 4;
    const uint16_t* in_tile = in + row_base * n;
    uint16_t* out_tile = out + row_base * n;

    bf16_frag h_low = make_h16_b_fragment(lane, 0.25f);
    bf16_frag h_high = make_h16_b_fragment(lane, rsqrtf(static_cast<float>(groups)));

    for (int idx = lane; idx < 16 * 256; idx += 32) {
        temp[idx] = 0;
    }
    __syncthreads();

    for (int group = 0; group < groups; ++group) {
        bf16_frag a = load_a_rowmajor_16x16(in_tile + group * 16, n, lane);
        f32_frag d = wmma_h16_bf16_f32(a, h_low);
        if (transpose_layout) {
            store_d_transposed_group_16x16(temp, 256, group, lane, d);
        } else {
            store_d_rowmajor_16x16(temp + group * 16, 256, lane, d);
        }
    }
    __syncthreads();

    for (int low = 0; low < 16; ++low) {
        bf16_frag a;
        if (transpose_layout) {
            a = load_a_contiguous_k_padded_16x16(temp + low * 16, 256, groups, lane);
        } else {
            a = load_a_strided_axis_padded_16x16(temp, 256, 16, groups, low, lane);
        }
        f32_frag d = wmma_h16_bf16_f32(a, h_high);
        store_d_strided_axis_16x16(out_tile, n, 16, low, groups, lane, d);
    }
#else
    extern __shared__ uint32_t smem_le256_fallback_u32[];
    float* tile = reinterpret_cast<float*>(smem_le256_fallback_u32);
    int logn = (n == 64) ? 6 : ((n == 128) ? 7 : 8);
    scalar_bf16_fht_tile(in, out, rows, n, logn, rsqrtf(static_cast<float>(n)), tile);
#endif
}

extern "C" __global__ void hadacore_bf16_wmma_h256_stage1_direct_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ temp_out,
    int rows,
    int n
) {
    int lane = threadIdx.x;
    if (lane >= 32) {
        return;
    }

#if HADACORE_HAS_F32_BF16_WMMA
    extern __shared__ uint32_t smem_h256_direct_u32[];
    uint16_t* temp = reinterpret_cast<uint16_t*>(smem_h256_direct_u32);

    int row_base = blockIdx.x * 16;
    int chunk = blockIdx.y;
    if (row_base >= rows) {
        return;
    }

    const uint16_t* in_tile = in + row_base * n + chunk * 256;
    uint16_t* dst_tile = temp_out + row_base * n + chunk * 256;
    bf16_frag h = make_h16_b_fragment(lane, 0.25f);

    for (int high = 0; high < 16; ++high) {
        bf16_frag a = load_a_rowmajor_16x16(in_tile + high * 16, n, lane);
        f32_frag d = wmma_h16_bf16_f32(a, h);
        store_d_rowmajor_16x16(temp + high * 16, 256, lane, d);
    }
    __syncthreads();

    for (int low = 0; low < 16; ++low) {
        bf16_frag a = load_a_strided_axis_padded_16x16(temp, 256, 16, 16, low, lane);
        f32_frag d = wmma_h16_bf16_f32(a, h);
        store_d_strided_axis_16x16(dst_tile, n, 16, low, 16, lane, d);
    }
#endif
}

extern "C" __global__ void hadacore_bf16_wmma_h256_stage1_transpose_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ temp_t,
    int rows,
    int n
) {
    int lane = threadIdx.x;
    if (lane >= 32) {
        return;
    }

#if HADACORE_HAS_F32_BF16_WMMA
    extern __shared__ uint32_t smem_h256_transpose_u32[];
    uint16_t* temp = reinterpret_cast<uint16_t*>(smem_h256_transpose_u32);

    int row_base = blockIdx.x * 16;
    int chunk = blockIdx.y;
    if (row_base >= rows) {
        return;
    }

    const uint16_t* in_tile = in + row_base * n + chunk * 256;
    uint16_t* dst_tile = temp_t + row_base * 4096;
    bf16_frag h = make_h16_b_fragment(lane, 0.25f);

    for (int high = 0; high < 16; ++high) {
        bf16_frag a = load_a_rowmajor_16x16(in_tile + high * 16, n, lane);
        f32_frag d = wmma_h16_bf16_f32(a, h);
        store_d_rowmajor_16x16(temp + high * 16, 256, lane, d);
    }
    __syncthreads();

    for (int low = 0; low < 16; ++low) {
        bf16_frag a = load_a_strided_axis_padded_16x16(temp, 256, 16, 16, low, lane);
        f32_frag d = wmma_h16_bf16_f32(a, h);
        store_d_h256_transposed_chunk_16x16(dst_tile, chunk, low, lane, d);
    }
#endif
}

extern "C" __global__ void hadacore_bf16_wmma_chunk_stage2_direct_kernel(
    const uint16_t* __restrict__ temp_in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int chunks
) {
    int lane = threadIdx.x;
    if (lane >= 32) {
        return;
    }

#if HADACORE_HAS_F32_BF16_WMMA
    int row_base = blockIdx.x * 16;
    int low = blockIdx.y;
    if (row_base >= rows) {
        return;
    }

    const uint16_t* temp_tile = temp_in + row_base * n;
    uint16_t* out_tile = out + row_base * n;
    bf16_frag h = make_h16_b_fragment(lane, rsqrtf(static_cast<float>(chunks)));
    bf16_frag a = load_a_strided_axis_padded_16x16(temp_tile, n, 256, chunks, low, lane);
    f32_frag d = wmma_h16_bf16_f32(a, h);
    store_d_strided_axis_16x16(out_tile, n, 256, low, chunks, lane, d);
#endif
}

extern "C" __global__ void hadacore_bf16_wmma_chunk_stage2_transpose_kernel(
    const uint16_t* __restrict__ temp_t,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int chunks
) {
    int lane = threadIdx.x;
    if (lane >= 32) {
        return;
    }

#if HADACORE_HAS_F32_BF16_WMMA
    int row_base = blockIdx.x * 16;
    int low = blockIdx.y;
    if (row_base >= rows) {
        return;
    }

    const uint16_t* temp_tile = temp_t + row_base * 4096 + low * 16;
    uint16_t* out_tile = out + row_base * n;
    bf16_frag h = make_h16_b_fragment(lane, rsqrtf(static_cast<float>(chunks)));
    bf16_frag a = load_a_contiguous_k_padded_16x16(temp_tile, 4096, chunks, lane);
    f32_frag d = wmma_h16_bf16_f32(a, h);
    store_d_strided_axis_16x16(out_tile, n, 256, low, chunks, lane, d);
#endif
}

extern "C" __global__ void hadacore_bf16_wmma_chunk_stage2_direct_smem_kernel(
    const uint16_t* __restrict__ temp_in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int chunks
) {
    int lane = threadIdx.x;
    if (lane >= 32) {
        return;
    }

#if HADACORE_HAS_F32_BF16_WMMA
    extern __shared__ uint32_t smem_stage2_direct_u32[];
    uint16_t* a_tile = reinterpret_cast<uint16_t*>(smem_stage2_direct_u32);

    int row_base = blockIdx.x * 16;
    int low = blockIdx.y;
    if (row_base >= rows) {
        return;
    }

    for (int idx = lane; idx < 16 * 16; idx += 32) {
        a_tile[idx] = 0;
    }
    __syncthreads();

    const uint16_t* temp_tile = temp_in + row_base * n;
    for (int idx = lane; idx < 16 * 16; idx += 32) {
        int row = idx >> 4;
        int k = idx & 15;
        if (k < chunks) {
            a_tile[row * 16 + k] = temp_tile[row * n + k * 256 + low];
        }
    }
    __syncthreads();

    uint16_t* out_tile = out + row_base * n;
    bf16_frag h = make_h16_b_fragment(lane, rsqrtf(static_cast<float>(chunks)));
    bf16_frag a = load_a_rowmajor_16x16(a_tile, 16, lane);
    f32_frag d = wmma_h16_bf16_f32(a, h);
    store_d_strided_axis_16x16(out_tile, n, 256, low, chunks, lane, d);
#endif
}

extern "C" __global__ void hadacore_bf16_wmma_chunk_stage2_transpose_smem_kernel(
    const uint16_t* __restrict__ temp_t,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int chunks
) {
    int lane = threadIdx.x;
    if (lane >= 32) {
        return;
    }

#if HADACORE_HAS_F32_BF16_WMMA
    extern __shared__ uint32_t smem_stage2_transpose_u32[];
    uint16_t* a_tile = reinterpret_cast<uint16_t*>(smem_stage2_transpose_u32);

    int row_base = blockIdx.x * 16;
    int low = blockIdx.y;
    if (row_base >= rows) {
        return;
    }

    for (int idx = lane; idx < 16 * 16; idx += 32) {
        a_tile[idx] = 0;
    }
    __syncthreads();

    const uint16_t* temp_tile = temp_t + row_base * 4096 + low * 16;
    for (int idx = lane; idx < 16 * 16; idx += 32) {
        int row = idx >> 4;
        int k = idx & 15;
        if (k < chunks) {
            a_tile[row * 16 + k] = temp_tile[row * 4096 + k];
        }
    }
    __syncthreads();

    uint16_t* out_tile = out + row_base * n;
    bf16_frag h = make_h16_b_fragment(lane, rsqrtf(static_cast<float>(chunks)));
    bf16_frag a = load_a_rowmajor_16x16(a_tile, 16, lane);
    f32_frag d = wmma_h16_bf16_f32(a, h);
    store_d_strided_axis_16x16(out_tile, n, 256, low, chunks, lane, d);
#endif
}

__device__ __forceinline__ void store_d_selected_low4_for_single_kernel(
    uint16_t* phase1,
    int chunk,
    int high,
    int low4,
    int lane,
    f32_frag frag
) {
    int col = lane & 15;
    if (col == low4) {
        int row_base = (lane >> 4) * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            phase1[(row_base + i) * 256 + chunk * 16 + high] =
                f32_to_bf16_bits(frag[i]);
        }
    }
}

__device__ __forceinline__ void store_d_h256_for_single_kernel(
    uint16_t* outer,
    int chunk,
    int lane,
    f32_frag frag
) {
    int high_out = lane & 15;
    int row_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        outer[(row_base + i) * 256 + high_out * 16 + chunk] =
            f32_to_bf16_bits(frag[i]);
    }
}

__device__ __forceinline__ void store_d_h256_store_t_for_single_kernel(
    uint16_t* outer_t,
    int chunk,
    int lane,
    f32_frag frag
) {
    int high_out = lane & 15;
    int row_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        outer_t[high_out * 16 * 16 + (row_base + i) * 16 + chunk] =
            f32_to_bf16_bits(frag[i]);
    }
}

__device__ __forceinline__ void store_d_h256_high8_low_tile_for_uint4_kernel(
    uint16_t* h_tile,
    int high_base,
    int low_off,
    int chunk,
    int lane,
    f32_frag frag
) {
    int high_out = lane & 15;
    if (high_out >= high_base && high_out < high_base + 8) {
        int high_off = high_out - high_base;
        int row_base = (lane >> 4) * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            h_tile[((high_off * 8 + low_off) * 16 + (row_base + i)) * 16 + chunk] =
                f32_to_bf16_bits(frag[i]);
        }
    }
}

__device__ __forceinline__ void row_style_h256_chunk_to_bf16(
    const uint16_t* x_chunk,
    uint16_t* temp,
    uint16_t* out_chunk,
    int lane
) {
    bf16_frag h16 = make_h16_b_fragment(lane, 0.25f);

    bf16_frag a0 = load_a_rowmajor_16x16(x_chunk, 16, lane);
    f32_frag d0 = wmma_h16_bf16_f32(a0, h16);
    store_d_rowmajor_16x16(temp, 16, lane, d0);
    __syncthreads();

    bf16_frag a1 = load_a_transposed_square_16x16(temp, lane);
    f32_frag d1 = wmma_h16_bf16_f32(a1, h16);
    store_d_transposed_square_16x16(out_chunk, lane, d1);
    __syncthreads();
}

__device__ __forceinline__ f16_frag load_a_rowmajor_f16_16x16(
    const uint16_t* base,
    int stride,
    int lane
) {
    f16_frag frag;
    int row = lane & 15;
    int k_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        frag[i] = f16_bits_to_f16(base[row * stride + k_base + i]);
    }
    return frag;
}

__device__ __forceinline__ void store_d_rowmajor_f16_16x16(
    uint16_t* base,
    int stride,
    int lane,
    f32_frag frag
) {
    int col = lane & 15;
    int row_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        base[(row_base + i) * stride + col] = f32_to_f16_bits(frag[i]);
    }
}

__device__ __forceinline__ f16_frag load_a_transposed_square_f16_16x16(
    const uint16_t* base,
    int lane
) {
    f16_frag frag;
    int row = lane & 15;
    int k_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        frag[i] = f16_bits_to_f16(base[(k_base + i) * 16 + row]);
    }
    return frag;
}

__device__ __forceinline__ void store_d_transposed_square_f16_16x16(
    uint16_t* base,
    int lane,
    f32_frag frag
) {
    int col = lane & 15;
    int row_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        base[col * 16 + row_base + i] = f32_to_f16_bits(frag[i]);
    }
}

__device__ __forceinline__ f16_frag make_h16_f16_b_fragment(int lane, float scale) {
    f16_frag frag;
    int col = lane & 15;
    int k_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int k = k_base + i;
        bool positive = (__builtin_popcount(static_cast<unsigned>(k & col)) & 1) == 0;
        float value = positive ? scale : -scale;
        frag[i] = static_cast<_Float16>(value);
    }
    return frag;
}

__device__ __forceinline__ f32_frag wmma_h16_f16_f32(f16_frag a, f16_frag h) {
    f32_frag c = zero_f32_frag();
#if HADACORE_HAS_F32_F16_WMMA
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, h, c);
#else
    return c;
#endif
}

__device__ __forceinline__ void row_style_h256_chunk_to_f16(
    const uint16_t* x_chunk,
    uint16_t* temp,
    uint16_t* out_chunk,
    int lane
) {
    f16_frag h16 = make_h16_f16_b_fragment(lane, 0.25f);

    f16_frag a0 = load_a_rowmajor_f16_16x16(x_chunk, 16, lane);
    f32_frag d0 = wmma_h16_f16_f32(a0, h16);
    store_d_rowmajor_f16_16x16(temp, 16, lane, d0);
    __syncthreads();

    f16_frag a1 = load_a_transposed_square_f16_16x16(temp, lane);
    f32_frag d1 = wmma_h16_f16_f32(a1, h16);
    store_d_transposed_square_f16_16x16(out_chunk, lane, d1);
    __syncthreads();
}

__device__ __forceinline__ bf16_frag load_row_style_outer_low_group_16x16(
    const uint16_t* h256_all,
    int low_base,
    int lane
) {
    bf16_frag frag;
    int low_off = lane & 15;
    int k_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int chunk = k_base + i;
        frag[i] = static_cast<short>(h256_all[chunk * 256 + low_base + low_off]);
    }
    return frag;
}

__device__ __forceinline__ void store_row_style_outer_low_group_16x16(
    uint16_t* out_row,
    int low_base,
    int lane,
    f32_frag frag
) {
    int chunk_out = lane & 15;
    int low_base_row = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        out_row[chunk_out * 256 + low_base + low_base_row + i] =
            f32_to_bf16_bits(frag[i]);
    }
}

__device__ __forceinline__ f16_frag load_row_style_outer_low_group_f16_16x16(
    const uint16_t* h256_all,
    int low_base,
    int lane
) {
    f16_frag frag;
    int low_off = lane & 15;
    int k_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int chunk = k_base + i;
        frag[i] = f16_bits_to_f16(h256_all[chunk * 256 + low_base + low_off]);
    }
    return frag;
}

__device__ __forceinline__ void store_row_style_outer_low_group_f16_16x16(
    uint16_t* out_row,
    int low_base,
    int lane,
    f32_frag frag
) {
    int chunk_out = lane & 15;
    int low_base_row = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        out_row[chunk_out * 256 + low_base + low_base_row + i] =
            f32_to_f16_bits(frag[i]);
    }
}

__device__ __forceinline__ f16_frag load_a_high_axis_f16_16x16(
    const uint16_t* base,
    int low_col,
    int lane
) {
    f16_frag frag;
    int row = lane & 15;
    int k_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int high = k_base + i;
        frag[i] = f16_bits_to_f16(base[row * 256 + high * 16 + low_col]);
    }
    return frag;
}

__device__ __forceinline__ f16_frag load_a_contiguous_k_padded_f16_16x16(
    const uint16_t* base,
    int row_stride,
    int valid_k,
    int lane
) {
    f16_frag frag;
    int row = lane & 15;
    int k_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int k = k_base + i;
        frag[i] = (k < valid_k) ? f16_bits_to_f16(base[row * row_stride + k]) : static_cast<_Float16>(0.0f);
    }
    return frag;
}

__device__ __forceinline__ f16_frag load_a_strided_axis_padded_f16_16x16(
    const uint16_t* base,
    int row_stride,
    int axis_stride,
    int valid_k,
    int low,
    int lane
) {
    f16_frag frag;
    int row = lane & 15;
    int k_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int k = k_base + i;
        frag[i] = (k < valid_k)
            ? f16_bits_to_f16(base[row * row_stride + k * axis_stride + low])
            : static_cast<_Float16>(0.0f);
    }
    return frag;
}

__device__ __forceinline__ void store_d_strided_axis_f16_16x16(
    uint16_t* base,
    int row_stride,
    int axis_stride,
    int low,
    int valid_cols,
    int lane,
    f32_frag frag
) {
    int col = lane & 15;
    int row_base = (lane >> 4) * 8;
    if (col < valid_cols) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            base[(row_base + i) * row_stride + col * axis_stride + low] =
                f32_to_f16_bits(frag[i]);
        }
    }
}

__device__ __forceinline__ void store_d_transposed_group_f16_16x16(
    uint16_t* base,
    int row_stride,
    int group,
    int lane,
    f32_frag frag
) {
    int low = lane & 15;
    int row_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        base[(row_base + i) * row_stride + low * 16 + group] =
            f32_to_f16_bits(frag[i]);
    }
}

__device__ __forceinline__ void store_d_selected_low4_for_single_kernel_f16(
    uint16_t* phase1,
    int chunk,
    int high,
    int low4,
    int lane,
    f32_frag frag
) {
    int col = lane & 15;
    if (col == low4) {
        int row_base = (lane >> 4) * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            phase1[(row_base + i) * 256 + chunk * 16 + high] =
                f32_to_f16_bits(frag[i]);
        }
    }
}

__device__ __forceinline__ void store_d_h256_store_t_for_single_kernel_f16(
    uint16_t* outer_t,
    int chunk,
    int lane,
    f32_frag frag
) {
    int high_out = lane & 15;
    int row_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        outer_t[high_out * 16 * 16 + (row_base + i) * 16 + chunk] =
            f32_to_f16_bits(frag[i]);
    }
}

__device__ __forceinline__ void store_d_low8_uint4_strided_axis_f16_16x16(
    uint16_t* base,
    int row_stride,
    int axis_stride,
    int low_base,
    int valid_cols,
    int lane,
    f32_frag d0,
    f32_frag d1,
    f32_frag d2,
    f32_frag d3,
    f32_frag d4,
    f32_frag d5,
    f32_frag d6,
    f32_frag d7
) {
    int row = lane & 15;
    int row_elem = row & 7;
    int src_lane_hi = (row >= 8) ? 16 : 0;

    for (int col_base = 0; col_base < valid_cols; col_base += 2) {
        int col = col_base + (lane >> 4);
        if (col < valid_cols) {
            int src_lane = src_lane_hi + col;
            uint4 packed;
            packed.x = pack_f16x2(
                shfl_frag_elem(d0, row_elem, src_lane),
                shfl_frag_elem(d1, row_elem, src_lane)
            );
            packed.y = pack_f16x2(
                shfl_frag_elem(d2, row_elem, src_lane),
                shfl_frag_elem(d3, row_elem, src_lane)
            );
            packed.z = pack_f16x2(
                shfl_frag_elem(d4, row_elem, src_lane),
                shfl_frag_elem(d5, row_elem, src_lane)
            );
            packed.w = pack_f16x2(
                shfl_frag_elem(d6, row_elem, src_lane),
                shfl_frag_elem(d7, row_elem, src_lane)
            );
            uint4* dst = reinterpret_cast<uint4*>(
                base + row * row_stride + col * axis_stride + low_base
            );
            *dst = packed;
        }
    }
}

__device__ __forceinline__ void clear_phase1_invalid_rows(
    uint16_t* phase1,
    int valid_rows,
    int lane
) {
    for (int idx = lane; idx < (16 - valid_rows) * 256; idx += 32) {
        int row = valid_rows + idx / 256;
        int col = idx & 255;
        phase1[row * 256 + col] = 0;
    }
}

__device__ __forceinline__ void clear_outer_t_invalid_rows(
    uint16_t* outer_t,
    int valid_rows,
    int lane
) {
    for (int idx = lane; idx < 16 * (16 - valid_rows) * 16; idx += 32) {
        int high = idx / ((16 - valid_rows) * 16);
        int rem = idx % ((16 - valid_rows) * 16);
        int row = valid_rows + rem / 16;
        int col = rem & 15;
        outer_t[high * 16 * 16 + row * 16 + col] = 0;
    }
}

extern "C" __global__ void hadacore_bf16_wmma_single_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int chunks
) {
    int lane = threadIdx.x;
    if (lane >= 32) {
        return;
    }

#if HADACORE_HAS_F32_BF16_WMMA
    extern __shared__ uint32_t smem_single_u32[];
    uint16_t* phase1 = reinterpret_cast<uint16_t*>(smem_single_u32);
    uint16_t* outer = phase1 + 16 * 16 * 16;

    int row_base = blockIdx.x * 16;
    if (row_base >= rows) {
        return;
    }

    const uint16_t* in_tile = in + row_base * n;
    uint16_t* out_tile = out + row_base * n;
    bf16_frag h16 = make_h16_b_fragment(lane, 0.25f);
    bf16_frag hchunk = make_h16_b_fragment(lane, rsqrtf(static_cast<float>(chunks)));

    for (int low4 = 0; low4 < 16; ++low4) {
        for (int idx = lane; idx < 2 * 16 * 16 * 16; idx += 32) {
            phase1[idx] = 0;
        }
        __syncthreads();

        for (int chunk = 0; chunk < chunks; ++chunk) {
            for (int high = 0; high < 16; ++high) {
                const uint16_t* src = in_tile + chunk * 256 + high * 16;
                bf16_frag a = load_a_rowmajor_16x16(src, n, lane);
                f32_frag d = wmma_h16_bf16_f32(a, h16);
                store_d_selected_low4_for_single_kernel(phase1, chunk, high, low4, lane, d);
            }
        }
        __syncthreads();

        for (int chunk = 0; chunk < chunks; ++chunk) {
            bf16_frag a = load_a_strided_axis_padded_16x16(
                phase1 + chunk * 16,
                256,
                1,
                16,
                0,
                lane
            );
            f32_frag d = wmma_h16_bf16_f32(a, h16);
            store_d_h256_for_single_kernel(outer, chunk, lane, d);
        }
        __syncthreads();

        for (int high_out = 0; high_out < 16; ++high_out) {
            uint16_t* a_tile = phase1;
            for (int idx = lane; idx < 16 * 16; idx += 32) {
                int row = idx >> 4;
                int k = idx & 15;
                a_tile[idx] = (k < chunks) ? outer[row * 256 + high_out * 16 + k] : 0;
            }
            __syncthreads();

            bf16_frag a = load_a_rowmajor_16x16(a_tile, 16, lane);
            f32_frag d = wmma_h16_bf16_f32(a, hchunk);
            int low256 = high_out * 16 + low4;
            store_d_strided_axis_16x16(out_tile, n, 256, low256, chunks, lane, d);
            __syncthreads();
        }
    }
#endif
}

extern "C" __global__ void hadacore_bf16_wmma_single_store_t_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int chunks
) {
    int lane = threadIdx.x;
    if (lane >= 32) {
        return;
    }

#if HADACORE_HAS_F32_BF16_WMMA
    extern __shared__ uint32_t smem_single_store_t_u32[];
    uint16_t* phase1 = reinterpret_cast<uint16_t*>(smem_single_store_t_u32);
    uint16_t* outer_t = phase1 + 16 * 16 * 16;

    int row_base = blockIdx.x * 16;
    if (row_base >= rows) {
        return;
    }

    const uint16_t* in_tile = in + row_base * n;
    uint16_t* out_tile = out + row_base * n;
    bf16_frag h16 = make_h16_b_fragment(lane, 0.25f);
    bf16_frag hchunk = make_h16_b_fragment(lane, rsqrtf(static_cast<float>(chunks)));

    for (int low4 = 0; low4 < 16; ++low4) {
        for (int idx = lane; idx < 2 * 16 * 16 * 16; idx += 32) {
            phase1[idx] = 0;
        }
        __syncthreads();

        for (int chunk = 0; chunk < chunks; ++chunk) {
            for (int high = 0; high < 16; ++high) {
                const uint16_t* src = in_tile + chunk * 256 + high * 16;
                bf16_frag a = load_a_rowmajor_16x16(src, n, lane);
                f32_frag d = wmma_h16_bf16_f32(a, h16);
                store_d_selected_low4_for_single_kernel(phase1, chunk, high, low4, lane, d);
            }
        }
        __syncthreads();

        for (int chunk = 0; chunk < chunks; ++chunk) {
            bf16_frag a = load_a_strided_axis_padded_16x16(
                phase1 + chunk * 16,
                256,
                1,
                16,
                0,
                lane
            );
            f32_frag d = wmma_h16_bf16_f32(a, h16);
            store_d_h256_store_t_for_single_kernel(outer_t, chunk, lane, d);
        }
        __syncthreads();

        for (int high_out = 0; high_out < 16; ++high_out) {
            bf16_frag a = load_a_rowmajor_16x16(
                outer_t + high_out * 16 * 16,
                16,
                lane
            );
            f32_frag d = wmma_h16_bf16_f32(a, hchunk);
            int low256 = high_out * 16 + low4;
            store_d_strided_axis_16x16(out_tile, n, 256, low256, chunks, lane, d);
        }
        __syncthreads();
    }
#endif
}

extern "C" __global__ void hadacore_bf16_wmma_single_store_t_tile_flush_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int chunks
) {
    int lane = threadIdx.x;
    if (lane >= 32) {
        return;
    }

#if HADACORE_HAS_F32_BF16_WMMA
    extern __shared__ uint4 smem_single_store_t_tile_flush_u4[];
    uint16_t* phase1 = reinterpret_cast<uint16_t*>(smem_single_store_t_tile_flush_u4);
    uint16_t* outer_t = phase1 + 16 * 16 * 16;
    uint16_t* out_smem = outer_t + 16 * 16 * 16;

    int row_base = blockIdx.x * 16;
    if (row_base >= rows) {
        return;
    }

    const uint16_t* in_tile = in + row_base * n;
    uint16_t* out_tile = out + row_base * n;
    bf16_frag h16 = make_h16_b_fragment(lane, 0.25f);
    bf16_frag hchunk = make_h16_b_fragment(lane, rsqrtf(static_cast<float>(chunks)));

    for (int low4 = 0; low4 < 16; ++low4) {
        for (int idx = lane; idx < 2 * 16 * 16 * 16; idx += 32) {
            phase1[idx] = 0;
        }
        __syncthreads();

        for (int chunk = 0; chunk < chunks; ++chunk) {
            for (int high = 0; high < 16; ++high) {
                const uint16_t* src = in_tile + chunk * 256 + high * 16;
                bf16_frag a = load_a_rowmajor_16x16(src, n, lane);
                f32_frag d = wmma_h16_bf16_f32(a, h16);
                store_d_selected_low4_for_single_kernel(phase1, chunk, high, low4, lane, d);
            }
        }
        __syncthreads();

        for (int chunk = 0; chunk < chunks; ++chunk) {
            bf16_frag a = load_a_strided_axis_padded_16x16(
                phase1 + chunk * 16,
                256,
                1,
                16,
                0,
                lane
            );
            f32_frag d = wmma_h16_bf16_f32(a, h16);
            store_d_h256_store_t_for_single_kernel(outer_t, chunk, lane, d);
        }
        __syncthreads();

        for (int high_out = 0; high_out < 16; ++high_out) {
            bf16_frag a = load_a_rowmajor_16x16(
                outer_t + high_out * 16 * 16,
                16,
                lane
            );
            f32_frag d = wmma_h16_bf16_f32(a, hchunk);
            int low256 = high_out * 16 + low4;
            store_d_strided_axis_16x16(out_smem, n, 256, low256, chunks, lane, d);
        }
        __syncthreads();
    }

    uint4* src4 = reinterpret_cast<uint4*>(out_smem);
    uint4* dst4 = reinterpret_cast<uint4*>(out_tile);
    int vec_count = (16 * n * static_cast<int>(sizeof(uint16_t))) / static_cast<int>(sizeof(uint4));
    for (int idx = lane; idx < vec_count; idx += 32) {
        dst4[idx] = src4[idx];
    }
#endif
}

extern "C" __global__ void hadacore_bf16_wmma_single_store_t_uint4_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int chunks
) {
    int lane = threadIdx.x;
    if (lane >= 32) {
        return;
    }

#if HADACORE_HAS_F32_BF16_WMMA
    extern __shared__ uint4 smem_single_store_t_uint4_u4[];
    uint16_t* phase1 = reinterpret_cast<uint16_t*>(smem_single_store_t_uint4_u4);
    uint16_t* h256 = phase1 + 16 * 256;

    int row_base = blockIdx.x * 16;
    if (row_base >= rows) {
        return;
    }

    const uint16_t* in_tile = in + row_base * n;
    uint16_t* out_tile = out + row_base * n;
    bf16_frag h16 = make_h16_b_fragment(lane, 0.25f);
    bf16_frag hchunk = make_h16_b_fragment(lane, rsqrtf(static_cast<float>(chunks)));

    for (int chunk = 0; chunk < chunks; ++chunk) {
        for (int idx = lane; idx < 16 * 256; idx += 32) {
            phase1[idx] = 0;
        }
        __syncthreads();

        for (int high = 0; high < 16; ++high) {
            const uint16_t* src = in_tile + chunk * 256 + high * 16;
            bf16_frag a = load_a_rowmajor_16x16(src, n, lane);
            f32_frag d = wmma_h16_bf16_f32(a, h16);
            store_d_rowmajor_16x16(phase1 + high * 16, 256, lane, d);
        }
        __syncthreads();

        for (int low = 0; low < 16; ++low) {
            bf16_frag a = load_a_high_axis_16x16(phase1, low, lane);
            f32_frag d = wmma_h16_bf16_f32(a, h16);
            store_d_strided_axis_16x16(h256 + chunk * 256, n, 16, low, 16, lane, d);
        }
        __syncthreads();
    }

    for (int high_out = 0; high_out < 16; ++high_out) {
        for (int low_base = 0; low_base < 16; low_base += 8) {
            bf16_frag a0 = load_a_strided_axis_padded_16x16(
                h256 + high_out * 16,
                n,
                256,
                chunks,
                low_base + 0,
                lane
            );
            bf16_frag a1 = load_a_strided_axis_padded_16x16(
                h256 + high_out * 16,
                n,
                256,
                chunks,
                low_base + 1,
                lane
            );
            bf16_frag a2 = load_a_strided_axis_padded_16x16(
                h256 + high_out * 16,
                n,
                256,
                chunks,
                low_base + 2,
                lane
            );
            bf16_frag a3 = load_a_strided_axis_padded_16x16(
                h256 + high_out * 16,
                n,
                256,
                chunks,
                low_base + 3,
                lane
            );
            bf16_frag a4 = load_a_strided_axis_padded_16x16(
                h256 + high_out * 16,
                n,
                256,
                chunks,
                low_base + 4,
                lane
            );
            bf16_frag a5 = load_a_strided_axis_padded_16x16(
                h256 + high_out * 16,
                n,
                256,
                chunks,
                low_base + 5,
                lane
            );
            bf16_frag a6 = load_a_strided_axis_padded_16x16(
                h256 + high_out * 16,
                n,
                256,
                chunks,
                low_base + 6,
                lane
            );
            bf16_frag a7 = load_a_strided_axis_padded_16x16(
                h256 + high_out * 16,
                n,
                256,
                chunks,
                low_base + 7,
                lane
            );

            f32_frag d0 = wmma_h16_bf16_f32(a0, hchunk);
            f32_frag d1 = wmma_h16_bf16_f32(a1, hchunk);
            f32_frag d2 = wmma_h16_bf16_f32(a2, hchunk);
            f32_frag d3 = wmma_h16_bf16_f32(a3, hchunk);
            f32_frag d4 = wmma_h16_bf16_f32(a4, hchunk);
            f32_frag d5 = wmma_h16_bf16_f32(a5, hchunk);
            f32_frag d6 = wmma_h16_bf16_f32(a6, hchunk);
            f32_frag d7 = wmma_h16_bf16_f32(a7, hchunk);

            store_d_low8_uint4_strided_axis_16x16(
                out_tile,
                n,
                256,
                high_out * 16 + low_base,
                chunks,
                lane,
                d0,
                d1,
                d2,
                d3,
                d4,
                d5,
                d6,
                d7
            );
        }
    }
#endif
}

extern "C" __global__ void hadacore_bf16_wmma_single_store_t_uint4_tiled_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int chunks
) {
    int lane = threadIdx.x;
    if (lane >= 32) {
        return;
    }

#if HADACORE_HAS_F32_BF16_WMMA
    extern __shared__ uint4 smem_single_store_t_uint4_tiled_u4[];
    uint16_t* phase1 = reinterpret_cast<uint16_t*>(smem_single_store_t_uint4_tiled_u4);
    uint16_t* h_tile = phase1 + 16 * 256;

    int row_base = blockIdx.x * 16;
    if (row_base >= rows) {
        return;
    }

    const uint16_t* in_tile = in + row_base * n;
    uint16_t* out_tile = out + row_base * n;
    bf16_frag h16 = make_h16_b_fragment(lane, 0.25f);
    bf16_frag hchunk = make_h16_b_fragment(lane, rsqrtf(static_cast<float>(chunks)));

    for (int high_base = 0; high_base < 16; high_base += 8) {
        for (int low_base = 0; low_base < 16; low_base += 8) {
            for (int idx = lane; idx < 8 * 8 * 16 * 16; idx += 32) {
                h_tile[idx] = 0;
            }
            __syncthreads();

            for (int chunk = 0; chunk < chunks; ++chunk) {
                for (int idx = lane; idx < 16 * 256; idx += 32) {
                    phase1[idx] = 0;
                }
                __syncthreads();

                for (int high = 0; high < 16; ++high) {
                    const uint16_t* src = in_tile + chunk * 256 + high * 16;
                    bf16_frag a = load_a_rowmajor_16x16(src, n, lane);
                    f32_frag d = wmma_h16_bf16_f32(a, h16);
                    store_d_rowmajor_16x16(phase1 + high * 16, 256, lane, d);
                }
                __syncthreads();

                for (int low_off = 0; low_off < 8; ++low_off) {
                    bf16_frag a = load_a_high_axis_16x16(phase1, low_base + low_off, lane);
                    f32_frag d = wmma_h16_bf16_f32(a, h16);
                    store_d_h256_high8_low_tile_for_uint4_kernel(
                        h_tile,
                        high_base,
                        low_off,
                        chunk,
                        lane,
                        d
                    );
                }
                __syncthreads();
            }

            for (int high_off = 0; high_off < 8; ++high_off) {
                uint16_t* h_base = h_tile + high_off * 8 * 16 * 16;
                bf16_frag a0 = load_a_rowmajor_16x16(h_base + 0 * 16 * 16, 16, lane);
                bf16_frag a1 = load_a_rowmajor_16x16(h_base + 1 * 16 * 16, 16, lane);
                bf16_frag a2 = load_a_rowmajor_16x16(h_base + 2 * 16 * 16, 16, lane);
                bf16_frag a3 = load_a_rowmajor_16x16(h_base + 3 * 16 * 16, 16, lane);
                bf16_frag a4 = load_a_rowmajor_16x16(h_base + 4 * 16 * 16, 16, lane);
                bf16_frag a5 = load_a_rowmajor_16x16(h_base + 5 * 16 * 16, 16, lane);
                bf16_frag a6 = load_a_rowmajor_16x16(h_base + 6 * 16 * 16, 16, lane);
                bf16_frag a7 = load_a_rowmajor_16x16(h_base + 7 * 16 * 16, 16, lane);

                f32_frag d0 = wmma_h16_bf16_f32(a0, hchunk);
                f32_frag d1 = wmma_h16_bf16_f32(a1, hchunk);
                f32_frag d2 = wmma_h16_bf16_f32(a2, hchunk);
                f32_frag d3 = wmma_h16_bf16_f32(a3, hchunk);
                f32_frag d4 = wmma_h16_bf16_f32(a4, hchunk);
                f32_frag d5 = wmma_h16_bf16_f32(a5, hchunk);
                f32_frag d6 = wmma_h16_bf16_f32(a6, hchunk);
                f32_frag d7 = wmma_h16_bf16_f32(a7, hchunk);

                store_d_low8_uint4_strided_axis_16x16(
                    out_tile,
                    n,
                    256,
                    (high_base + high_off) * 16 + low_base,
                    chunks,
                    lane,
                    d0,
                    d1,
                    d2,
                    d3,
                    d4,
                    d5,
                    d6,
                    d7
                );
            }
            __syncthreads();
        }
    }
#endif
}

extern "C" __global__ void hadacore_bf16_wmma_cuda_style_4096_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows
) {
    int lane = threadIdx.x & 31;
    int wave = threadIdx.x >> 5;
    int row = blockIdx.x;
    if (row >= rows || wave >= 8) {
        return;
    }

#if HADACORE_HAS_F32_BF16_WMMA
    extern __shared__ uint32_t smem_cuda_style_4096_u32[];
    uint16_t* temp_all = reinterpret_cast<uint16_t*>(smem_cuda_style_4096_u32);
    uint16_t* h256_all = temp_all + 8 * 256;
    uint16_t* temp = temp_all + wave * 256;

    const uint16_t* in_row = in + row * 4096;
    uint16_t* out_row = out + row * 4096;

    for (int local = 0; local < 2; ++local) {
        int chunk = wave * 2 + local;
        row_style_h256_chunk_to_bf16(
            in_row + chunk * 256,
            temp,
            h256_all + chunk * 256,
            lane
        );
    }
    __syncthreads();

    bf16_frag hchunk = make_h16_b_fragment(lane, 0.25f);
    for (int low_group = wave; low_group < 16; low_group += 8) {
        int low_base = low_group * 16;
        bf16_frag a = load_row_style_outer_low_group_16x16(h256_all, low_base, lane);
        f32_frag d = wmma_h16_bf16_f32(a, hchunk);
        store_row_style_outer_low_group_16x16(out_row, low_base, lane, d);
    }
#endif
}

extern "C" __global__ void hadacore_f16_wmma_cuda_style_4096_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows
) {
    int lane = threadIdx.x & 31;
    int wave = threadIdx.x >> 5;
    int row = blockIdx.x;
    if (row >= rows || wave >= 8) {
        return;
    }

#if HADACORE_HAS_F32_F16_WMMA
    extern __shared__ uint32_t smem_f16_cuda_style_4096_u32[];
    uint16_t* temp_all = reinterpret_cast<uint16_t*>(smem_f16_cuda_style_4096_u32);
    uint16_t* h256_all = temp_all + 8 * 256;
    uint16_t* temp = temp_all + wave * 256;

    const uint16_t* in_row = in + row * 4096;
    uint16_t* out_row = out + row * 4096;

    for (int local = 0; local < 2; ++local) {
        int chunk = wave * 2 + local;
        row_style_h256_chunk_to_f16(
            in_row + chunk * 256,
            temp,
            h256_all + chunk * 256,
            lane
        );
    }
    __syncthreads();

    f16_frag hchunk = make_h16_f16_b_fragment(lane, 0.25f);
    for (int low_group = wave; low_group < 16; low_group += 8) {
        int low_base = low_group * 16;
        f16_frag a = load_row_style_outer_low_group_f16_16x16(
            h256_all,
            low_base,
            lane
        );
        f32_frag d = wmma_h16_f16_f32(a, hchunk);
        store_row_style_outer_low_group_f16_16x16(out_row, low_base, lane, d);
    }
#endif
}

extern "C" __global__ void hadacore_f16_wmma_le256_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int transpose_layout
) {
    int lane = threadIdx.x;
    if (lane >= 32) {
        return;
    }

#if HADACORE_HAS_F32_F16_WMMA
    extern __shared__ uint32_t smem_f16_le256_u32[];
    uint16_t* temp = reinterpret_cast<uint16_t*>(smem_f16_le256_u32);

    int row_base = blockIdx.x * 16;
    if (row_base >= rows) {
        return;
    }

    int groups = n >> 4;
    const uint16_t* in_tile = in + row_base * n;
    uint16_t* out_tile = out + row_base * n;

    f16_frag h_low = make_h16_f16_b_fragment(lane, 0.25f);
    f16_frag h_high = make_h16_f16_b_fragment(lane, rsqrtf(static_cast<float>(groups)));

    for (int idx = lane; idx < 16 * 256; idx += 32) {
        temp[idx] = 0;
    }
    __syncthreads();

    for (int group = 0; group < groups; ++group) {
        f16_frag a = load_a_rowmajor_f16_16x16(in_tile + group * 16, n, lane);
        f32_frag d = wmma_h16_f16_f32(a, h_low);
        if (transpose_layout) {
            store_d_transposed_group_f16_16x16(temp, 256, group, lane, d);
        } else {
            store_d_rowmajor_f16_16x16(temp + group * 16, 256, lane, d);
        }
    }
    __syncthreads();

    for (int low = 0; low < 16; ++low) {
        f16_frag a;
        if (transpose_layout) {
            a = load_a_contiguous_k_padded_f16_16x16(temp + low * 16, 256, groups, lane);
        } else {
            a = load_a_strided_axis_padded_f16_16x16(temp, 256, 16, groups, low, lane);
        }
        f32_frag d = wmma_h16_f16_f32(a, h_high);
        store_d_strided_axis_f16_16x16(out_tile, n, 16, low, groups, lane, d);
    }
#endif
}

// Dedicated H512 path for CodeLlama. One workgroup owns a row and computes
// two H256 transforms concurrently. The corresponding half-results are loaded
// into registers for the outer H2 butterfly, normalization, and FP16 store.
// This avoids WMMA fragment transposes and any global intermediate.
extern "C" __global__ void hadacore_f16_h512_direct_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows
) {
    __shared__ float halves[512];
    const int lane = threadIdx.x;
    const int row = blockIdx.x;
    if (row >= rows || lane >= 256) {
        return;
    }
    const uint16_t* in_row = in + static_cast<size_t>(row) * 512;
    uint16_t* out_row = out + static_cast<size_t>(row) * 512;
    halves[lane] = static_cast<float>(f16_bits_to_f16(in_row[lane]));
    halves[256 + lane] =
        static_cast<float>(f16_bits_to_f16(in_row[256 + lane]));
    __syncthreads();

#pragma unroll
    for (int stride = 1; stride < 256; stride <<= 1) {
        const int low = (lane / stride) * (stride << 1) +
                        (lane & (stride - 1));
        const float left_a = halves[low];
        const float left_b = halves[low + stride];
        const float right_a = halves[256 + low];
        const float right_b = halves[256 + low + stride];
        halves[low] = left_a + left_b;
        halves[low + stride] = left_a - left_b;
        halves[256 + low] = right_a + right_b;
        halves[256 + low + stride] = right_a - right_b;
        __syncthreads();
    }

    const float left = halves[lane];
    const float right = halves[256 + lane];
    constexpr float scale = 0.04419417382415922f;  // 1/sqrt(512)
    out_row[lane] = f32_to_f16_bits((left + right) * scale);
    out_row[256 + lane] = f32_to_f16_bits((left - right) * scale);
}

extern "C" __global__ void hadacore_f16_wmma_single_store_t_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int chunks
) {
    int lane = threadIdx.x;
    if (lane >= 32) {
        return;
    }

#if HADACORE_HAS_F32_F16_WMMA
    extern __shared__ uint32_t smem_f16_single_store_t_u32[];
    uint16_t* phase1 = reinterpret_cast<uint16_t*>(smem_f16_single_store_t_u32);
    uint16_t* outer_t = phase1 + 16 * 16 * 16;

    int row_base = blockIdx.x * 16;
    if (row_base >= rows) {
        return;
    }

    const uint16_t* in_tile = in + row_base * n;
    uint16_t* out_tile = out + row_base * n;
    f16_frag h16 = make_h16_f16_b_fragment(lane, 0.25f);
    f16_frag hchunk = make_h16_f16_b_fragment(lane, rsqrtf(static_cast<float>(chunks)));

    for (int low4 = 0; low4 < 16; ++low4) {
        for (int idx = lane; idx < 2 * 16 * 16 * 16; idx += 32) {
            phase1[idx] = 0;
        }
        __syncthreads();

        for (int chunk = 0; chunk < chunks; ++chunk) {
            for (int high = 0; high < 16; ++high) {
                const uint16_t* src = in_tile + chunk * 256 + high * 16;
                f16_frag a = load_a_rowmajor_f16_16x16(src, n, lane);
                f32_frag d = wmma_h16_f16_f32(a, h16);
                store_d_selected_low4_for_single_kernel_f16(phase1, chunk, high, low4, lane, d);
            }
        }
        __syncthreads();

        for (int chunk = 0; chunk < chunks; ++chunk) {
            f16_frag a = load_a_strided_axis_padded_f16_16x16(
                phase1 + chunk * 16,
                256,
                1,
                16,
                0,
                lane
            );
            f32_frag d = wmma_h16_f16_f32(a, h16);
            store_d_h256_store_t_for_single_kernel_f16(outer_t, chunk, lane, d);
        }
        __syncthreads();

        for (int high_out = 0; high_out < 16; ++high_out) {
            f16_frag a = load_a_rowmajor_f16_16x16(
                outer_t + high_out * 16 * 16,
                16,
                lane
            );
            f32_frag d = wmma_h16_f16_f32(a, hchunk);
            int low256 = high_out * 16 + low4;
            store_d_strided_axis_f16_16x16(out_tile, n, 256, low256, chunks, lane, d);
        }
        __syncthreads();
    }
#endif
}

extern "C" __global__ void hadacore_f16_wmma_single_store_t_uint4_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int chunks
) {
    int lane = threadIdx.x;
    if (lane >= 32) {
        return;
    }

#if HADACORE_HAS_F32_F16_WMMA
    extern __shared__ uint4 smem_f16_single_store_t_uint4_u4[];
    uint16_t* phase1 = reinterpret_cast<uint16_t*>(smem_f16_single_store_t_uint4_u4);
    uint16_t* h256 = phase1 + 16 * 256;

    int row_base = blockIdx.x * 16;
    if (row_base >= rows) {
        return;
    }

    const uint16_t* in_tile = in + row_base * n;
    uint16_t* out_tile = out + row_base * n;
    f16_frag h16 = make_h16_f16_b_fragment(lane, 0.25f);
    f16_frag hchunk = make_h16_f16_b_fragment(lane, rsqrtf(static_cast<float>(chunks)));

    for (int chunk = 0; chunk < chunks; ++chunk) {
        for (int idx = lane; idx < 16 * 256; idx += 32) {
            phase1[idx] = 0;
        }
        __syncthreads();

        for (int high = 0; high < 16; ++high) {
            const uint16_t* src = in_tile + chunk * 256 + high * 16;
            f16_frag a = load_a_rowmajor_f16_16x16(src, n, lane);
            f32_frag d = wmma_h16_f16_f32(a, h16);
            store_d_rowmajor_f16_16x16(phase1 + high * 16, 256, lane, d);
        }
        __syncthreads();

        for (int low = 0; low < 16; ++low) {
            f16_frag a = load_a_high_axis_f16_16x16(phase1, low, lane);
            f32_frag d = wmma_h16_f16_f32(a, h16);
            store_d_strided_axis_f16_16x16(h256 + chunk * 256, n, 16, low, 16, lane, d);
        }
        __syncthreads();
    }

    for (int high_out = 0; high_out < 16; ++high_out) {
        for (int low_base = 0; low_base < 16; low_base += 8) {
            f16_frag a0 = load_a_strided_axis_padded_f16_16x16(
                h256 + high_out * 16,
                n,
                256,
                chunks,
                low_base + 0,
                lane
            );
            f16_frag a1 = load_a_strided_axis_padded_f16_16x16(
                h256 + high_out * 16,
                n,
                256,
                chunks,
                low_base + 1,
                lane
            );
            f16_frag a2 = load_a_strided_axis_padded_f16_16x16(
                h256 + high_out * 16,
                n,
                256,
                chunks,
                low_base + 2,
                lane
            );
            f16_frag a3 = load_a_strided_axis_padded_f16_16x16(
                h256 + high_out * 16,
                n,
                256,
                chunks,
                low_base + 3,
                lane
            );
            f16_frag a4 = load_a_strided_axis_padded_f16_16x16(
                h256 + high_out * 16,
                n,
                256,
                chunks,
                low_base + 4,
                lane
            );
            f16_frag a5 = load_a_strided_axis_padded_f16_16x16(
                h256 + high_out * 16,
                n,
                256,
                chunks,
                low_base + 5,
                lane
            );
            f16_frag a6 = load_a_strided_axis_padded_f16_16x16(
                h256 + high_out * 16,
                n,
                256,
                chunks,
                low_base + 6,
                lane
            );
            f16_frag a7 = load_a_strided_axis_padded_f16_16x16(
                h256 + high_out * 16,
                n,
                256,
                chunks,
                low_base + 7,
                lane
            );

            f32_frag d0 = wmma_h16_f16_f32(a0, hchunk);
            f32_frag d1 = wmma_h16_f16_f32(a1, hchunk);
            f32_frag d2 = wmma_h16_f16_f32(a2, hchunk);
            f32_frag d3 = wmma_h16_f16_f32(a3, hchunk);
            f32_frag d4 = wmma_h16_f16_f32(a4, hchunk);
            f32_frag d5 = wmma_h16_f16_f32(a5, hchunk);
            f32_frag d6 = wmma_h16_f16_f32(a6, hchunk);
            f32_frag d7 = wmma_h16_f16_f32(a7, hchunk);

            store_d_low8_uint4_strided_axis_f16_16x16(
                out_tile,
                n,
                256,
                high_out * 16 + low_base,
                chunks,
                lane,
                d0,
                d1,
                d2,
                d3,
                d4,
                d5,
                d6,
                d7
            );
        }
    }
#endif
}

extern "C" __global__ void hadacore_bf16_wmma_single_store_t_4row_flush_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int chunks
) {
    int lane = threadIdx.x;
    if (lane >= 32) {
        return;
    }

#if HADACORE_HAS_F32_BF16_WMMA
    constexpr int rows_per_wave = 4;
    extern __shared__ uint32_t smem_single_store_t_4row_u32[];
    uint16_t* phase1 = reinterpret_cast<uint16_t*>(smem_single_store_t_4row_u32);
    uint16_t* outer_t = phase1 + 16 * 16 * 16;
    uint16_t* out_smem = outer_t + 16 * 16 * 16;

    int row_base = blockIdx.x * rows_per_wave;
    if (row_base >= rows) {
        return;
    }

    const uint16_t* in_tile = in + row_base * n;
    uint16_t* out_tile = out + row_base * n;
    bf16_frag h16 = make_h16_b_fragment(lane, 0.25f);
    bf16_frag hchunk = make_h16_b_fragment(lane, rsqrtf(static_cast<float>(chunks)));

    for (int idx = lane; idx < rows_per_wave * n; idx += 32) {
        out_smem[idx] = 0;
    }
    __syncthreads();

    for (int low4 = 0; low4 < 16; ++low4) {
        for (int idx = lane; idx < 2 * 16 * 16 * 16; idx += 32) {
            phase1[idx] = 0;
        }
        __syncthreads();

        for (int chunk = 0; chunk < chunks; ++chunk) {
            for (int high = 0; high < 16; ++high) {
                const uint16_t* src = in_tile + chunk * 256 + high * 16;
                bf16_frag a = load_a_rowmajor_rows_padded_16x16(
                    src,
                    n,
                    rows_per_wave,
                    lane
                );
                f32_frag d = wmma_h16_bf16_f32(a, h16);
                store_d_selected_low4_for_single_kernel(phase1, chunk, high, low4, lane, d);
            }
        }
        __syncthreads();
        clear_phase1_invalid_rows(phase1, rows_per_wave, lane);
        __syncthreads();

        for (int chunk = 0; chunk < chunks; ++chunk) {
            bf16_frag a = load_a_strided_axis_padded_16x16(
                phase1 + chunk * 16,
                256,
                1,
                16,
                0,
                lane
            );
            f32_frag d = wmma_h16_bf16_f32(a, h16);
            store_d_h256_store_t_for_single_kernel(outer_t, chunk, lane, d);
        }
        __syncthreads();
        clear_outer_t_invalid_rows(outer_t, rows_per_wave, lane);
        __syncthreads();

        for (int high_out = 0; high_out < 16; ++high_out) {
            bf16_frag a = load_a_rowmajor_16x16(
                outer_t + high_out * 16 * 16,
                16,
                lane
            );
            f32_frag d = wmma_h16_bf16_f32(a, hchunk);
            int low256 = high_out * 16 + low4;
            store_d_strided_axis_rows_16x16(
                out_smem,
                n,
                256,
                low256,
                chunks,
                rows_per_wave,
                lane,
                d
            );
        }
        __syncthreads();
    }

    for (int idx = lane; idx < rows_per_wave * n; idx += 32) {
        out_tile[idx] = out_smem[idx];
    }
#endif
}

extern "C" __global__ void hadacore_bf16_wmma_single_store_t_4row_direct_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int chunks
) {
    int lane = threadIdx.x;
    if (lane >= 32) {
        return;
    }

#if HADACORE_HAS_F32_BF16_WMMA
    constexpr int rows_per_wave = 4;
    extern __shared__ uint32_t smem_single_store_t_4row_direct_u32[];
    uint16_t* phase1 = reinterpret_cast<uint16_t*>(smem_single_store_t_4row_direct_u32);
    uint16_t* outer_t = phase1 + 16 * 16 * 16;

    int row_base = blockIdx.x * rows_per_wave;
    if (row_base >= rows) {
        return;
    }

    const uint16_t* in_tile = in + row_base * n;
    uint16_t* out_tile = out + row_base * n;
    bf16_frag h16 = make_h16_b_fragment(lane, 0.25f);
    bf16_frag hchunk = make_h16_b_fragment(lane, rsqrtf(static_cast<float>(chunks)));

    for (int low4 = 0; low4 < 16; ++low4) {
        for (int idx = lane; idx < 2 * 16 * 16 * 16; idx += 32) {
            phase1[idx] = 0;
        }
        __syncthreads();

        for (int chunk = 0; chunk < chunks; ++chunk) {
            for (int high = 0; high < 16; ++high) {
                const uint16_t* src = in_tile + chunk * 256 + high * 16;
                bf16_frag a = load_a_rowmajor_rows_padded_16x16(
                    src,
                    n,
                    rows_per_wave,
                    lane
                );
                f32_frag d = wmma_h16_bf16_f32(a, h16);
                store_d_selected_low4_for_single_kernel(phase1, chunk, high, low4, lane, d);
            }
        }
        __syncthreads();
        clear_phase1_invalid_rows(phase1, rows_per_wave, lane);
        __syncthreads();

        for (int chunk = 0; chunk < chunks; ++chunk) {
            bf16_frag a = load_a_strided_axis_padded_16x16(
                phase1 + chunk * 16,
                256,
                1,
                16,
                0,
                lane
            );
            f32_frag d = wmma_h16_bf16_f32(a, h16);
            store_d_h256_store_t_for_single_kernel(outer_t, chunk, lane, d);
        }
        __syncthreads();
        clear_outer_t_invalid_rows(outer_t, rows_per_wave, lane);
        __syncthreads();

        for (int high_out = 0; high_out < 16; ++high_out) {
            bf16_frag a = load_a_rowmajor_16x16(
                outer_t + high_out * 16 * 16,
                16,
                lane
            );
            f32_frag d = wmma_h16_bf16_f32(a, hchunk);
            int low256 = high_out * 16 + low4;
            store_d_strided_axis_rows_16x16(
                out_tile,
                n,
                256,
                low256,
                chunks,
                rows_per_wave,
                lane,
                d
            );
        }
        __syncthreads();
    }
#endif
}

extern "C" __global__ void hadacore_bf16_wmma_single_store_t_4row_inner_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int chunks
) {
    int lane = threadIdx.x;
    if (lane >= 32) {
        return;
    }

#if HADACORE_HAS_F32_BF16_WMMA
    constexpr int rows_per_wave = 4;
    extern __shared__ uint32_t smem_single_store_t_4row_inner_u32[];
    uint16_t* phase1 = reinterpret_cast<uint16_t*>(smem_single_store_t_4row_inner_u32);

    int row_base = blockIdx.x * rows_per_wave;
    if (row_base >= rows) {
        return;
    }

    const uint16_t* in_tile = in + row_base * n;
    uint16_t* out_tile = out + row_base * n;
    bf16_frag h16 = make_h16_b_fragment(lane, 0.25f);

    for (int low4 = 0; low4 < 16; ++low4) {
        for (int idx = lane; idx < 16 * 16 * 16; idx += 32) {
            phase1[idx] = 0;
        }
        __syncthreads();

        for (int chunk = 0; chunk < chunks; ++chunk) {
            for (int high = 0; high < 16; ++high) {
                const uint16_t* src = in_tile + chunk * 256 + high * 16;
                bf16_frag a = load_a_rowmajor_rows_padded_16x16(
                    src,
                    n,
                    rows_per_wave,
                    lane
                );
                f32_frag d = wmma_h16_bf16_f32(a, h16);
                store_d_selected_low4_for_single_kernel(phase1, chunk, high, low4, lane, d);
            }
        }
        __syncthreads();
        clear_phase1_invalid_rows(phase1, rows_per_wave, lane);
        __syncthreads();

        for (int chunk = 0; chunk < chunks; ++chunk) {
            bf16_frag a = load_a_strided_axis_padded_16x16(
                phase1 + chunk * 16,
                256,
                1,
                16,
                0,
                lane
            );
            f32_frag d = wmma_h16_bf16_f32(a, h16);
            store_d_strided_axis_rows_16x16(
                out_tile + chunk * 256,
                n,
                16,
                low4,
                16,
                rows_per_wave,
                lane,
                d
            );
        }
        __syncthreads();
    }
#endif
}

extern "C" __global__ void hadacore_bf16_wmma_single_inner_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int chunks
) {
    int lane = threadIdx.x;
    if (lane >= 32) {
        return;
    }

#if HADACORE_HAS_F32_BF16_WMMA
    extern __shared__ uint32_t smem_single_inner_u32[];
    uint16_t* phase1 = reinterpret_cast<uint16_t*>(smem_single_inner_u32);

    int row_base = blockIdx.x * 16;
    if (row_base >= rows) {
        return;
    }

    const uint16_t* in_tile = in + row_base * n;
    uint16_t* out_tile = out + row_base * n;
    bf16_frag h16 = make_h16_b_fragment(lane, 0.25f);

    for (int low4 = 0; low4 < 16; ++low4) {
        for (int idx = lane; idx < 16 * 16 * 16; idx += 32) {
            phase1[idx] = 0;
        }
        __syncthreads();

        for (int chunk = 0; chunk < chunks; ++chunk) {
            for (int high = 0; high < 16; ++high) {
                const uint16_t* src = in_tile + chunk * 256 + high * 16;
                bf16_frag a = load_a_rowmajor_16x16(src, n, lane);
                f32_frag d = wmma_h16_bf16_f32(a, h16);
                store_d_selected_low4_for_single_kernel(phase1, chunk, high, low4, lane, d);
            }
        }
        __syncthreads();

        for (int chunk = 0; chunk < chunks; ++chunk) {
            bf16_frag a = load_a_strided_axis_padded_16x16(
                phase1 + chunk * 16,
                256,
                1,
                16,
                0,
                lane
            );
            f32_frag d = wmma_h16_bf16_f32(a, h16);
            store_d_strided_axis_16x16(out_tile + chunk * 256, n, 16, low4, 16, lane, d);
        }
        __syncthreads();
    }
#endif
}

extern "C" __global__ void hadacore_bf16_chunk_stage2_scalar_kernel(
    const uint16_t* __restrict__ temp_in,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int chunks
) {
    int row = blockIdx.x;
    int low = threadIdx.x;
    if (row >= rows || low >= 256) {
        return;
    }

    float vals[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        vals[i] = 0.0f;
    }

    const uint16_t* src = temp_in + static_cast<size_t>(row) * n + low;
#pragma unroll
    for (int c = 0; c < 16; ++c) {
        if (c < chunks) {
            vals[c] = bf16_bits_to_f32(src[c * 256]);
        }
    }

    for (int stride = 1; stride < chunks; stride <<= 1) {
        for (int base = 0; base < 16; base += 2 * stride) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (j < stride && base + stride + j < chunks) {
                    float a = vals[base + j];
                    float b = vals[base + stride + j];
                    vals[base + j] = a + b;
                    vals[base + stride + j] = a - b;
                }
            }
        }
    }

    float scale = rsqrtf(static_cast<float>(chunks));
    uint16_t* dst = out + static_cast<size_t>(row) * n + low;
#pragma unroll
    for (int c = 0; c < 16; ++c) {
        if (c < chunks) {
            dst[c * 256] = f32_to_bf16_bits(vals[c] * scale);
        }
    }
}

extern "C" __global__ void hadacore_bf16_chunk_stage2_scalar_transpose_kernel(
    const uint16_t* __restrict__ temp_t,
    uint16_t* __restrict__ out,
    int rows,
    int n,
    int chunks
) {
    int row = blockIdx.x;
    int low = threadIdx.x;
    if (row >= rows || low >= 256) {
        return;
    }

    float vals[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        vals[i] = 0.0f;
    }

    const uint16_t* src = temp_t + static_cast<size_t>(row) * 4096 + low * 16;
#pragma unroll
    for (int c = 0; c < 16; ++c) {
        if (c < chunks) {
            vals[c] = bf16_bits_to_f32(src[c]);
        }
    }

    for (int stride = 1; stride < chunks; stride <<= 1) {
        for (int base = 0; base < 16; base += 2 * stride) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (j < stride && base + stride + j < chunks) {
                    float a = vals[base + j];
                    float b = vals[base + stride + j];
                    vals[base + j] = a + b;
                    vals[base + stride + j] = a - b;
                }
            }
        }
    }

    float scale = rsqrtf(static_cast<float>(chunks));
    uint16_t* dst = out + static_cast<size_t>(row) * n + low;
#pragma unroll
    for (int c = 0; c < 16; ++c) {
        if (c < chunks) {
            dst[c * 256] = f32_to_bf16_bits(vals[c] * scale);
        }
    }
}

extern "C" __global__ void hadacore_fallback_fp32_kernel(
    float* __restrict__ data,
    int n,
    float scale,
    int logn
) {
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    float* base = data + static_cast<size_t>(row) * n;

    for (int i = tid; i < n; i += blockDim.x) {
        sdata[i] = base[i];
    }
    __syncthreads();

    for (int step = 0; step < logn; ++step) {
        int stride = 1 << step;
        for (int p = tid; p < (n >> 1); p += blockDim.x) {
            int idx = (p / stride) * (2 * stride) + (p & (stride - 1));
            int pair = idx + stride;
            float a = sdata[idx];
            float b = sdata[pair];
            sdata[idx] = a + b;
            sdata[pair] = a - b;
        }
        __syncthreads();
    }

    for (int i = tid; i < n; i += blockDim.x) {
        base[i] = sdata[i] * scale;
    }
}