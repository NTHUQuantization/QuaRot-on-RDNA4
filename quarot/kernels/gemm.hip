#include <ATen/hip/HIPContext.h>
#include <gemm.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

using I32x8       = int32_t __attribute__((ext_vector_type(8)));
using I32x2       = int32_t __attribute__((ext_vector_type(2)));

static constexpr uint32_t kWmmaK = 32;
static constexpr uint32_t kLaneK = kWmmaK / 2u;

namespace {

#if !defined(QUAROT_BPRE_GFX12) &&                                               \
    (defined(__gfx12__) || defined(__gfx1200__) || defined(__gfx1201__) ||       \
     defined(__gfx12_generic__))
#define QUAROT_BPRE_GFX12 1
#endif

void check_hip(hipError_t status, const char* what)
{
    if(status != hipSuccess)
    {
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
    }
}

size_t b_prepacked_size_bytes(uint32_t N, uint32_t K)
{
    const uint32_t n_tiles16 = (N + 15u) / 16u;
    const uint32_t k_groups  = K / kWmmaK;
    return static_cast<size_t>(n_tiles16) * k_groups * 2u * 16u * 8u;
}

__device__ __forceinline__ I32x2 unpack_u64_to_i32x2(uint64_t packed)
{
    return {static_cast<int32_t>(static_cast<uint32_t>(packed)),
            static_cast<int32_t>(static_cast<uint32_t>(packed >> 32))};
}

__device__ __forceinline__ I32x2 load_row_i4x16_as_i32x2(const Int4Storage* base)
{
    uint32_t packed0 = 0;
    uint32_t packed1 = 0;
#pragma unroll
    for(int i = 0; i < 4; ++i)
    {
        packed0 |= static_cast<uint32_t>(base[i]) << (8 * i);
        packed1 |= static_cast<uint32_t>(base[4 + i]) << (8 * i);
    }
    return {static_cast<int32_t>(packed0), static_cast<int32_t>(packed1)};
}

__device__ __forceinline__ I32x2 load_pre_i4x16_as_i32x2(const Int4Storage* pre,
                                                         uint32_t tile16,
                                                         uint32_t k_group,
                                                         uint32_t row_blk,
                                                         uint32_t lane16,
                                                         uint32_t k_groups)
{
    const size_t offset =
        (((static_cast<size_t>(tile16) * k_groups + k_group) * 2u + row_blk) * 16u +
         lane16) *
        8u;
    return unpack_u64_to_i32x2(*reinterpret_cast<const uint64_t*>(pre + offset));
}

__device__ __forceinline__ uint64_t load_u64_or_tail(const Int4Storage* base,
                                                     uint32_t available_bytes)
{
    if(available_bytes >= 8u)
    {
        return *reinterpret_cast<const uint64_t*>(base);
    }

    uint64_t packed = 0;
#pragma unroll
    for(uint32_t i = 0; i < 8u; ++i)
    {
        if(i < available_bytes)
        {
            packed |= static_cast<uint64_t>(base[i]) << (8u * i);
        }
    }
    return packed;
}

__global__ void prepack_b_kernel(const Int4Storage* __restrict__ B,
                                 Int4Storage* __restrict__ BPre,
                                 uint32_t N,
                                 uint32_t K)
{
    const uint32_t k_groups = K / kWmmaK;
    const uint32_t n_tiles  = (N + 15u) / 16u;
    const uint32_t units    = n_tiles * k_groups * 2u * 16u;
    const uint32_t idx      = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= units)
    {
        return;
    }

    const uint32_t lane16  = idx & 15u;
    const uint32_t tmp0    = idx >> 4u;
    const uint32_t row_blk = tmp0 & 1u;
    const uint32_t tmp1    = tmp0 >> 1u;
    const uint32_t kg      = tmp1 % k_groups;
    const uint32_t tile    = tmp1 / k_groups;
    const uint32_t n       = tile * 16u + lane16;
    const uint32_t k_bytes = K / 2u;
    const uint32_t byte_k  = kg * 16u + row_blk * 8u;

    uint64_t value = 0;
    if(n < N)
    {
        value = *reinterpret_cast<const uint64_t*>(B + static_cast<size_t>(n) * k_bytes + byte_k);
    }
    *reinterpret_cast<uint64_t*>(BPre + static_cast<size_t>(idx) * 8u) = value;
}

template <uint32_t BlockMWaves,
          uint32_t BlockNWaves,
          uint32_t MTilesPerWave,
          uint32_t NTilesPerWave>
__global__ void bpre_kernel(const Int4Storage* __restrict__ A,
                            const Int4Storage* __restrict__ BPre,
                            int32_t* __restrict__ C,
                            uint32_t M,
                            uint32_t N,
                            uint32_t K)
{
#if defined(QUAROT_BPRE_GFX12)
    constexpr uint32_t kWavesPerBlock = BlockMWaves * BlockNWaves;
    const uint32_t wave_id = threadIdx.x >> 5u;
    if(wave_id >= kWavesPerBlock)
    {
        return;
    }

    const uint32_t lane    = threadIdx.x & 31u;
    const uint32_t lane16  = lane & 15u;
    const uint32_t row_blk = lane >> 4u;
    const uint32_t wave_m  = wave_id / BlockNWaves;
    const uint32_t wave_n  = wave_id - wave_m * BlockNWaves;

    const uint32_t m_tile_base =
        (blockIdx.y * BlockMWaves * MTilesPerWave + wave_m * MTilesPerWave) * 16u;
    const uint32_t n_tile_base =
        (blockIdx.x * BlockNWaves * NTilesPerWave + wave_n * NTilesPerWave) * 16u;

    const uint32_t k_bytes  = K / 2u;
    const uint32_t k_groups = K / kWmmaK;

    I32x8 acc[MTilesPerWave][NTilesPerWave];
    const I32x8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for(uint32_t mt = 0; mt < MTilesPerWave; ++mt)
    {
#pragma unroll
        for(uint32_t nt = 0; nt < NTilesPerWave; ++nt)
        {
            acc[mt][nt] = zero;
        }
    }

    for(uint32_t k = 0; k < K; k += kWmmaK)
    {
        const uint32_t k_group = k / kWmmaK;
        const uint32_t byte_k  = (k + row_blk * kLaneK) / 2u;

        I32x2 b_frags[NTilesPerWave];
#pragma unroll
        for(uint32_t nt = 0; nt < NTilesPerWave; ++nt)
        {
            const uint32_t b_col = n_tile_base + nt * 16u + lane16;
            I32x2 b_frag         = {0, 0};
            if(b_col < N)
            {
                const uint32_t b_tile16 = (n_tile_base + nt * 16u) / 16u;
                b_frag =
                    load_pre_i4x16_as_i32x2(BPre, b_tile16, k_group, row_blk, lane16,
                                            k_groups);
            }
            b_frags[nt] = b_frag;
        }

#pragma unroll
        for(uint32_t mt = 0; mt < MTilesPerWave; ++mt)
        {
            const uint32_t a_row = m_tile_base + mt * 16u + lane16;
            I32x2 a_frag         = {0, 0};
            if(a_row < M)
            {
                a_frag =
                    load_row_i4x16_as_i32x2(A + static_cast<size_t>(a_row) * k_bytes +
                                             byte_k);
            }

#pragma unroll
            for(uint32_t nt = 0; nt < NTilesPerWave; ++nt)
            {
                acc[mt][nt] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                    true, a_frag, true, b_frags[nt], acc[mt][nt], false);
            }
        }
    }

#pragma unroll
    for(uint32_t mt = 0; mt < MTilesPerWave; ++mt)
    {
#pragma unroll
        for(uint32_t i = 0; i < 8u; ++i)
        {
            const uint32_t row = m_tile_base + mt * 16u + row_blk * 8u + i;
#pragma unroll
            for(uint32_t nt = 0; nt < NTilesPerWave; ++nt)
            {
                const uint32_t col = n_tile_base + nt * 16u + lane16;
                if(row < M && col < N)
                {
                    C[static_cast<size_t>(row) * N + col] = acc[mt][nt][i];
                }
            }
        }
    }
#else
    (void)A;
    (void)BPre;
    (void)C;
    (void)M;
    (void)N;
    (void)K;
#endif
}

// One wave computes a 16x16 output tile. Integer WMMA accumulation is closed
// every 256 K elements so the corresponding activation-group scale can be
// applied before partial sums from different groups are combined.
template <uint32_t BlockMWaves, uint32_t BlockNWaves>
__global__ void bpre_grouped_scale_kernel(
    const Int4Storage* __restrict__ A,
    const Int4Storage* __restrict__ BPre,
    const half* __restrict__ scale_group,
    const half* __restrict__ scale_weight,
    half* __restrict__ C,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t scale_groups)
{
#if defined(QUAROT_BPRE_GFX12)
    const uint32_t wave = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t wave_m = wave / BlockNWaves;
    const uint32_t wave_n = wave - wave_m * BlockNWaves;
    const uint32_t lane16 = lane & 15u;
    const uint32_t row_blk = lane >> 4u;
    const uint32_t row_base = (blockIdx.y * BlockMWaves + wave_m) * 16u;
    const uint32_t n_tile = blockIdx.x * BlockNWaves + wave_n;
    const uint32_t col = n_tile * 16u + lane16;
    const uint32_t k_bytes = K / 2u;
    const uint32_t k_groups = K / kWmmaK;
    const uint32_t wmmas_per_scale = k_groups / scale_groups;
    float output[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const I32x8 zero = {0, 0, 0, 0, 0, 0, 0, 0};

    for(uint32_t sg = 0; sg < scale_groups; ++sg)
    {
        I32x8 acc = zero;
#pragma unroll
        for(uint32_t sub = 0; sub < wmmas_per_scale; ++sub)
        {
            const uint32_t kg = sg * wmmas_per_scale + sub;
            const uint32_t byte_k = (kg * kWmmaK + row_blk * kLaneK) / 2u;
            I32x2 b_frag = {0, 0};
            if(col < N)
                b_frag = load_pre_i4x16_as_i32x2(
                    BPre, n_tile, kg, row_blk, lane16, k_groups);
            const uint32_t a_row = row_base + lane16;
            I32x2 a_frag = {0, 0};
            if(a_row < M)
                a_frag = load_row_i4x16_as_i32x2(
                    A + static_cast<size_t>(a_row) * k_bytes + byte_k);
            acc = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                true, a_frag, true, b_frag, acc, false);
        }
#pragma unroll
        for(uint32_t i = 0; i < 8u; ++i)
        {
            const uint32_t row = row_base + row_blk * 8u + i;
            if(row < M)
                output[i] += static_cast<float>(acc[i]) *
                             __half2float(scale_group[static_cast<size_t>(row) *
                                                      scale_groups + sg]);
        }
    }

    if(col < N)
    {
        const float ws = __half2float(scale_weight[col]);
#pragma unroll
        for(uint32_t i = 0; i < 8u; ++i)
        {
            const uint32_t row = row_base + row_blk * 8u + i;
            if(row < M)
                C[static_cast<size_t>(row) * N + col] = __float2half(output[i] * ws);
        }
    }
#endif
}

// Shared-input decode projection. One launch covers two or three independently
// scaled weight matrices and writes a contiguous [B0 | B1 | B2] result. All
// supported model widths are 16-column aligned, so a WMMA tile never straddles
// two matrices.
template <uint32_t BlockNWaves>
__global__ void bpre_multi_scale_kernel(
    const Int4Storage* __restrict__ A,
    const Int4Storage* __restrict__ B0,
    const Int4Storage* __restrict__ B1,
    const Int4Storage* __restrict__ B2,
    const half* __restrict__ scale_activation,
    const half* __restrict__ scale_weight0,
    const half* __restrict__ scale_weight1,
    const half* __restrict__ scale_weight2,
    half* __restrict__ C,
    uint32_t M, uint32_t N0, uint32_t N1, uint32_t N2, uint32_t K,
    bool allow_interleaved)
{
#if defined(QUAROT_BPRE_GFX12)
    const uint32_t wave = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t lane16 = lane & 15u;
    const uint32_t row_blk = lane >> 4u;
    const uint32_t global_tile = blockIdx.x * BlockNWaves + wave;
    const uint32_t global_col = global_tile * 16u + lane16;
    const uint32_t total_n = N0 + N1 + N2;
    if(global_col >= total_n) return;

    const Int4Storage* B;
    const half* scale_weight;
    uint32_t local_col;
    uint32_t local_tile;
    const size_t matrix_stride = static_cast<size_t>(K) / 2u;
    const bool interleaved = allow_interleaved &&
        B1 == B0 + static_cast<size_t>(N0) * matrix_stride &&
        scale_weight1 == scale_weight0 + N0 &&
        (N2 == 0 ||
         (B2 == B1 + static_cast<size_t>(N1) * matrix_stride &&
          scale_weight2 == scale_weight1 + N1));
    if(interleaved) {
        B = B0; scale_weight = scale_weight0; local_col = global_col;
        local_tile = global_tile;
    } else if(global_col < N0) {
        B = B0; scale_weight = scale_weight0; local_col = global_col;
        local_tile = global_tile;
    } else if(global_col < N0 + N1) {
        B = B1; scale_weight = scale_weight1; local_col = global_col - N0;
        local_tile = local_col / 16u;
    } else {
        B = B2; scale_weight = scale_weight2; local_col = global_col - N0 - N1;
        local_tile = local_col / 16u;
    }

    const uint32_t row_base = blockIdx.y * 16u;
    const uint32_t k_bytes = K / 2u;
    const uint32_t k_groups = K / kWmmaK;
    const uint32_t a_row = row_base + lane16;
    I32x8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
    for(uint32_t kg = 0; kg < k_groups; ++kg) {
        const uint32_t byte_k = (kg * kWmmaK + row_blk * kLaneK) / 2u;
        const I32x2 b_frag = load_pre_i4x16_as_i32x2(
            B, local_tile, kg, row_blk, lane16, k_groups);
        I32x2 a_frag = {0, 0};
        if(a_row < M)
            a_frag = load_row_i4x16_as_i32x2(
                A + static_cast<size_t>(a_row) * k_bytes + byte_k);
        acc = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
            true, a_frag, true, b_frag, acc, false);
    }
    const float ws = __half2float(scale_weight[local_col]);
    for(uint32_t i = 0; i < 8u; ++i) {
        const uint32_t row = row_base + row_blk * 8u + i;
        if(row < M)
            C[static_cast<size_t>(row) * total_n + global_col] = __float2half(
                static_cast<float>(acc[i]) * __half2float(scale_activation[row]) * ws);
    }
#endif
}

template <uint32_t BlockMWaves,
          uint32_t BlockNWaves,
          uint32_t MTilesPerWave,
          uint32_t NTilesPerWave,
          uint32_t LdsK>
__global__ void lds_swizzle_kernel(const Int4Storage* __restrict__ A,
                                   const Int4Storage* __restrict__ B,
                                   int32_t* __restrict__ C,
                                   uint32_t M,
                                   uint32_t N,
                                   uint32_t K)
{
#if defined(QUAROT_BPRE_GFX12)
    constexpr uint32_t kWavesPerBlock  = BlockMWaves * BlockNWaves;
    constexpr uint32_t kThreadsPerBlock = 32u * kWavesPerBlock;
    constexpr uint32_t kLdsKBytes      = LdsK / 2u;
    constexpr uint32_t kLdsKGroups     = LdsK / kWmmaK;
    constexpr uint32_t kBlockMTiles16  = BlockMWaves * MTilesPerWave;
    constexpr uint32_t kBlockNTiles16  = BlockNWaves * NTilesPerWave;
    constexpr uint32_t kAUnits         = kBlockMTiles16 * kLdsKGroups * 2u * 16u;
    constexpr uint32_t kBUnits         = kBlockNTiles16 * kLdsKGroups * 2u * 16u;

    const uint32_t wave_id = threadIdx.x >> 5u;
    if(wave_id >= kWavesPerBlock)
    {
        return;
    }

    const uint32_t lane    = threadIdx.x & 31u;
    const uint32_t lane16  = lane & 15u;
    const uint32_t row_blk = lane >> 4u;
    const uint32_t wave_m  = wave_id / BlockNWaves;
    const uint32_t wave_n  = wave_id - wave_m * BlockNWaves;

    __shared__ uint64_t s_A_u64[kAUnits];
    __shared__ uint64_t s_B_u64[kBUnits];
    Int4Storage* s_A = reinterpret_cast<Int4Storage*>(s_A_u64);
    Int4Storage* s_B = reinterpret_cast<Int4Storage*>(s_B_u64);

    const uint32_t m_tile_base =
        (blockIdx.y * BlockMWaves * MTilesPerWave + wave_m * MTilesPerWave) * 16u;
    const uint32_t n_tile_base =
        (blockIdx.x * BlockNWaves * NTilesPerWave + wave_n * NTilesPerWave) * 16u;
    const uint32_t block_m_base = blockIdx.y * kBlockMTiles16 * 16u;
    const uint32_t block_n_base = blockIdx.x * kBlockNTiles16 * 16u;
    const uint32_t k_bytes      = K / 2u;

    I32x8 acc[MTilesPerWave][NTilesPerWave];
    const I32x8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for(uint32_t mt = 0; mt < MTilesPerWave; ++mt)
    {
#pragma unroll
        for(uint32_t nt = 0; nt < NTilesPerWave; ++nt)
        {
            acc[mt][nt] = zero;
        }
    }

    for(uint32_t k_base = 0; k_base < K; k_base += LdsK)
    {
        const uint32_t a_units_row = kBlockMTiles16 * 16u * (kLdsKBytes / 8u);
        for(uint32_t idx = threadIdx.x; idx < a_units_row; idx += kThreadsPerBlock)
        {
            const uint32_t u64_per_row  = kLdsKBytes / 8u;
            const uint32_t row_in_block = idx / u64_per_row;
            const uint32_t u64_in_k     = idx - row_in_block * u64_per_row;
            const uint32_t tile16       = row_in_block / 16u;
            const uint32_t lane_in_tile = row_in_block - tile16 * 16u;
            const uint32_t kg_local     = u64_in_k / 2u;
            const uint32_t rb           = u64_in_k & 1u;
            const uint32_t row          = block_m_base + row_in_block;
            const uint32_t global_byte  = k_base / 2u + u64_in_k * 8u;
            uint64_t value              = 0;
            if(row < M && global_byte < k_bytes)
            {
                const uint32_t available =
                    ((k_bytes - global_byte) < 8u) ? (k_bytes - global_byte) : 8u;
                value = load_u64_or_tail(A + static_cast<size_t>(row) * k_bytes + global_byte,
                                         available);
            }
            const uint32_t swiz =
                (((tile16 * kLdsKGroups + kg_local) * 2u + rb) * 16u + lane_in_tile);
            s_A_u64[swiz] = value;
        }

        const uint32_t b_units_row = kBlockNTiles16 * 16u * (kLdsKBytes / 8u);
        for(uint32_t idx = threadIdx.x; idx < b_units_row; idx += kThreadsPerBlock)
        {
            const uint32_t u64_per_col  = kLdsKBytes / 8u;
            const uint32_t col_in_block = idx / u64_per_col;
            const uint32_t u64_in_k     = idx - col_in_block * u64_per_col;
            const uint32_t tile16       = col_in_block / 16u;
            const uint32_t lane_in_tile = col_in_block - tile16 * 16u;
            const uint32_t kg_local     = u64_in_k / 2u;
            const uint32_t rb           = u64_in_k & 1u;
            const uint32_t col          = block_n_base + col_in_block;
            const uint32_t global_byte  = k_base / 2u + u64_in_k * 8u;
            uint64_t value              = 0;
            if(col < N && global_byte < k_bytes)
            {
                const uint32_t available =
                    ((k_bytes - global_byte) < 8u) ? (k_bytes - global_byte) : 8u;
                value = load_u64_or_tail(B + static_cast<size_t>(col) * k_bytes + global_byte,
                                         available);
            }
            const uint32_t swiz =
                (((tile16 * kLdsKGroups + kg_local) * 2u + rb) * 16u + lane_in_tile);
            s_B_u64[swiz] = value;
        }
        __syncthreads();

        const uint32_t tile_k = (K - k_base) < LdsK ? (K - k_base) : LdsK;
        for(uint32_t q = 0; q < tile_k; q += kWmmaK)
        {
            const uint32_t kg_local = q / kWmmaK;
            I32x2 b_frags[NTilesPerWave];
#pragma unroll
            for(uint32_t nt = 0; nt < NTilesPerWave; ++nt)
            {
                const uint32_t b_tile16 = wave_n * NTilesPerWave + nt;
                b_frags[nt] =
                    load_pre_i4x16_as_i32x2(s_B, b_tile16, kg_local, row_blk, lane16,
                                            kLdsKGroups);
            }

#pragma unroll
            for(uint32_t mt = 0; mt < MTilesPerWave; ++mt)
            {
                const uint32_t a_tile16 = wave_m * MTilesPerWave + mt;
                const I32x2 a_frag =
                    load_pre_i4x16_as_i32x2(s_A, a_tile16, kg_local, row_blk, lane16,
                                            kLdsKGroups);
#pragma unroll
                for(uint32_t nt = 0; nt < NTilesPerWave; ++nt)
                {
                    acc[mt][nt] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                        true, a_frag, true, b_frags[nt], acc[mt][nt], false);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for(uint32_t mt = 0; mt < MTilesPerWave; ++mt)
    {
#pragma unroll
        for(uint32_t i = 0; i < 8u; ++i)
        {
            const uint32_t row = m_tile_base + mt * 16u + row_blk * 8u + i;
#pragma unroll
            for(uint32_t nt = 0; nt < NTilesPerWave; ++nt)
            {
                const uint32_t col = n_tile_base + nt * 16u + lane16;
                if(row < M && col < N)
                {
                    C[static_cast<size_t>(row) * N + col] = acc[mt][nt][i];
                }
            }
        }
    }
#else
    (void)A;
    (void)B;
    (void)C;
    (void)M;
    (void)N;
    (void)K;
#endif
}

enum class DispatchVariant
{
    Kv1024LdsSwizzle,
    N4096LdsSwizzle,
    N5120BPre,
    N6656BPre,
    N8192BPre,
    N11008BPre,
    N14336BPre,
    N17920BPre,
    N22016BPre,
    N25600BPre,
    FallbackBPre,
};

DispatchVariant select_variant(uint32_t N)
{
    switch(N)
    {
    case 1024: return DispatchVariant::Kv1024LdsSwizzle;
    case 4096: return DispatchVariant::N4096LdsSwizzle;
    case 5120: return DispatchVariant::N5120BPre;
    case 6656: return DispatchVariant::N6656BPre;
    case 8192: return DispatchVariant::N8192BPre;
    case 11008: return DispatchVariant::N11008BPre;
    case 14336: return DispatchVariant::N14336BPre;
    case 17920: return DispatchVariant::N17920BPre;
    case 22016: return DispatchVariant::N22016BPre;
    case 25600: return DispatchVariant::N25600BPre;
    default: return DispatchVariant::FallbackBPre;
    }
}

bool env_flag_enabled(const char* name)
{
    const char* value = std::getenv(name);
    if(value == nullptr)
    {
        return false;
    }
    return std::string(value) == "1" || std::string(value) == "true" ||
           std::string(value) == "TRUE" || std::string(value) == "on" ||
           std::string(value) == "ON" || std::string(value) == "yes" ||
           std::string(value) == "YES";
}

bool env_flag_disabled(const char* name)
{
    const char* value = std::getenv(name);
    if(value == nullptr)
    {
        return false;
    }
    return std::string(value) == "0" || std::string(value) == "false" ||
           std::string(value) == "FALSE" || std::string(value) == "off" ||
           std::string(value) == "OFF" || std::string(value) == "no" ||
           std::string(value) == "NO";
}

uint32_t read_nwaves_override(const char* name, uint32_t fallback)
{
    const char* value = std::getenv(name);
    if(value == nullptr)
    {
        return fallback;
    }
    const unsigned long parsed = std::strtoul(value, nullptr, 10);
    if(parsed != 1u && parsed != 2u && parsed != 4u && parsed != 8u)
    {
        throw std::invalid_argument(
            std::string(name) + " must be one of 1, 2, 4, or 8");
    }
    return static_cast<uint32_t>(parsed);
}

uint32_t qwen3_32b_grouped_scale_nwaves()
{
    static const uint32_t value = read_nwaves_override(
        "QUAROT_QWEN3_32B_GROUPED_NWAVES", 4u);
    return value;
}

uint32_t qwen3_32b_multi_scale_nwaves()
{
    static const uint32_t value = read_nwaves_override(
        "QUAROT_QWEN3_32B_MULTI_NWAVES", 2u);
    return value;
}

bool is_qwen3_32b_multi_scale(uint32_t N0, uint32_t N1, uint32_t N2, uint32_t K)
{
    return K == 5120u &&
           ((N0 == 8192u && N1 == 1024u && N2 == 1024u) ||
            (N0 == 25600u && N1 == 25600u && N2 == 0u));
}

bool bpre_cache_disabled()
{
    if(env_flag_enabled("GEMM_INT4_HIP_DISABLE_BPRE") ||
       env_flag_enabled("GEMM_INT4_HIP_DISABLE_BPRE_CACHE"))
    {
        return true;
    }
    // Runtime caching duplicates the row-packed weight. Keep it opt-in;
    // Linear4bit uses replacement prepacked storage instead.
    return !env_flag_enabled("GEMM_INT4_HIP_BPRE_CACHE");
}

bool bpre_input_enabled()
{
    const char* layout = std::getenv("GEMM_INT4_HIP_B_LAYOUT");
    return env_flag_enabled("GEMM_INT4_HIP_ASSUME_BPRE_INPUT") ||
           (layout != nullptr &&
            (std::string(layout) == "bpre" || std::string(layout) == "BPRE" ||
             std::string(layout) == "bpre-only" || std::string(layout) == "offline-bpre"));
}

const char* variant_name(DispatchVariant variant)
{
    switch(variant)
    {
    case DispatchVariant::Kv1024LdsSwizzle: return "N=1024 LDS swizzle, block4x4 mt2 nt2 LDS64";
    case DispatchVariant::N4096LdsSwizzle: return "N=4096 LDS swizzle, block4x4 mt2 nt2 LDS64";
    case DispatchVariant::N5120BPre: return "N=5120 Qwen3-32B cached B-prepack, block2x2 mt2 nt4";
    case DispatchVariant::N6656BPre: return "N=6656 cached B-prepack, block2x2 mt2 nt4";
    case DispatchVariant::N8192BPre: return "N=8192 cached B-prepack, block2x2 mt2 nt4";
    case DispatchVariant::N11008BPre: return "N=11008 cached B-prepack, block2x2 mt2 nt4";
    case DispatchVariant::N14336BPre: return "N=14336 cached B-prepack, block2x2 mt2 nt4";
    case DispatchVariant::N17920BPre: return "N=17920 cached B-prepack, block2x2 mt2 nt4";
    case DispatchVariant::N22016BPre: return "N=22016 cached B-prepack, block2x2 mt2 nt4";
    case DispatchVariant::N25600BPre: return "N=25600 Qwen3-32B cached B-prepack, block2x2 mt2 nt4";
    case DispatchVariant::FallbackBPre: return "fallback cached B-prepack, block4x4 mt1 nt2";
    }
    return "unknown";
}

const char* variant_name_memory_mode(DispatchVariant variant)
{
    if(!bpre_cache_disabled())
    {
        return variant_name(variant);
    }

    switch(variant)
    {
    case DispatchVariant::Kv1024LdsSwizzle: return "N=1024 LDS swizzle, block4x4 mt2 nt2 LDS64";
    case DispatchVariant::N4096LdsSwizzle: return "N=4096 LDS swizzle, block4x4 mt2 nt2 LDS64";
    case DispatchVariant::N5120BPre:
        return "N=5120 Qwen3-32B memory mode, row-packed LDS swizzle, block2x2 mt2 nt4 LDS64";
    case DispatchVariant::N6656BPre:
        return "N=6656 memory mode, row-packed LDS swizzle, block2x2 mt2 nt4 LDS64";
    case DispatchVariant::N8192BPre:
        return "N=8192 memory mode, row-packed LDS swizzle, block2x2 mt2 nt4 LDS64";
    case DispatchVariant::N11008BPre:
        return "N=11008 memory mode, row-packed LDS swizzle, block2x2 mt2 nt4 LDS64";
    case DispatchVariant::N14336BPre:
        return "N=14336 memory mode, row-packed LDS swizzle, block2x4 mt4 nt4 LDS128";
    case DispatchVariant::N17920BPre:
        return "N=17920 memory mode, row-packed LDS swizzle, block2x4 mt4 nt4 LDS128";
    case DispatchVariant::N22016BPre:
        return "N=22016 memory mode, row-packed LDS swizzle, block2x4 mt4 nt4 LDS128";
    case DispatchVariant::N25600BPre:
        return "N=25600 Qwen3-32B memory mode, row-packed LDS swizzle, block2x4 mt4 nt4 LDS128";
    case DispatchVariant::FallbackBPre:
        return "fallback memory mode, row-packed LDS swizzle, block4x4 mt1 nt2 LDS64";
    }
    return "unknown";
}

const char* variant_name_bpre_only(DispatchVariant variant)
{
    switch(variant)
    {
    case DispatchVariant::Kv1024LdsSwizzle:
        return "N=1024 BPre-only, block2x2 mt2 nt4";
    case DispatchVariant::N4096LdsSwizzle:
        return "N=4096 BPre-only, block2x2 mt2 nt4";
    case DispatchVariant::N5120BPre:
        return "N=5120 Qwen3-32B BPre-only, block2x2 mt2 nt4";
    case DispatchVariant::N6656BPre:
        return "N=6656 BPre-only, block2x2 mt2 nt4";
    case DispatchVariant::N8192BPre:
        return "N=8192 BPre-only, block2x2 mt2 nt4";
    case DispatchVariant::N11008BPre:
        return "N=11008 BPre-only, block2x2 mt2 nt4";
    case DispatchVariant::N14336BPre:
        return "N=14336 BPre-only, block2x2 mt2 nt4";
    case DispatchVariant::N17920BPre:
        return "N=17920 BPre-only, block2x2 mt2 nt4";
    case DispatchVariant::N22016BPre:
        return "N=22016 BPre-only, block2x2 mt2 nt4";
    case DispatchVariant::N25600BPre:
        return "N=25600 Qwen3-32B BPre-only, block2x2 mt2 nt4";
    case DispatchVariant::FallbackBPre:
        return "fallback BPre-only, block4x4 mt1 nt2";
    }
    return "unknown";
}

bool variant_uses_bpre(DispatchVariant variant)
{
    switch(variant)
    {
    case DispatchVariant::Kv1024LdsSwizzle:
    case DispatchVariant::N4096LdsSwizzle: return false;
    case DispatchVariant::N5120BPre:
    case DispatchVariant::N6656BPre:
    case DispatchVariant::N8192BPre:
    case DispatchVariant::N11008BPre:
    case DispatchVariant::N14336BPre:
    case DispatchVariant::N17920BPre:
    case DispatchVariant::N22016BPre:
    case DispatchVariant::N25600BPre:
    case DispatchVariant::FallbackBPre: return true;
    }
    return true;
}

struct BPrepackKey
{
    const Int4Storage* b;
    uint32_t n;
    uint32_t k;
    int device;

    bool operator==(const BPrepackKey& other) const
    {
        return b == other.b && n == other.n && k == other.k && device == other.device;
    }
};

struct BPrepackKeyHash
{
    size_t operator()(const BPrepackKey& key) const
    {
        size_t h = reinterpret_cast<size_t>(key.b);
        h ^= static_cast<size_t>(key.n) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= static_cast<size_t>(key.k) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= static_cast<size_t>(key.device) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        return h;
    }
};

struct BPrepackEntry
{
    Int4Storage* ptr = nullptr;
    size_t bytes     = 0;
};

std::mutex g_bpre_mutex;
std::unordered_map<BPrepackKey, BPrepackEntry, BPrepackKeyHash> g_bpre_cache;

void launch_prepack_b(const Int4Storage* B, Int4Storage* BPre, uint32_t N, uint32_t K)
{
    const uint32_t units = static_cast<uint32_t>(b_prepacked_size_bytes(N, K) / 8u);
    const uint32_t block = 256;
    const uint32_t grid  = (units + block - 1u) / block;
    prepack_b_kernel<<<grid, block, 0, at::hip::getCurrentHIPStream().stream()>>>(B, BPre, N, K);
    check_hip(hipGetLastError(), "prepack_b_kernel launch");
}

const Int4Storage* get_or_create_bpre(const Int4Storage* B, uint32_t N, uint32_t K)
{
    int device = 0;
    check_hip(hipGetDevice(&device), "hipGetDevice");

    const BPrepackKey key{B, N, K, device};
    std::lock_guard<std::mutex> lock(g_bpre_mutex);

    auto it = g_bpre_cache.find(key);
    if(it != g_bpre_cache.end())
    {
        return it->second.ptr;
    }

    BPrepackEntry entry;
    entry.bytes = b_prepacked_size_bytes(N, K);
    check_hip(hipMalloc(reinterpret_cast<void**>(&entry.ptr), entry.bytes), "hipMalloc B prepack");
    launch_prepack_b(B, entry.ptr, N, K);

    const auto inserted = g_bpre_cache.emplace(key, entry);
    return inserted.first->second.ptr;
}

template <uint32_t BlockMWaves,
          uint32_t BlockNWaves,
          uint32_t MTilesPerWave,
          uint32_t NTilesPerWave>
void launch_bpre(const Int4Storage* A,
                 const Int4Storage* BPre,
                 int32_t* C,
                 uint32_t M,
                 uint32_t N,
                 uint32_t K)
{
    constexpr uint32_t kThreadsPerBlock = 32u * BlockMWaves * BlockNWaves;
    const dim3 block(kThreadsPerBlock);
    const dim3 grid((N + 16u * BlockNWaves * NTilesPerWave - 1u) /
                        (16u * BlockNWaves * NTilesPerWave),
                    (M + 16u * BlockMWaves * MTilesPerWave - 1u) /
                        (16u * BlockMWaves * MTilesPerWave));

    bpre_kernel<BlockMWaves, BlockNWaves, MTilesPerWave, NTilesPerWave>
        <<<grid, block, 0, at::hip::getCurrentHIPStream().stream()>>>(A, BPre, C, M, N, K);
}

template <uint32_t BlockMWaves,
          uint32_t BlockNWaves,
          uint32_t MTilesPerWave,
          uint32_t NTilesPerWave,
          uint32_t LdsK>
void launch_lds_swizzle(const Int4Storage* A,
                        const Int4Storage* B,
                        int32_t* C,
                        uint32_t M,
                        uint32_t N,
                        uint32_t K)
{
    constexpr uint32_t kThreadsPerBlock = 32u * BlockMWaves * BlockNWaves;
    const dim3 block(kThreadsPerBlock);
    const dim3 grid((N + 16u * BlockNWaves * NTilesPerWave - 1u) /
                        (16u * BlockNWaves * NTilesPerWave),
                    (M + 16u * BlockMWaves * MTilesPerWave - 1u) /
                        (16u * BlockMWaves * MTilesPerWave));

    lds_swizzle_kernel<BlockMWaves, BlockNWaves, MTilesPerWave, NTilesPerWave, LdsK>
        <<<grid, block, 0, at::hip::getCurrentHIPStream().stream()>>>(A, B, C, M, N, K);
}

template <uint32_t BlockNWaves>
void launch_bpre_grouped_scale(const Int4Storage* A,
                               const Int4Storage* BPre,
                               const half* scale_group,
                               const half* scale_weight,
                               half* C,
                               uint32_t M,
                               uint32_t N,
                               uint32_t K,
                               uint32_t scale_groups)
{
    constexpr uint32_t kThreadsPerBlock = 32u * BlockNWaves;
    const dim3 block(kThreadsPerBlock);
    const dim3 grid((N + 16u * BlockNWaves - 1u) / (16u * BlockNWaves),
                    (M + 15u) / 16u);

    bpre_grouped_scale_kernel<1, BlockNWaves><<<grid, block, 0, at::hip::getCurrentHIPStream().stream()>>>(
        A, BPre, scale_group, scale_weight, C, M, N, K, scale_groups);
}

template <uint32_t BlockNWaves>
void launch_bpre_multi_scale(const Int4Storage* A,
                             const Int4Storage* B0,
                             const Int4Storage* B1,
                             const Int4Storage* B2,
                             const half* scale_activation,
                             const half* scale_weight0,
                             const half* scale_weight1,
                             const half* scale_weight2,
                             half* C,
                             uint32_t M, uint32_t N0, uint32_t N1,
                             uint32_t N2, uint32_t K,
                             bool allow_interleaved)
{
    constexpr uint32_t kThreadsPerBlock = 32u * BlockNWaves;
    const uint32_t total_n = N0 + N1 + N2;
    const dim3 block(kThreadsPerBlock);
    const dim3 grid((total_n + 16u * BlockNWaves - 1u) /
                        (16u * BlockNWaves),
                    (M + 15u) / 16u);

    bpre_multi_scale_kernel<BlockNWaves><<<grid, block, 0, at::hip::getCurrentHIPStream().stream()>>>(
        A, B0, B1, B2, scale_activation, scale_weight0, scale_weight1,
        scale_weight2, C, M, N0, N1, N2, K, allow_interleaved);
}

void dispatch_matmul(const Int4Storage* A,
                     const Int4Storage* BRow,
                     const Int4Storage* BPre,
                     uint32_t M,
                     uint32_t N,
                     uint32_t K,
                     int32_t* C)
{
    if((K % kWmmaK) != 0)
    {
        throw std::runtime_error("K must be a multiple of 32");
    }

    switch(select_variant(N))
    {
    case DispatchVariant::Kv1024LdsSwizzle:
        launch_lds_swizzle<4, 4, 2, 2, 64>(A, BRow, C, M, N, K);
        break;
    case DispatchVariant::N4096LdsSwizzle:
        launch_lds_swizzle<4, 4, 2, 2, 64>(A, BRow, C, M, N, K);
        break;
    case DispatchVariant::N5120BPre:
    case DispatchVariant::N6656BPre:
        if(BPre != nullptr)
        {
            launch_bpre<2, 2, 2, 4>(A, BPre, C, M, N, K);
        }
        else
        {
            launch_lds_swizzle<2, 2, 2, 4, 64>(A, BRow, C, M, N, K);
        }
        break;
    case DispatchVariant::N8192BPre:
        if(BPre != nullptr)
        {
            launch_bpre<2, 2, 2, 4>(A, BPre, C, M, N, K);
        }
        else
        {
            launch_lds_swizzle<2, 2, 2, 4, 64>(A, BRow, C, M, N, K);
        }
        break;
    case DispatchVariant::N11008BPre:
        if(BPre != nullptr)
        {
            launch_bpre<2, 2, 2, 4>(A, BPre, C, M, N, K);
        }
        else
        {
            launch_lds_swizzle<2, 2, 2, 4, 64>(A, BRow, C, M, N, K);
        }
        break;
    case DispatchVariant::N25600BPre:
    case DispatchVariant::N14336BPre:
        if(BPre != nullptr)
        {
            launch_bpre<2, 2, 2, 4>(A, BPre, C, M, N, K);
        }
        else
        {
            launch_lds_swizzle<2, 4, 4, 4, 128>(A, BRow, C, M, N, K);
        }
        break;
    case DispatchVariant::N17920BPre:
        if(BPre != nullptr)
        {
            launch_bpre<2, 2, 2, 4>(A, BPre, C, M, N, K);
        }
        else
        {
            launch_lds_swizzle<2, 4, 4, 4, 128>(A, BRow, C, M, N, K);
        }
        break;
    case DispatchVariant::N22016BPre:
        if(BPre != nullptr)
        {
            launch_bpre<2, 2, 2, 4>(A, BPre, C, M, N, K);
        }
        else
        {
            launch_lds_swizzle<2, 4, 4, 4, 128>(A, BRow, C, M, N, K);
        }
        break;
    case DispatchVariant::FallbackBPre:
        if(BPre != nullptr)
        {
            launch_bpre<4, 4, 1, 2>(A, BPre, C, M, N, K);
        }
        else
        {
            launch_lds_swizzle<4, 4, 1, 2, 64>(A, BRow, C, M, N, K);
        }
        break;
    }
    check_hip(hipGetLastError(), "dispatch_matmul launch");
}

void dispatch_matmul_bpre_only(const Int4Storage* A,
                               const Int4Storage* BPre,
                               uint32_t M,
                               uint32_t N,
                               uint32_t K,
                               int32_t* C)
{
    if((K % kWmmaK) != 0)
    {
        throw std::runtime_error("K must be a multiple of 32");
    }

    if(M <= 16u && env_flag_enabled("QUAROT_ENABLE_M16_BPRE"))
    {
        // PARD2 verification is padded to one WMMA M tile. Keep all four
        // waves on that tile and distribute them across N.
        launch_bpre<1, 4, 1, 2>(A, BPre, C, M, N, K);
        check_hip(hipGetLastError(), "dispatch_matmul_bpre_only M16 launch");
        return;
    }

    switch(select_variant(N))
    {
    case DispatchVariant::Kv1024LdsSwizzle:
    case DispatchVariant::N4096LdsSwizzle:
    case DispatchVariant::N5120BPre:
    case DispatchVariant::N6656BPre:
    case DispatchVariant::N8192BPre:
    case DispatchVariant::N11008BPre:
    case DispatchVariant::N14336BPre:
    case DispatchVariant::N17920BPre:
    case DispatchVariant::N22016BPre:
    case DispatchVariant::N25600BPre:
        launch_bpre<2, 2, 2, 4>(A, BPre, C, M, N, K);
        break;
    case DispatchVariant::FallbackBPre:
        launch_bpre<4, 4, 1, 2>(A, BPre, C, M, N, K);
        break;
    }
    check_hip(hipGetLastError(), "dispatch_matmul_bpre_only launch");
}

} // namespace

void matmul_host(const Int4Storage* A,
                 const Int4Storage* B,
                 uint32_t M,
                 uint32_t N,
                 uint32_t K,
                 int32_t* C)
{
#if !defined(QUAROT_BPRE_GFX12)
    (void)A;
    (void)B;
    (void)M;
    (void)N;
    (void)K;
    (void)C;
    throw std::runtime_error("GEMM_int4_for_hip requires gfx12/gfx1201 WMMA int4 support");
#else
    if(A == nullptr || B == nullptr || C == nullptr)
    {
        throw std::invalid_argument("matmul_host received a null device pointer");
    }
    if((K % kWmmaK) != 0)
    {
        throw std::invalid_argument("K must be a multiple of 32");
    }

    const DispatchVariant variant = select_variant(N);
    if(bpre_input_enabled() && variant != DispatchVariant::Kv1024LdsSwizzle)
    {
        dispatch_matmul_bpre_only(A, B, M, N, K, C);
        return;
    }

    const bool use_bpre_cache = variant_uses_bpre(variant) && !bpre_cache_disabled();
    const Int4Storage* BPre   = use_bpre_cache ? get_or_create_bpre(B, N, K) : nullptr;
    dispatch_matmul(A, B, BPre, M, N, K, C);
#endif
}

size_t prepack_b_host_size_bytes(uint32_t N, uint32_t K)
{
    if((K % kWmmaK) != 0)
    {
        throw std::invalid_argument("K must be a multiple of 32");
    }
    return b_prepacked_size_bytes(N, K);
}

void prepack_b_cpu_host(const Int4Storage* BRowHost,
                        uint32_t N,
                        uint32_t K,
                        Int4Storage* BPreHost)
{
#if !defined(QUAROT_BPRE_GFX12)
    (void)BRowHost;
    (void)N;
    (void)K;
    (void)BPreHost;
    throw std::runtime_error("GEMM_int4_for_hip requires gfx12/gfx1201 WMMA int4 support");
#else
    if(BRowHost == nullptr || BPreHost == nullptr)
    {
        throw std::invalid_argument("prepack_b_cpu_host received a null host pointer");
    }
    if((K % kWmmaK) != 0)
    {
        throw std::invalid_argument("K must be a multiple of 32");
    }

    const uint32_t n_tiles16 = (N + 15u) / 16u;
    const uint32_t k_groups  = K / kWmmaK;
    const uint32_t k_bytes   = K / 2u;
    const size_t dst_bytes   = b_prepacked_size_bytes(N, K);

    std::memset(BPreHost, 0, dst_bytes);
    for(uint32_t tile = 0; tile < n_tiles16; ++tile)
    {
        for(uint32_t kg = 0; kg < k_groups; ++kg)
        {
            for(uint32_t row_blk = 0; row_blk < 2u; ++row_blk)
            {
                for(uint32_t lane16 = 0; lane16 < 16u; ++lane16)
                {
                    const uint32_t n      = tile * 16u + lane16;
                    const uint32_t byte_k = kg * 16u + row_blk * 8u;
                    const size_t dst_offset =
                        (((static_cast<size_t>(tile) * k_groups + kg) * 2u + row_blk) *
                             16u +
                         lane16) *
                        8u;
                    if(n < N)
                    {
                        const size_t src_offset = static_cast<size_t>(n) * k_bytes + byte_k;
                        std::memcpy(BPreHost + dst_offset, BRowHost + src_offset, 8u);
                    }
                }
            }
        }
    }
#endif
}

void prepack_b_device_host(const Int4Storage* BRow,
                           uint32_t N,
                           uint32_t K,
                           Int4Storage* BPre)
{
#if !defined(QUAROT_BPRE_GFX12)
    (void)BRow; (void)N; (void)K; (void)BPre;
    throw std::runtime_error("GEMM_int4_for_hip requires gfx12/gfx1201 WMMA int4 support");
#else
    if(BRow == nullptr || BPre == nullptr)
        throw std::invalid_argument("prepack_b_device_host received a null device pointer");
    if((K % kWmmaK) != 0)
        throw std::invalid_argument("K must be a multiple of 32");
    launch_prepack_b(BRow, BPre, N, K);
#endif
}

void matmul_bpre_host(const Int4Storage* A,
                      const Int4Storage* BPre,
                      uint32_t M,
                      uint32_t N,
                      uint32_t K,
                      int32_t* C)
{
#if !defined(QUAROT_BPRE_GFX12)
    (void)A;
    (void)BPre;
    (void)M;
    (void)N;
    (void)K;
    (void)C;
    throw std::runtime_error("GEMM_int4_for_hip requires gfx12/gfx1201 WMMA int4 support");
#else
    if(A == nullptr || BPre == nullptr || C == nullptr)
    {
        throw std::invalid_argument("matmul_bpre_host received a null device pointer");
    }
    if((K % kWmmaK) != 0)
    {
        throw std::invalid_argument("K must be a multiple of 32");
    }

    dispatch_matmul_bpre_only(A, BPre, M, N, K, C);
#endif
}

void matmul_bpre_grouped_scale_host(const Int4Storage* A,
                                    const Int4Storage* BPre,
                                    const half* scale_group,
                                    const half* scale_weight,
                                    uint32_t M,
                                    uint32_t N,
                                    uint32_t K,
                                    half* C,
                                    uint32_t scale_groups)
{
#if !defined(QUAROT_BPRE_GFX12)
    throw std::runtime_error("grouped-scale GEMM requires gfx12/gfx1201 WMMA int4 support");
#else
    if(A == nullptr || BPre == nullptr || scale_group == nullptr ||
       scale_weight == nullptr || C == nullptr)
        throw std::invalid_argument("grouped-scale GEMM received a null device pointer");
    if(scale_groups == 0 || K % scale_groups != 0 ||
       ((K / scale_groups) % kWmmaK) != 0)
        throw std::invalid_argument("invalid grouped-scale GEMM scale layout");
    if(M >= 512u)
    {
        const dim3 block(128u);
        const dim3 grid((N + 31u) / 32u, (M + 31u) / 32u);
        bpre_grouped_scale_kernel<2, 2><<<grid, block, 0, at::hip::getCurrentHIPStream().stream()>>>(
            A, BPre, scale_group, scale_weight, C, M, N, K, scale_groups);
    }
    else if(M <= 16u)
    {
        const bool exact_qwen3_32b =
            (N == 5120u && (K == 8192u || K == 25600u)) ||
            (N == 25600u && K == 5120u);
        const uint32_t nwaves = exact_qwen3_32b
            ? qwen3_32b_grouped_scale_nwaves() : 4u;
        if(nwaves == 8u)
            launch_bpre_grouped_scale<8>(
                A, BPre, scale_group, scale_weight, C, M, N, K,
                scale_groups);
        else if(nwaves == 4u)
            launch_bpre_grouped_scale<4>(
                A, BPre, scale_group, scale_weight, C, M, N, K,
                scale_groups);
        else if(nwaves == 2u)
            launch_bpre_grouped_scale<2>(
                A, BPre, scale_group, scale_weight, C, M, N, K,
                scale_groups);
        else
            launch_bpre_grouped_scale<1>(
                A, BPre, scale_group, scale_weight, C, M, N, K,
                scale_groups);
    }
    else
    {
        const dim3 block(32u);
        const dim3 grid((N + 15u) / 16u, (M + 15u) / 16u);
        bpre_grouped_scale_kernel<1, 1><<<grid, block, 0, at::hip::getCurrentHIPStream().stream()>>>(
            A, BPre, scale_group, scale_weight, C, M, N, K, scale_groups);
    }
    check_hip(hipGetLastError(), "bpre_grouped_scale_kernel launch");
#endif
}

void matmul_bpre_multi_scale_host(const Int4Storage* A,
                                  const Int4Storage* B0,
                                  const Int4Storage* B1,
                                  const Int4Storage* B2,
                                  const half* scale_activation,
                                  const half* scale_weight0,
                                  const half* scale_weight1,
                                  const half* scale_weight2,
                                  uint32_t M, uint32_t N0, uint32_t N1,
                                  uint32_t N2, uint32_t K, half* C)
{
#if !defined(QUAROT_BPRE_GFX12)
    throw std::runtime_error("multi-scale GEMM requires gfx12/gfx1201 WMMA int4 support");
#else
    if(A == nullptr || B0 == nullptr || B1 == nullptr ||
       scale_activation == nullptr || scale_weight0 == nullptr ||
       scale_weight1 == nullptr || C == nullptr ||
       (N2 != 0 && (B2 == nullptr || scale_weight2 == nullptr)))
        throw std::invalid_argument("multi-scale GEMM received a null device pointer");
    if((N0 % 16u) || (N1 % 16u) || (N2 % 16u) || (K % kWmmaK))
        throw std::invalid_argument("multi-scale GEMM dimensions must be WMMA aligned");
    const char* interleaved_env = std::getenv(
        "QUAROT_INTERLEAVED_PROJECTIONS");
    const bool allow_interleaved = interleaved_env == nullptr ||
        std::strcmp(interleaved_env, "0") != 0;
    const uint32_t nwaves = is_qwen3_32b_multi_scale(N0, N1, N2, K)
        ? qwen3_32b_multi_scale_nwaves() : 1u;
    if(nwaves == 8u)
        launch_bpre_multi_scale<8>(
            A, B0, B1, B2, scale_activation, scale_weight0, scale_weight1,
            scale_weight2, C, M, N0, N1, N2, K, allow_interleaved);
    else if(nwaves == 4u)
        launch_bpre_multi_scale<4>(
            A, B0, B1, B2, scale_activation, scale_weight0, scale_weight1,
            scale_weight2, C, M, N0, N1, N2, K, allow_interleaved);
    else if(nwaves == 2u)
        launch_bpre_multi_scale<2>(
            A, B0, B1, B2, scale_activation, scale_weight0, scale_weight1,
            scale_weight2, C, M, N0, N1, N2, K, allow_interleaved);
    else
        launch_bpre_multi_scale<1>(
            A, B0, B1, B2, scale_activation, scale_weight0, scale_weight1,
            scale_weight2, C, M, N0, N1, N2, K, allow_interleaved);
    check_hip(hipGetLastError(), "bpre_multi_scale_kernel launch");
#endif
}

void clear_prepacked_weight_cache()
{
    std::lock_guard<std::mutex> lock(g_bpre_mutex);
    for(auto& item : g_bpre_cache)
    {
        if(item.second.ptr != nullptr)
        {
            check_hip(hipFree(item.second.ptr), "hipFree cached prepacked B");
        }
    }
    g_bpre_cache.clear();
}

size_t prepacked_weight_cache_bytes()
{
    std::lock_guard<std::mutex> lock(g_bpre_mutex);
    size_t total = 0;
    for(const auto& item : g_bpre_cache)
    {
        total += item.second.bytes;
    }
    return total;
}

const char* selected_kernel_name(uint32_t N)
{
    const DispatchVariant variant = select_variant(N);
    if(bpre_input_enabled() && variant != DispatchVariant::Kv1024LdsSwizzle)
    {
        return variant_name_bpre_only(variant);
    }
    return variant_name_memory_mode(variant);
}

const char* selected_bpre_kernel_name(uint32_t N)
{
    return variant_name_bpre_only(select_variant(N));
}
