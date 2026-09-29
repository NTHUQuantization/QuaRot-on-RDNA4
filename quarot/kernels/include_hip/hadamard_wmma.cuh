#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>

namespace quarot_hadamard_wmma {
using f16_frag = _Float16 __attribute__((ext_vector_type(8)));
using f32_frag = float __attribute__((ext_vector_type(8)));

__device__ __forceinline__ _Float16 bits_to_f16(uint16_t x) {
  union { uint16_t u; _Float16 h; } v; v.u = x; return v.h;
}
__device__ __forceinline__ uint16_t f16_to_bits(_Float16 x) {
  union { _Float16 h; uint16_t u; } v; v.h = x; return v.u;
}
__device__ __forceinline__ uint16_t f32_to_bits(float x) {
  return f16_to_bits(static_cast<_Float16>(x));
}
__device__ __forceinline__ float bits_to_f32(uint16_t x) {
  return static_cast<float>(bits_to_f16(x));
}
__device__ __forceinline__ f16_frag h16(int lane) {
  f16_frag h;
  const int col = lane & 15;
  const int base = (lane >> 4) * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const bool positive = (__builtin_popcount(
        static_cast<unsigned>((base + i) & col)) & 1) == 0;
    h[i] = static_cast<_Float16>(positive ? 0.25f : -0.25f);
  }
  return h;
}
__device__ __forceinline__ f16_frag load_row(
    const uint16_t* base, int lane) {
  f16_frag x; const int row = lane & 15; const int k = (lane >> 4) * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) x[i] = bits_to_f16(base[row * 16 + k + i]);
  return x;
}
__device__ __forceinline__ f16_frag load_transposed(
    const uint16_t* base, int lane) {
  f16_frag x; const int row = lane & 15; const int k = (lane >> 4) * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) x[i] = bits_to_f16(base[(k + i) * 16 + row]);
  return x;
}
__device__ __forceinline__ void store_row(
    uint16_t* base, int lane, f32_frag x) {
  const int col = lane & 15; const int row = (lane >> 4) * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) base[(row + i) * 16 + col] = f32_to_bits(x[i]);
}
__device__ __forceinline__ void store_transposed(
    uint16_t* base, int lane, f32_frag x) {
  const int col = lane & 15; const int row = (lane >> 4) * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) base[col * 16 + row + i] = f32_to_bits(x[i]);
}
__device__ __forceinline__ void h256(
    const uint16_t* input, uint16_t* temporary, uint16_t* output, int lane) {
  f32_frag zero = {0, 0, 0, 0, 0, 0, 0, 0};
  const f16_frag h = h16(lane);
  f32_frag first = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
      load_row(input, lane), h, zero);
  store_row(temporary, lane, first);
  __syncthreads();
  f32_frag second = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
      load_transposed(temporary, lane), h, zero);
  store_transposed(output, lane, second);
  __syncthreads();
}
}  // namespace quarot_hadamard_wmma
