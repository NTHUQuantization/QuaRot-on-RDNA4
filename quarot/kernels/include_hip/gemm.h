#pragma once

#include <common.h>

#include <cstddef>

// Drop-in replacement for QuaRot's CUTLASS int4 GEMM entry point.
//
// Inputs are device pointers:
//   A: row-packed signed int4, shape [M, K / 2]
//   B: row-packed signed int4, shape [N, K / 2]
//   C: int32 output, shape [M, N]
//
// Packing order matches QuaRot/CUTLASS int4 storage: low nibble is even K,
// high nibble is odd K. Accumulation is exact int4 x int4 -> int32.
void matmul_host(const Int4Storage* A,
                 const Int4Storage* B,
                 uint32_t M,
                 uint32_t N,
                 uint32_t K,
                 int32_t* C);

// Offline/CPU B-prepack support.
//
// Use this path when the model checkpoint stores the static B weight directly
// in the gfx12 WMMA-friendly layout. In that mode, only the prepacked B tensor
// needs to live on the GPU; the original row-packed B tensor can stay on disk or
// CPU and does not need a second device copy.
size_t prepack_b_host_size_bytes(uint32_t N, uint32_t K);

void prepack_b_cpu_host(const Int4Storage* BRowHost,
                        uint32_t N,
                        uint32_t K,
                        Int4Storage* BPreHost);

void prepack_b_device_host(const Int4Storage* BRow,
                           uint32_t N,
                           uint32_t K,
                           Int4Storage* BPre);

void matmul_bpre_host(const Int4Storage* A,
                      const Int4Storage* BPre,
                      uint32_t M,
                      uint32_t N,
                      uint32_t K,
                      int32_t* C);

void matmul_bpre_grouped_scale_host(const Int4Storage* A,
                                    const Int4Storage* BPre,
                                    const half* scale_group,
                                    const half* scale_weight,
                                    uint32_t M,
                                    uint32_t N,
                                    uint32_t K,
                                    half* C,
                                    uint32_t scale_groups = 0);

void matmul_bpre_multi_scale_host(const Int4Storage* A,
                                  const Int4Storage* B0,
                                  const Int4Storage* B1,
                                  const Int4Storage* B2,
                                  const half* scale_activation,
                                  const half* scale_weight0,
                                  const half* scale_weight1,
                                  const half* scale_weight2,
                                  uint32_t M,
                                  uint32_t N0,
                                  uint32_t N1,
                                  uint32_t N2,
                                  uint32_t K,
                                  half* C);


// Clear cached prepacked B weights. Call this if weight device pointers are
// destroyed/reallocated during a long-running process.
void clear_prepacked_weight_cache();

// Current device memory retained by the internal B-prepack cache. This remains zero by default; set GEMM_INT4_HIP_BPRE_CACHE=1 only for
// callers that explicitly accept a second device copy.
size_t prepacked_weight_cache_bytes();

// Human-readable dispatch choice for a given output width N.
// Useful for smoke tests and integration checks.
const char* selected_kernel_name(uint32_t N);

// Human-readable dispatch choice for matmul_bpre_host().
const char* selected_bpre_kernel_name(uint32_t N);
