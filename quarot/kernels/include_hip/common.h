#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cstdint>

using Int4Storage = uint8_t;

constexpr uint32_t kElementsPerVector = 2;
constexpr int qmin = -8;
constexpr int qmax = 7;

template <typename T>
constexpr T cdiv(T x, T y)
{
    return (x + y - 1) / y;
}

__host__ __device__ inline int clamp(
    int value, int lower, int upper)
{
    return value < lower ? lower :
           (value > upper ? upper : value);
}