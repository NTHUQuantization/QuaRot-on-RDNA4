#pragma once

#include <cstdint>

// QuaRot stores two signed, two's-complement 4-bit values in every byte.
// Keep this header independent of CK so that bindings.cpp can be compiled by
// the host C++ compiler used by PyTorch's extension build.
using Int4Storage = std::uint8_t;
constexpr std::uint32_t kElementsPerVector = 2;

__host__ __device__ inline std::int8_t unpack_int4(Int4Storage packed, int lane)
{
    std::int8_t value = static_cast<std::int8_t>((packed >> (lane * 4)) & 0x0f);
    return value >= 8 ? static_cast<std::int8_t>(value - 16) : value;
}

__host__ __device__ inline Int4Storage pack_int4(std::int8_t value, int lane)
{
    return static_cast<Int4Storage>((static_cast<std::uint8_t>(value) & 0x0f) << (lane * 4));
}
