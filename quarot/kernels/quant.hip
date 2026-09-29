#include <ATen/hip/HIPContext.h>
#include <quant.h>

__device__ __forceinline__ int half_to_int_rn(half value)
{
    return __float2int_rn(__half2float(value));
}

__global__ void sym_quantize_f16_i4_kernel(
        const half *__restrict__ x,
        const half *__restrict__ scale,
        uint32_t rows,
        uint32_t colsSrc,
        uint32_t colsDst,
        Int4Storage *__restrict__ q
)
{
    uint32_t r = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t c = threadIdx.x + blockIdx.x * blockDim.x;

    if (r < rows && 2 * c < colsSrc) // boundary
    {
        half data_low = __hdiv(x[r * colsSrc + 2 * c], scale[r]);
        int8_t low = clamp(half_to_int_rn(data_low), qmin, qmax);

        int8_t up = 0;
        if (2 * c + 1 < colsSrc)
        {
            half data_up = __hdiv(x[r * colsSrc + 2 * c + 1], scale[r]);
            up = clamp(half_to_int_rn(data_up), qmin, qmax);
        }

        q[r * colsDst + c] = (low & 0x0F) | ((up & 0x0F) << 4);
    }
}

__global__ void sym_dequantize_i32_f16_kernel(
        const int32_t *__restrict__ q,
        const half *__restrict__ scale_row,
        const half *__restrict__ scale_col,
        uint32_t rows, uint32_t cols,
        half *__restrict__ x
)
{
    uint32_t r = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t c = threadIdx.x + blockIdx.x * blockDim.x;
    if (r < rows && c < cols) // boundary
    {
        half acc = __int2half_rn(q[r * cols + c]);
        // x[r * cols + c] = scale_row[r] * scale_col[c] * acc;
        float scale =
            __half2float(scale_row[r]) *
            __half2float(scale_col[c]);

        x[r * cols + c] =
            __float2half(scale * __half2float(acc));
    }
}

void sym_quant_host(
        const half *x,
        const half *scale,
        uint32_t rows,
        uint32_t colsSrc,
        uint32_t colsDst,
        Int4Storage *q
)
{

    dim3 block{std::min<uint32_t>(colsDst, 32), std::min<uint32_t>(rows, 16)};
    dim3 grid{cdiv(colsDst, block.x), cdiv(rows, block.y)};
    sym_quantize_f16_i4_kernel<<<grid, block, 0, at::hip::getCurrentHIPStream().stream()>>>(x, scale, rows, colsSrc, colsDst, q);
}


void sym_dequant_host(const int32_t *q,
                                 const half *scale_row,
                                 const half *scale_col,
                                 uint32_t rows,
                                 uint32_t cols,
                                 half *x
)
{
    dim3 block{std::min<uint32_t>(cols, 16), std::min<uint32_t>(rows, 16)};
    dim3 grid{cdiv(cols, block.x), cdiv(rows, block.y)};
    sym_dequantize_i32_f16_kernel<<<grid, block, 0, at::hip::getCurrentHIPStream().stream()>>>(
            q,
            scale_row, scale_col,
            rows, cols, x);
}