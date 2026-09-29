import torch
from . import nn
from . import functional


import quarot._HIP


__all__ = [
           "matmul", "matmul_bpre", #int-4 matmul
           "sym_quant", "sym_dequant", "PackedQuantizedTensor", # Quantization
]

class ShapeHandler:
    def __init__(self, x: torch.Tensor):
        self.size_excl_last = x.numel()//x.shape[-1]
        self.shape_excl_last = tuple(x.shape[:-1])

    # Keep the last dim unchanged, flatten all previous dims
    def flatten(self, x: torch.Tensor):
        return x.view(self.size_excl_last, -1)

    # Recover back to the original shape.
    def unflatten(self, x: torch.Tensor):
        return x.view(self.shape_excl_last + (-1,))

    def unflatten_scale(self, x: torch.Tensor):
        return x.view(self.shape_excl_last)


def flatten_last_dim_and_return_shape(x: torch.Tensor):
    shape_excl_last = x.shape[:-1]
    x = x.view(-1, x.shape[-1])
    return x, shape_excl_last


def matmul(A, B):
    assert A.shape[-1] % 32 == 0

    A, A_shape_excl_last = flatten_last_dim_and_return_shape(A)
    B, B_shape_excl_last = flatten_last_dim_and_return_shape(B)

    original_m = A.shape[0]
    padded_m = (original_m + 15) // 16 * 16
    if padded_m != original_m:
        A = torch.nn.functional.pad(A, (0, 0, 0, padded_m - original_m))

    result = _HIP.matmul(A.contiguous(), B.contiguous())
    result = result[:original_m]

    return result.view(*A_shape_excl_last, *B_shape_excl_last)

def matmul_bpre(A, B, out_features, in_features):
    A, A_shape_excl_last = flatten_last_dim_and_return_shape(A)
    original_m = A.shape[0]
    padded_m = (original_m + 15) // 16 * 16
    if padded_m != original_m:
        A = torch.nn.functional.pad(A, (0, 0, 0, padded_m - original_m))
    result = _HIP.matmul_bpre(
        A.contiguous(), B.contiguous(), out_features, in_features)
    return result[:original_m].view(*A_shape_excl_last, out_features)


def matmul_bpre_grouped_scale(A, B, scales, weight_scales,
                              out_features, in_features):
    A, A_shape_excl_last = flatten_last_dim_and_return_shape(A)
    scales = scales.view(A.shape[0], -1)
    result = _HIP.matmul_bpre_grouped_scale(
        A.contiguous(), B.contiguous(), scales.contiguous(),
        weight_scales.view(-1).contiguous(), out_features, in_features)
    return result.view(*A_shape_excl_last, out_features)


def sym_quant(x, scale):
    assert x.dtype == scale.dtype == torch.float16
    x, x_shape_excl_last = flatten_last_dim_and_return_shape(x)
    return quarot._HIP.sym_quant(x, scale.view(-1)).view(*x_shape_excl_last, -1)

def sym_dequant(q, scale_row, scale_col, bits=32):
    assert q.dtype == torch.int32
    assert scale_row.dtype == scale_col.dtype == torch.float16
    q, q_shape_excl_last = flatten_last_dim_and_return_shape(q)
    return quarot._HIP.sym_dequant(q, scale_row.view(-1), scale_col, bits).view(*q_shape_excl_last, -1)


class PackedQuantizedTensor:
    def __init__(self,
                 quantized_x: torch.Tensor,
                 scales_x: torch.Tensor,
                 logical_shape=None):
        self.quantized_x = quantized_x
        self.scales_x = scales_x
        self.logical_shape = (None if logical_shape is None else
                              tuple(int(dim) for dim in logical_shape))

    def size(self):
        # Keep the historical storage-shape contract.
        return self.quantized_x.size()

    @property
    def device(self):
        return self.quantized_x.device

    @property
    def dtype(self):
        return self.quantized_x.dtype
