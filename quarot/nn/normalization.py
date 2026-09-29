import torch
import quarot
import os

class RMSNorm(torch.nn.Module):
    """
    This class implements the Root Mean Square Normalization (RMSN) layer.
    We use the implementation from LLAMARMSNorm here:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
    """

    def __init__(self, mean_dim: int, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)


class FusedRMSNormQuant(torch.nn.Module):
    def __init__(self, mean_dim: int, eps=1e-5, input_clip_ratio=0.9):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim
        self.input_clip_ratio = input_clip_ratio

    def forward(self, x: torch.Tensor):
        if os.getenv("QUAROT_FUSED_NORM_QUANT", "1") == "0":
            normalized = RMSNorm(self.mean_dim, self.eps)(x)
            return quarot.nn.Quantizer(self.input_clip_ratio)(normalized)
        packed, scales = quarot._HIP.fused_rmsnorm_quant_i4(
            x.contiguous(), self.eps, self.input_clip_ratio)
        return quarot.PackedQuantizedTensor(
            packed, scales, logical_shape=x.shape)
