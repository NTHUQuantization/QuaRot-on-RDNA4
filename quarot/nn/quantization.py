import quarot
import torch


class Quantizer(torch.nn.Module):
    def __init__(self, input_clip_ratio=1.0):
        super().__init__()
        self.input_clip_ratio = float(input_clip_ratio)
        if not 0.0 < self.input_clip_ratio <= 1.0:
            raise ValueError("input_clip_ratio must be in (0, 1]")

    def forward(self, x):
        if isinstance(x, quarot.PackedQuantizedTensor):
            return x
        scales_x = (
            x.abs().amax(dim=-1, keepdim=True).div(7)
            * self.input_clip_ratio
        ).to(torch.float16)
        scales_x.clamp_min_(torch.finfo(torch.float16).tiny)
        quantized_x = quarot.sym_quant(x, scales_x)
        return quarot.PackedQuantizedTensor(
            quantized_x, scales_x, logical_shape=x.shape)
