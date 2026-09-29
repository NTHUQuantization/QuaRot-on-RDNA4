"""Accuracy coverage for Hadacore shapes used by benchmark.py models."""
import math

import pytest
import torch

fht = pytest.importorskip("fast_hadamard_transform")
if torch.version.hip is None or not torch.cuda.is_available():
    pytest.skip("requires a ROCm/HIP GPU", allow_module_level=True)


def _reference(x):
    rows, width = x.shape
    result = x.float()
    stride = 1
    while stride < width:
        view = result.view(rows, -1, stride * 2)
        low = view[..., :stride].clone()
        high = view[..., stride:].clone()
        view[..., :stride] = low + high
        view[..., stride:] = low - high
        stride <<= 1
    return (result / math.sqrt(width)).half()


@pytest.mark.parametrize("rows,width", [
    (43, 256),   # Llama-2 7B active FFN
    (27, 512),   # Llama-2 13B active FFN
    (172, 128),  # CodeLlama 34B active FFN
    (43, 512),   # optimized H512 candidate coverage
    (27, 1024),  # Qwen2.5-32B FFN
    (25, 1024),  # Qwen3-32B FFN
    (28, 1024),  # Llama-2 70B FFN
])
def test_hadacore_model_inner_shapes(rows, width):
    torch.manual_seed(rows * width)
    x = torch.randn(rows, width, device="cuda", dtype=torch.float16)
