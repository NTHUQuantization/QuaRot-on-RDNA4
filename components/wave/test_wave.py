"""Check Wave against the mathematical Sylvester Hadamard transform."""
import math

import pytest
import torch

wave = pytest.importorskip("quarot_wave_fht")
if torch.version.hip is None or not torch.cuda.is_available():
    pytest.skip("requires a ROCm/HIP GPU", allow_module_level=True)


def reference(x, scale):
    matrix = torch.ones((1, 1), device=x.device, dtype=torch.float32)
    while matrix.shape[0] < x.shape[-1]:
        matrix = torch.cat(
            (torch.cat((matrix, matrix), dim=1),
             torch.cat((matrix, -matrix), dim=1)), dim=0)
    return (x.float() @ matrix * scale).to(x.dtype)


@pytest.mark.parametrize("width", [32, 64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 7, 8, 9, 17])
def test_normalized_transform(width, dtype, rows):
    torch.manual_seed(width + rows)
    x = torch.randn(rows, width, device="cuda", dtype=dtype)
    result = wave.hadamard_transform(x, scale=1 / math.sqrt(width))
    tolerance = 2e-3 if dtype == torch.float16 else 2e-2
    assert result.dtype == x.dtype
    assert result.shape == x.shape
    torch.testing.assert_close(result, reference(x, 1 / math.sqrt(width)),
                               atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("width", [32, 64, 128, 256])
def test_noncontiguous_leading_dimensions(width):
    x = torch.randn(2, 3, width * 2, device="cuda", dtype=torch.float16)[..., ::2]
    result = wave.hadamard_transform(x, scale=1 / math.sqrt(width))
    torch.testing.assert_close(result, reference(x, 1 / math.sqrt(width)),
                               atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("scale", [1.0, 0.125, 0.0, -0.25])
def test_scale(scale):
    x = torch.randn(9, 64, device="cuda", dtype=torch.float16)
    result = wave.hadamard_transform(x, scale=scale)
    # Scaling follows the original half-precision normalized intermediate.
    torch.testing.assert_close(result, reference(x, scale), atol=2e-2, rtol=2e-3)


def test_empty_batch():
    x = torch.empty(0, 128, device="cuda", dtype=torch.bfloat16)
    result = wave.hadamard_transform(x)
    assert result.shape == x.shape
    assert result.dtype == x.dtype


def test_unsupported_width():
    with pytest.raises(RuntimeError, match="widths"):
        wave.hadamard_transform(torch.ones(1, 512, device="cuda", dtype=torch.float16))


def test_unsupported_dtype():
    with pytest.raises(RuntimeError, match="float16 and bfloat16"):
        wave.hadamard_transform(torch.ones(1, 128, device="cuda", dtype=torch.float32))


def test_cpu_rejected():
    with pytest.raises(RuntimeError, match="CUDA/HIP"):
        wave.hadamard_transform(torch.ones(1, 128, dtype=torch.float16))
