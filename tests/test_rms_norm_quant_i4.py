"""Isolated parity tests for fused row RMSNorm plus signed INT4."""

import pytest
import torch

_HIP = pytest.importorskip("quarot._HIP")
if torch.version.hip is None or not torch.cuda.is_available():
    pytest.skip("requires a ROCm/HIP GPU", allow_module_level=True)


def _oracle(value, width, eps):
    normalized = _HIP.rms_norm_rows(value, width, eps)
    scales = (normalized.abs().amax(dim=-1, keepdim=True) / 7).to(torch.float16)
    packed = _HIP.sym_quant(normalized.reshape(-1, width), scales.reshape(-1))
    return packed.view(*value.shape[:-1], width // 2), scales


@pytest.mark.parametrize("shape", [(1, 2), (3, 128), (1, 1, 5120), (2, 15, 4096), (1, 16, 8192)])
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_bit_exact(shape, eps):
    torch.manual_seed(sum(shape))
    value = torch.randn(*shape, device="cuda", dtype=torch.float16).contiguous()
    packed, scales = _HIP.rms_norm_quant_i4_rows(value, shape[-1], eps)
    expected_packed, expected_scales = _oracle(value, shape[-1], eps)
    torch.cuda.synchronize()
    assert torch.equal(scales, expected_scales)
    assert torch.equal(packed, expected_packed)


@pytest.mark.parametrize("fill", [0.0, 1.0, -1.0, 65504.0])
def test_special_values(fill):
    value = torch.full((2, 4096), fill, device="cuda", dtype=torch.float16)
    packed, scales = _HIP.rms_norm_quant_i4_rows(value, 4096, 1e-6)
    expected_packed, expected_scales = _oracle(value, 4096, 1e-6)
    torch.cuda.synchronize()
    assert torch.equal(scales, expected_scales)
    assert torch.equal(packed, expected_packed)


def test_contract_rejections():
    with pytest.raises(RuntimeError, match="float16"):
        _HIP.rms_norm_quant_i4_rows(
            torch.ones(1, 128, device="cuda", dtype=torch.bfloat16), 128, 1e-6)
    base = torch.ones(1, 256, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="contiguous"):
        _HIP.rms_norm_quant_i4_rows(base[:, ::2], 128, 1e-6)
    with pytest.raises(RuntimeError, match="mean_dim"):
        _HIP.rms_norm_quant_i4_rows(base, 128, 1e-6)
