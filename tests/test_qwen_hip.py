"""ROCm kernel coverage for dense Qwen head dimensions."""
import pytest
import torch

_HIP = pytest.importorskip("quarot._HIP")
if torch.version.hip is None or not torch.cuda.is_available():
    pytest.skip("requires a ROCm/HIP GPU", allow_module_level=True)

@pytest.mark.parametrize("disable_quant", [False, True])
def test_qwen_head_dim_64_cache_prefill_and_decode(disable_quant):
    from quarot.transformers.kv_cache import MultiLayerPagedKVCache4Bit
    cache = MultiLayerPagedKVCache4Bit(
        batch_size=1, page_size=8, max_seq_len=8, device="cuda",
        n_layers=1, num_heads=2, num_kv_heads=1, head_dim=64,
        disable_quant=disable_quant,
        hadamard_dtype=None if disable_quant else torch.float16)
    key = torch.randn(1, 4, 1, 64, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    cache.update(key, value, 0, {})
    next_key = torch.randn(1, 1, 1, 64, device="cuda", dtype=torch.float16)
    next_value = torch.randn_like(next_key)
    attention = cache.update(next_key, next_value, 0, {})
    query = torch.randn(1, 1, 2, 64, device="cuda", dtype=torch.float16)
    output = attention(query)
    torch.cuda.synchronize()
    assert output.shape == query.shape
    assert torch.isfinite(output).all()

@pytest.mark.parametrize("heads", [12, 14, 28, 40])
def test_general_fused_attention_heads(heads):
    import quarot
    matrix, order = quarot.functional.hadamard.get_hadK(heads)
    assert matrix is not None
    matrix = matrix.to(device="cuda", dtype=torch.float16)
    value = torch.randn(1, 2, heads, 64, device="cuda", dtype=torch.float16)
    packed, scale = _HIP.fused_attention_hadamard_quant_general(
        value, heads, matrix)
    inner = heads // order
    staged = value.transpose(-1, -2).contiguous().float().view(
        1, 2, 64, order, inner)
    stride = 1
    while stride < inner:
        shaped = staged.view(1, 2, 64, order, -1, stride * 2)
        left = shaped[..., :stride].clone()
        right = shaped[..., stride:].clone()
        shaped[..., :stride] = left + right
        shaped[..., stride:] = left - right
        stride *= 2
    transformed = torch.einsum(
        "ok,bsdkp->bsdop", matrix.float(), staged
    ).reshape(1, 2, 64, heads).transpose(-1, -2).reshape(1, 2, -1)
    transformed /= heads ** 0.5
    expected_scale = (transformed.abs().amax(-1, keepdim=True) / 7).half()
    expected_scale.clamp_min_(torch.finfo(torch.float16).tiny)
    quantized = torch.round(transformed / expected_scale).clamp(-8, 7).to(torch.int8)
    expected = ((quantized[..., 0::2].to(torch.uint8) & 15) |
                ((quantized[..., 1::2].to(torch.uint8) & 15) << 4))
    torch.cuda.synchronize()
    assert torch.equal(scale, expected_scale)
    assert torch.equal(packed, expected)
