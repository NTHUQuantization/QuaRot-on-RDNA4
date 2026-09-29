"""Independent numerical and stream regressions for pard_fix's wave kernels."""

import math

import pytest
import torch

if torch.version.hip is None or not torch.cuda.is_available():
    pytest.skip("requires a ROCm/HIP GPU", allow_module_level=True)

from quarot import _HIP


def _hadamard(value):
    result = value.float().contiguous().clone()
    width = result.shape[-1]
    stride = 1
    while stride < width:
        blocks = result.reshape(*result.shape[:-1], -1, 2 * stride)
        low, high = blocks[..., :stride].clone(), blocks[..., stride:].clone()
        blocks[..., :stride], blocks[..., stride:] = low + high, low - high
        stride *= 2
    return result


def _signed_quant(value):
    scale = (value.abs().amax(-1, keepdim=True) / 7).clamp_min(
        torch.finfo(torch.float16).tiny).half()
    codes = (value / scale.float()).round().clamp(-8, 7).to(torch.uint8)
    return (codes[..., ::2] & 15) | ((codes[..., 1::2] & 15) << 4), scale


def _ffn_reference(gate, up, matrix=None, *, fp16_inner=False):
    values = torch.nn.functional.silu(gate.float()) * up.float()
    width = values.shape[-1]
    if fp16_inner:
        values = values.half().float()
    if matrix is None:
        values = _hadamard(values)
    else:
        values = _hadamard(values.reshape(*values.shape[:-1], len(matrix), -1))
        if fp16_inner:
            values = values.half().float()
        values = torch.matmul(matrix.float(), values).reshape_as(gate)
    values = values / math.sqrt(width)
    if fp16_inner:
        values = values.half().float()
    return _signed_quant(values)


@pytest.mark.parametrize("rows", [1, 15, 16])
@pytest.mark.parametrize("width", [32, 64, 128, 256])
def test_narrow_ffn_wave_matches_fp32_reference(rows, width):
    torch.manual_seed(23000 + width + rows)
    gate = torch.randn(rows, width, device="cuda", dtype=torch.float16)
    up = torch.randn_like(gate)
    actual = _HIP.fused_ffn_silu_hadamard_quant(gate, up)
    expected = _ffn_reference(gate, up)
    for got, want in zip(actual, expected):
        assert torch.equal(got, want)


@pytest.mark.parametrize("rows", [1, 15, 16])
@pytest.mark.parametrize("heads,head_dim", [(128, 32), (128, 64), (256, 32)])
def test_attention_h128_h256_preserves_head_layout(rows, heads, head_dim):
    torch.manual_seed(24000 + rows + heads + head_dim)
    value = torch.randn(1, rows, heads, head_dim, device="cuda", dtype=torch.float16)
    transformed = _hadamard(value.transpose(-1, -2)) / math.sqrt(heads)
    expected = _signed_quant(transformed.transpose(-1, -2).reshape(1, rows, -1))
    actual = _HIP.fused_attention_hadamard_quant(value, heads)
    for got, want in zip(actual, expected):
        assert torch.equal(got, want)


@pytest.mark.parametrize("rows,inner,remainder", [
    (1, 128, 43), (15, 128, 43), (16, 256, 43),
    (1, 128, 129), (16, 256, 65),
])
def test_general_ffn_wave_inner_and_large_fallback(rows, inner, remainder):
    torch.manual_seed(25000 + rows + inner + remainder)
    width = inner * remainder
    gate = torch.randn(rows, width, device="cuda", dtype=torch.float16)
    up = torch.randn_like(gate)
    # A permuted, scaled identity isolates inner-transform accuracy and
    # remainder addressing without making the oracle depend on GEMM's
    # floating-point accumulation order. The last two shapes exceed 14336.
    matrix = (torch.eye(remainder, device="cuda").roll(1, 0)
              * math.sqrt(remainder)).half()
    expected = _ffn_reference(gate, up, matrix)
    actual = _HIP.fused_ffn_silu_hadamard_quant_general(gate, up, matrix)
    for got, want in zip(actual, expected):
        assert torch.equal(got, want)


@pytest.mark.parametrize("inner", [128, 256])
def test_single_wmma_ffn_wave_inner_respects_fp16_lds(inner):
    from quarot.functional.hadamard import get_hadK

    torch.manual_seed(26000 + inner)
    width = 43 * inner
    # Explicit H43 exercises both legal inner widths. get_hadK(5504)
    # selects a different decomposition (H172 x H32) for the model wrapper.
    matrix, order = get_hadK(11008)
    assert order == 43
    matrix = matrix.cuda().half()
    gate = torch.randn(1, width, device="cuda", dtype=torch.float16)
    up = torch.randn_like(gate)
    expected = _ffn_reference(gate, up, matrix, fp16_inner=True)
    actual = _HIP.fused_ffn_silu_hadamard_quant_single_fp16lds(gate, up, matrix)
    for got, want in zip(actual, expected):
        assert torch.equal(got, want)


def _rope(value, cos, sin, *, round_products=True, round_sum=True):
    half = value.shape[-1] // 2
    rotated = torch.cat((-value[..., half:], value[..., :half]), -1)
    first = value.float() * cos[:, None, None, :].float()
    second = rotated.float() * sin[:, None, None, :].float()
    if round_products:
        first, second = first.half().float(), second.half().float()
    result = first + second
    return result.half() if round_sum else result


def _unsigned_quant(value):
    low, high = value.amin(-1, keepdim=True), value.amax(-1, keepdim=True)
    scale = ((high - low) / 15).clamp_min(1e-5).half()
    zero = (-low).half()
    codes = ((value + zero.float()) / scale.float()).round().clamp(0, 15).byte()
    packed = codes[..., ::2] | (codes[..., 1::2] << 4)
    return packed, torch.cat((scale, zero), -1)


def _kv_inputs(head_dim, tokens):
    # CPU generation and reference arithmetic are independent of HIP's
    # compiler and RNG. Several GQA heads expose both product and sum rounding.
    rng = torch.Generator().manual_seed(27000 + head_dim + tokens)
    query = torch.randn(2, 1, 16, head_dim, generator=rng).half()
    key = torch.randn(2, tokens, 8, head_dim, generator=rng).half()
    value = torch.randn(2, tokens, 8, head_dim, generator=rng).half()
    angles = torch.randn(2, head_dim, generator=rng)
    return query, key, value, angles.cos().half(), angles.sin().half()


def _cache_reference(key, value, final_length):
    batch, tokens, heads, width = key.shape
    page_size, layers, layer = 8, 2, 1
    pages = math.ceil(final_length / page_size)
    shape = (batch * pages, layers, 2, heads, page_size, width // 2)
    data = torch.full(shape, 0xA5, dtype=torch.uint8)
    params = torch.full((*shape[:-1], 2), 3.0, dtype=torch.float16)
    indices = torch.arange(batch * pages - 1, -1, -1, dtype=torch.int32)
    indptr = torch.arange(batch + 1, dtype=torch.int32) * pages
    last = torch.full((batch,), (final_length - 1) % page_size + 1, dtype=torch.int32)
    for plane, source in enumerate((_hadamard(key) / math.sqrt(width), value.float())):
        packed, quant_params = _unsigned_quant(source)
        for b in range(batch):
            for token in range(tokens):
                position = final_length - tokens + token
                page = indices[b * pages + position // page_size]
                slot = position % page_size
                data[page, layer, plane, :, slot] = packed[b, token]
                params[page, layer, plane, :, slot] = quant_params[b, token]
    return data, params, indptr, indices, last


@pytest.mark.parametrize("head_dim", [64, 128])
def test_rope_append_preserves_each_fp16_materialization(head_dim):
    query, key, value, cos, sin = _kv_inputs(head_dim, 1)
    expected_query = _rope(query, cos, sin)
    expected_key = _rope(key, cos, sin)
    expected = _cache_reference(expected_key, value, 9)
    # Ensure these fixtures actually catch either missing rounding boundary;
    # comparing fused RoPE with another fused kernel could hide the defect.
    assert not torch.equal(expected_query, _rope(query, cos, sin, round_products=False))
    for kwargs in ({"round_products": False}, {"round_sum": False}):
        wrong = _cache_reference(_rope(key, cos, sin, **kwargs), value, 9)
        assert not all(torch.equal(got, want) for got, want in zip(wrong[:2], expected[:2]))
    data, params = torch.full_like(expected[0], 0xA5).cuda(), torch.full_like(expected[1], 3).cuda()
    indptr, indices, last = (item.cuda() for item in expected[2:])
    actual_query = _HIP.fused_rope_append_kv_i4(
        *(item.cuda() for item in (query, key, value, cos, sin)),
        data, params, indptr, indices, last, 2, 1, 8)
    assert torch.equal(actual_query.cpu(), expected_query)
    assert torch.equal(data.cpu(), expected[0])
    assert torch.equal(params.cpu(), expected[1])


@pytest.mark.parametrize("operation", ["append", "rope"])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_kv_wave_current_stream_and_graph_replay(operation, head_dim):
    tokens = 15 if operation == "append" else 1
    cpu_inputs = _kv_inputs(head_dim, tokens)
    query, key, value, cos, sin = (item.cuda() for item in cpu_inputs)
    final_length = 7 + tokens + 1
    template = _cache_reference(cpu_inputs[1], cpu_inputs[2], final_length)
    data, params = template[0].cuda(), template[1].cuda()
    indptr, indices, last = (item.cuda() for item in template[2:])

    def launch():
        if operation == "rope":
            return _HIP.fused_rope_append_kv_i4(
                query, key, value, cos, sin, data, params,
                indptr, indices, last, 2, 1, 8)
        _HIP.fused_append_kv_i4(
            data, params, indptr, indices, last, key, value, 2, 1, 8, 8, 2)
        return query

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            launch()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual_query = launch()
        # Consumers on the capture stream also expose accidental launches on
        # the default stream, even if that stream finishes before assertions.
        observed_data, observed_params = data.clone(), params.clone()
        observed_query = actual_query.clone()
    for step, factor in enumerate((1.0, -0.5)):
        updated = tuple((item * factor).half() for item in cpu_inputs[:3])
        expected_key = (_rope(updated[1], cpu_inputs[3], cpu_inputs[4])
                        if operation == "rope" else updated[1])
        expected = _cache_reference(expected_key, updated[2], final_length + step)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for target, source in zip((query, key, value), updated):
                target.copy_(source)
            data.fill_(0xA5)
            params.fill_(3)
            last.copy_(expected[4])
            graph.replay()
        stream.synchronize()
        assert torch.equal(observed_data.cpu(), expected[0])
        assert torch.equal(observed_params.cpu(), expected[1])
        if operation == "rope":
            expected_query = _rope(updated[0], cpu_inputs[3], cpu_inputs[4])
            assert torch.equal(observed_query.cpu(), expected_query)


@pytest.mark.parametrize("head_dim", [64, 128])
def test_rope_half_products_cancel_without_fma_residue(head_dim):
    # Rounded products cancel exactly. A half FMA instead leaves a nonzero
    # 0.000202178955078125 residue, depending on which product contracts.
    query = torch.full((1, 1, 2, head_dim), 1.0009765625, dtype=torch.float16)
    key, value = query[:, :, :1].clone(), torch.zeros(1, 1, 1, head_dim).half()
    cos = torch.full((1, head_dim), 0.70703125, dtype=torch.float16)
    sin = cos.clone()
    expected_query = _rope(query, cos, sin)
    expected = _cache_reference(_rope(key, cos, sin), value, 1)
    data = torch.full_like(expected[0], 0xA5).cuda()
    params = torch.full_like(expected[1], 3.0).cuda()
    actual = _HIP.fused_rope_append_kv_i4(
        *(item.cuda() for item in (query, key, value, cos, sin)),
        data, params, *(item.cuda() for item in expected[2:]), 2, 1, 8)
    assert torch.count_nonzero(actual[..., :head_dim // 2]).item() == 0
    assert torch.equal(actual.cpu(), expected_query)
    assert torch.equal(data.cpu(), expected[0])
    assert torch.equal(params.cpu(), expected[1])
