"""HIP-only correctness checks for the unified QuaRot fused extension."""

import math

import pytest
import torch


_HIP = pytest.importorskip("quarot._HIP")
if torch.version.hip is None or not torch.cuda.is_available():
    pytest.skip("requires a ROCm/HIP GPU", allow_module_level=True)


def _hadamard(x, *, output_dtype=torch.float32):
    width = x.size(-1)
    y = x.float().clone()
    stride = 1
    while stride < width:
        y = y.reshape(*y.shape[:-1], -1, stride * 2)
        left, right = y[..., :stride].clone(), y[..., stride:].clone()
        y[..., :stride], y[..., stride:] = left + right, left - right
        y = y.reshape_as(x)
        stride <<= 1
    return (y / math.sqrt(width)).to(output_dtype)


def _pack_s4(x, scale):
    q = torch.round(x / scale).clamp(-8, 7).to(torch.int8)
    return (q[..., 0::2].to(torch.uint8) & 0x0F) | ((q[..., 1::2].to(torch.uint8) & 0x0F) << 4)


@pytest.mark.parametrize("rows", [1, 2, 15, 16])
def test_rms_norm_rows_matches_independent_m1_launches(rows):
    torch.manual_seed(9100 + rows)
    value = torch.randn(
        1, rows, 4096, device="cuda", dtype=torch.float16).contiguous()
    actual = _HIP.rms_norm_rows(value, 4096, 1e-6)
    expected = torch.cat([
        _HIP.rms_norm_rows(value[:, index:index + 1], 4096, 1e-6)
        for index in range(rows)
    ], dim=1)
    torch.cuda.synchronize()
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("rows", [1, 15, 16])
@pytest.mark.parametrize("out_features", [1024, 4096, 12288])
def test_m16_bpre_gemm_matches_row_packed_oracle(rows, out_features, monkeypatch):
    monkeypatch.setenv("QUAROT_ENABLE_M16_BPRE", "1")
    torch.manual_seed(rows + out_features)
    in_features = 4096
    activation = torch.randint(
        -8, 8, (rows, in_features), device="cuda", dtype=torch.int8)
    weight = torch.randint(
        -8, 8, (out_features, in_features), device="cuda", dtype=torch.int8)
    packed_activation = _pack_s4(activation, 1).contiguous()
    packed_weight = _pack_s4(weight, 1).contiguous()
    prepacked_weight = _HIP.prepack_b(packed_weight)
    actual = _HIP.matmul_bpre(
        packed_activation, prepacked_weight, out_features, in_features)
    expected = _HIP.matmul(packed_activation, packed_weight)
    torch.cuda.synchronize()
    assert torch.equal(actual, expected)


def _reference_attention(x):
    # Same layout as OnlineHadamard(num_heads): rotate across heads for every
    # head-dimension coordinate, then quantize the flattened projection input.
    b, s, h, d = x.shape
    y = _hadamard(x.transpose(-1, -2).contiguous()).transpose(-1, -2).reshape(b, s, h * d)
    scale = (y.abs().amax(dim=-1, keepdim=True) / 7).half().clamp_min(torch.finfo(torch.float16).tiny)
    return _pack_s4(y, scale), scale


@pytest.mark.parametrize("batch,seq,heads,head_dim", [(1, 1, 2, 128), (2, 16, 32, 128), (1, 128, 64, 128)])
def test_fused_attention_matches_quarot_layout(batch, seq, heads, head_dim):
    torch.manual_seed(heads)
    x = torch.randn(batch, seq, heads, head_dim, device="cuda", dtype=torch.float16)
    packed, scale = _HIP.fused_attention_hadamard_quant(x.contiguous(), heads)
    expected_packed, expected_scale = _reference_attention(x)
    torch.cuda.synchronize()
    assert torch.equal(packed, expected_packed)
    assert torch.equal(scale, expected_scale)


@pytest.mark.parametrize("rows,width", [
    (1, 11008), (4, 14336), (1, 25600), (2, 28672), (1, 29696),
])
def test_grouped256_ffn_matches_per_group_scale_contract(rows, width):
    torch.manual_seed(width - rows)
    gate = torch.randn(rows, width, device="cuda", dtype=torch.float16)
    up = torch.randn_like(gate)
    packed, scale = _HIP.fused_ffn_silu_hadamard_quant_grouped256(gate, up)
    values = torch.nn.functional.silu(gate.float()) * up.float()
    rotated = _hadamard(values.view(rows, -1, 256)).view_as(values)
    scale_float = (rotated.view(rows, -1, 256).abs().amax(-1) / 7)
    scale_float.clamp_min_(torch.finfo(torch.float16).tiny)
    expected_scale = scale_float.half()
    torch.cuda.synchronize()
    expected_packed = _pack_s4(
        rotated, scale_float.repeat_interleave(256, dim=-1))
    # Every shape now uses the FP32 warp butterfly. Quantization divides by
    # the FP32 group scale; only the returned scale is rounded to FP16.
    assert torch.equal(packed, expected_packed)
    assert torch.equal(scale, expected_scale)


@pytest.mark.parametrize("rows", [15, 16])
@pytest.mark.parametrize("width", [11008, 14336, 25600])
def test_grouped256_verification_matches_independent_decode_rows(rows, width):
    torch.manual_seed(rows + width)
    gate = torch.randn(rows, width, device="cuda", dtype=torch.float16)
    up = torch.randn_like(gate)
    actual = _HIP.fused_ffn_silu_hadamard_quant_grouped256(gate, up)
    independent = [_HIP.fused_ffn_silu_hadamard_quant_grouped256(
        gate[index:index + 1], up[index:index + 1]) for index in range(rows)]
    for part in (0, 1):
        assert torch.equal(actual[part], torch.cat(
            [row[part] for row in independent], dim=0))


def test_grouped_scale_bpre_gemm_matches_partial_sum_reference():
    from quarot.functional import pack_i4

    torch.manual_seed(25616)
    rows, outputs, width = 16, 32, 512
    a_i4 = torch.randint(-8, 8, (rows, width), device="cuda", dtype=torch.int8)
    b_i4 = torch.randint(-8, 8, (outputs, width), device="cuda", dtype=torch.int8)
    a = pack_i4(a_i4).contiguous()
    b = _HIP.prepack_b(pack_i4(b_i4).contiguous())
    group_scale = (torch.rand(rows, width // 256, device="cuda") + 0.1).half()
    weight_scale = (torch.rand(outputs, device="cuda") + 0.1).half()
    actual = _HIP.matmul_bpre_grouped_scale(
        a, b, group_scale, weight_scale, outputs, width)
    partial = torch.einsum(
        "mgk,ngk->mng", a_i4.view(rows, -1, 256).float(),
        b_i4.view(outputs, -1, 256).float())
    expected = (partial * group_scale.float()[:, None, :]).sum(-1)
    expected = (expected * weight_scale.float()[None, :]).half()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=0.5)


@pytest.mark.parametrize("rows", [1, 15, 16])
def test_grouped_scale_bpre_matches_bpre_dequant_oracle(rows):
    """Gate the fused FP16 epilogue against the legacy INT32 path."""
    from quarot.functional import pack_i4

    torch.manual_seed(3000 + rows)
    outputs, width = 128, 4096
    a_i4 = torch.randint(
        -8, 8, (rows, width), device="cuda", dtype=torch.int8)
    b_i4 = torch.randint(
        -8, 8, (outputs, width), device="cuda", dtype=torch.int8)
    packed_a = pack_i4(a_i4).contiguous()
    packed_b = _HIP.prepack_b(pack_i4(b_i4).contiguous())
    activation_scale = (
        torch.rand(rows, 1, device="cuda") * 0.05 + 0.001).half()
    weight_scale = (
        torch.rand(outputs, device="cuda") * 0.05 + 0.001).half()

    actual = _HIP.matmul_bpre_grouped_scale(
        packed_a, packed_b, activation_scale, weight_scale, outputs, width)
    accum_i32 = _HIP.matmul_bpre(packed_a, packed_b, outputs, width)
    expected = _HIP.sym_dequant(
        accum_i32, activation_scale.view(-1), weight_scale, 32)
    torch.cuda.synchronize()

    # At realistic K=4096 this is intentionally not bit-exact: sym_dequant
    # first rounds the INT32 accumulator through FP16, whereas the fused
    # epilogue applies both scales in FP32 and casts only the final value.
    bit_exact = torch.equal(actual, expected)
    assert not bit_exact
    assert torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=1e-2)


@pytest.mark.parametrize("outputs,k,m", [
    ((64, 32), 256, 3),
    ((64, 32, 32), 256, 3),
    ((8192, 1024, 1024), 5120, 1),
])
def test_multi_scale_bpre_matches_independent_projections(outputs, k, m):
    from quarot.functional import pack_i4

    torch.manual_seed(19)
    a_i4 = torch.randint(-8, 8, (m, k), device="cuda", dtype=torch.int8)
    packed_a = pack_i4(a_i4).contiguous()
    activation_scale = torch.rand(m, 1, device="cuda", dtype=torch.float16)
    packed_weights, weight_scales, expected = [], [], []
    for n in outputs:
        weight = torch.randint(-8, 8, (n, k), device="cuda", dtype=torch.int8)
        packed = _HIP.prepack_b(pack_i4(weight).contiguous())
        scale = torch.rand(n, device="cuda", dtype=torch.float16)
        packed_weights.append(packed)
        weight_scales.append(scale)
        expected.append(_HIP.matmul_bpre_grouped_scale(
            packed_a, packed, activation_scale, scale, n, k))
    actual = _HIP.matmul_bpre_multi_scale(
        packed_a, activation_scale, packed_weights[0], weight_scales[0],
        packed_weights[1], weight_scales[1],
        packed_weights[2] if len(outputs) == 3 else None,
        weight_scales[2] if len(outputs) == 3 else None,
        outputs[0], outputs[1], outputs[2] if len(outputs) == 3 else 0, k)
    for got, want in zip(actual.split(outputs, dim=-1), expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)



@pytest.mark.parametrize("rows", [1, 15, 16])
@pytest.mark.parametrize("outputs", [(64, 32), (64, 32, 48)])
def test_multi_scale_bpre_interleaved_views_match_separate_projections(
        rows, outputs, monkeypatch):
    """Exercise the pointer-adjacent two/three-projection fast path."""
    from quarot.functional import pack_i4

    monkeypatch.setenv("QUAROT_INTERLEAVED_PROJECTIONS", "1")
    torch.manual_seed(4000 + rows + sum(outputs))
    width = 256
    a_i4 = torch.randint(
        -8, 8, (rows, width), device="cuda", dtype=torch.int8)
    packed_a = pack_i4(a_i4).contiguous()
    activation_scale = (
        torch.rand(rows, 1, device="cuda") + 0.1).half()

    packed_weights, weight_scales, expected = [], [], []
    for output_width in outputs:
        weight = torch.randint(
            -8, 8, (output_width, width), device="cuda", dtype=torch.int8)
        packed = _HIP.prepack_b(pack_i4(weight).contiguous())
        scale = (
            torch.rand(output_width, device="cuda") + 0.1).half()
        assert packed.numel() == output_width * width // 2
        packed_weights.append(packed)
        weight_scales.append(scale)
        expected.append(_HIP.matmul_bpre_grouped_scale(
            packed_a, packed, activation_scale, scale,
            output_width, width))

    packed_backing = torch.cat(packed_weights).contiguous()
    scale_backing = torch.cat(weight_scales).contiguous()
    packed_views, scale_views = [], []
    packed_offset = scale_offset = 0
    for packed, scale in zip(packed_weights, weight_scales):
        packed_view = packed_backing.narrow(
            0, packed_offset, packed.numel())
        scale_view = scale_backing.narrow(0, scale_offset, scale.numel())
        assert packed_view.is_contiguous() and scale_view.is_contiguous()
        assert packed_view.data_ptr() == packed_backing.data_ptr() + packed_offset
        assert scale_view.data_ptr() == (
            scale_backing.data_ptr() +
            scale_offset * scale_backing.element_size())
        packed_views.append(packed_view)
        scale_views.append(scale_view)
        packed_offset += packed.numel()
        scale_offset += scale.numel()

    has_third = len(outputs) == 3
    actual = _HIP.matmul_bpre_multi_scale(
        packed_a, activation_scale,
        packed_views[0], scale_views[0], packed_views[1], scale_views[1],
        packed_views[2] if has_third else None,
        scale_views[2] if has_third else None,
        outputs[0], outputs[1], outputs[2] if has_third else 0, width)
    torch.cuda.synchronize()
    for got, want in zip(actual.split(outputs, dim=-1), expected):
        assert torch.equal(got, want)


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("page_size,positions", [(4, [1, 2, 4, 5]), (8, [1, 8, 9])])
def test_fused_k1_writes_target_asymmetric_cache(head_dim, page_size, positions):
    batch, heads, pages, layers = 2, 4, 2, 1
    data = torch.empty((batch * pages, layers, 2, heads, page_size, head_dim // 2), device="cuda", dtype=torch.uint8)
    params = torch.empty((batch * pages, layers, 2, heads, page_size, 2), device="cuda", dtype=torch.float16)
    for position in positions:
        key = torch.randn(batch, heads, head_dim, device="cuda", dtype=torch.float16)
        value = torch.randn_like(key)
        current_pages = (position + page_size - 1) // page_size
        indptr = torch.arange(0, batch + 1, device="cuda", dtype=torch.int32) * current_pages
        indices = ((torch.arange(current_pages, device="cuda", dtype=torch.int32) * batch).unsqueeze(0) + torch.arange(batch, device="cuda", dtype=torch.int32).unsqueeze(1)).flatten()
        offset = (position - 1) % page_size + 1
        last = torch.full((batch,), offset, device="cuda", dtype=torch.int32)
        query = torch.zeros(batch, 1, heads, head_dim, device="cuda", dtype=torch.float16)
        cos = torch.ones(batch, head_dim, device="cuda", dtype=torch.float16)
        sin = torch.zeros_like(cos)
        _HIP.fused_rope_append_kv_i4(
            query, key.unsqueeze(1), value.unsqueeze(1), cos, sin,
            data, params, indptr, indices, last, layers, 0, page_size)
        page = (position - 1) // page_size
        slot = (position - 1) % page_size
        page_ids = indices[indptr[:-1] + page]
        for source, plane in ((key, 0), (value, 1)):
            rotated = (_hadamard(source) if plane == 0
                       else source.float())
            xmin, xmax = rotated.amin(dim=-1, keepdim=True), rotated.amax(dim=-1, keepdim=True)
            scale = ((xmax - xmin).clamp_min(1e-5) / 15).half()
            zero = (-xmin).half()
            expected = torch.round((rotated + zero) / scale).clamp(0, 15).to(torch.uint8)
            expected = expected[..., 0::2] | (expected[..., 1::2] << 4)
            assert torch.equal(data[page_ids, 0, plane, :, slot], expected)
            assert torch.equal(params[page_ids, 0, plane, :, slot, 0], scale.squeeze(-1))
            assert torch.equal(params[page_ids, 0, plane, :, slot, 1], zero.squeeze(-1))


@pytest.mark.parametrize("position", [1, 31, 32, 33, 63, 64, 65, 1024])
def test_fused_k1_page_boundaries(position):
    page_size, batch, heads, layers = 32, 1, 2, 1
    page_count = (position + page_size - 1) // page_size
    data = torch.full((page_count, layers, 2, heads, page_size, 64), 0xA5, device="cuda", dtype=torch.uint8)
    params = torch.full((page_count, layers, 2, heads, page_size, 2), 3.0, device="cuda", dtype=torch.float16)
    key = torch.zeros(batch, heads, 128, device="cuda", dtype=torch.float16)
    value = torch.full_like(key, 2.0)
    indptr = torch.tensor([0, page_count], device="cuda", dtype=torch.int32)
    indices = torch.arange(page_count, device="cuda", dtype=torch.int32)
    last = torch.tensor([(position - 1) % page_size + 1], device="cuda", dtype=torch.int32)
    query = torch.zeros(batch, 1, heads, 128, device="cuda", dtype=torch.float16)
    cos = torch.ones(batch, 128, device="cuda", dtype=torch.float16)
    sin = torch.zeros_like(cos)
    _HIP.fused_rope_append_kv_i4(
        query, key.unsqueeze(1), value.unsqueeze(1), cos, sin,
        data, params, indptr, indices, last, layers, 0, page_size)
    torch.cuda.synchronize()
    page, slot = (position - 1) // page_size, (position - 1) % page_size
    assert torch.all(data[page, 0, :, :, slot] != 0xA5)
    mask = torch.ones_like(data, dtype=torch.bool)
    mask[page, 0, :, :, slot] = False
    assert torch.all(data[mask] == 0xA5), "fused append wrote outside the target cache slot"



@pytest.mark.parametrize("head_dim", [64, 128])
def test_fused_k1_matches_tensor_reference(head_dim):
    torch.manual_seed(9100 + head_dim)
    batch, query_heads, kv_heads = 2, 4, 2
    layers, page_size, position = 1, 8, 5
    query = torch.randn(
        batch, 1, query_heads, head_dim,
        device="cuda", dtype=torch.float16)
    key = torch.randn(
        batch, 1, kv_heads, head_dim,
        device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    angles = torch.randn(batch, head_dim, device="cuda").float()
    cos, sin = angles.cos().half(), angles.sin().half()
    shape = (
        batch, layers, 2, kv_heads, page_size, head_dim // 2)
    param_shape = (batch, layers, 2, kv_heads, page_size, 2)
    reference_data = torch.zeros(
        shape, device="cuda", dtype=torch.uint8)
    actual_data = torch.zeros_like(reference_data)
    reference_params = torch.zeros(
        param_shape, device="cuda", dtype=torch.float16)
    actual_params = torch.zeros_like(reference_params)
    indptr = torch.arange(
        batch + 1, device="cuda", dtype=torch.int32)
    indices = torch.arange(batch, device="cuda", dtype=torch.int32)
    last = torch.full(
        (batch,), position, device="cuda", dtype=torch.int32)

    half = head_dim // 2
    reference_query = (
        query * cos[:, None, None, :]
        + torch.cat((-query[..., half:], query[..., :half]), dim=-1)
        * sin[:, None, None, :]
    ).half()
    rotated_key = (
        key * cos[:, None, None, :]
        + torch.cat((-key[..., half:], key[..., :half]), dim=-1)
        * sin[:, None, None, :]
    ).half()
    page_ids = indices[indptr[:-1]]
    slot = position - 1
    for source, plane in ((_hadamard(rotated_key.squeeze(1)), 0),
                          (value.squeeze(1).float(), 1)):
        xmin = source.amin(dim=-1, keepdim=True)
        xmax = source.amax(dim=-1, keepdim=True)
        scale = ((xmax - xmin).clamp_min(1e-5) / 15).half()
        zero = (-xmin).half()
        packed = torch.round((source + zero) / scale).clamp(0, 15).to(torch.uint8)
        packed = packed[..., 0::2] | (packed[..., 1::2] << 4)
        reference_data[page_ids, 0, plane, :, slot] = packed
        reference_params[page_ids, 0, plane, :, slot, 0] = scale.squeeze(-1)
        reference_params[page_ids, 0, plane, :, slot, 1] = zero.squeeze(-1)
    actual_query = _HIP.fused_rope_append_kv_i4(
        query.contiguous(), key.contiguous(), value.contiguous(),
        cos.contiguous(), sin.contiguous(), actual_data, actual_params,
        indptr, indices, last, layers, 0, page_size)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual_query, reference_query, rtol=0, atol=2 ** -8)
    # PyTorch decomposes RoPE/reduction differently from the HIP kernel, so
    # values on an INT4 rounding boundary may differ by one code. Identity
    # RoPE packing is checked bit-exactly in the tests above.
    actual_written = actual_data[page_ids, 0, :, :, slot]
    reference_written = reference_data[page_ids, 0, :, :, slot]
    assert (actual_written == reference_written).float().mean() > 0.99
    torch.testing.assert_close(
        actual_params[page_ids, 0, :, :, slot],
        reference_params[page_ids, 0, :, :, slot],
        rtol=0, atol=2 ** -8)

def test_special_values_and_int4_endpoints():
    for fill in (0.0, 1.0, -1.0, 100.0):
        gate = torch.full((2, 256), fill, device="cuda", dtype=torch.float16)
        up = torch.ones_like(gate)
        packed, scale = _HIP.fused_ffn_silu_hadamard_quant_grouped256(gate, up)
        gate_f = gate.float()
        y = _hadamard(
            (torch.nn.functional.silu(gate_f) * up.float()).view(2, -1, 256),
            output_dtype=torch.float32).view_as(gate)
        scale_float = (y.view(2, -1, 256).abs().amax(-1) / 7).clamp_min(
            torch.finfo(torch.float16).tiny)
        expected_scale = scale_float.half()
        torch.cuda.synchronize()
        assert torch.equal(
            packed, _pack_s4(y, scale_float.repeat_interleave(256, -1)))
        assert torch.equal(scale, expected_scale)
    endpoints = torch.tensor([-8, -7, -1, 0, 1, 7], device="cuda", dtype=torch.int8)
    expected = torch.tensor([0x98, 0x0F, 0x71], device="cuda", dtype=torch.uint8)
    assert torch.equal(_pack_s4(endpoints, torch.tensor(1.0, device="cuda")), expected)


def test_unsupported_contracts_fail_before_launch():
    with pytest.raises(RuntimeError, match="float16"):
        _HIP.fused_ffn_silu_hadamard_quant_grouped256(
            torch.ones(1, 128, device="cuda", dtype=torch.bfloat16),
            torch.ones(1, 128, device="cuda", dtype=torch.bfloat16),
        )
    base = torch.ones(1, 256, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="contiguous"):
        _HIP.fused_ffn_silu_hadamard_quant_grouped256(base[:, ::2], base[:, ::2])
    data = torch.empty((1, 1, 2, 1, 1, 64), device="cuda", dtype=torch.uint8)
    params = torch.empty((1, 1, 2, 1, 1, 2), device="cuda", dtype=torch.float16)
    key = torch.zeros(1, 1, 128, device="cuda", dtype=torch.float16)
    query = key[:, None]
    cos = torch.ones(1, 128, device="cuda", dtype=torch.float16)
    sin = torch.zeros_like(cos)
    with pytest.raises(RuntimeError, match="metadata must be contiguous int32 CUDA/HIP"):
        _HIP.fused_rope_append_kv_i4(
            query, query, query, cos, sin, data, params,
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32), 1, 0, 1)


def test_cache_hadamard_preserves_qk_across_query_and_key_ranks():
    from quarot.transformers.kv_cache import matmul_had_HIP

    torch.manual_seed(12832)
    query = torch.randn(1, 32, 128, device="cuda", dtype=torch.float16)
    key = torch.randn(1, 6, 32, 128, device="cuda", dtype=torch.float16)
    query_before, key_before = query.clone(), key.clone()
    transformed_query = matmul_had_HIP(query, torch.float16)
    transformed_key = matmul_had_HIP(key, torch.float16)
    reference = torch.einsum("bhd,bshd->bhs", query.float(), key.float())
    transformed = torch.einsum(
        "bhd,bshd->bhs", transformed_query.float(), transformed_key.float())

    assert torch.equal(query, query_before)
    assert torch.equal(key, key_before)
    torch.testing.assert_close(transformed, reference, rtol=2e-3, atol=4e-2)


def test_gqa_cache_keeps_native_kv_heads(monkeypatch):
    import quarot.transformers.kv_cache as kv_cache

    observed_heads = []
    original_quantize = kv_cache.asym_quantize_and_pack_i4

    def recording_quantize(x):
        observed_heads.append(x.shape[2])
        return original_quantize(x)

    monkeypatch.setattr(kv_cache, "asym_quantize_and_pack_i4", recording_quantize)
    cache = kv_cache.MultiLayerPagedKVCache4Bit(
        batch_size=2, page_size=16, max_seq_len=16, device="cuda",
        n_layers=1, num_heads=4, num_kv_heads=2, head_dim=128,
        native_gqa=True, fused_decode_append=False)
    torch.manual_seed(3408)
    key = torch.randn(2, 16, 2, 128, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    returned_key, returned_value = cache.update(key, value, 0, {})
    torch.cuda.synchronize()

    assert observed_heads == [2, 2]
    assert returned_key is key and returned_value is value
    assert cache.pages.shape[3] == 2
    assert cache.scales.shape[3] == 2


@pytest.mark.parametrize("disable_quant", [False, True])
@pytest.mark.parametrize("num_q_heads,num_kv_heads,head_dim", [
    (4, 2, 64),
    (32, 8, 128),  # meta-llama/Llama-3.1-8B
    (64, 8, 128),  # meta-llama/CodeLlama-34b-hf
])
def test_native_gqa_fused_decode_matches_explicit_grouped_reference(
        disable_quant, num_q_heads, num_kv_heads, head_dim):
    from quarot.transformers.kv_cache import (
        MultiLayerPagedKVCache4Bit, matmul_had_HIP,
        unpack_i4_and_asym_dequantize)

    torch.manual_seed(num_q_heads * 1000 + head_dim + int(disable_quant))
    cache = MultiLayerPagedKVCache4Bit(
        batch_size=1, page_size=8, max_seq_len=8, device="cuda",
        n_layers=1, num_heads=num_q_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, disable_quant=disable_quant,
        hadamard_dtype=None if disable_quant else torch.float16)
    key = torch.randn(1, 4, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    cache.update(key, value, 0, {})
    next_key = torch.randn(1, 1, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    next_value = torch.randn_like(next_key)
    attention = cache.update(next_key, next_value, 0, {})
    query = torch.randn(1, 1, num_q_heads, head_dim, device="cuda", dtype=torch.float16)
    actual = attention(query).squeeze(1)

    assert cache.pages.shape[3] == num_kv_heads
    if disable_quant:
        cached_key = cache.pages[0, 0, 0, :, :5].permute(1, 0, 2)
        cached_value = cache.pages[0, 0, 1, :, :5].permute(1, 0, 2)
        reference_query = query.squeeze(1)
    else:
        cached = []
        for plane in (0, 1):
            packed = cache.pages[0, 0, plane, :, :5].permute(1, 0, 2)
            params = cache.scales[0, 0, plane, :, :5].permute(1, 0, 2)
            cached.append(unpack_i4_and_asym_dequantize(
                packed, params[..., :1], params[..., 1:]))
        cached_key, cached_value = cached
        reference_query = matmul_had_HIP(query.squeeze(1), torch.float16)

    repeats = num_q_heads // num_kv_heads
    cached_key = cached_key.repeat_interleave(repeats, dim=1).float()
    cached_value = cached_value.repeat_interleave(repeats, dim=1).float()
    scores = torch.einsum("bhd,shd->bhs", reference_query.float(), cached_key)
    scores /= math.sqrt(head_dim)
    expected = torch.einsum("bhs,shd->bhd", scores.softmax(dim=-1), cached_value)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual.float(), expected, rtol=3e-3, atol=3e-3)

# Qualified PARD2 full-row/chunk contracts retained alongside the new kernels.

@pytest.mark.parametrize("batch,width", [(1, 128), (2, 256), (2, 1024), (4, 4096), (1, 8192)])
def test_fused_ffn_matches_silu_hadamard_quant(batch, width):
    torch.manual_seed(width)
    gate = torch.randn(batch, width, device="cuda", dtype=torch.float16)
    up = torch.randn_like(gate)
    packed, scale = _HIP.fused_ffn_silu_hadamard_quant(gate.contiguous(), up.contiguous())
    # The fused reference promotes FP16 inputs before SiLU, multiplication,
    # and every Hadamard butterfly. FP16 intermediates change values at
    # quantization thresholds and are not the fused mathematical contract.
    gate_f = gate.float()
    y = _hadamard(torch.nn.functional.silu(gate_f) * up.float())
    expected_scale = (y.abs().amax(dim=-1, keepdim=True) / 7).half().clamp_min(torch.finfo(torch.float16).tiny)
    torch.cuda.synchronize()
    assert torch.equal(packed, _pack_s4(y, expected_scale))
    assert torch.equal(scale, expected_scale)

@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("page_size,positions", [(4, [1, 2, 4, 5]), (8, [1, 8, 9])])
def test_fused_kv_append_writes_target_asymmetric_cache(head_dim, page_size, positions):
    batch, heads, pages, layers = 2, 4, 2, 1
    data = torch.empty((batch * pages, layers, 2, heads, page_size, head_dim // 2), device="cuda", dtype=torch.uint8)
    params = torch.empty((batch * pages, layers, 2, heads, page_size, 2), device="cuda", dtype=torch.float16)
    for position in positions:
        key = torch.randn(batch, heads, head_dim, device="cuda", dtype=torch.float16)
        value = torch.randn_like(key)
        current_pages = (position + page_size - 1) // page_size
        indptr = torch.arange(0, batch + 1, device="cuda", dtype=torch.int32) * current_pages
        indices = ((torch.arange(current_pages, device="cuda", dtype=torch.int32) * batch).unsqueeze(0) + torch.arange(batch, device="cuda", dtype=torch.int32).unsqueeze(1)).flatten()
        offset = (position - 1) % page_size + 1
        last = torch.full((batch,), offset, device="cuda", dtype=torch.int32)
        _HIP.fused_append_kv_i4(data, params, indptr, indices, last, key, value, layers, 0, heads, page_size, batch)
        page = (position - 1) // page_size
        slot = (position - 1) % page_size
        page_ids = indices[indptr[:-1] + page]
        for source, plane in ((key, 0), (value, 1)):
            rotated = (_hadamard(source) if plane == 0
                       else source.float())
            xmin, xmax = rotated.amin(dim=-1, keepdim=True), rotated.amax(dim=-1, keepdim=True)
            scale = ((xmax - xmin).clamp_min(1e-5) / 15).half()
            zero = (-xmin).half()
            expected = torch.round((rotated + zero) / scale).clamp(0, 15).to(torch.uint8)
            expected = expected[..., 0::2] | (expected[..., 1::2] << 4)
            assert torch.equal(data[page_ids, 0, plane, :, slot], expected)
            assert torch.equal(params[page_ids, 0, plane, :, slot, 0], scale.squeeze(-1))
            assert torch.equal(params[page_ids, 0, plane, :, slot, 1], zero.squeeze(-1))

@pytest.mark.parametrize("position", [1, 31, 32, 33, 63, 64, 65, 1024])
def test_fused_kv_append_page_boundaries(position):
    page_size, batch, heads, layers = 32, 1, 2, 1
    page_count = (position + page_size - 1) // page_size
    data = torch.full((page_count, layers, 2, heads, page_size, 64), 0xA5, device="cuda", dtype=torch.uint8)
    params = torch.full((page_count, layers, 2, heads, page_size, 2), 3.0, device="cuda", dtype=torch.float16)
    key = torch.zeros(batch, heads, 128, device="cuda", dtype=torch.float16)
    value = torch.full_like(key, 2.0)
    indptr = torch.tensor([0, page_count], device="cuda", dtype=torch.int32)
    indices = torch.arange(page_count, device="cuda", dtype=torch.int32)
    last = torch.tensor([(position - 1) % page_size + 1], device="cuda", dtype=torch.int32)
    _HIP.fused_append_kv_i4(data, params, indptr, indices, last, key, value, layers, 0, heads, page_size, batch)
    torch.cuda.synchronize()
    page, slot = (position - 1) // page_size, (position - 1) % page_size
    assert torch.all(data[page, 0, :, :, slot] != 0xA5)
    mask = torch.ones_like(data, dtype=torch.bool)
    mask[page, 0, :, :, slot] = False
    assert torch.all(data[mask] == 0xA5), "fused append wrote outside the target cache slot"

def test_legacy_full_row_special_values_and_int4_endpoints():
    for fill in (0.0, 1.0, -1.0, 100.0):
        gate = torch.full((2, 256), fill, device="cuda", dtype=torch.float16)
        up = torch.ones_like(gate)
        packed, scale = _HIP.fused_ffn_silu_hadamard_quant(gate, up)
        gate_f = gate.float()
        y = _hadamard(torch.nn.functional.silu(gate_f) * up.float())
        expected_scale = (y.abs().amax(dim=-1, keepdim=True) / 7).half().clamp_min(torch.finfo(torch.float16).tiny)
        torch.cuda.synchronize()
        assert torch.equal(packed, _pack_s4(y, expected_scale))
        assert torch.equal(scale, expected_scale)
    endpoints = torch.tensor([-8, -7, -1, 0, 1, 7], device="cuda", dtype=torch.int8)
    expected = torch.tensor([0x98, 0x0F, 0x71], device="cuda", dtype=torch.uint8)
    assert torch.equal(_pack_s4(endpoints, torch.tensor(1.0, device="cuda")), expected)

def test_legacy_unsupported_contracts_fail_before_launch():
    with pytest.raises(RuntimeError, match="float16"):
        _HIP.fused_ffn_silu_hadamard_quant(
            torch.ones(1, 128, device="cuda", dtype=torch.bfloat16),
            torch.ones(1, 128, device="cuda", dtype=torch.bfloat16),
        )
    base = torch.ones(1, 256, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="contiguous"):
        _HIP.fused_ffn_silu_hadamard_quant(base[:, ::2], base[:, ::2])
    data = torch.empty((1, 1, 2, 1, 1, 64), device="cuda", dtype=torch.uint8)
    params = torch.empty((1, 1, 2, 1, 1, 2), device="cuda", dtype=torch.float16)
    key = torch.zeros(1, 1, 128, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="metadata must be CUDA/HIP"):
        _HIP.fused_append_kv_i4(data, params, torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32), torch.tensor([1], dtype=torch.int32),
            key, key, 1, 0, 1, 1, 1)

def test_fused_ffn_generalized_llama2_7b_width():
    from quarot.functional.hadamard import get_hadK

    width, remainder = 11008, 43
    torch.manual_seed(width)
    gate = torch.randn(1, width, device="cuda", dtype=torch.float16)
    up = torch.randn_like(gate)
    hadamard, order = get_hadK(width)
    assert order == remainder
    hadamard = hadamard.to(device="cuda", dtype=torch.float16)
    packed, scale = _HIP.fused_ffn_silu_hadamard_quant_general(
        gate, up, hadamard)

    inner = width // remainder
    values = torch.nn.functional.silu(gate.float()) * up.float()
    values = values.view(-1, remainder, inner)
    stride = 1
    while stride < inner:
        view = values.view(-1, remainder, inner // (2 * stride), 2, stride)
        low, high = view[..., 0, :].clone(), view[..., 1, :].clone()
        view[..., 0, :], view[..., 1, :] = low + high, low - high
        stride <<= 1
    expected = torch.matmul(hadamard.float(), values).reshape_as(gate) / width**0.5
    expected_scale = (expected.abs().amax(dim=-1, keepdim=True) / 7).half()
    expected_scale.clamp_min_(torch.finfo(torch.float16).tiny)
    torch.cuda.synchronize()
    assert torch.equal(packed, _pack_s4(expected, expected_scale))
    assert torch.equal(scale, expected_scale)

@pytest.mark.parametrize("width,remainder", [
    (13824, 27),    # Llama-2 13B: DCT27 x H512
    (22016, 43),    # CodeLlama 34B: DCT43 x H512
    (25600, 25),    # Qwen3-32B: DCT25 x H1024
    (27648, 27),    # Qwen2.5-32B: DCT27 x H1024
    (28672, 28),    # Llama-2 70B: H28 x H1024
])
def test_fused_ffn_generalized_benchmark_widths(width, remainder):
    from quarot.functional.hadamard import get_hadK

    torch.manual_seed(width)
    gate = torch.randn(1, width, device="cuda", dtype=torch.float16)
    up = torch.randn_like(gate)
    hadamard, order = get_hadK(width)
    assert order == remainder
    hadamard = hadamard.to(device="cuda", dtype=torch.float16)
    packed, scale = _HIP.fused_ffn_silu_hadamard_quant_general(
        gate, up, hadamard)

    inner = width // remainder
    values = torch.nn.functional.silu(gate.float()) * up.float()
    values = _hadamard(
        values.view(-1, remainder, inner), output_dtype=torch.float32)
    expected = torch.matmul(hadamard.float(), values).reshape_as(gate)
    expected /= math.sqrt(remainder)
    expected_scale = (expected.abs().amax(dim=-1, keepdim=True) / 7).half()
    expected_scale.clamp_min_(torch.finfo(torch.float16).tiny)
    torch.cuda.synchronize()
    assert torch.equal(packed, _pack_s4(expected, expected_scale))
    assert torch.equal(scale, expected_scale)

@pytest.mark.parametrize("width,remainder", [
    (11008, 43), (13824, 27), (22016, 43),
    (25600, 25), (27648, 27), (28672, 28),
])
def test_single_fp16lds_large_ffn_agrees_with_fp32_contract(width, remainder):
    from quarot.functional.hadamard import get_hadK, _dct_remainder

    torch.manual_seed(width + 17)
    gate = torch.randn(1, width, device="cuda", dtype=torch.float16)
    up = torch.randn_like(gate)
    matrix = get_hadK(width)[0]
    matrix = matrix.cuda().half()
    packed, scale = _HIP.fused_ffn_silu_hadamard_quant_single_fp16lds(
        gate, up, matrix)
    reference_packed, reference_scale = (
        _HIP.fused_ffn_silu_hadamard_quant_general(gate, up, matrix))
    torch.cuda.synchronize()
    assert torch.equal(scale, reference_scale)
    # FP16 LDS intentionally rounds at the inner/remainder boundary. Packed
    # decisions should remain stable except for values at INT4 thresholds.
    assert (packed == reference_packed).float().mean().item() >= 0.995

def test_gqa_prefill_quantizes_before_cache_head_expansion(monkeypatch):
    import quarot.transformers.kv_cache as kv_cache

    observed_heads = []
    original_quantize = kv_cache.asym_quantize_and_pack_i4

    def recording_quantize(x):
        observed_heads.append(x.shape[2])
        return original_quantize(x)

    monkeypatch.setattr(kv_cache, "asym_quantize_and_pack_i4", recording_quantize)
    cache = kv_cache.MultiLayerPagedKVCache4Bit(
        batch_size=2, page_size=16, max_seq_len=16, device="cuda",
        n_layers=1, num_heads=4, num_kv_heads=2, head_dim=128,
        native_gqa=False, fused_decode_append=False)
    torch.manual_seed(3408)
    key = torch.randn(2, 16, 2, 128, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    returned_key, returned_value = cache.update(key, value, 0, {})
    torch.cuda.synchronize()

    assert observed_heads == [2, 2]
    assert returned_key is key and returned_value is value
    for plane in (0, 1):
        assert torch.equal(cache.pages[:, 0, plane, 0], cache.pages[:, 0, plane, 1])
        assert torch.equal(cache.pages[:, 0, plane, 2], cache.pages[:, 0, plane, 3])
        assert torch.equal(cache.scales[:, 0, plane, 0], cache.scales[:, 0, plane, 1])
        assert torch.equal(cache.scales[:, 0, plane, 2], cache.scales[:, 0, plane, 3])

@pytest.mark.parametrize("kind", ["i4", "f16"])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_legacy_decode_name_dispatches_native_gqa(kind, head_dim):
    torch.manual_seed(4800 + head_dim)
    batch, query_heads, kv_heads, page_size = 1, 4, 2, 8
    storage_dim = head_dim // 2 if kind == "i4" else head_dim
    if kind == "i4":
        pages = torch.randint(
            0, 16, (1, 1, 2, kv_heads, page_size, storage_dim),
            device="cuda", dtype=torch.uint8)
    else:
        pages = torch.randn(
            1, 1, 2, kv_heads, page_size, storage_dim,
            device="cuda", dtype=torch.float16)
    params = torch.zeros(
        1, 1, 2, kv_heads, page_size, 2,
        device="cuda", dtype=torch.float16)
    params[..., 0], params[..., 1] = 0.125, 1.0
    indptr = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    indices = torch.tensor([0], device="cuda", dtype=torch.int32)
    last = torch.tensor([4], device="cuda", dtype=torch.int32)
    query = torch.randn(
        batch, query_heads, head_dim, device="cuda", dtype=torch.float16)
    legacy, explicit = torch.empty_like(query), torch.empty_like(query)
    args = (pages, params, indptr, indices, last, 0)
    getattr(_HIP, f"batch_decode_{kind}")(legacy, query, *args)
    getattr(_HIP, f"batch_decode_{kind}_gqa")(explicit, query, *args)
    torch.cuda.synchronize()
    assert torch.equal(legacy, explicit)
