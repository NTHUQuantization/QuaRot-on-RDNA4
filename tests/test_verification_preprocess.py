"""Non-GEMM verification kernels against the pre-existing execution paths.

Run on an idle ROCm GPU after rebuilding quarot._HIP:
  PYTHONPATH=.:third-party/hadacore pytest -q tests/test_verification_preprocess.py
The H128 oracle deliberately retains the backend's eight-row safety limit.
"""
import math

import pytest
import torch

if torch.version.hip is None or not torch.cuda.is_available():
    pytest.skip("requires a ROCm/HIP GPU", allow_module_level=True)

from fast_hadamard_transform import hadamard_transform
from quarot import _HIP

if not hasattr(_HIP, "hadamard_h128"):
    pytest.skip("rebuild the verification preprocessing extension", allow_module_level=True)


def _safe_hadamard(x):
    rows = x.contiguous().view(-1, 128)
    return torch.cat([
        hadamard_transform(rows[start:start + 8], scale=1 / math.sqrt(128))
        for start in range(0, len(rows), 8)
    ]).view(x.shape)


def _norm_rope(x, weight, cos, sin, eps=1e-6):
    fp32 = x.float()
    normalized = (fp32 * torch.rsqrt(
        fp32.square().mean(-1, keepdim=True) + eps)).half() * weight
    rotated = torch.cat((-normalized[..., 64:], normalized[..., :64]), -1)
    return normalized * cos[:, :, None, :] + rotated * sin[:, :, None, :]


@pytest.mark.parametrize("operation", ["hadamard", "query", "kv"])
def test_preprocess_rejects_int32_index_overflow_without_large_allocation(operation):
    # Logical size is 2**31 elements, but the expanded view owns only 256
    # bytes. Call the C++ API directly: the Python Hadamard wrapper would
    # materialize a contiguous copy before reaching the C++ size guard.
    seed = torch.empty((1, 1, 1, 128), device="cuda", dtype=torch.float16)
    huge = seed.expand(1, 1 << 24, 1, 128)
    assert huge.numel() == 1 << 31
    assert huge.untyped_storage().nbytes() == 256
    if operation == "hadamard":
        with pytest.raises(RuntimeError, match="int32 indexing capacity"):
            _HIP.hadamard_h128(huge)
        return

    weight = torch.empty(128, device="cuda", dtype=torch.float16)
    cos = torch.empty((1, 1, 128), device="cuda", dtype=torch.float16)
    sin = torch.empty_like(cos)
    if operation == "query":
        with pytest.raises(RuntimeError, match="int32 indexing capacity"):
            _HIP.chunk_q_norm_rope_hadamard(huge, weight, cos, sin, 1e-6)
        return

    data = torch.empty((1, 1, 2, 1, 1, 64), device="cuda", dtype=torch.uint8)
    params = torch.empty((1, 1, 2, 1, 1, 2), device="cuda", dtype=torch.float16)
    indptr = torch.empty(2, device="cuda", dtype=torch.int32)
    indices = torch.empty(1, device="cuda", dtype=torch.int32)
    last = torch.empty(1, device="cuda", dtype=torch.int32)
    # Metadata remains uninitialized: rejecting the oversized input must
    # happen before any launch, metadata access, or cache write.
    with pytest.raises(RuntimeError, match="int32 indexing capacity"):
        _HIP.chunk_k_norm_rope_append_i4(
            huge, huge, weight, cos, sin, data, params,
            indptr, indices, last, 0, 1e-6)


@pytest.mark.parametrize("rows", [1, 7, 8, 9, 15, 16, 32, 120, 128, 480, 512])
def test_batched_h128_bit_exact_safe_dispatch(rows):
    torch.manual_seed(7310 + rows)
    x = torch.randn(rows, 128, device="cuda", dtype=torch.float16)
    expected = _safe_hadamard(x)
    actual = _HIP.hadamard_h128(x)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("tokens", [1, 15, 16])
def test_batched_h128_rows_do_not_mix(tokens):
    rows = 2 * tokens * 32
    x = torch.zeros(rows, 128, device="cuda", dtype=torch.float16)
    row = torch.arange(rows, device="cuda")
    x[row, row % 128] = (1 + row % 17).half()
    assert torch.equal(_HIP.hadamard_h128(x), _safe_hadamard(x))


@pytest.mark.parametrize("tokens", [1, 15, 16])
@pytest.mark.parametrize("strided", [False, True])
def test_chunk_q_preserves_norm_rope_hadamard(tokens, strided):
    torch.manual_seed(8400 + tokens)
    source = torch.randn(2, tokens, 40, 128, device="cuda", dtype=torch.float16)
    query = source[:, :, :32] if strided else source[:, :, :32].contiguous()
    weight = (1 + 0.1 * torch.randn(128, device="cuda")).half()
    angles = torch.randn(2, tokens, 128, device="cuda")
    cos, sin = angles.cos().half(), angles.sin().half()
    expected = _safe_hadamard(_norm_rope(query, weight, cos, sin))
    actual = _HIP.chunk_q_norm_rope_hadamard(query, weight, cos, sin, 1e-6)
    # The kernel mirrors ATen's reduction tree and materialization boundaries.
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("heads", [1, 8, 32])
@pytest.mark.parametrize("fill", [0.0, 1.0, -1.0])
def test_chunk_q_zero_and_constant_no_additive_bias(heads, fill):
    query = torch.full((1, 1, heads, 128), fill,
                       device="cuda", dtype=torch.float16)
    weight = torch.ones(128, device="cuda", dtype=torch.float16)
    cos = torch.ones(1, 1, 128, device="cuda", dtype=torch.float16)
    sin = torch.zeros_like(cos)
    actual = _HIP.chunk_q_norm_rope_hadamard(query, weight, cos, sin, 1e-6)
    expected = _safe_hadamard(_norm_rope(query, weight, cos, sin))
    assert torch.equal(actual, expected)


def _metadata(batch, final_length, page_size):
    pages = math.ceil(final_length / page_size)
    indptr = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * pages
    # Reverse physical pages to catch an accidental logical-page address.
    indices = torch.arange(batch * pages - 1, -1, -1,
                           device="cuda", dtype=torch.int32)
    last = torch.full((batch,), (final_length - 1) % page_size + 1,
                      device="cuda", dtype=torch.int32)
    return pages, indptr, indices, last


@pytest.mark.parametrize("tokens", [1, 15, 16])
@pytest.mark.parametrize("prefix,page_size", [(7, 8), (31, 32), (32, 32)])
def test_chunk_k_matches_existing_append_across_pages(tokens, prefix, page_size):
    torch.manual_seed(9200 + tokens + prefix)
    batch, heads, layers, layer = 2, 8, 3, 1
    pages, indptr, indices, last = _metadata(batch, prefix + tokens, page_size)
    # Slices deliberately retain larger row strides, as merged QKV may do.
    source = torch.randn(batch, tokens, 16, 128, device="cuda", dtype=torch.float16)
    key, value = source[:, :, :8], source[:, :, 8:]
    weight = (1 + 0.1 * torch.randn(128, device="cuda")).half()
    angles = torch.randn(batch, tokens, 128, device="cuda")
    cos, sin = angles.cos().half(), angles.sin().half()
    shape = (batch * pages, layers, 2, heads, page_size, 64)
    expected = torch.full(shape, 0xA5, dtype=torch.uint8, device="cuda")
    actual = torch.full_like(expected, 0xA5)
    params_shape = (*shape[:-1], 2)
    expected_params = torch.full(params_shape, 3.0, dtype=torch.float16, device="cuda")
    actual_params = torch.full_like(expected_params, 3.0)
    rotated = _norm_rope(key, weight, cos, sin)
    _HIP.fused_append_kv_i4(
        expected, expected_params, indptr, indices, last,
        rotated.contiguous(), value.contiguous(),
        layers, layer, heads, page_size, batch)
    _HIP.chunk_k_norm_rope_append_i4(
        key, value, weight, cos, sin, actual, actual_params,
        indptr, indices, last, layer, 1e-6)
    written = torch.zeros(shape[:-1], device="cuda", dtype=torch.bool)
    for b in range(batch):
        for token in range(tokens):
            position = prefix + token
            physical = int(indices[b * pages + position // page_size])
            written[physical, layer, :, :, position % page_size] = True
    assert torch.all(actual[~written] == 0xA5), "write outside provisional chunk"
    assert torch.all(actual_params[~written] == 3.0)
    assert torch.equal(actual_params[written], expected_params[written])
    assert torch.equal(actual[written], expected[written])


@pytest.mark.parametrize("tokens", [1, 15, 16])
def test_chunk_k_constant_and_zero_values_match_existing_append(tokens):
    batch, heads, layers, page_size, prefix = 1, 2, 1, 8, 7
    pages, indptr, indices, last = _metadata(batch, prefix + tokens, page_size)
    key = torch.zeros(batch, tokens, heads, 128, device="cuda", dtype=torch.float16)
    value = torch.full_like(key, 2.0)
    weight = torch.ones(128, device="cuda", dtype=torch.float16)
    cos = torch.ones(batch, tokens, 128, device="cuda", dtype=torch.float16)
    sin = torch.zeros_like(cos)
    shape = (pages, layers, 2, heads, page_size, 64)
    expected = torch.full(shape, 0xA5, device="cuda", dtype=torch.uint8)
    actual = expected.clone()
    expected_params = torch.full((*shape[:-1], 2), 3.0,
                                  device="cuda", dtype=torch.float16)
    actual_params = expected_params.clone()
    _HIP.fused_append_kv_i4(expected, expected_params, indptr, indices, last,
                          key, value, layers, 0, heads, page_size, batch)
    _HIP.chunk_k_norm_rope_append_i4(
        key, value, weight, cos, sin, actual, actual_params,
        indptr, indices, last, 0, 1e-6)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_params, expected_params)
