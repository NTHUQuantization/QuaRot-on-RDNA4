import pytest
import torch

from e2e.speculative import (
    _inverse_hadamard_cuda, _normalized_hadamard_cpu)
from quarot.transformers.kv_cache import (
    MultiLayerPagedKVCache4Bit, matmul_had_HIP)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")


def cache(native, max_len):
    return MultiLayerPagedKVCache4Bit(
        batch_size=1, page_size=128, max_seq_len=max_len, device="cuda",
        n_layers=1, num_heads=32, num_kv_heads=8, head_dim=128,
        native_gqa=native, disable_quant=False, hadamard_dtype=torch.float16)

def test_batched_kv_hadamard_matches_independent_rows():
    torch.manual_seed(5)
    values = torch.randn(
        1, 16, 8, 128, device="cuda", dtype=torch.float16)
    batched = matmul_had_HIP(values, torch.float16)
    independent = torch.cat([
        matmul_had_HIP(values[:, index:index + 1], torch.float16)
        for index in range(values.shape[1])
    ], dim=1)
    assert torch.equal(batched, independent)


def test_qwen3_32b_generalized_inverse_hadamard_matches_cpu_oracle():
    torch.manual_seed(19)
    width = 5120
    # Nine rows exercises the gfx1201 eight-row dispatch boundary.
    source = torch.randn(9, width)
    signs = torch.tensor(
        [1, -1] * (width // 2), dtype=torch.float32)
    rotated = _normalized_hadamard_cpu(source * signs)
    restored = _inverse_hadamard_cuda(
        rotated.to(device="cuda", dtype=torch.float16))
    actual = restored.float().cpu() * signs
    torch.testing.assert_close(
        actual, source, atol=2e-2, rtol=2e-2)


def test_fused_chunk_writer_matches_sequential_slots():
    torch.manual_seed(13)
    sequential, chunked = cache(True, 192), cache(True, 192)
    context_k = torch.randn(
        1, 37, 8, 128, device="cuda", dtype=torch.float16)
    context_v = torch.randn_like(context_k)
    sequential.update(context_k, context_v, 0, {})
    chunked.update(context_k, context_v, 0, {})
    key = torch.randn(1, 16, 8, 128, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    for index in range(key.shape[1]):
        sequential.update(
            key[:, index:index + 1], value[:, index:index + 1], 0, {})
    chunked.update(key, value, 0, {})
    end = sequential.length
    assert end == chunked.length
    for position in range(end):
        page, slot = divmod(position, sequential.page_size)
        assert torch.equal(sequential.pages[page, ..., slot, :],
                           chunked.pages[page, ..., slot, :])
        assert torch.equal(sequential.scales[page, ..., slot, :],
                           chunked.scales[page, ..., slot, :])

@pytest.mark.parametrize("context", [1, 127, 128, 129, 1024, 4096])
def test_native_gqa_matches_expanded_mha_oracle(context):
    torch.manual_seed(7)
    native, oracle = cache(True, context + 2), cache(False, context + 2)
    key = torch.randn(1, context, 8, 128, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    native.update(key, value, 0, {})
    oracle.update(key, value, 0, {})
    new_key = torch.randn(1, 1, 8, 128, device="cuda", dtype=torch.float16)
    new_value = torch.randn_like(new_key)
    native_attn = native.update(new_key, new_value, 0, {})
    oracle_attn = oracle.update(new_key, new_value, 0, {})
    query = torch.randn(1, 1, 32, 128, device="cuda", dtype=torch.float16)
    actual, expected = native_attn(query), oracle_attn(query)
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=4e-2)
    assert torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0) > 0.999


@pytest.mark.parametrize("chunk", [1, 15, 16])
def test_provisional_chunk_transaction_and_oracle(chunk):
    torch.manual_seed(11)
    native, oracle = cache(True, 160), cache(False, 160)
    key = torch.randn(1, 129, 8, 128, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    native.update(key, value, 0, {})
    oracle.update(key, value, 0, {})
    transactions = (native.begin(), oracle.begin())
    key = torch.randn(1, chunk, 8, 128, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    native_attn = native.update(key, value, 0, {})
    oracle_attn = oracle.update(key, value, 0, {})
    query = torch.randn(1, chunk, 32, 128, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(native_attn(query), oracle_attn(query), rtol=2e-2, atol=2e-2)
    keep = chunk // 2
    for transaction in transactions:
        transaction.commit(keep)
    assert native.length == oracle.length == 129 + keep
    rollback = native.begin()
    native.length += 1
    rollback.proposed_length += 1
    rollback.rollback()
    assert native.length == 129 + keep
