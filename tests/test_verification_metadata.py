"""Compact causal metadata must remain correct across pages and rollback."""
import pytest
import torch
from quarot.transformers.kv_cache import MultiLayerPagedKVCache4Bit

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm required")


def make_cache(monkeypatch, batch=1, enabled=True):
    monkeypatch.setenv("QUAROT_STATIC_KV_METADATA", str(int(enabled)))
    return MultiLayerPagedKVCache4Bit(batch, 128, 400, "cuda", 2, 4, 128,
                                     num_kv_heads=2)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("rows", [1, 15, 16])
def test_gpu_metadata_matches_cpu_causal_oracle(monkeypatch, batch, rows):
    cache = make_cache(monkeypatch, batch)
    buffers = cache._chunk_metadata[rows]
    addresses = {name: {k: t.data_ptr() for k, t in buffers[name].items()}
                 for name in ("append", "causal")}
    for base in (0, 1, 113, 127, 128, 129, 255, 256, 270, 128, 127):
        cache.length = base
        positions = cache.prepare_verification_metadata(rows)
        assert positions.tolist() == list(range(base, base + rows))
        cache.length += rows
        actual = cache.get_virtual_cache_specs(rows)
        cache._active_chunk_metadata = None
        expected = cache.get_virtual_cache_specs(rows)
        cache._active_chunk_metadata = buffers
        for name in ("kv_indptr", "last_page_offset"):
            assert torch.equal(actual[name], expected[name])
        assert torch.equal(actual["kv_indices"][:expected["kv_indices"].numel()],
                           expected["kv_indices"])
        append = cache.get_cache_specs_for_flash_infer(None)
        pages = (base + rows + 127) // 128
        assert append["kv_indptr"].tolist() == [b * pages for b in range(batch + 1)]
        assert append["kv_indices"][:batch * pages].tolist() == [
            p * batch + b for b in range(batch) for p in range(pages)]
        assert append["last_page_offset"].tolist() == [(base + rows - 1) % 128 + 1] * batch
        assert addresses == {name: {k: t.data_ptr() for k, t in buffers[name].items()}
                             for name in ("append", "causal")}


@pytest.mark.parametrize("rows", [15, 16])
def test_metadata_graph_updates_positions_and_rewinds(monkeypatch, rows):
    cache = make_cache(monkeypatch)
    cache.length = 120
    cache.prepare_verification_metadata(rows)
    cache._verification_graph_capture = True
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            positions = cache.prepare_verification_metadata(rows)
    finally:
        cache._verification_graph_capture = False
    for base in (120, 130, 125, 255, 1):
        cache._metadata_base.fill_(base)
        graph.replay()
        assert positions.tolist() == list(range(base, base + rows))
        assert cache._chunk_metadata[rows]["causal"]["last_page_offset"].tolist() == [
            (base + token) % 128 + 1 for token in range(rows)]


@pytest.mark.parametrize("rows", [1, 15, 16])
def test_static_metadata_cache_rollback_and_causality(monkeypatch, rows):
    torch.manual_seed(82)
    cache = make_cache(monkeypatch)
    oracle = make_cache(monkeypatch, enabled=False)
    key = torch.randn(1, 127, 2, 128, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    for layer in range(2):
        cache.update(key, value, layer)
        oracle.update(key, value, layer)
    query = torch.randn(1, rows, 4, 128, device="cuda", dtype=torch.float16)
    key = torch.randn(1, rows, 2, 128, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    for keep in (0, rows // 2, rows):
        cache.length = oracle.length = 127
        transactions = (cache.begin(), oracle.begin())
        for layer in range(2):
            actual = cache.update(key, value, layer)(query)
            expected = oracle.update(key, value, layer)(query)
            assert torch.equal(actual, expected)
        for transaction in transactions:
            transaction.commit(keep)
        assert cache.length == oracle.length == 127 + keep
        replacement = key[:, :1] * -1
        for layer in range(2):
            actual = cache.update(replacement, value[:, :1], layer)(query[:, :1])
            expected = oracle.update(replacement, value[:, :1], layer)(query[:, :1])
            assert torch.equal(actual, expected)
        for pos in range(128 + keep):
            page, slot = divmod(pos, 128)
            assert torch.equal(cache.pages[page, ..., slot, :], oracle.pages[page, ..., slot, :])
            assert torch.equal(cache.scales[page, ..., slot, :], oracle.scales[page, ..., slot, :])


def test_capacity_and_fallback(monkeypatch):
    cache = make_cache(monkeypatch)
    cache.length = 390
    with pytest.raises(ValueError, match="capacity"):
        cache.prepare_verification_metadata(16)
    assert cache.prepare_verification_metadata(17) is None
    assert make_cache(monkeypatch, enabled=False).prepare_verification_metadata(16) is None
