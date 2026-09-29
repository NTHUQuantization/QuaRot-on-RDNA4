"""CPU-only attention dispatch checks; no model or checkpoint is constructed."""
from types import SimpleNamespace

import pytest
import torch

from e2e.quantized_common import QuarotAttentionMixin


class _DispatchSelected(RuntimeError):
    pass


class _NormProbe:
    def __init__(self, name, events):
        self.name, self.events = name, events
        self.weight = torch.ones(128, device="cpu", dtype=torch.float16)
        self.variance_epsilon = 1e-6

    def __call__(self, value):
        self.events.append(self.name)
        return value


def _stop_at_route(name, events):
    def selected(*args, **kwargs):
        events.append(name)
        # Stop before any attention/kernel work. The production forward has
        # already chosen its route and performed any required eager norms.
        raise _DispatchSelected(name)
    return selected


@pytest.mark.parametrize("rows,k1_enabled,transaction,chunk_flag,expected", [
    (1, True, False, "1", ("q_norm", "k_norm", "k1")),
    (1, True, True, "1", ("q_norm", "k_norm", "eager")),
    (1, False, False, "1", ("chunk",)),
    (15, False, False, "1", ("chunk",)),
    (16, False, False, "1", ("chunk",)),
    (15, True, False, "1", ("chunk",)),
    (16, True, True, "1", ("chunk",)),
    (1, True, False, "0", ("q_norm", "k_norm", "k1")),
    (16, False, False, "0", ("q_norm", "k_norm", "eager")),
], ids=[
    "grouped_ar_preserves_k1", "grouped_ar_transaction_preserves_eager",
    "legacy_ar_chunk", "legacy_m15_chunk", "legacy_m16_chunk",
    "grouped_m15_chunk", "grouped_m16_transaction_chunk",
    "disabled_chunk_preserves_k1", "disabled_chunk_verifier_eager",
])
def test_chunk_dispatch_preserves_k1_and_norm_order(
        monkeypatch, rows, k1_enabled, transaction, chunk_flag, expected):
    monkeypatch.setenv("QUAROT_CHUNK_PREPROCESS", chunk_flag)
    events = []
    identity = lambda value: value
    attention = SimpleNamespace(
        head_dim=128, layer_idx=0,
        _fused_projections_enabled=False, _fused_k1_enabled=k1_enabled,
        quantizer=identity, q_proj=identity, k_proj=identity, v_proj=identity,
        q_norm=_NormProbe("q_norm", events),
        k_norm=_NormProbe("k_norm", events),
        _quarot_apply_rotary=_stop_at_route("eager", events))
    cache = SimpleNamespace(
        can_fuse_chunk=lambda layer, mask, count: (
            mask is None and count in (1, 15, 16)),
        can_fuse_k1=lambda layer, mask: mask is None and not transaction,
        update_fused_chunk=_stop_at_route("chunk", events),
        update_fused_k1=_stop_at_route("k1", events))
    hidden = torch.zeros((1, rows, 128), device="cpu", dtype=torch.float16)
    cos = torch.ones((1, rows, 128), device="cpu", dtype=torch.float16)
    sin = torch.zeros_like(cos)

    with pytest.raises(_DispatchSelected, match=f"^{expected[-1]}$"):
        QuarotAttentionMixin.forward(
            attention, hidden, (cos, sin), past_key_value=cache)
    assert events == list(expected)
