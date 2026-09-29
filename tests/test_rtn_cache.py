from argparse import Namespace
from copy import deepcopy

import torch

from e2e.checkpoint_utils import gptq_utils


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(4, 3, bias=False)),
            torch.nn.Sequential(torch.nn.Linear(3, 2, bias=False)),
        ])


def _args(cache_dir, **overrides):
    values = dict(
        model="toy", seed=7, rotation_dtype="float32", w_bits=4,
        w_groupsize=-1, w_asym=False, w_clip=False,
        rtn_cache_dir=cache_dir)
    values.update(overrides)
    return Namespace(**values)


def test_rtn_cache_resumes_without_requantizing(tmp_path, monkeypatch):
    original = _ToyModel()
    resumed = deepcopy(original)

    expected_quantizers = gptq_utils.rtn_fwrd(
        original, torch.device("cpu"), _args(tmp_path))
    expected_weights = [layer[0].weight.detach().clone()
                        for layer in original.model.layers]

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "layer_00000.pt", "layer_00001.pt"]

    def unexpected_quantization(self, weight):
        raise AssertionError("a compatible cached layer was recomputed")

    monkeypatch.setattr(
        gptq_utils.WeightQuantizer, "find_params", unexpected_quantization)
    actual_quantizers = gptq_utils.rtn_fwrd(
        resumed, torch.device("cpu"), _args(tmp_path))

    assert actual_quantizers.keys() == expected_quantizers.keys()
    for actual, expected in zip(resumed.model.layers, expected_weights):
        assert torch.equal(actual[0].weight, expected)
    for name in actual_quantizers:
        assert torch.equal(actual_quantizers[name].scale,
                           expected_quantizers[name].scale)


def test_rtn_cache_recomputes_incompatible_settings(tmp_path, monkeypatch):
    gptq_utils.rtn_fwrd(_ToyModel(), torch.device("cpu"), _args(tmp_path))
    calls = 0
    original_find_params = gptq_utils.WeightQuantizer.find_params

    def count_quantization(self, weight):
        nonlocal calls
        calls += 1
        return original_find_params(self, weight)

    monkeypatch.setattr(gptq_utils.WeightQuantizer, "find_params",
                        count_quantization)
    gptq_utils.rtn_fwrd(
        _ToyModel(), torch.device("cpu"), _args(tmp_path, w_asym=True))

    assert calls == 2
