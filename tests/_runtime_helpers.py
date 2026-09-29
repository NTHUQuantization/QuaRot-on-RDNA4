"""Deterministic CPU stand-ins for generation correctness tests."""
from types import SimpleNamespace, MethodType

import torch

from e2e import speculative


class UntimedStages:
    def record(self, name, fn):
        return fn()

    def totals(self):
        return {}


class ToyModel:
    def __init__(self, draft=False):
        self.draft = draft
        self.features = None
        self.calls = []

    def __call__(self, input_ids, past_key_values, **kwargs):
        self.calls.append(input_ids.clone())
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, 128)
        for row, values in enumerate(input_ids.tolist()):
            last = 0
            for i, token in enumerate(values):
                last = last + 1 if self.draft and token == 99 else token
                logits[row, i, (last + 1) % 128] = 1
        self.features = torch.ones(batch, length, 4)
        past_key_values.length += length
        if kwargs.get('logits_to_keep') == 1:
            logits = logits[:, -1:]
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


def toy_runtime(mode):
    target, draft = ToyModel(), ToyModel(draft=True)
    runtime = SimpleNamespace(mode=mode, target=target, draft=draft,
        ignore_eos=True, max_cache_len=128, spec=SimpleNamespace(draft_k=3, pard_token=99),
        adaptive_k=None, verification_graph=False, td_unique_projection=False,
        td_lazy_features=False, collector=None)
    if mode == 'pard2-td':
        runtime.collector = SimpleNamespace(reset=lambda: None, features=lambda: target.features)
    def target_call(ids, cache, positions, materialize_features=True):
        out = target(ids, cache)
        return out, target.features if runtime.collector else None
    runtime._target_call = target_call
    runtime._target_cache = lambda: SimpleNamespace(length=0)
    runtime._draft_cache = lambda: SimpleNamespace(length=0)
    runtime.draft_forward = draft
    runtime._stop = lambda tokens: False
    return runtime


class FeatureTarget(ToyModel):
    def __call__(self, input_ids, past_key_values, **kwargs):
        output = super().__call__(input_ids, past_key_values, **kwargs)
        self.features = input_ids.float()[..., None].expand(-1, -1, 4)
        return output

class CheckedDraft(ToyModel):
    def __init__(self, fail_pattern, td):
        super().__init__(draft=True)
        self.fail_pattern, self.td = fail_pattern, td
        self.prefill_logits = []

    def project_features(self, features):
        return features

    def __call__(self, input_ids, past_key_values, cache_position, **kwargs):
        if self.td:
            features = kwargs.get('target_feat', kwargs.get('projected_target_feat'))
            assert features is not None
            real = input_ids.shape[1] if kwargs.get('logits_to_keep') == 1 else input_ids.shape[1] - 2
            start = int(cache_position[0])
            previous = (past_key_values.tokens[:, start - 1] if start else
                        torch.zeros(input_ids.shape[0], dtype=torch.long))
            expected = torch.cat((previous[:, None], input_ids[:, :real-1]), dim=1).float()
            torch.testing.assert_close(features[:, :real, 0], expected)
            if real < input_ids.shape[1]:
                torch.testing.assert_close(features[:, real:, 0], expected[:, -1:].expand(-1, 2))
        past_key_values.tokens[:, cache_position] = input_ids
        output = super().__call__(input_ids, past_key_values, **kwargs)
        if kwargs.get('logits_to_keep') == 1:
            self.prefill_logits.append(output.logits.shape[1])
        else:
            for row in range(input_ids.shape[0]):
                fail = self.fail_pattern[row % len(self.fail_pattern)]
                if fail < 3:
                    output.logits[row, -3 + fail].zero_()
                    output.logits[row, -3 + fail, 100] = 1
        return output

def runtime_for(mode, batch, pattern, lazy=False, projected=False):
    runtime = toy_runtime(mode)
    runtime.target = FeatureTarget()
    runtime.draft = CheckedDraft(pattern, mode == 'pard2-td')
    runtime.draft_forward = runtime.draft
    runtime.max_cache_len = 384
    def cache(size):
        return SimpleNamespace(length=0, batch_size=size, tokens=torch.zeros(size, 384, dtype=torch.long))
    runtime._target_cache = lambda size=1: cache(size)
    runtime._draft_cache = lambda size=1: cache(size)
    def target_call(ids, cache, positions, materialize_features=True):
        out = runtime.target(ids, cache)
        return out, runtime.target.features if materialize_features and mode == 'pard2-td' else None
    runtime._target_call = target_call
    if mode == 'pard2-td':
        runtime.collector = SimpleNamespace(reset=lambda: None,
            features=lambda rows=None: runtime.target.features if rows is None else runtime.target.features[:, rows])
    runtime.td_lazy_features, runtime.td_unique_projection = lazy, projected
    runtime._accepted_td_features = MethodType(speculative.FusedPardRuntime._accepted_td_features, runtime)
    runtime._project_td_features = MethodType(speculative.FusedPardRuntime._project_td_features, runtime)
    return runtime
