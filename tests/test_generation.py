"""Single-request PARD2 output must match the independent greedy target."""
from unittest.mock import patch

import pytest
import torch

from e2e import speculative
from _runtime_helpers import UntimedStages, runtime_for, toy_runtime


@pytest.mark.parametrize("mode", ["pard2-ti", "pard2-td"])
@pytest.mark.parametrize("accepted", [0, 1, 2, 3])
@pytest.mark.parametrize("prompt_length,new_tokens", [(1, 1), (4, 7), (127, 31)])
def test_speculative_generation_matches_greedy_target(
        mode, accepted, prompt_length, new_tokens):
    prompt = torch.arange(prompt_length).remainder(50).add(3).unsqueeze(0)
    runtime = runtime_for(mode, 1, [accepted])
    with patch.object(speculative, "_StageTimer", UntimedStages), patch.multiple(
            torch.cuda, synchronize=lambda: None,
            reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 0):
        expected = speculative.FusedPardRuntime._generate_ar(
            toy_runtime("ar"), prompt, new_tokens)
        actual = speculative.FusedPardRuntime._generate_spec(
            runtime, prompt, new_tokens)
    assert actual.output_ids == expected.output_ids
    assert len(actual.output_ids) == new_tokens
    assert sum(actual.emitted_tokens_per_step) == new_tokens
    assert actual.accepted_draft_tokens == accepted * actual.verifier_steps
