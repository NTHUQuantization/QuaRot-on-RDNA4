"""Independent AR oracle, divergent acceptance, and TD alignment contracts."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from e2e import speculative
from e2e.synchronous_pard import accepted_prefix_lengths, generate_synchronous
from _runtime_helpers import UntimedStages, toy_runtime, runtime_for


def finish(state):
    assert next(state) == 'prefill_complete'
    try:
        next(state)
    except StopIteration as stopped:
        return stopped.value
    raise AssertionError('extra phase')


class SynchronousContracts(unittest.TestCase):
    def test_target_cache_batch_switch_releases_old_graphs(self):
        def build(batch, page, length, **kwargs):
            return SimpleNamespace(batch_size=batch, length=0, n_layers=1,
                _needs_init=[True], _persistent_metadata_enabled=True)
        runtime = SimpleNamespace(verification_graph=True, _verification_cache=None,
            _verification_graphs={}, target=SimpleNamespace(build_cache=build),
            page_size=128, max_cache_len=384, native_gqa=True, fused_decode_append=True)
        for batch in (1, 2, 4, 1):
            cache = speculative.FusedPardRuntime._target_cache(runtime, batch)
            self.assertEqual(cache.batch_size, batch)
            self.assertEqual(runtime._verification_graphs, {})
            self.assertEqual(cache._persistent_metadata_enabled, batch == 1)
            runtime._verification_graphs['sentinel'] = object()
            cache.length = 19
            reused = speculative.FusedPardRuntime._target_cache(runtime, batch)
            self.assertIs(reused, cache)
            self.assertEqual(reused.length, 0)
            self.assertIn('sentinel', runtime._verification_graphs)

    def test_per_row_acceptance(self):
        candidates = torch.tensor([[1, 2, 3]] * 4)
        predictions = torch.tensor([[0, 2, 3, 4], [1, 0, 3, 4], [1, 2, 0, 4], [1, 2, 3, 4]])
        self.assertEqual(accepted_prefix_lengths(candidates, predictions).tolist(), [0, 1, 2, 3])
        with self.assertRaises(ValueError):
            accepted_prefix_lengths(candidates, predictions[:, :3])

    def test_outputs_and_td_alignment_across_partial_rounds_and_pages(self):
        with patch.object(speculative, '_StageTimer', UntimedStages), patch.multiple(
                torch.cuda, synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
                max_memory_allocated=lambda: 0):
            for batch in (2, 4, 8, 16):
                for mode in ('pard2-ti', 'pard2-td'):
                    for tokens, length, pattern in ((1, 1, [3]), (7, 4, [1, 3]),
                                                  (31, 127, [0, 1, 2, 3]), (31, 129, [3])):
                        for lazy, projected in ((False, False), (True, True)):
                            with self.subTest(batch=batch, mode=mode, tokens=tokens, length=length,
                                              pattern=pattern, lazy=lazy):
                                prompt = torch.arange(batch * length).reshape(batch, length).remainder(50) + 3
                                expected = [speculative.FusedPardRuntime._generate_ar(
                                    toy_runtime('ar'), row[None], tokens).output_ids for row in prompt]
                                runtime = runtime_for(mode, batch, pattern, lazy, projected)
                                result = finish(generate_synchronous(runtime, prompt, tokens, UntimedStages()))
                                self.assertEqual(result.output_ids, expected)
                                self.assertEqual(sum(result.emitted_tokens_per_sequence_by_round), tokens)
                                self.assertTrue(all(c == min(a) for c, a in zip(
                                    result.common_accepted_by_round, result.accepted_lengths_by_round)))
                                if length > 1:
                                    self.assertEqual(runtime.draft.prefill_logits, [1])

    def test_public_runtime_guard_and_result(self):
        runtime = runtime_for('pard2-ti', 2, [1, 3])
        prompt = torch.tensor([[3, 4], [7, 8]])
        with patch.object(speculative, '_StageTimer', UntimedStages), patch.multiple(
                torch.cuda, synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
                max_memory_allocated=lambda: 0):
            result = speculative.FusedPardRuntime.generate(runtime, prompt, 7)
            self.assertEqual(result.output_ids, [list(range(5, 12)), list(range(9, 16))])
            runtime.ignore_eos = False
            with self.assertRaisesRegex(ValueError, 'ignore_eos'):
                speculative.FusedPardRuntime.generate(runtime, prompt, 7)


if __name__ == '__main__':
    unittest.main()
