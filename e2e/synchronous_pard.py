"""Equal-length, fixed-output PARD2 batches with a common accepted prefix.

All rows execute real batched target/draft forwards. The shortest accepted
prefix determines common progress; each row emits its own next target token.
The generator exposes a prefill boundary to the shared benchmark harness.
"""
from dataclasses import dataclass, field
import time

import torch


def accepted_prefix_lengths(candidates, predictions):
    if (candidates.ndim != 2 or predictions.ndim != 2
            or candidates.shape[0] != predictions.shape[0]
            or candidates.shape[0] < 1 or candidates.shape[1] < 1
            or predictions.shape[1] != candidates.shape[1] + 1):
        raise ValueError('Expected candidates [B,K] and predictions [B,K+1]')
    equal = candidates.eq(predictions[:, :-1])
    return equal.to(torch.int32).cumprod(dim=1).sum(dim=1)


@dataclass
class BatchedGenerationResult:
    output_ids: list[list[int]]
    target_forwards: int
    draft_forwards: int
    proposed_draft_tokens: int
    accepted_draft_tokens: int
    accepted_lengths_by_round: list[list[int]]
    common_accepted_by_round: list[int]
    emitted_tokens_per_sequence_by_round: list[int]
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    total_ms: float = 0.0
    peak_vram_bytes: int = 0
    stage_ms: dict = field(default_factory=dict)


def generate_synchronous(runtime, input_ids, max_new_tokens, timer):
    if (input_ids.ndim != 2 or min(input_ids.shape) < 1 or max_new_tokens < 1
            or not runtime.ignore_eos or runtime.adaptive_k is not None
            or runtime.mode not in ('pard2-ti', 'pard2-td')):
        raise ValueError('Synchronous generation needs equal-length nonempty inputs, fixed K and ignore_eos')
    batch = input_ids.shape[0]
    if input_ids.shape[1] + max_new_tokens + runtime.spec.draft_k > runtime.max_cache_len:
        raise ValueError('Insufficient cache capacity for verification lookahead')
    target_cache = runtime._target_cache(batch)
    draft_cache = runtime._draft_cache(batch)
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    if runtime.collector is not None:
        runtime.collector.reset()
    prefill_out = timer.record('target_prefill', lambda: runtime.target(
        input_ids=input_ids, past_key_values=target_cache, cache_position=positions,
        use_cache=True, attention_mask=None, return_dict=True,
        output_hidden_states=False, logits_to_keep=1))
    target_cache = prefill_out.past_key_values
    prefill_features = runtime.collector.features() if runtime.collector is not None else None
    draft_features = None
    if runtime.mode == 'pard2-td':
        draft_features = torch.cat((torch.zeros_like(prefill_features[:, :1]),
                                    prefill_features[:, :-1]), dim=1)
    draft_input, draft_cache_len = input_ids, 0
    target_forwards, draft_forwards = 1, 0
    if input_ids.shape[1] > 1:
        prefix = input_ids[:, :-1]
        kwargs = {}
        if runtime.mode == 'pard2-td':
            if runtime.td_unique_projection:
                kwargs['projected_target_feat'] = timer.record('td_feature_project_prefill',
                    lambda: runtime._project_td_features(draft_features[:, :-1]))
            else:
                kwargs['target_feat'] = draft_features[:, :-1]
        timer.record('draft_prefill', lambda: runtime.draft(input_ids=prefix,
            past_key_values=draft_cache, cache_position=positions[:-1],
            use_cache=True, attention_mask=None, return_dict=True,
            logits_to_keep=getattr(runtime, 'draft_prefill_logits_to_keep', 1), **kwargs))
        draft_forwards += 1
        draft_cache_len = prefix.shape[1]
        draft_input = input_ids[:, -1:]
        if runtime.mode == 'pard2-td':
            draft_features = draft_features[:, -1:]
    yield 'prefill_complete'

    pending_prediction = prefill_out.logits[:, -1:].argmax(-1)
    generated = [[] for _ in range(batch)]
    lengths_by_round, common_by_round, emitted_by_round = [], [], []
    accepted_total = proposed = emitted_count = 0
    target_input, first_round = None, True
    k = runtime.spec.draft_k
    while emitted_count < max_new_tokens:
        masks = torch.full((batch, k - 1), runtime.spec.pard_token,
                           device=input_ids.device, dtype=input_ids.dtype)
        draft_ids = torch.cat((draft_input, masks), dim=1)
        draft_pos = torch.arange(draft_cache_len, draft_cache_len + draft_ids.shape[1],
                                 device=input_ids.device)
        if hasattr(draft_cache, 'length'):
            draft_cache.length = draft_cache_len
        kwargs = {}
        if runtime.mode == 'pard2-td':
            if runtime.td_unique_projection:
                kwargs['projected_target_feat'] = timer.record('td_feature_project',
                    lambda: runtime._project_td_features(draft_features, k - 1))
            else:
                kwargs['target_feat'] = torch.cat((draft_features,
                    draft_features[:, -1:].expand(-1, k - 1, -1)), dim=1)
        draft_out = timer.record('draft', lambda: runtime.draft_forward(input_ids=draft_ids,
            past_key_values=draft_cache, cache_position=draft_pos, use_cache=True,
            attention_mask=None, return_dict=True, **kwargs))
        draft_forwards += 1
        draft_cache_len += draft_ids.shape[1]
        candidates = draft_out.logits[:, -k:].argmax(-1)
        proposed += batch * k
        verify_ids = candidates if first_round else torch.cat((target_input, candidates), dim=1)
        base_length = target_cache.length
        target_pos = (None if runtime.verification_graph
            and getattr(target_cache, '_static_metadata_enabled', False)
            and verify_ids.shape[1] in (15, 16) else
            torch.arange(base_length, base_length + verify_ids.shape[1], device=input_ids.device))
        target_out, new_features = timer.record('target_verify', lambda: runtime._target_call(
            verify_ids, target_cache, target_pos, materialize_features=not runtime.td_lazy_features))
        predictions = (torch.cat((pending_prediction, target_out.logits[:, -k:].argmax(-1)), dim=1)
                       if first_round else target_out.logits[:, -(k+1):].argmax(-1))
        individual = accepted_prefix_lengths(candidates, predictions).tolist()
        accepted = min(individual)
        lengths_by_round.append(individual)
        common_by_round.append(accepted)
        accepted_total += batch * accepted
        target_cache = target_out.past_key_values
        target_cache.length = base_length + (0 if first_round else 1) + accepted
        target_forwards += 1
        take = min(accepted + 1, max_new_tokens - emitted_count)
        emitted = predictions[:, :take].tolist()
        for output, row in zip(generated, emitted):
            output.extend(row)
        emitted_count += take
        emitted_by_round.append(take)
        if runtime.mode == 'pard2-td':
            if runtime.td_lazy_features:
                draft_features = runtime._accepted_td_features(prefill_features, take, first_round, timer)
            else:
                source = (torch.cat((prefill_features[:, -1:], new_features), dim=1)
                          if first_round else new_features)
                draft_features = source[:, :take]
        if emitted_count >= max_new_tokens:
            break
        draft_input = torch.tensor(emitted, device=input_ids.device, dtype=input_ids.dtype)
        target_input = draft_input[:, -1:]
        draft_cache_len = max(0, draft_cache_len - (k - 1))
        if hasattr(draft_cache, 'length'):
            draft_cache.length = draft_cache_len
        first_round = False
    return BatchedGenerationResult(generated, target_forwards, draft_forwards, proposed,
        accepted_total, lengths_by_round, common_by_round, emitted_by_round)


def run_synchronous(runtime, input_ids, max_new_tokens):
    from e2e.speculative import _StageTimer
    timer = _StageTimer()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    state = generate_synchronous(runtime, input_ids, max_new_tokens, timer)
    assert next(state) == 'prefill_complete'
    torch.cuda.synchronize()
    prefilled = time.perf_counter()
    try:
        next(state)
    except StopIteration as stopped:
        result = stopped.value
    else:
        raise RuntimeError('Unexpected extra phase boundary')
    torch.cuda.synchronize()
    ended = time.perf_counter()
    result.prefill_ms, result.decode_ms = (prefilled-started)*1000, (ended-prefilled)*1000
    result.total_ms = (ended-started)*1000
    result.peak_vram_bytes = torch.cuda.max_memory_allocated()
    result.stage_ms = timer.totals()
    return result
