from transformers.cache_utils import Cache
from typing import Optional, Tuple, Dict, Any
import math
import os
import torch
from .. import _HIP
import functools
from fast_hadamard_transform import hadamard_transform
from quarot.functional.quantization import get_minq_maxq

@torch.jit.script
def asym_quantize_and_pack_i4(x: torch.Tensor):
    minq, maxq = get_minq_maxq(bits=4, sym=False)
    xmax = torch.amax(x, dim=-1, keepdim=True)
    xmin = torch.amin(x, dim=-1, keepdim=True)
    scale = ((xmax - xmin).clamp(min=1e-5) / maxq)
    zero = -xmin
    q = torch.clamp(torch.round((x + zero) / scale), 0, maxq)

    # pack int4
    q = q.to(dtype=torch.uint8)
    q = q[..., 0::2] | (q[..., 1::2] << 4)
    return q, scale, zero

def unpack_i4_and_asym_dequantize(q, scale, zero):
    #unpack int4
    assert q.dtype == torch.uint8
    q = torch.stack((q & 0x0f, (q >> 4) & 0x0f), dim=-1).view(*q.shape[:-1], q.shape[-1] * 2)
    return q * scale - zero

def matmul_had_HIP(X, dtype):
    n = X.shape[-1]
    # Flatten explicitly so every KV head/token is an independent FHT row.
    # The HIP backend otherwise mixes chunk row 1+ across leading axes.
    rows = X.to(dtype).contiguous().view(-1, n)
    if (n == 128 and rows.dtype == torch.float16 and rows.is_cuda
            and os.getenv("QUAROT_BATCHED_H128", "1") != "0"
            and hasattr(_HIP, "hadamard_h128")):
        # Eight valid rows per workgroup preserve the safe WMMA arithmetic;
        # one launch processes all groups without padded tensors or cat.
        return _HIP.hadamard_h128(rows).to(X.dtype).view(X.shape)
    # gfx1201 hadacore is exact for at most eight rows per dispatch.
    output = torch.cat([
        hadamard_transform(rows[start:start + 8], scale=1/math.sqrt(n))
        for start in range(0, rows.shape[0], 8)
    ], dim=0)
    return output.to(X.dtype).view(X.shape)


def init_kv_i4(kv_data, kv_param,
               kv_indptr, kv_indices,
               last_page_offset, k,
               v, k_param, v_param,
               seqlen_indptr, layer_idx):
    return _HIP.init_kv_i4(
        kv_data, kv_param,
        kv_indptr, kv_indices,
        last_page_offset, k,
        v, k_param, v_param,
        seqlen_indptr, layer_idx)


def append_kv_i4(kv_data, kv_param,
               kv_indptr, kv_indices,
               last_page_offset, k,
               v, k_param, v_param,
               layer_idx):
    return _HIP.append_kv_i4(
        kv_data, kv_param,
        kv_indptr, kv_indices,
        last_page_offset, k,
        v, k_param, v_param,
        layer_idx)


def fused_append_kv_i4(kv_data, kv_param,
                       kv_indptr, kv_indices,
                       last_page_offset, k, v,
                       num_layers, layer_idx, num_heads,
                       page_size, batch_size):
    """Decode-only K/V path matching this package's asymmetric FlashInfer cache."""
    return _HIP.fused_append_kv_i4(
        kv_data, kv_param, kv_indptr, kv_indices, last_page_offset, k, v,
        num_layers, layer_idx, num_heads, page_size, batch_size)

def batch_decode_i4(o, q, kv_data, kv_param,
               kv_indptr, kv_indices,
               last_page_offset, layer_idx):
    return _HIP.batch_decode_i4(
        o, q, kv_data, kv_param,
        kv_indptr, kv_indices,
        last_page_offset, layer_idx)

def batch_decode_i4_gqa(o, q, kv_data, kv_param,
               kv_indptr, kv_indices, last_page_offset, layer_idx):
    return _HIP.batch_decode_i4_gqa(
        o, q, kv_data, kv_param, kv_indptr, kv_indices,
        last_page_offset, layer_idx)


def init_kv_f16(kv_data, kv_param,
               kv_indptr, kv_indices,
               last_page_offset, k,
               v, k_param, v_param,
               seqlen_indptr, layer_idx):
    return _HIP.init_kv_f16(
        kv_data, kv_param,
        kv_indptr, kv_indices,
        last_page_offset, k,
        v, k_param, v_param,
        seqlen_indptr, layer_idx)


def append_kv_f16(kv_data, kv_param,
               kv_indptr, kv_indices,
               last_page_offset, k,
               v, k_param, v_param,
               layer_idx):
    return _HIP.append_kv_f16(
        kv_data, kv_param,
        kv_indptr, kv_indices,
        last_page_offset, k,
        v, k_param, v_param,
        layer_idx)

def batch_decode_f16(o, q, kv_data, kv_param,
               kv_indptr, kv_indices,
               last_page_offset, layer_idx):
    return _HIP.batch_decode_f16(
        o, q, kv_data, kv_param,
        kv_indptr, kv_indices,
        last_page_offset, layer_idx)

def batch_decode_f16_gqa(o, q, kv_data, kv_param,
               kv_indptr, kv_indices, last_page_offset, layer_idx):
    return _HIP.batch_decode_f16_gqa(
        o, q, kv_data, kv_param, kv_indptr, kv_indices,
        last_page_offset, layer_idx)


class _AttentionStub(object):
    def __init__(self, cache_page_size, device, n_layers, disable_quant, hadamard_dtype):
        self.cache_page_size = cache_page_size
        self.n_layers = n_layers
        self.disable_quant = disable_quant
        self.hadamard_dtype = hadamard_dtype

    def forward(self, q, num_kv_heads, attention_kwargs, layer_idx,
                hadamard_applied=False):
        batch_size, q_len, num_qo_heads, head_dim = q.shape
        q = q.view(batch_size * q_len, num_qo_heads, head_dim)
        if self.hadamard_dtype is not None and not hadamard_applied:
            q = matmul_had_HIP(q, dtype=self.hadamard_dtype)
        attn_output = torch.empty_like(q)
        if self.disable_quant:
            batch_decode = (batch_decode_f16_gqa if num_qo_heads != num_kv_heads
                            else batch_decode_f16)
        else:
            batch_decode = (batch_decode_i4_gqa if num_qo_heads != num_kv_heads
                            else batch_decode_i4)
        batch_decode(
            attn_output, q,
            **attention_kwargs, layer_idx=layer_idx
        )
        attn_output = attn_output.view(batch_size, q_len, num_qo_heads, head_dim)
        return attn_output


class CacheTransaction:
    """Logical KV transaction; stale provisional slots are overwritten later."""

    def __init__(self, cache):
        if cache._transaction is not None:
            raise RuntimeError("a cache transaction is already active")
        self.cache = cache
        self.start_length = cache.length
        self.proposed_length = self.start_length
        self.closed = False
        cache._transaction = self

    @property
    def proposed_tokens(self):
        return self.proposed_length - self.start_length

    def commit(self, keep_tokens):
        if self.closed:
            raise RuntimeError("cache transaction is already closed")
        if not 0 <= keep_tokens <= self.proposed_tokens:
            raise ValueError(
                f"keep_tokens must be in [0, {self.proposed_tokens}]")
        self.cache.length = self.start_length + keep_tokens
        self.cache._transaction = None
        self.closed = True

    def rollback(self):
        self.commit(0)

    def __del__(self):
        if not self.closed:
            self.cache.length = self.start_length
            self.cache._transaction = None
            self.closed = True


class MultiLayerPagedKVCache4Bit(Cache):
    def __init__(
        self, batch_size, page_size, max_seq_len,
        device, n_layers, num_heads, head_dim,
        num_kv_heads=None,
        native_gqa=True,
        fused_decode_append=True,
        fused_k1=None,
        disable_quant=False, hadamard_dtype=torch.float16 ):
        self.page_size = page_size
        self.n_layers = n_layers
        self.num_q_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.native_gqa = bool(native_gqa)
        self.fused_decode_append = bool(fused_decode_append)
        self.fused_k1 = (
            os.getenv("QUAROT_FUSED_K1", "1") != "0"
            if fused_k1 is None else bool(fused_k1))
        # Expanded MHA remains available as a correctness oracle/ablation.
        self.cache_heads = (self.num_kv_heads if self.native_gqa
                            else self.num_q_heads)
        # transformers.Cache exposes ``batch_size`` as a read-only property.
        # Store the value privately and expose it below so this cache keeps the
        # same public API without assigning to the base-class descriptor.
        self._batch_size = batch_size
        max_page_cnt = self.page_cnt_from_length(max_seq_len)
        self.disable_quant = disable_quant
        self.pages = torch.empty(
            (
                max_page_cnt * batch_size,
                n_layers,
                2,
                self.cache_heads,
                page_size,
                head_dim if disable_quant else head_dim // 2
            ),
            dtype=torch.float16 if disable_quant else torch.uint8, device=device)

        self.scales = torch.empty((max_page_cnt * batch_size, n_layers, 2, self.cache_heads, page_size, 2), dtype=torch.float16, device=device)
        self.page_size = page_size
        self.max_seq_len = max_seq_len
        self._needs_init = [True] * n_layers
        self.length = 0
        self.device = device
        self.hadamard_dtype = hadamard_dtype
        self._transaction = None
        # Equal-length decode metadata is immutable except for selecting a
        # length row. Precomputing it avoids repeated decoder-layer allocation
        # while rollback remains a logical length change.
        self._persistent_metadata_enabled = (
            os.getenv("QUAROT_PERSISTENT_KV_METADATA", "1") != "0")
        self._cuda_graph_decode = False
        self._static_metadata_enabled = (
            os.getenv("QUAROT_STATIC_KV_METADATA", "0") != "0"
            and hasattr(_HIP, "verification_metadata"))
        self._verification_graph_capture = False
        self._metadata_prepared_key = None
        self._active_chunk_metadata = None
        self._chunk_metadata = {}
        if self._static_metadata_enabled:
            self._metadata_base = torch.empty((), device=device, dtype=torch.int64)
            for rows in (1, 15, 16):
                def integers(size):
                    return torch.empty(size, device=device, dtype=torch.int32)
                self._chunk_metadata[rows] = {
                    "positions": torch.empty(rows, device=device, dtype=torch.int64),
                    "append": dict(kv_data=self.pages, kv_param=self.scales,
                        kv_indptr=integers(batch_size + 1),
                        kv_indices=integers(batch_size * max_page_cnt),
                        last_page_offset=integers(batch_size)),
                    "causal": dict(kv_data=self.pages, kv_param=self.scales,
                        kv_indptr=integers(batch_size * rows + 1),
                        kv_indices=integers(batch_size * rows * max_page_cnt),
                        last_page_offset=integers(batch_size * rows)),
                }
        if self._persistent_metadata_enabled:
            lengths = torch.arange(
                max_seq_len + 1, device=device, dtype=torch.int32)
            page_counts = torch.div(
                lengths + page_size - 1, page_size, rounding_mode="floor")
            batch_offsets = torch.arange(
                batch_size + 1, device=device, dtype=torch.int32)
            self._decode_kv_indptr = (
                page_counts[:, None] * batch_offsets[None, :])
            self._decode_kv_indices = torch.arange(
                max_page_cnt * batch_size, device=device, dtype=torch.int32)
            page_offsets = lengths.remainder(page_size)
            page_offsets = torch.where(
                (lengths != 0) & (page_offsets == 0), page_size, page_offsets)
            self._decode_last_page_offset = page_offsets[:, None].repeat(
                1, batch_size)
        self._stub = _AttentionStub(
            self.page_size, device, n_layers,
            disable_quant=self.disable_quant,
            hadamard_dtype=self.hadamard_dtype)

    def page_cnt_from_length(self, length):
        return (length + self.page_size - 1) // self.page_size

    @property
    def batch_size(self):
        return self._batch_size

    def _ensure_page_cnt_per_batch(self, expected_page_cnt_per_batch):
        expected_page_cnt = expected_page_cnt_per_batch * self.batch_size
        if expected_page_cnt <= self.pages.shape[0]:
            return
        raise NotImplementedError

    @property
    def seen_tokens(self):
        return self.length

    def begin(self):
        if self._cuda_graph_decode:
            raise RuntimeError(
                "cache transactions cannot use graph decode metadata")
        return CacheTransaction(self)

    def prepare_verification_metadata(self, rows):
        """Update shared fixed buffers once per forward, including after rollback."""
        if (not self._static_metadata_enabled or self._cuda_graph_decode
                or rows not in self._chunk_metadata):
            self._active_chunk_metadata = None
            self._metadata_prepared_key = None
            return None
        if self.length < 0 or self.length + rows > self.max_seq_len:
            raise ValueError("verification exceeds KV cache capacity")
        metadata = self._chunk_metadata[rows]
        if not self._verification_graph_capture:
            self._metadata_base.fill_(self.length)
        append, causal = metadata["append"], metadata["causal"]
        _HIP.verification_metadata(
            self._metadata_base, metadata["positions"],
            append["kv_indptr"], append["kv_indices"], append["last_page_offset"],
            causal["kv_indptr"], causal["kv_indices"], causal["last_page_offset"],
            self.batch_size, self.page_size,
            self.page_cnt_from_length(self.max_seq_len))
        self._active_chunk_metadata = metadata
        self._metadata_prepared_key = (self.length, rows)
        return metadata["positions"]

    def _prepare_layer_metadata(self, layer_idx, rows, attention_mask):
        if layer_idx == 0:
            if attention_mask is not None:
                self._active_chunk_metadata = None
                self._metadata_prepared_key = None
            elif self._metadata_prepared_key != (self.length, rows):
                self.prepare_verification_metadata(rows)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ):
        cache_kwargs = cache_kwargs or {}

        b_sz, added_length, num_heads, head_dim = key_states.shape
        self._prepare_layer_metadata(
            layer_idx, added_length, cache_kwargs.get("attention_mask"))
        if num_heads != self.num_kv_heads:
            raise ValueError("KV tensor head count does not match cache configuration")

        orig_key_states = key_states
        orig_value_states = value_states

        use_fused_append = (
            self.fused_decode_append and not self.disable_quant and
            cache_kwargs.get("attention_mask") is None
        )

        if use_fused_append and self.cache_heads != num_heads:
            repeats = self.cache_heads // num_heads
            key_states = key_states.repeat_interleave(repeats, dim=2)
            value_states = value_states.repeat_interleave(repeats, dim=2)
            num_heads = self.cache_heads

        if not use_fused_append and self.hadamard_dtype is not None:
            key_states = matmul_had_HIP(key_states, dtype=self.hadamard_dtype)

        if use_fused_append:
            # The HIP kernel applies the exact head-wise Hadamard, target-cache
            # asymmetric quantization, and writes directly to the selected page.
            pass
        elif self.disable_quant:
            k_scale = key_states.new_ones((b_sz, added_length, num_heads, 1))
            k_zero = key_states.new_zeros((b_sz, added_length, num_heads, 1))
            v_scale = value_states.new_ones((b_sz, added_length, num_heads, 1))
            v_zero = value_states.new_zeros((b_sz, added_length, num_heads, 1))
        else:
            key_states, k_scale, k_zero = asym_quantize_and_pack_i4(key_states)
            value_states, v_scale, v_zero = asym_quantize_and_pack_i4(value_states)

        if not use_fused_append and self.cache_heads != num_heads:
            repeats = self.cache_heads // num_heads
            key_states = key_states.repeat_interleave(repeats, dim=2)
            value_states = value_states.repeat_interleave(repeats, dim=2)
            k_scale = k_scale.repeat_interleave(repeats, dim=2)
            k_zero = k_zero.repeat_interleave(repeats, dim=2)
            v_scale = v_scale.repeat_interleave(repeats, dim=2)
            v_zero = v_zero.repeat_interleave(repeats, dim=2)
            num_heads = self.cache_heads

        if not use_fused_append:
            k_param = torch.cat([k_scale, k_zero], dim=-1).view(self.batch_size * added_length, num_heads, 2)
            v_param = torch.cat([v_scale, v_zero], dim=-1).view(self.batch_size * added_length, num_heads, 2)

        quantized_head_dim = self.pages.shape[-1]

        assert b_sz == self.batch_size
        if layer_idx == 0:
            current_length = self.length
            new_length = current_length + added_length
            if new_length > self.max_seq_len:
                raise ValueError("KV cache capacity exceeded")
            self._ensure_page_cnt_per_batch(self.page_cnt_from_length(new_length))
            self.length = new_length
            if self._transaction is not None:
                self._transaction.proposed_length = new_length
        attention_mask = cache_kwargs.get("attention_mask")
        if self._needs_init[layer_idx]:
            self._needs_init[layer_idx] = False
            if use_fused_append:
                fused_append_kv_i4(
                    **self.get_cache_specs_for_flash_infer(None),
                    k=key_states.contiguous(), v=value_states.contiguous(),
                    num_layers=self.n_layers, layer_idx=layer_idx,
                    num_heads=num_heads, page_size=self.page_size,
                    batch_size=self.batch_size,
                )
                return orig_key_states, orig_value_states
            if attention_mask is not None:
                nonzero_indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten().view(-1, 1)
                key_states = key_states.view(self.batch_size * added_length, num_heads * quantized_head_dim)
                value_states = value_states.view(self.batch_size * added_length, num_heads * quantized_head_dim)
                key_states = torch.gather(key_states, 0, nonzero_indices.expand(-1, num_heads * quantized_head_dim))
                value_states = torch.gather(value_states, 0, nonzero_indices.expand(-1, num_heads * quantized_head_dim))

                k_param = k_param.view(self.batch_size * added_length, num_heads * 2)
                v_param = v_param.view(self.batch_size * added_length, num_heads * 2)
                k_param = torch.gather(k_param, 0, nonzero_indices.expand(-1, num_heads * 2))
                v_param = torch.gather(v_param, 0, nonzero_indices.expand(-1, num_heads * 2))

                seqlens_in_batch = torch.nn.functional.pad(torch.cumsum(attention_mask.sum(dim=-1, dtype=torch.int32), dim=0, dtype=torch.int32), (1, 0))
            else:
                seqlens_in_batch = torch.arange(self.batch_size + 1, device=self.device, dtype=torch.int) * added_length

            init_kv = init_kv_f16 if self.disable_quant else init_kv_i4
            init_kv(
                **self.get_cache_specs_for_flash_infer(attention_mask),
                k=key_states.view(-1, num_heads, quantized_head_dim),
                v=value_states.view(-1, num_heads, quantized_head_dim),
                k_param=k_param.view(-1, num_heads, 2),
                v_param=v_param.view(-1, num_heads, 2),
                seqlen_indptr=seqlens_in_batch,
                layer_idx=layer_idx
            )
            return orig_key_states, orig_value_states
        else:
            specs = self.get_cache_specs_for_flash_infer(attention_mask)
            if use_fused_append:
                fused_append_kv_i4(
                    **specs, k=key_states.contiguous(),
                    v=value_states.contiguous(),
                    num_layers=self.n_layers, layer_idx=layer_idx, num_heads=num_heads,
                    page_size=self.page_size, batch_size=self.batch_size,
                )
            elif added_length == 1:
                append_kv = append_kv_f16 if self.disable_quant else append_kv_i4
                append_kv(
                    **specs,
                    k=key_states.view(self.batch_size, num_heads, quantized_head_dim),
                    v=value_states.view(self.batch_size, num_heads, quantized_head_dim),
                    k_param=k_param.view(-1, num_heads, 2),
                    v_param=v_param.view(-1, num_heads, 2),
                    layer_idx=layer_idx,
                )
            else:
                # FlashInfer's prefill append kernel writes the supplied suffix
                # at (final sequence length - suffix length). This provides a
                # single-launch provisional chunk append without reallocating
                # or copying the existing cache.
                seqlen_indptr = (
                    torch.arange(self.batch_size + 1, device=self.device,
                                 dtype=torch.int32) * added_length)
                init_kv = init_kv_f16 if self.disable_quant else init_kv_i4
                init_kv(
                    **specs,
                    k=key_states.view(-1, num_heads, quantized_head_dim),
                    v=value_states.view(-1, num_heads, quantized_head_dim),
                    k_param=k_param.view(-1, num_heads, 2),
                    v_param=v_param.view(-1, num_heads, 2),
                    seqlen_indptr=seqlen_indptr,
                    layer_idx=layer_idx,
                )
            attention_specs = (self.get_virtual_cache_specs(added_length)
                               if added_length > 1 else specs)
        return functools.partial(
            self._stub.forward,
            num_kv_heads=num_heads,
            attention_kwargs=(attention_specs if not self._needs_init[layer_idx]
                              else self.get_cache_specs_for_flash_infer(attention_mask)),
            layer_idx=layer_idx,
        )

    def can_fuse_k1(self, layer_idx, attention_mask):
        """Whether the incoming AR-only RoPE/KV append kernel is safe."""
        return (
            self.fused_k1 and self.fused_decode_append and self.native_gqa
            and attention_mask is None and not self.disable_quant
            and self._transaction is None
            and not self._needs_init[layer_idx])

    def can_fuse_chunk(self, layer_idx, attention_mask, rows):
        return (rows in (1, 15, 16) and self.fused_decode_append
                and self.native_gqa and attention_mask is None
                and not self.disable_quant and self.hadamard_dtype == torch.float16
                and not self._needs_init[layer_idx]
                and hasattr(_HIP, "chunk_q_norm_rope_hadamard"))

    def update_fused_chunk(self, query, key, value, cos, sin, layer_idx,
                           q_weight, k_weight, q_eps, k_eps):
        batch, rows, kv_heads, head_dim = key.shape
        if (not self.can_fuse_chunk(layer_idx, None, rows)
                or batch != self.batch_size or head_dim != 128
                or kv_heads != self.num_kv_heads or query.shape[2] != self.num_q_heads):
            raise ValueError("unsupported fused chunk configuration")
        self._prepare_layer_metadata(layer_idx, rows, None)
        if layer_idx == 0:
            if self.length + rows > self.max_seq_len:
                raise ValueError("KV cache capacity exceeded")
            self.length += rows
            if self._transaction is not None:
                self._transaction.proposed_length = self.length
        specs = self.get_cache_specs_for_flash_infer(None)
        query_out = _HIP.chunk_q_norm_rope_hadamard(
            query, q_weight, cos.contiguous(), sin.contiguous(), q_eps)
        _HIP.chunk_k_norm_rope_append_i4(
            key, value, k_weight, cos.contiguous(), sin.contiguous(),
            specs["kv_data"], specs["kv_param"], specs["kv_indptr"],
            specs["kv_indices"], specs["last_page_offset"], layer_idx, k_eps)
        attention = functools.partial(
            self._stub.forward, num_kv_heads=kv_heads,
            attention_kwargs=(self.get_virtual_cache_specs(rows) if rows > 1 else specs),
            layer_idx=layer_idx, hadamard_applied=True)
        return query_out, attention

    def update_fused_k1(
            self, query_states, key_states, value_states, cos, sin, layer_idx):
        """Decode-only Q/K RoPE plus direct native-GQA KV4 append."""
        batch, added_length, kv_heads, head_dim = key_states.shape
        query_heads = query_states.shape[2]
        self._prepare_layer_metadata(layer_idx, added_length, None)
        if (added_length != 1 or batch != self.batch_size
                or kv_heads != self.num_kv_heads
                or query_heads != self.num_q_heads
                or not self.can_fuse_k1(layer_idx, None)):
            raise ValueError(
                "K1 requires one non-transactional native-GQA decode token")
        if layer_idx == 0:
            new_length = self.length + 1
            if new_length > self.max_seq_len:
                raise ValueError("KV cache capacity exceeded")
            self._ensure_page_cnt_per_batch(
                self.page_cnt_from_length(new_length))
            self.length = new_length
            if self._transaction is not None:
                self._transaction.proposed_length = new_length
        specs = self.get_cache_specs_for_flash_infer(None)
        query_out = _HIP.fused_rope_append_kv_i4(
            query_states.contiguous(), key_states.contiguous(),
            value_states.contiguous(), cos.contiguous(), sin.contiguous(),
            specs["kv_data"], specs["kv_param"], specs["kv_indptr"],
            specs["kv_indices"], specs["last_page_offset"],
            self.n_layers, layer_idx, self.page_size)
        attention = functools.partial(
            self._stub.forward, num_kv_heads=kv_heads,
            attention_kwargs=specs, layer_idx=layer_idx)
        return query_out, attention

    def get_virtual_cache_specs(self, chunk_length):
        """Build causal paged metadata for B*chunk independent decode rows."""
        if (self._active_chunk_metadata is not None and
                self._metadata_prepared_key == (self.length - chunk_length, chunk_length)):
            return self._active_chunk_metadata["causal"]
        if chunk_length < 1:
            raise ValueError("chunk_length must be positive")
        start = self.length - chunk_length
        indptr = [0]
        indices = []
        offsets = []
        for batch in range(self.batch_size):
            for token in range(chunk_length):
                seq_len = start + token + 1
                pages = self.page_cnt_from_length(seq_len)
                indices.extend(page * self.batch_size + batch
                               for page in range(pages))
                indptr.append(len(indices))
                offset = seq_len % self.page_size
                offsets.append(self.page_size if seq_len and offset == 0
                               else offset)
        return {
            "kv_data": self.pages,
            "kv_indptr": torch.tensor(indptr, device=self.device,
                                      dtype=torch.int32),
            "kv_indices": torch.tensor(indices, device=self.device,
                                       dtype=torch.int32),
            "last_page_offset": torch.tensor(offsets, device=self.device,
                                             dtype=torch.int32),
            "kv_param": self.scales,
        }

    def get_cache_specs_for_flash_infer(self, attention_mask):
        if attention_mask is None and self._cuda_graph_decode:
            return self._cuda_graph_specs
        if (attention_mask is None and self._active_chunk_metadata is not None
                and sum(self._metadata_prepared_key) == self.length):
            return self._active_chunk_metadata["append"]
        if attention_mask is None and self._persistent_metadata_enabled:
            if self.length > self.max_seq_len:
                raise ValueError("cache length exceeds preallocated metadata")
            page_cnt = self.page_cnt_from_length(self.length)
            return {
                "kv_data": self.pages,
                "kv_indptr": self._decode_kv_indptr[self.length],
                "kv_indices": self._decode_kv_indices[
                    :page_cnt * self.batch_size],
                "last_page_offset": self._decode_last_page_offset[self.length],
                "kv_param": self.scales,
            }
        if attention_mask is not None:
            seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        else:
            seqlens_in_batch = torch.tensor([self.length], dtype=torch.int32, device=self.device).expand(self.batch_size)
        page_cnt = self.page_cnt_from_length(seqlens_in_batch)
        if (page_cnt[0] != page_cnt).any():
            raise NotImplementedError("Current implementation does not support the case where batches have different number of pages")
        page_cnt = page_cnt[0]
        page_ptr = seqlens_in_batch % self.page_size
        page_ptr = torch.where((seqlens_in_batch != 0) & (page_ptr == 0), self.page_size, page_ptr)
        return {
            f"kv_data": self.pages,
            f"kv_indptr": torch.arange(0, self.batch_size + 1, device=self.device, dtype=torch.int) * page_cnt,
            f"kv_indices": (
                (torch.arange(page_cnt, device=self.device, dtype=torch.int) * self.batch_size).unsqueeze(0) +
                torch.arange(self.batch_size, device=self.device, dtype=torch.int).unsqueeze(1)).view(-1),
            f"last_page_offset": page_ptr, #torch.full((self.batch_size, ), page_ptr, device=self.device, dtype=torch.int),
            f"kv_param": self.scales,
        }

    def enable_cuda_graph_decode(self):
        """Freeze single-page decode metadata addresses for graph replay."""
        if not self._persistent_metadata_enabled:
            raise RuntimeError("graph decode requires persistent KV metadata")
        if self._transaction is not None:
            raise RuntimeError("graph decode cannot start during a transaction")
        if self.page_cnt_from_length(self.max_seq_len) != 1:
            raise NotImplementedError(
                "graph decode currently requires a single cache page")
        next_length = self.length + 1
        if next_length > self.max_seq_len:
            raise ValueError("cache is already at capacity")
        self._cuda_graph_specs = {
            "kv_data": self.pages,
            "kv_indptr": self._decode_kv_indptr[next_length].clone(),
            "kv_indices": self._decode_kv_indices[:self.batch_size],
            "last_page_offset": self._decode_last_page_offset[
                next_length].clone(),
            "kv_param": self.scales,
        }
        self._cuda_graph_decode = True

    def advance_cuda_graph_decode(self):
        if not self._cuda_graph_decode:
            raise RuntimeError("CUDA graph decode metadata is not enabled")
        self._cuda_graph_specs["last_page_offset"].add_(1)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        return self.length

    def get_mask_sizes(self, cache_position: torch.Tensor,
                       layer_idx: int) -> Tuple[int, int]:
        """Transformers >=4.53 causal-mask compatibility for the custom cache."""
        del layer_idx
        return self.length + cache_position.shape[0], 0

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states, if there is any."""
        return None

    def to_legacy_cache(self):
        return self
