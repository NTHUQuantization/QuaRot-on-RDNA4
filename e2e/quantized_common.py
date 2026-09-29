"""Shared dense decoder runtime for rotated Llama, Qwen2, and Qwen3."""
import os
import torch
from transformers.modeling_flash_attention_utils import _flash_attention_forward
import quarot
import quarot.transformers

def config_head_dim(config):
    return getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

LEGACY_FFN_FORMAT = "legacy_hadk"
GROUPED_FFN_FORMAT = "grouped_h256_v1"


def config_ffn_format(config):
    version = getattr(config, "quarot_checkpoint_format_version", None)
    value = getattr(config, "quarot_ffn_format", None)
    if version is None:
        if value not in (None, LEGACY_FFN_FORMAT):
            raise ValueError(
                "non-legacy QuaRot FFN format requires checkpoint format "
                "version 2")
        return LEGACY_FFN_FORMAT
    if version != 2:
        raise ValueError(
            f"unsupported QuaRot checkpoint format version {version!r}")
    if value != GROUPED_FFN_FORMAT:
        raise ValueError(
            "QuaRot checkpoint format version 2 requires "
            f"quarot_ffn_format={GROUPED_FFN_FORMAT!r}, got {value!r}")
    return GROUPED_FFN_FORMAT


def config_activation_clip_ratio(config):
    ffn_format = config_ffn_format(config)
    expected = 0.9 if ffn_format == GROUPED_FFN_FORMAT else 1.0
    value = float(getattr(
        config, "quarot_activation_clip_ratio", expected))
    if value != expected:
        raise ValueError(
            f"QuaRot FFN format {ffn_format!r} requires "
            f"quarot_activation_clip_ratio={expected}, got {value}")
    return expected


def packed_logical_shape(value):
    if value.logical_shape is not None:
        return value.logical_shape
    storage_shape = value.quantized_x.shape
    return (*storage_shape[:-1], storage_shape[-1] * 2)


class QuarotAttentionMixin:
    def _init_quarot_attention(self, quantized):
        self._quarot_quantized = quantized
        self._quarot_ffn_format = config_ffn_format(self.config)
        self._quarot_clip_ratio = config_activation_clip_ratio(self.config)
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.hidden_size = self.config.hidden_size
        self.quantizer = (quarot.nn.Quantizer(self._quarot_clip_ratio)
                          if quantized else torch.nn.Identity())
        self.o_proj_hadamard = quarot.nn.OnlineHadamard(self.num_heads)
        if (self._quarot_ffn_format == LEGACY_FFN_FORMAT and
                self.o_proj_hadamard.had_rem_dim is not None):
            self.o_proj_hadamard._non_persistent_buffers_set.discard(
                "had_rem_dim")
        if quantized:
            self.q_proj = quarot.nn.Linear4bit.from_float(self.q_proj)
            self.k_proj = quarot.nn.Linear4bit.from_float(self.k_proj)
            self.v_proj = quarot.nn.Linear4bit.from_float(self.v_proj)
            self.o_proj = torch.nn.Sequential(
                quarot.nn.Quantizer(self._quarot_clip_ratio),
                quarot.nn.Linear4bit.from_float(self.o_proj))
        self._fused_attention_output = quantized and self.hidden_size <= 8192
        self._fused_projections_enabled = (
            quantized and
            os.getenv("QUAROT_FUSED_PROJECTIONS", "1") != "0")
        k1_override = os.getenv("QUAROT_FUSED_K1")
        self._fused_k1_enabled = (
            quantized and (
                self._quarot_ffn_format == GROUPED_FFN_FORMAT
                if k1_override is None
                else k1_override != "0"))

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_value=None, cache_position=None, **kwargs):
        # Transformers 4.57 Qwen3 passes the cache as past_key_values
        # (plural), while Llama/Qwen2 releases used the singular spelling.
        # Normalize both names here; otherwise Qwen3 silently runs every token
        # as a cache-less prefill and never updates the paged KV4 state.
        plural_cache = kwargs.pop("past_key_values", None)
        if past_key_value is None:
            past_key_value = plural_cache
        if isinstance(hidden_states, quarot.PackedQuantizedTensor):
            bsz, q_len, _ = packed_logical_shape(hidden_states)
        else:
            bsz, q_len, _ = hidden_states.shape
        hidden_states = self.quantizer(hidden_states)
        shape = (bsz, q_len, -1, self.head_dim)
        if self._fused_projections_enabled:
            query_states, key_states, value_states = (
                quarot.nn.Linear4bit.fused_forward(
                    hidden_states, self.q_proj, self.k_proj, self.v_proj))
            query_states = query_states.view(shape)
            key_states = key_states.view(shape)
            value_states = value_states.view(shape)
        else:
            query_states = self.q_proj(hidden_states).view(shape)
            key_states = self.k_proj(hidden_states).view(shape)
            value_states = self.v_proj(hidden_states).view(shape)
        cos, sin = position_embeddings
        use_chunk_preprocess = (
            os.getenv("QUAROT_CHUNK_PREPROCESS", "0") != "0"
            and hasattr(self, "q_norm") and past_key_value is not None
            # Preserve the existing grouped-checkpoint M=1 rounding path.
            and not (q_len == 1 and self._fused_k1_enabled)
            and query_states.dtype == torch.float16 and self.head_dim == 128
            and self.q_norm.weight.dtype == torch.float16
            and self.k_norm.weight.dtype == torch.float16
            and self.q_norm.weight.is_contiguous() and self.k_norm.weight.is_contiguous()
            and cos.dtype == torch.float16 and sin.dtype == torch.float16
            and cos.numel() == bsz * q_len * self.head_dim
            and hasattr(past_key_value, "can_fuse_chunk")
            and past_key_value.can_fuse_chunk(self.layer_idx, attention_mask, q_len))
        if hasattr(self, "q_norm") and not use_chunk_preprocess:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)
        use_fused_k1 = (
            self._fused_k1_enabled and past_key_value is not None and
            q_len == 1 and query_states.dtype == torch.float16 and
            cos.numel() == bsz * self.head_dim and
            hasattr(past_key_value, "can_fuse_k1") and
            past_key_value.can_fuse_k1(self.layer_idx, attention_mask))
        if use_chunk_preprocess:
            query_states, cache_out = past_key_value.update_fused_chunk(
                query_states, key_states, value_states, cos, sin, self.layer_idx,
                self.q_norm.weight, self.k_norm.weight,
                self.q_norm.variance_epsilon, self.k_norm.variance_epsilon)
            attn_output = cache_out(query_states)
        elif use_fused_k1:
            query_states, cache_out = past_key_value.update_fused_k1(
                query_states, key_states, value_states, cos, sin,
                self.layer_idx)
            attn_output = cache_out(query_states)
        else:
            query_states, key_states = self._quarot_apply_rotary(
                query_states, key_states, cos, sin, unsqueeze_dim=2)
            if past_key_value is None:
                attn_output = _flash_attention_forward(
                    query_states, key_states, value_states, attention_mask,
                    query_length=q_len, is_causal=True,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    position_ids=kwargs.get("position_ids"), softmax_scale=self.scaling,
                    sliding_window=getattr(self, "sliding_window", None),
                    attn_implementation="flash_attention_2")
            else:
                cache_out = past_key_value.update(
                    key_states, value_states, self.layer_idx,
                    {"sin": sin, "cos": cos, "cache_position": cache_position,
                     "attention_mask": attention_mask})
                if isinstance(cache_out, tuple):
                    cached_key, cached_value = cache_out
                    attn_output = _flash_attention_forward(
                        query_states, cached_key, cached_value, attention_mask,
                        query_length=q_len, is_causal=True,
                        dropout=0.0 if not self.training else self.attention_dropout,
                        position_ids=kwargs.get("position_ids"),
                        softmax_scale=self.scaling,
                        sliding_window=getattr(self, "sliding_window", None),
                        attn_implementation="flash_attention_2")
                else:
                    attn_output = cache_out(query_states)
        if self._fused_attention_output:
            if self.o_proj_hadamard.had_rem_dim is None:
                packed, scales = quarot._HIP.fused_attention_hadamard_quant(
                    attn_output.contiguous(), self.num_heads)
            else:
                packed, scales = quarot._HIP.fused_attention_hadamard_quant_general(
                    attn_output.contiguous(), self.num_heads,
                    self.o_proj_hadamard.had_rem_dim.contiguous())
            attn_output = self.o_proj[1](
                quarot.PackedQuantizedTensor(packed, scales))
        elif self._quarot_quantized:
            attn_output = self.o_proj_hadamard(
                attn_output.transpose(-1, -2)).transpose(-1, -2)
            attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
            attn_output = self.o_proj(attn_output)
        else:
            # The FP16 baseline loads ordinary, unrotated Hugging Face
            # weights. Applying the online Hadamard here is only valid when
            # o_proj was transformed by the offline checkpoint converter.
            attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
            attn_output = self.o_proj(attn_output)
        return attn_output, None

class QuarotMLPMixin:
    # Measured at B=1, S=2048 on gfx1201. These widths are faster in prefill
    # with the library FHT + quantizer pipeline; decode remains fused.
    _unfused_prefill_widths = frozenset((11008, 13824, 22016))

    def _init_quarot_mlp(self):
        self._quarot_ffn_format = config_ffn_format(self.config)
        self._quarot_clip_ratio = config_activation_clip_ratio(self.config)
        self.quantizer = quarot.nn.Quantizer(self._quarot_clip_ratio)
        self._fused_projections_enabled = (
            os.getenv("QUAROT_FUSED_PROJECTIONS", "1") != "0")
        if self._quarot_ffn_format == GROUPED_FFN_FORMAT:
            physical = quarot.functional.hadamard.grouped_ffn_physical_width(
                self.intermediate_size)
            up, gate, down = self.up_proj, self.gate_proj, self.down_proj
            self.ffn_physical_size = physical
            self.up_proj = quarot.nn.Linear4bit(
                up.in_features, physical, bias=up.bias is not None,
                dtype=up.weight.dtype)
            self.gate_proj = quarot.nn.Linear4bit(
                gate.in_features, physical, bias=gate.bias is not None,
                dtype=gate.weight.dtype)
            self.down_proj = quarot.nn.Linear4bit(
                physical, down.out_features, bias=down.bias is not None,
                dtype=down.weight.dtype)
            self._fused_ffn = True
            return

        self.ffn_physical_size = self.intermediate_size
        self.up_proj = quarot.nn.Linear4bit.from_float(self.up_proj)
        self.gate_proj = quarot.nn.Linear4bit.from_float(self.gate_proj)
        self.down_proj = torch.nn.Sequential(
            quarot.nn.OnlineHadamard(self.intermediate_size),
            quarot.nn.Quantizer(self._quarot_clip_ratio),
            quarot.nn.Linear4bit.from_float(self.down_proj))
        hadamard = self.down_proj[0]
        if hadamard.had_rem_dim is not None:
            hadamard._non_persistent_buffers_set.discard("had_rem_dim")
        inner = self.intermediate_size // hadamard.rem_dim
        self._fused_ffn = (
            self.intermediate_size <= 28672 and self.intermediate_size % 2 == 0
            and inner in (32, 64, 128, 256, 512, 1024))

    def _should_use_fused_ffn(self, x):
        if self._quarot_ffn_format == GROUPED_FFN_FORMAT:
            return True
        if isinstance(x, quarot.PackedQuantizedTensor):
            logical_shape = packed_logical_shape(x)
        else:
            logical_shape = x.shape
        is_prefill = len(logical_shape) >= 3 and logical_shape[-2] > 1
        return self._fused_ffn and not (
            is_prefill and
            self.intermediate_size in self._unfused_prefill_widths)

    def _project_gate_up(self, x):
        if self._fused_projections_enabled:
            return quarot.nn.Linear4bit.fused_forward(
                x, self.gate_proj, self.up_proj)
        return self.gate_proj(x), self.up_proj(x)

    def forward(self, x):
        use_fused = self._should_use_fused_ffn(x)
        x = self.quantizer(x)
        if not use_fused:
            return super().forward(x)
        if self._quarot_ffn_format == GROUPED_FFN_FORMAT:
            gate, up = self._project_gate_up(x)
            packed, scales = quarot._HIP.fused_ffn_silu_hadamard_quant_grouped256(
                gate.contiguous(), up.contiguous())
            return self.down_proj(quarot.PackedQuantizedTensor(
                packed, scales, logical_shape=gate.shape))

        gate, up = self._project_gate_up(x)
        hadamard = self.down_proj[0]
        if hadamard.had_rem_dim is None:
            packed, scales = quarot._HIP.fused_ffn_silu_hadamard_quant(
                gate.contiguous(), up.contiguous())
        elif (hadamard.rem_dim <= 48 and
              self.intermediate_size // hadamard.rem_dim <= 1024):
            packed, scales = (
                quarot._HIP.fused_ffn_silu_hadamard_quant_single_fp16lds(
                    gate.contiguous(), up.contiguous(),
                    hadamard.had_rem_dim.contiguous()))
        else:
            packed, scales = quarot._HIP.fused_ffn_silu_hadamard_quant_general(
                gate.contiguous(), up.contiguous(),
                hadamard.had_rem_dim.contiguous())
        return self.down_proj[2](quarot.PackedQuantizedTensor(
            packed, scales, logical_shape=gate.shape))

class QuarotCausalLMMixin:
    def _init_quarot_model(self, attention_cls, mlp_cls=None, norm_cls=None):
        ffn_format = config_ffn_format(self.config)
        clip_ratio = config_activation_clip_ratio(self.config)
        use_fused_norm = (
            ffn_format == GROUPED_FFN_FORMAT and
            norm_cls is quarot.nn.FusedRMSNormQuant and
            os.getenv("QUAROT_FUSED_NORM_QUANT", "1") != "0")
        layer_norm_cls = (quarot.nn.FusedRMSNormQuant
                          if use_fused_norm else quarot.nn.RMSNorm)
        norm_kwargs = ({"input_clip_ratio": clip_ratio}
                       if use_fused_norm else {})
        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn = attention_cls(self.config, layer_idx)
            if mlp_cls is not None:
                layer.mlp = mlp_cls(self.config)
                layer.input_layernorm = layer_norm_cls(
                    self.config.hidden_size, eps=self.config.rms_norm_eps,
                    **norm_kwargs)
                layer.post_attention_layernorm = layer_norm_cls(
                    self.config.hidden_size, eps=self.config.rms_norm_eps,
                    **norm_kwargs)
        if mlp_cls is not None:
            self.model.norm = quarot.nn.RMSNorm(
                self.config.hidden_size, eps=self.config.rms_norm_eps)
        self._expected_max_length = None

    def build_cache(self, batch_size, page_size, max_length, native_gqa=True,
                    fused_decode_append=True):
        projection = self.model.layers[0].self_attn.v_proj
        if isinstance(projection, quarot.nn.Linear4bit):
            device, dtype = projection.weight.device, torch.float16
        else:
            device, dtype = projection.weight.device, projection.weight.dtype
        disable_quant = self.cache_dtype == "float16"
        return quarot.transformers.MultiLayerPagedKVCache4Bit(
            batch_size=batch_size, page_size=page_size, max_seq_len=max_length,
            device=device, n_layers=len(self.model.layers),
            num_heads=self.config.num_attention_heads,
            num_kv_heads=self.config.num_key_value_heads,
            native_gqa=native_gqa,
            fused_decode_append=fused_decode_append,
            fused_k1=self.model.layers[0].self_attn._fused_k1_enabled,
            head_dim=config_head_dim(self.config), disable_quant=disable_quant,
            hadamard_dtype=None if disable_quant else dtype)

    def _get_logits_processor(self, generation_config, *args, **kwargs):
        self._expected_max_length = generation_config.max_length
        return super()._get_logits_processor(generation_config, *args, **kwargs)

    def forward(self, input_ids=None, *args, past_key_values=None, **kwargs):
        use_cache = kwargs.get("use_cache", self.config.use_cache)
        if past_key_values is None and use_cache and input_ids is not None:
            max_length = self._expected_max_length or input_ids.shape[1]
            self._expected_max_length = None
            past_key_values = self.build_cache(
                input_ids.shape[0], max_length, max_length)
        if (past_key_values is not None and input_ids is not None
                and kwargs.get("attention_mask") is None
                and hasattr(past_key_values, "prepare_verification_metadata")):
            positions = past_key_values.prepare_verification_metadata(input_ids.shape[1])
            # Explicit positions remain authoritative for eager callers.
            if positions is not None and (kwargs.get("cache_position") is None
                    or past_key_values._verification_graph_capture):
                kwargs["cache_position"] = positions
        return super().forward(
            input_ids, *args, past_key_values=past_key_values, **kwargs)
