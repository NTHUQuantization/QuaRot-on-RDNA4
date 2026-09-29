import pytest
import torch
import quarot
from transformers import LlamaConfig, Qwen2Config, Qwen3Config
from quarot.functional.hadamard import get_hadK
from e2e.quantized_common import (
    GROUPED_FFN_FORMAT, LEGACY_FFN_FORMAT, config_activation_clip_ratio,
    config_ffn_format, config_head_dim)
from e2e.quantized_llama.modeling_llama import (
    QuarotFP16LlamaForCausalLM, QuarotLlamaForCausalLM)
from e2e.quantized_qwen2.modeling_qwen2 import QuarotQwen2ForCausalLM
from e2e.quantized_qwen3.modeling_qwen3 import QuarotQwen3ForCausalLM

@pytest.mark.parametrize("width", [
    4096, 5120, 8192, 11008, 13824, 22016, 28672,
    4864, 8960, 18944, 29568, 3072, 6144, 9728, 12288, 17408, 25600,
    12, 14, 28, 32, 40, 64,
])
def test_supported_model_hadamards_are_orthogonal(width):
    matrix, order = get_hadK(width)
    assert width % order == 0
    assert (width // order) & (width // order - 1) == 0
    if matrix is not None:
        identity = torch.eye(order) * order
        assert torch.allclose(matrix @ matrix.T, identity, atol=2e-4 * order)

@pytest.mark.parametrize("config_cls,model_cls,extra", [
    (LlamaConfig, QuarotLlamaForCausalLM, {}),
    (Qwen2Config, QuarotQwen2ForCausalLM, {}),
    (Qwen3Config, QuarotQwen3ForCausalLM, {"head_dim": 64}),
])
def test_dense_runtime_construction(config_cls, model_cls, extra):
    config = config_cls(
        vocab_size=128, hidden_size=128, intermediate_size=256,
        num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, max_position_embeddings=128, **extra)
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = model_cls(config)
    attention = model.model.layers[0].self_attn
    assert config_head_dim(config) == attention.head_dim
    assert attention.num_key_value_heads == 1
    if config_cls is Qwen3Config:
        assert hasattr(attention, "q_norm") and hasattr(attention, "k_norm")


def _tiny_llama_config(intermediate_size=256):
    config = LlamaConfig(
        vocab_size=128, hidden_size=128,
        intermediate_size=intermediate_size,
        num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, max_position_embeddings=128)
    config._attn_implementation = "eager"
    return config


def test_missing_checkpoint_format_metadata_uses_legacy_contract():
    config = _tiny_llama_config()
    with torch.device("meta"):
        model = QuarotLlamaForCausalLM(config)
    layer = model.model.layers[0]

    assert config_ffn_format(config) == LEGACY_FFN_FORMAT
    assert isinstance(layer.mlp.down_proj, torch.nn.Sequential)
    assert layer.mlp.quantizer.input_clip_ratio == 1.0
    assert isinstance(layer.input_layernorm, quarot.nn.RMSNorm)
    assert isinstance(layer.post_attention_layernorm, quarot.nn.RMSNorm)
    assert isinstance(model.model.norm, quarot.nn.RMSNorm)
    assert layer.self_attn._fused_k1_enabled is False
    assert model.build_cache(1, 8, 16).fused_k1 is False


def test_version_two_uses_grouped_h256_contract(monkeypatch):
    monkeypatch.setenv("QUAROT_FUSED_NORM_QUANT", "1")
    config = _tiny_llama_config(intermediate_size=300)
    config.quarot_checkpoint_format_version = 2
    config.quarot_ffn_format = GROUPED_FFN_FORMAT
    with torch.device("meta"):
        model = QuarotLlamaForCausalLM(config)
    layer = model.model.layers[0]

    assert config_ffn_format(config) == GROUPED_FFN_FORMAT
    assert isinstance(layer.mlp.down_proj, quarot.nn.Linear4bit)
    assert layer.mlp.ffn_physical_size == 512
    assert layer.mlp.down_proj.in_features == 512
    assert layer.mlp.quantizer.input_clip_ratio == 0.9
    assert isinstance(layer.input_layernorm, quarot.nn.FusedRMSNormQuant)
    assert isinstance(
        layer.post_attention_layernorm, quarot.nn.FusedRMSNormQuant)
    assert isinstance(model.model.norm, quarot.nn.RMSNorm)
    assert layer.self_attn._fused_k1_enabled is True
    assert model.build_cache(1, 8, 16).fused_k1 is True


@pytest.mark.parametrize("attrs", [
    {"quarot_ffn_format": GROUPED_FFN_FORMAT},
    {"quarot_checkpoint_format_version": 1},
    {"quarot_checkpoint_format_version": 3,
     "quarot_ffn_format": GROUPED_FFN_FORMAT},
    {"quarot_checkpoint_format_version": 2},
    {"quarot_checkpoint_format_version": 2,
     "quarot_ffn_format": LEGACY_FFN_FORMAT},
    {"quarot_checkpoint_format_version": 2,
     "quarot_ffn_format": "unknown"},
])
def test_unknown_or_mismatched_checkpoint_format_fails_fast(attrs):
    from types import SimpleNamespace

    with pytest.raises(ValueError):
        config_ffn_format(SimpleNamespace(**attrs))


def test_supported_checkpoint_format_combinations_are_explicit():
    from types import SimpleNamespace

    legacy = SimpleNamespace()
    grouped = SimpleNamespace(
        quarot_checkpoint_format_version=2,
        quarot_ffn_format=GROUPED_FFN_FORMAT)
    assert config_ffn_format(legacy) == LEGACY_FFN_FORMAT
    assert config_ffn_format(SimpleNamespace(
        quarot_ffn_format=LEGACY_FFN_FORMAT)) == LEGACY_FFN_FORMAT
    assert config_ffn_format(grouped) == GROUPED_FFN_FORMAT
    assert config_activation_clip_ratio(legacy) == 1.0
    assert config_activation_clip_ratio(grouped) == 0.9


@pytest.mark.parametrize("attrs", [
    {"quarot_activation_clip_ratio": 0.9},
    {"quarot_checkpoint_format_version": 2,
     "quarot_ffn_format": GROUPED_FFN_FORMAT,
     "quarot_activation_clip_ratio": 1.0},
])
def test_checkpoint_format_rejects_mismatched_activation_clip(attrs):
    from types import SimpleNamespace

    with pytest.raises(ValueError):
        config_activation_clip_ratio(SimpleNamespace(**attrs))


@pytest.mark.parametrize("grouped", [False, True])
def test_fused_projection_and_k1_environment_fallbacks(monkeypatch, grouped):
    monkeypatch.setenv("QUAROT_FUSED_PROJECTIONS", "0")
    monkeypatch.setenv("QUAROT_FUSED_K1", "0")
    config = _tiny_llama_config(intermediate_size=300 if grouped else 256)
    if grouped:
        config.quarot_checkpoint_format_version = 2
        config.quarot_ffn_format = GROUPED_FFN_FORMAT
    with torch.device("meta"):
        model = QuarotLlamaForCausalLM(config)

    layer = model.model.layers[0]
    assert layer.self_attn._fused_projections_enabled is False
    assert layer.self_attn._fused_k1_enabled is False
    assert layer.mlp._fused_projections_enabled is False
    cache = model.build_cache(
        batch_size=1, page_size=8, max_length=16)
    assert cache.fused_k1 is False


def test_fp16_reference_does_not_apply_rotated_output_hadamard():
    config = LlamaConfig(
        vocab_size=128, hidden_size=128, intermediate_size=256,
        num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, max_position_embeddings=128)
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = QuarotFP16LlamaForCausalLM(config)
    assert not model.model.layers[0].self_attn._quarot_quantized


@pytest.mark.parametrize("config_cls,model_cls,hidden,ffn,heads,kv_heads", [
    (LlamaConfig, QuarotLlamaForCausalLM, 4096, 11008, 32, 32),
    (LlamaConfig, QuarotLlamaForCausalLM, 5120, 13824, 40, 40),
    (LlamaConfig, QuarotLlamaForCausalLM, 8192, 28672, 64, 8),
    (LlamaConfig, QuarotLlamaForCausalLM, 8192, 22016, 64, 8),
    (Qwen3Config, QuarotQwen3ForCausalLM, 5120, 25600, 64, 8),
    (Qwen2Config, QuarotQwen2ForCausalLM, 5120, 27648, 40, 8),
])
def test_benchmark_models_select_int4_fusions(
        config_cls, model_cls, hidden, ffn, heads, kv_heads):
    kwargs = dict(
        vocab_size=128, hidden_size=hidden, intermediate_size=ffn,
        num_hidden_layers=1, num_attention_heads=heads,
        num_key_value_heads=kv_heads, max_position_embeddings=128)
    if config_cls is Qwen3Config:
        kwargs["head_dim"] = 128
    config = config_cls(**kwargs)
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = model_cls(config)
    layer = model.model.layers[0]
    assert layer.self_attn._fused_attention_output
    assert layer.mlp._fused_ffn


def test_qwen3_32b_exact_kernel_dimensions_are_specialized():
    config = Qwen3Config(
        vocab_size=151936, hidden_size=5120, intermediate_size=25600,
        num_hidden_layers=1, num_attention_heads=64,
        num_key_value_heads=8, head_dim=128,
        max_position_embeddings=40960)
    config._attn_implementation = "eager"
    config.quarot_checkpoint_format_version = 2
    config.quarot_ffn_format = GROUPED_FFN_FORMAT
    with torch.device("meta"):
        model = QuarotQwen3ForCausalLM(config)

    layer = model.model.layers[0]
    attention = layer.self_attn
    mlp = layer.mlp
    assert (attention.q_proj.in_features,
            attention.q_proj.out_features) == (5120, 8192)
    assert (attention.k_proj.in_features,
            attention.k_proj.out_features) == (5120, 1024)
    assert (attention.v_proj.in_features,
            attention.v_proj.out_features) == (5120, 1024)
    assert (attention.o_proj[1].in_features,
            attention.o_proj[1].out_features) == (8192, 5120)
    assert (mlp.gate_proj.in_features,
            mlp.gate_proj.out_features) == (5120, 25600)
    assert (mlp.down_proj.in_features,
            mlp.down_proj.out_features) == (25600, 5120)
    assert mlp.ffn_physical_size == 25600
    assert "Qwen3-32B" in quarot._HIP.selected_bpre_kernel_name(5120)
    assert "Qwen3-32B" in quarot._HIP.selected_bpre_kernel_name(25600)


@pytest.mark.parametrize("width", [
    11008, 13824, 14336, 22016, 25600, 27648, 28672, 29568,
])
def test_every_phase_uses_universal_ffn_dispatch(width):
    from types import SimpleNamespace
    from e2e.quantized_common import QuarotMLPMixin

    mlp = SimpleNamespace(
        intermediate_size=width,
        _fused_ffn=True,
        _quarot_ffn_format=GROUPED_FFN_FORMAT)
    decide = QuarotMLPMixin._should_use_fused_ffn
    assert decide(mlp, torch.empty(1, 2048, 1)) is True
    assert decide(mlp, torch.empty(1, 1, 1)) is True


def test_universal_ffn_pads_physical_width_to_h256():
    from quarot.functional.hadamard import grouped_ffn_physical_width

    assert grouped_ffn_physical_width(29568) == 29696
    config = LlamaConfig(
        vocab_size=128, hidden_size=128, intermediate_size=29568,
        num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, max_position_embeddings=128)
    config._attn_implementation = "eager"
    config.quarot_checkpoint_format_version = 2
    config.quarot_ffn_format = GROUPED_FFN_FORMAT
    with torch.device("meta"):
        model = QuarotLlamaForCausalLM(config)
    mlp = model.model.layers[0].mlp
    assert mlp.ffn_physical_size == 29696
    assert mlp.gate_proj.out_features == 29696
    assert mlp.up_proj.out_features == 29696
    assert mlp.down_proj.in_features == 29696


def test_padded_grouped_h256_preserves_dense_down_projection():
    from quarot.functional.hadamard import (
        grouped_ffn_physical_width, matmul_grouped_h256)

    torch.manual_seed(256)
    logical, hidden = 300, 32
    physical = grouped_ffn_physical_width(logical)
    activation = torch.randn(3, logical, dtype=torch.float64)
    weight = torch.randn(hidden, logical, dtype=torch.float64)
    activation = torch.nn.functional.pad(
        activation, (0, physical - logical))
    weight = torch.nn.functional.pad(weight, (0, physical - logical))
    expected = activation @ weight.T
    rotated_activation = matmul_grouped_h256(activation)
    rotated_weight = matmul_grouped_h256(weight)
    torch.testing.assert_close(
        rotated_activation @ rotated_weight.T, expected,
        rtol=1e-12, atol=1e-12)
