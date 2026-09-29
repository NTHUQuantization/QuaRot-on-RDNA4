"""Configuration and family-dispatch coverage for Llama 3.1."""
from transformers import LlamaConfig

from e2e.checkpoint_utils.quantize_checkpoint import runtime_types
from e2e.model_registry import RUNTIMES


def test_llama31_config_uses_supported_llama_runtime_and_gqa_geometry():
    config = LlamaConfig(
        hidden_size=4096, intermediate_size=14336, num_hidden_layers=32,
        num_attention_heads=32, num_key_value_heads=8, head_dim=128,
        max_position_embeddings=131072, rope_theta=500000.0,
        rope_scaling={
            "rope_type": "llama3", "factor": 8.0,
            "low_freq_factor": 1.0, "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
    )
    config_cls, model_cls, output_type, _ = runtime_types(config.model_type)

    assert config.model_type == "llama"
    assert config.num_attention_heads == 32
    assert config.num_key_value_heads == 8
    assert config.num_attention_heads // config.num_key_value_heads == 4
    assert config.head_dim == 128
    assert config.rope_scaling["rope_type"] == "llama3"
    assert output_type == "llama_quarot"
    assert config_cls is RUNTIMES["llama"][0]
    assert model_cls is RUNTIMES["llama"][1]
