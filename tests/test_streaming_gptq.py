import json
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from e2e.checkpoint_utils import quantize_checkpoint
from e2e.checkpoint_utils import streaming_gptq
from e2e.checkpoint_utils import streaming_rtn
from e2e.checkpoint_utils import rotation_utils
from quarot.functional.hadamard import (
    matmul_grouped_h256, matmul_hadU, random_hadamard_matrix)


class _ToyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.o_proj = torch.nn.Linear(4, 4, bias=False)
        self.mlp = torch.nn.Module()
        self.mlp.down_proj = torch.nn.Linear(256, 256, bias=False)
        with torch.no_grad():
            self.self_attn.o_proj.weight.copy_(torch.eye(4))
            self.mlp.down_proj.weight.copy_(torch.eye(256))


class _ToyNormLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.q_norm = torch.nn.LayerNorm(
            4, elementwise_affine=True, dtype=torch.float16)
        self.self_attn.k_norm = torch.nn.LayerNorm(
            4, elementwise_affine=True, dtype=torch.float16)


def test_pack_layer_preserves_source_norm_dtype_and_values():
    prefix = "model.layers.0."
    layer = _ToyNormLayer()
    q_norm = torch.tensor(
        [0.5, 0.75, 1.0, 1.25], dtype=torch.bfloat16)
    k_norm = torch.tensor(
        [0.625, 0.875, 1.125, 1.375], dtype=torch.bfloat16)
    preserved = {
        prefix + "self_attn.q_norm.weight": q_norm,
        prefix + "self_attn.k_norm.weight": k_norm,
    }
    packed = streaming_gptq._pack_layer(
        layer, prefix, quantizers={}, preserved_tensors=preserved)
    assert packed[prefix + "self_attn.q_norm.weight"].dtype == torch.bfloat16
    assert packed[prefix + "self_attn.k_norm.weight"].dtype == torch.bfloat16
    assert torch.equal(packed[prefix + "self_attn.q_norm.weight"], q_norm)
    assert torch.equal(packed[prefix + "self_attn.k_norm.weight"], k_norm)


def test_calibration_layer_installs_runtime_hadamards_without_changing_keys(
        monkeypatch):
    layer = _ToyLayer()
    original_keys = tuple(layer.state_dict())
    config = SimpleNamespace(
        num_attention_heads=2, hidden_size=4, head_dim=2,
        intermediate_size=256)

    # A last-dimension reversal makes the attention head-axis transform easy
    # to distinguish from an incorrect full-hidden-dimension transform.
    monkeypatch.setattr(
        streaming_gptq, "_apply_hadamard",
        lambda x, had_rem_dim, rem_dim: x.flip(-1))
    streaming_gptq._install_online_hadamards(layer, config)

    x = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4)
    attention = layer.self_attn.o_proj(x)
    mlp_x = torch.arange(256, dtype=torch.float32).reshape(1, 1, 256)
    mlp = layer.mlp.down_proj(mlp_x)

    assert torch.equal(attention, torch.tensor([[[2., 3., 0., 1.]]]))
    assert torch.equal(mlp, matmul_grouped_h256(mlp_x))
    assert tuple(layer.state_dict()) == original_keys
    assert dict(layer.named_modules())["self_attn.o_proj"] is layer.self_attn.o_proj
    assert dict(layer.named_modules())["mlp.down_proj"] is layer.mlp.down_proj


import pytest
from transformers import LlamaConfig, Qwen2Config, Qwen3Config
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer


@pytest.mark.parametrize(
    ("config", "decoder_cls"),
    [
        (LlamaConfig(
            hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2),
         LlamaDecoderLayer),
        (Qwen2Config(
            hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2),
         Qwen2DecoderLayer),
        (Qwen3Config(
            hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8),
         Qwen3DecoderLayer),
    ],
)
def test_supported_family_layer_preserves_state_and_installs_hooks(
        config, decoder_cls):
    source = decoder_cls(config, 0).to(dtype=torch.float16)
    rotation_utils.pad_mlp_modules(source)
    prefix = "model.layers.0."
    tensors = {prefix + key: value.clone()
               for key, value in source.state_dict().items()}

    actual = streaming_gptq._layer_from_tensors(
        config, 0, tensors, prefix)

    assert isinstance(actual, decoder_cls)
    assert actual.state_dict().keys() == source.state_dict().keys()
    assert len(actual.self_attn.o_proj._forward_pre_hooks) == 1
    assert len(actual.mlp.down_proj._forward_pre_hooks) == 1
    _, rotary_cls = streaming_gptq._calibration_types(config)
    rotary = rotary_cls(config=config)
    x = torch.zeros(1, 4, config.hidden_size, dtype=torch.float16)
    positions = torch.arange(4).unsqueeze(0)
    cos, sin = rotary(x, positions)
    expected_head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads)
    assert cos.shape[-1] == expected_head_dim
    assert sin.shape == cos.shape


def test_unsupported_gptq_family_is_rejected():
    config = SimpleNamespace(model_type="unsupported")
    with pytest.raises(ValueError, match="unsupported"):
        streaming_gptq._calibration_types(config)

def test_fp32_embedding_produces_fp16_calibration_activations(tmp_path):
    global_path = tmp_path / "model-global.safetensors"
    embedding = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    save_file({"model.embed_tokens.weight": embedding}, global_path)
    loader = [
        (torch.tensor([[0, 3, 7]], dtype=torch.long),),
        (torch.tensor([[2, 5, 1]], dtype=torch.long),),
    ]
    config = SimpleNamespace(hidden_size=4)

    inps, outs = streaming_gptq._initial_activations(
        global_path, loader, config, torch.device("cpu"))

    assert inps.dtype == torch.float16
    assert outs.dtype == torch.float16
    assert torch.equal(inps[0], embedding[loader[0][0][0]].half())


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(torch.device("cpu"), id="cpu"),
        pytest.param(
            torch.device("cuda", 0), id="cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA/ROCm GPU required")),
    ],
)
def test_programmatic_resume_recreates_rotation_without_consuming_caller_rng(
        device):
    """RNG drift between partial conversion runs must not change Q."""
    config = SimpleNamespace(hidden_size=32)
    args = SimpleNamespace(seed=719)

    if device.type == "cuda":
        torch.cuda.manual_seed(11)
        torch.rand(17, device=device)
        state_before_first = torch.cuda.get_rng_state(device)
    else:
        torch.manual_seed(11)
        torch.rand(17)
        state_before_first = torch.random.get_rng_state()

    first = streaming_rtn._seeded_rotation_matrix(
        config, args, device, torch.float32)

    if device.type == "cuda":
        state_after_first = torch.cuda.get_rng_state(device)
        torch.cuda.manual_seed(991)
        torch.rand(29, device=device)
        state_before_resume = torch.cuda.get_rng_state(device)
    else:
        state_after_first = torch.random.get_rng_state()
        torch.manual_seed(991)
        torch.rand(29)
        state_before_resume = torch.random.get_rng_state()
    assert torch.equal(state_after_first, state_before_first)

    resumed = streaming_rtn._seeded_rotation_matrix(
        config, args, device, torch.float32)

    if device.type == "cuda":
        state_after_resume = torch.cuda.get_rng_state(device)
    else:
        state_after_resume = torch.random.get_rng_state()
    assert torch.equal(state_after_resume, state_before_resume)
    assert torch.equal(resumed, first)


def test_seeded_generalized_rotation_retains_exact_random_signs():
    config = SimpleNamespace(hidden_size=40)
    args = SimpleNamespace(seed=913)
    q, signs = streaming_rtn._seeded_rotation(
        config, args, torch.device("cpu"), torch.float32)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed)
        legacy = random_hadamard_matrix(
            config.hidden_size, torch.device("cpu"), dtype=torch.float32)

    torch.manual_seed(18)
    values = torch.randn(3, config.hidden_size)
    expected = matmul_hadU(values * signs.to(torch.float32))
    actual = values @ q

    assert signs.dtype == torch.int8
    assert signs.shape == (config.hidden_size,)
    assert set(signs.tolist()) == {-1, 1}
    assert torch.equal(q, legacy)
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_rotation_checkpoint_metadata_records_qwen3_32b_hadk_contract():
    class FakeSource:
        @staticmethod
        def tensors(keys):
            assert keys == ["model.norm.weight"]
            return {"model.norm.weight": torch.arange(
                5120, dtype=torch.bfloat16)}

    signs = torch.ones(5120, dtype=torch.int8)
    signs[1::2] = -1
    metadata = streaming_rtn._rotation_checkpoint_metadata(
        FakeSource(), SimpleNamespace(hidden_size=5120), signs,
        SimpleNamespace(
            seed=0, rotation_device="cuda", rotation_dtype="float32"))

    assert metadata["quarot_rotation_format"] == "hadk_v1"
    assert metadata["quarot_rotation_width"] == 5120
    assert metadata["quarot_rotation_remainder"] == 40
    assert metadata["quarot_rotation_inner"] == 128
    assert metadata["quarot_rotation_signs"][:4] == [1, -1, 1, -1]
    assert len(metadata["quarot_final_norm_weight"]) == 5120


@pytest.mark.parametrize(
    ("signs", "norm", "message"),
    [
        ([1.5, -1, 1, -1], [1] * 4, r"only -1 or \+1"),
        ([1, -1, 1], [1] * 4, "rotation sign width"),
        ([1, -1, 1, -1], [1] * 3, "final norm width"),
        ([1, -1, 1, -1], [1, 1, float("nan"), 1], "non-finite"),
    ],
)
def test_rotation_checkpoint_metadata_rejects_corrupt_basis(
        signs, norm, message):
    class FakeSource:
        def tensors(self, keys):
            assert keys == ["model.norm.weight"]
            return {"model.norm.weight": torch.tensor(norm)}

    args = SimpleNamespace(
        seed=0, rotation_device="cpu", rotation_dtype="float32")
    with pytest.raises(ValueError, match=message):
        streaming_rtn._rotation_checkpoint_metadata(
            FakeSource(), SimpleNamespace(hidden_size=4),
            torch.tensor(signs), args)


def test_finalize_output_records_streaming_checkpoint_contract(
        tmp_path, monkeypatch):
    class FakeConfig:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            assert model == "source/model"
            assert kwargs == {"attn_implementation": "flash_attention_2"}
            return cls()

        def save_pretrained(self, output):
            output.mkdir(parents=True, exist_ok=True)
            (output / "config.json").write_text(
                json.dumps({"hidden_size": 32}) + "\n")

    class FakeRuntime:
        pass

    class FakeTokenizer:
        def save_pretrained(self, output):
            (output / "tokenizer_config.json").write_text("{}\n")

    monkeypatch.setattr(
        quantize_checkpoint.transformers, "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda model: FakeTokenizer()))
    runtime_source = tmp_path / "runtime_source.py"
    runtime_source.write_text("# runtime source\n")
    output = tmp_path / "converted"

    quantize_checkpoint.finalize_output(
        output, "source/model", FakeConfig, FakeRuntime, "qwen3_quarot",
        SimpleNamespace(__file__=str(runtime_source)), {
            "streaming_signature": {"method": "rtn", "seed": 0},
            "rotation_metadata": {
                "quarot_rotation_format": "hadk_v1",
                "quarot_rotation_width": 32,
                "quarot_rotation_remainder": 1,
                "quarot_rotation_inner": 32,
                "quarot_rotation_seed": 0,
                "quarot_rotation_device": "cuda",
                "quarot_rotation_dtype": "float32",
                "quarot_rotation_signs": [1] * 32,
                "quarot_final_norm_weight": [1.0] * 32,
            }
        })

    config = json.loads((output / "config.json").read_text())
    assert config["model_type"] == "qwen3_quarot"
    assert config["tokenizer_name_or_path"] == "source/model"
    assert config["auto_map"] == {
        "AutoConfig": "quarot.FakeConfig",
        "AutoModelForCausalLM": "quarot.FakeRuntime",
    }
    assert config["quarot_checkpoint_format_version"] == 2
    assert config["quarot_ffn_format"] == "grouped_h256_v1"
    assert config["quarot_activation_clip_ratio"] == 0.9
    assert config["quarot_source_model_id"] == "source/model"
    assert config["quarot_conversion"] == {"method": "rtn", "seed": 0}
    assert config["quarot_rotation_format"] == "hadk_v1"
    assert config["quarot_rotation_width"] == 32
    assert config["quarot_rotation_remainder"] == 1
    assert config["quarot_rotation_inner"] == 32
    assert config["quarot_rotation_signs"] == [1] * 32
    assert config["quarot_final_norm_weight"] == [1.0] * 32
    assert (output / "quarot.py").read_text() == "# runtime source\n"
