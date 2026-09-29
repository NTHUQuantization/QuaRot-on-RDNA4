"""Dense model-family registry shared by e2e conversion and benchmarks."""
import json
from pathlib import Path
import transformers
from e2e.quantized_llama import modeling_llama
from e2e.quantized_qwen2 import modeling_qwen2
from e2e.quantized_qwen3 import modeling_qwen3

RUNTIMES = {
    "llama": (modeling_llama.QuarotLlamaConfig,
              modeling_llama.QuarotLlamaForCausalLM,
              modeling_llama.QuarotFP16LlamaForCausalLM),
    "llama_quarot": (modeling_llama.QuarotLlamaConfig,
                     modeling_llama.QuarotLlamaForCausalLM,
                     modeling_llama.QuarotFP16LlamaForCausalLM),
    "qwen2": (modeling_qwen2.QuarotQwen2Config,
              modeling_qwen2.QuarotQwen2ForCausalLM,
              modeling_qwen2.QuarotFP16Qwen2ForCausalLM),
    "qwen2_quarot": (modeling_qwen2.QuarotQwen2Config,
                     modeling_qwen2.QuarotQwen2ForCausalLM,
                     modeling_qwen2.QuarotFP16Qwen2ForCausalLM),
    "qwen3": (modeling_qwen3.QuarotQwen3Config,
              modeling_qwen3.QuarotQwen3ForCausalLM,
              modeling_qwen3.QuarotFP16Qwen3ForCausalLM),
    "qwen3_quarot": (modeling_qwen3.QuarotQwen3Config,
                     modeling_qwen3.QuarotQwen3ForCausalLM,
                     modeling_qwen3.QuarotFP16Qwen3ForCausalLM),
}

TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
)


def _read_config(path, local_files_only=False):
    config_path = Path(path) / "config.json"
    if config_path.is_file():
        return json.loads(config_path.read_text())
    return transformers.AutoConfig.from_pretrained(
        path, local_files_only=local_files_only,
        trust_remote_code=False).to_dict()


def tokenizer_source(path, local_files_only=False):
    """Resolve the tokenizer paired with a dense or converted checkpoint."""
    checkpoint = Path(path)
    if checkpoint.is_dir() and any(
            (checkpoint / name).is_file() for name in TOKENIZER_FILES):
        return str(checkpoint)
    if not (checkpoint / "config.json").is_file():
        # A Hub model ID is already the authoritative tokenizer source.
        return str(path)

    config = _read_config(path, local_files_only=local_files_only)
    for key in ("tokenizer_name_or_path", "base_model_name_or_path",
                "_name_or_path"):
        source = config.get(key)
        if source and source != str(path):
            return source

    model_type = config.get("model_type", "").removesuffix("_quarot")
    vocab_size = config.get("vocab_size")
    known = {
        ("llama", 128256): "meta-llama/Llama-3.1-8B",
        ("llama", 32000): "meta-llama/Llama-2-7b-hf",
        ("qwen2", 151936): "Qwen/Qwen2.5-7B",
        ("qwen3", 151936): "Qwen/Qwen3-8B",
    }
    try:
        return known[(model_type, vocab_size)]
    except KeyError as error:
        raise ValueError(
            f"cannot infer tokenizer for model_type={model_type!r}, "
            f"vocab_size={vocab_size!r}; save tokenizer files alongside "
            "the checkpoint or add tokenizer_name_or_path to config.json"
        ) from error


def runtime_types(path, local_files_only=False):
    model_type = _read_config(path, local_files_only)["model_type"]
    try:
        return RUNTIMES[model_type]
    except KeyError as error:
        raise ValueError(f"unsupported dense model_type {model_type!r}") from error
