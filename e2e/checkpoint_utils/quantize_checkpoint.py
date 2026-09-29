"""Convert dense Llama 2/3.1, CodeLlama, Qwen2.5, or Qwen3 to QuaRot INT4."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
import torch
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import transformers
from e2e.checkpoint_utils import streaming_gptq, streaming_rtn

QUAROT_CHECKPOINT_FORMAT_VERSION = 2
QUAROT_FFN_FORMAT = "grouped_h256_v1"
QUAROT_ACTIVATION_CLIP_RATIO = 0.9

FAMILIES = {
    "llama": ("e2e.quantized_llama.modeling_llama", "QuarotLlamaConfig",
              "QuarotLlamaForCausalLM", "llama_quarot"),
    "qwen2": ("e2e.quantized_qwen2.modeling_qwen2", "QuarotQwen2Config",
              "QuarotQwen2ForCausalLM", "qwen2_quarot"),
    "qwen3": ("e2e.quantized_qwen3.modeling_qwen3", "QuarotQwen3Config",
              "QuarotQwen3ForCausalLM", "qwen3_quarot"),
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity(model):
    """Record stable Hub provenance without relying on an absolute path."""
    text = str(model)
    parts = Path(text).parts
    model_id = None
    revision = None
    for index, part in enumerate(parts):
        if part.startswith("models--"):
            components = part.removeprefix("models--").split("--")
            if len(components) >= 2:
                model_id = "/".join(components)
        if part == "snapshots" and index + 1 < len(parts):
            revision = parts[index + 1]
    if model_id is None and not Path(text).is_absolute() and text.count("/") == 1:
        model_id = text

    result = {}
    if model_id:
        result["quarot_source_model_id"] = model_id
    if revision:
        result["quarot_source_revision"] = revision
    source = Path(text)
    config_path = source / "config.json"
    if config_path.is_file():
        result["quarot_source_config_sha256"] = _sha256(config_path)
    index_path = source / "model.safetensors.index.json"
    if index_path.is_file():
        result["quarot_source_index_sha256"] = _sha256(index_path)
    return result


def runtime_types(model_type):
    if model_type not in FAMILIES:
        raise ValueError(f"unsupported dense model_type {model_type!r}")
    module_name, config_name, model_name, output_type = FAMILIES[model_type]
    module = __import__(module_name, fromlist=[config_name, model_name])
    return getattr(module, config_name), getattr(module, model_name), output_type, module

def main(args):
    # A resumed run must recreate the rotation used by its cached layers.
    torch.manual_seed(args.seed)
    config = transformers.AutoConfig.from_pretrained(args.model)
    if getattr(config, "num_experts", 0):
        raise ValueError("MoE checkpoints are intentionally not supported")
    config_cls, runtime_cls, output_type, runtime_module = runtime_types(config.model_type)
    if args.quant_method == "rtn":
        print(f"Streaming RtN: source={args.model}, output={args.output}, "
              f"rotation_device={args.rotation_device}, "
              f"rotation_dtype={args.rotation_dtype}", flush=True)
        conversion_metadata = streaming_rtn.convert(
            args.model, args.output, config, args)
    else:
        print(f"Streaming GPTQ: source={args.model}, output={args.output}, "
              f"rotation_device={args.rotation_device}, "
              f"rotation_dtype={args.rotation_dtype}", flush=True)
        conversion_metadata = streaming_gptq.convert(
            args.model, args.output, config, args)

    finalize_output(args.output, args.model, config_cls, runtime_cls,
                    output_type, runtime_module, conversion_metadata)


def finalize_output(output_path, model, config_cls, runtime_cls, output_type,
                    runtime_module, conversion_metadata=None):
    output = Path(output_path)
    runtime_config = config_cls.from_pretrained(
        model, attn_implementation="flash_attention_2")
    runtime_config.save_pretrained(output)
    config_path = output / "config.json"
    saved_config = json.loads(config_path.read_text())
    saved_config["auto_map"] = {
        "AutoConfig": f"quarot.{config_cls.__name__}",
        "AutoModelForCausalLM": f"quarot.{runtime_cls.__name__}"}
    saved_config["model_type"] = output_type
    saved_config["tokenizer_name_or_path"] = model
    saved_config["quarot_checkpoint_format_version"] = (
        QUAROT_CHECKPOINT_FORMAT_VERSION)
    saved_config["quarot_ffn_format"] = QUAROT_FFN_FORMAT
    saved_config["quarot_activation_clip_ratio"] = (
        QUAROT_ACTIVATION_CLIP_RATIO)
    saved_config.update(_source_identity(model))
    if conversion_metadata is not None:
        signature = conversion_metadata.get("streaming_signature")
        if signature is not None:
            saved_config["quarot_conversion"] = signature
        saved_config.update(conversion_metadata.get("rotation_metadata", {}))
    config_path.write_text(json.dumps(saved_config, indent=2) + "\n")
    transformers.AutoTokenizer.from_pretrained(model).save_pretrained(output)
    source = Path(runtime_module.__file__)
    shutil.copy(source, output / "quarot.py")
    shutil.copy(Path(__file__).parents[1] / "quantized_common.py",
                output / "quantized_common.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cal-dataset", default="wikitext2",
                        choices=("wikitext2", "ptb", "c4"))
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument(
        "--tokenizer-model",
        help="Optional tokenizer ID/path for calibration when --model uses a "
             "gated or unavailable tokenizer.")
    parser.add_argument(
        "--quant-method", choices=("rtn", "gptq"), default="rtn",
        help="Weight quantization method (default: rtn). Both methods stream "
             "the checkpoint one layer at a time.")
    parser.add_argument("--w-groupsize", type=int, default=-1)
    parser.add_argument("--w-asym", action="store_true")
    parser.add_argument(
        "--w-clip", action=argparse.BooleanOptionalAction, default=True,
        help="Optimize per-channel weight clipping (default: enabled; use "
             "--no-w-clip only for diagnostic unclipped RTN/GPTQ runs).")
    parser.add_argument("--percdamp", type=float, default=.01)
    parser.add_argument("--act-order", action="store_true")
    parser.add_argument(
        "--rotation-device", choices=("cuda", "cpu"), default="cuda",
        help="Device for offline LayerNorm fusion and randomized rotations "
             "(default: cuda). Weights are copied back to CPU after each op.")
    parser.add_argument(
        "--rotation-dtype", choices=("float32", "float64"),
        default="float32",
        help="Compute dtype for offline fusion and rotation (default: float32). "
             "Use float64 for legacy conversion numerics.")
    args = parser.parse_args()
    args.w_bits = 4
    main(args)
