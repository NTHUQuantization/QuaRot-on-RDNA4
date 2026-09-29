"""Low-host-memory, resumable RtN conversion for safetensors checkpoints."""
import gc
import json
import os
import shutil
from pathlib import Path

import torch
import tqdm
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME
from transformers.utils.hub import cached_file

from e2e.checkpoint_utils import gptq_utils
from quarot.functional import apply_exact_had_to_linear, pack_i4
from quarot.functional.hadamard import (
    get_hadK, grouped_ffn_physical_width, matmul_grouped_h256, matmul_hadU)


_STREAM_VERSION = 2
_REMOVED_NORMS = (
    "post_attention_layernorm.weight",
    "input_layernorm.weight",
    "model.norm.weight",
)


def _remap(key):
    return key.replace("self_attn.o_proj", "self_attn.o_proj.1")


class _SafeTensorSource:
    def __init__(self, model):
        index_path = cached_file(model, SAFE_WEIGHTS_INDEX_NAME, _raise_exceptions_for_missing_entries=False)
        if index_path:
            index = json.loads(Path(index_path).read_text())
            self.weight_map = index["weight_map"]
            filenames = set(self.weight_map.values())
        else:
            single = cached_file(model, SAFE_WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
            if not single:
                raise ValueError(
                    "Streaming RtN requires a safetensors checkpoint; no "
                    f"{SAFE_WEIGHTS_INDEX_NAME} or {SAFE_WEIGHTS_NAME} was found")
            with safe_open(single, framework="pt", device="cpu") as handle:
                self.weight_map = {key: SAFE_WEIGHTS_NAME for key in handle.keys()}
            filenames = {SAFE_WEIGHTS_NAME}
        self.files = {name: Path(cached_file(model, name)) for name in filenames}

    def keys(self):
        return self.weight_map.keys()

    def tensors(self, keys):
        by_file = {}
        for key in keys:
            by_file.setdefault(self.weight_map[key], []).append(key)
        result = {}
        for filename, names in by_file.items():
            with safe_open(self.files[filename], framework="pt", device="cpu") as handle:
                for name in names:
                    result[name] = handle.get_tensor(name)
        return result


def _atomic_save(tensors, path, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        try:
            save_file(
                {key: value.detach().cpu().contiguous() for key, value in tensors.items()},
                temporary, metadata={key: str(value) for key, value in metadata.items()})
        except (OSError, RuntimeError) as error:
            free = shutil.disk_usage(path.parent).free
            partial = temporary.stat().st_size if temporary.exists() else 0
            raise RuntimeError(
                f"Failed to write streaming shard {path} "
                f"({free / 2**30:.2f} GiB free; partial write "
                f"{partial / 2**20:.1f} MiB)") from error
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _matmul(left, right, device, compute_dtype, output_dtype):
    return torch.matmul(
        left.to(device=device, dtype=compute_dtype),
        right.to(device=device, dtype=compute_dtype),
    ).to(device="cpu", dtype=output_dtype)


def _random_hadamard_matrix_with_signs(size, device, dtype):
    """Build ``Q = D U`` and retain the exact randomized diagonal ``D``."""
    signs = torch.randint(low=0, high=2, size=(size,), device=device).to(dtype)
    signs = signs * 2 - 1
    q = matmul_hadU(torch.diag(signs))
    return q, signs.to(device="cpu", dtype=torch.int8)


def _seeded_rotation(config, args, rotation_device, compute_dtype):
    """Build the checkpoint rotation without consuming caller RNG state.

    Streaming conversion may be resumed from a process whose RNG history is
    unrelated to the process that wrote the existing shards.  The rotation is
    therefore derived from ``args.seed`` locally rather than from the ambient
    default generator.  ``fork_rng`` also makes programmatic conversion
    side-effect free for callers that use either the CPU or selected GPU RNG.
    """
    rotation_device = torch.device(rotation_device)
    if rotation_device.type == "cuda":
        device_index = rotation_device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        rotation_device = torch.device("cuda", device_index)
        with torch.random.fork_rng(devices=[device_index], device_type="cuda"):
            with torch.cuda.device(device_index):
                torch.cuda.manual_seed(int(args.seed))
                return _random_hadamard_matrix_with_signs(
                    config.hidden_size, rotation_device, dtype=compute_dtype)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(args.seed))
        return _random_hadamard_matrix_with_signs(
            config.hidden_size, rotation_device, dtype=compute_dtype)


def _seeded_rotation_matrix(config, args, rotation_device, compute_dtype):
    return _seeded_rotation(config, args, rotation_device, compute_dtype)[0]


def _hadamard(weight, *, output=False, had_dim=-1):
    # Reuse the established offline transform without allocating a second
    # initialized Linear weight.
    module = torch.nn.Linear(
        weight.shape[1], weight.shape[0], bias=False,
        dtype=weight.dtype, device="meta")
    module.weight = torch.nn.Parameter(weight, requires_grad=False)
    apply_exact_had_to_linear(module, had_dim=had_dim, output=output)
    return module.weight.detach()


def _fuse_norm(tensors, norm_key, weight_keys, device, compute_dtype):
    norm = tensors[norm_key]
    for key in weight_keys:
        dtype = tensors[key].dtype
        tensors[key] = (
            tensors[key].to(device=device, dtype=compute_dtype)
            * norm.to(device=device, dtype=compute_dtype)
        ).to(device="cpu", dtype=dtype)


def _rotate_layer(tensors, prefix, config, q, device, compute_dtype,
                  remove_norms=True):
    input_norm = prefix + "input_layernorm.weight"
    post_norm = prefix + "post_attention_layernorm.weight"
    qkv = [prefix + f"self_attn.{name}_proj.weight" for name in "qkv"]
    mlp_inputs = [prefix + "mlp.up_proj.weight", prefix + "mlp.gate_proj.weight"]
    _fuse_norm(tensors, input_norm, qkv, device, compute_dtype)
    _fuse_norm(tensors, post_norm, mlp_inputs, device, compute_dtype)

    for key in qkv + mlp_inputs:
        tensors[key] = _matmul(tensors[key], q, device, compute_dtype, tensors[key].dtype)

    o_key = prefix + "self_attn.o_proj.weight"
    tensors[o_key] = _matmul(q.T, tensors[o_key], device, compute_dtype, tensors[o_key].dtype)
    o_bias = prefix + "self_attn.o_proj.bias"
    if o_bias in tensors:
        tensors[o_bias] = _matmul(q.T, tensors[o_bias], device, compute_dtype, tensors[o_bias].dtype)

    down_key = prefix + "mlp.down_proj.weight"
    tensors[down_key] = _matmul(q.T, tensors[down_key], device, compute_dtype, tensors[down_key].dtype)
    down_bias = prefix + "mlp.down_proj.bias"
    if down_bias in tensors:
        tensors[down_bias] = _matmul(q.T, tensors[down_bias], device, compute_dtype, tensors[down_bias].dtype)

    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    v_key = prefix + "self_attn.v_proj.weight"
    tensors[v_key] = _hadamard(tensors[v_key], output=True, had_dim=head_dim)
    tensors[o_key] = _hadamard(tensors[o_key])
    logical = tensors[mlp_inputs[0]].shape[0]
    physical = grouped_ffn_physical_width(logical)
    padding = physical - logical
    if padding:
        for key in mlp_inputs:
            tensors[key] = torch.nn.functional.pad(
                tensors[key], (0, 0, 0, padding))
            bias_key = key[:-len("weight")] + "bias"
            if bias_key in tensors:
                tensors[bias_key] = torch.nn.functional.pad(
                    tensors[bias_key], (0, padding))
        tensors[down_key] = torch.nn.functional.pad(
            tensors[down_key], (0, padding))
    down_dtype = tensors[down_key].dtype
    tensors[down_key] = matmul_grouped_h256(
        tensors[down_key].to(device=device, dtype=compute_dtype)).to(
            device="cpu", dtype=down_dtype)
    if remove_norms:
        del tensors[input_norm], tensors[post_norm]
    else:
        tensors[input_norm] = torch.ones_like(tensors[input_norm])
        tensors[post_norm] = torch.ones_like(tensors[post_norm])


def _pack_linear(weight, args, device):
    quantizer = gptq_utils.WeightQuantizer()
    quantizer.configure(
        args.w_bits, perchannel=True, sym=not args.w_asym, mse=args.w_clip)
    gpu_weight = weight.to(device)
    quantizer.find_params(gpu_weight)
    dequantized = quantizer.quantize(gpu_weight)
    packed = pack_i4((dequantized / quantizer.scale).round().to(torch.int8)).cpu()
    scale = quantizer.scale.cpu()
    del gpu_weight, dequantized, quantizer
    torch.cuda.empty_cache()
    return packed, scale


def _packed_layer(tensors, prefix, args, device):
    output = {}
    for key, value in tensors.items():
        if key.endswith(".weight") and value.ndim == 2:
            packed, scale = _pack_linear(value, args, device)
            mapped = _remap(key)
            output[mapped] = packed
            output[mapped[:-len("weight")] + "weight_scales"] = scale
        elif not any(name in key for name in _REMOVED_NORMS):
            output[_remap(key)] = value
    return output


def _valid_shard(path, signature, expected_prefix):
    if not path.is_file():
        return False
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            return (metadata.get("quarot_signature") == signature
                    and any(key.startswith(expected_prefix) for key in handle.keys()))
    except Exception:
        return False


def _global_tensors(source, config, q, device, compute_dtype):
    keys = [key for key in source.keys() if not key.startswith("model.layers.")]
    tensors = source.tensors(keys)
    embed_key = "model.embed_tokens.weight"
    embed = tensors[embed_key]
    dtype = embed.dtype
    embed_gpu = embed.to(device=device, dtype=compute_dtype)
    embed_gpu -= embed_gpu.mean(dim=-1, keepdim=True)
    tensors[embed_key] = torch.matmul(embed_gpu, q).to("cpu", dtype=dtype)
    del embed_gpu, embed

    head_key = "lm_head.weight"
    norm_key = "model.norm.weight"
    if head_key in tensors:
        head = tensors[head_key]
        head_dtype = head.dtype
        if norm_key in tensors:
            head = (head.to(device=device, dtype=compute_dtype)
                    * tensors[norm_key].to(device=device, dtype=compute_dtype))
        tensors[head_key] = torch.matmul(head, q).to("cpu", dtype=head_dtype)
        del head
    tensors.pop(norm_key, None)
    return {_remap(key): value for key, value in tensors.items()}


def _rotation_checkpoint_metadata(source, config, rotation_signs, args):
    """Return compact metadata needed to restore PARD2 target features.

    The offline rotation is Q = D U. Persisting D and the final RMSNorm scale
    avoids reopening the dense source checkpoint at runtime, which is
    especially important for a streaming 32B conversion.
    """
    signs = torch.as_tensor(
        rotation_signs, device="cpu").reshape(-1)
    width = int(config.hidden_size)
    if signs.numel() != width:
        raise ValueError(f"rotation sign width {signs.numel()} != {width}")
    if not torch.isfinite(signs).all():
        raise ValueError("rotation signs contain non-finite values")
    if not torch.all((signs == -1) | (signs == 1)):
        raise ValueError("rotation signs must contain only -1 or +1")
    signs = signs.to(dtype=torch.int8)
    final_norm = source.tensors(["model.norm.weight"])[
        "model.norm.weight"].to(
            device="cpu", dtype=torch.float32).reshape(-1)
    if final_norm.numel() != width:
        raise ValueError(f"final norm width {final_norm.numel()} != {width}")
    if not torch.isfinite(final_norm).all():
        raise ValueError("final norm contains non-finite values")
    _, remainder = get_hadK(int(config.hidden_size))
    return {
        "quarot_rotation_format": "hadk_v1",
        "quarot_rotation_width": int(config.hidden_size),
        "quarot_rotation_remainder": int(remainder),
        "quarot_rotation_inner": int(config.hidden_size) // int(remainder),
        "quarot_rotation_seed": int(args.seed),
        "quarot_rotation_device": str(args.rotation_device),
        "quarot_rotation_dtype": str(args.rotation_dtype),
        "quarot_rotation_signs": signs.tolist(),
        "quarot_final_norm_weight": final_norm.tolist(),
    }


@torch.inference_mode()
def convert(model, output, config, args):
    """Stream a rotated RtN checkpoint to one resumable shard per layer."""
    if args.w_groupsize != -1:
        raise ValueError("Groupsize is not supported in RtN")
    source = _SafeTensorSource(model)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    rotation_device = device if args.rotation_device == "cuda" else torch.device("cpu")
    compute_dtype = {"float32": torch.float32, "float64": torch.float64}[args.rotation_dtype]
    q, rotation_signs = _seeded_rotation(
        config, args, rotation_device, compute_dtype)
    rotation_metadata = _rotation_checkpoint_metadata(
        source, config, rotation_signs, args)
    signature_data = {
        "version": _STREAM_VERSION, "method": "rtn",
        "model": str(model), "seed": args.seed,
        "rotation_device": args.rotation_device,
        "rotation_dtype": args.rotation_dtype, "w_bits": args.w_bits,
        "w_groupsize": args.w_groupsize, "w_asym": args.w_asym,
        "w_clip": args.w_clip,
    }
    signature = json.dumps(signature_data, sort_keys=True, separators=(",", ":"))
    weight_map = {}

    global_path = output / "model-global.safetensors"
    if not _valid_shard(global_path, signature, "model.embed_tokens"):
        tensors = _global_tensors(source, config, q, rotation_device, compute_dtype)
        _atomic_save(tensors, global_path, {"format": "pt", "quarot_signature": signature})
        del tensors
        gc.collect()
    with safe_open(global_path, framework="pt", device="cpu") as handle:
        weight_map.update({key: global_path.name for key in handle.keys()})

    for index in tqdm.tqdm(range(config.num_hidden_layers), desc="Streaming RtN", unit="layer"):
        source_prefix = f"model.layers.{index}."
        mapped_prefix = source_prefix
        shard = output / f"model-layer-{index:05d}.safetensors"
        if not _valid_shard(shard, signature, mapped_prefix):
            keys = [key for key in source.keys() if key.startswith(source_prefix)]
            tensors = source.tensors(keys)
            _rotate_layer(tensors, source_prefix, config, q, rotation_device, compute_dtype)
            packed = _packed_layer(tensors, source_prefix, args, device)
            _atomic_save(packed, shard, {"format": "pt", "quarot_signature": signature})
            del tensors, packed
            gc.collect()
            torch.cuda.empty_cache()
        with safe_open(shard, framework="pt", device="cpu") as handle:
            weight_map.update({key: shard.name for key in handle.keys()})

    index = {"metadata": {"total_size": sum(path.stat().st_size for path in output.glob("model-*.safetensors"))},
             "weight_map": weight_map}
    index_path = output / SAFE_WEIGHTS_INDEX_NAME
    temporary = index_path.with_suffix(index_path.suffix + ".tmp")
    temporary.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, index_path)
    return {
        "streaming_signature": signature_data,
        "rotation_metadata": rotation_metadata,
    }
