"""Layer-streamed GPTQ conversion for dense Llama and Qwen checkpoints."""
import gc
import json
import os
from pathlib import Path

import torch
import tqdm
from safetensors import safe_open
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer, LlamaRotaryEmbedding)
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer, Qwen2RotaryEmbedding)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer, Qwen3RotaryEmbedding)
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from e2e.checkpoint_utils import data_utils, gptq_utils, rotation_utils
from e2e.checkpoint_utils import streaming_rtn as stream
from quarot.functional import pack_i4, unpack_i4
from quarot.functional.hadamard import (
    get_hadK, matmul_grouped_h256, matmul_hadU_cuda)

_VERSION = 5
_CALIBRATION_DTYPE = torch.float16
_FAMILIES = {
    "llama": (LlamaDecoderLayer, LlamaRotaryEmbedding),
    "qwen2": (Qwen2DecoderLayer, Qwen2RotaryEmbedding),
    "qwen3": (Qwen3DecoderLayer, Qwen3RotaryEmbedding),
}
_GROUPS = (
    ("self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"),
    ("self_attn.o_proj",),
    ("mlp.up_proj", "mlp.gate_proj"),
    ("mlp.down_proj",),
)
_PRESERVED_LAYER_SUFFIXES = (
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
)


def _apply_hadamard(x, had_rem_dim, rem_dim):
    return matmul_hadU_cuda(x, had_rem_dim, rem_dim)


def _install_online_hadamards(layer, config):
    """Make the FP16 calibration layer match the transformed INT4 runtime."""
    head_had, head_rem_dim = get_hadK(config.num_attention_heads)
    num_heads = config.num_attention_heads
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads)

    def attention_input_hadamard(_, inputs):
        (x,) = inputs
        shape = x.shape
        x = x.view(*shape[:-1], num_heads, head_dim)
        x = _apply_hadamard(
            x.transpose(-1, -2), head_had, head_rem_dim).transpose(-1, -2)
        return (x.reshape(shape),)

    def mlp_input_hadamard(_, inputs):
        (x,) = inputs
        return (matmul_grouped_h256(x),)

    # Hooks preserve the original Linear module names used by GPTQ and add no
    # buffers to state_dict(), unlike wrapping the projections in Sequential.
    layer.self_attn.o_proj.register_forward_pre_hook(attention_input_hadamard)
    layer.mlp.down_proj.register_forward_pre_hook(mlp_input_hadamard)


def _calibration_types(config):
    try:
        return _FAMILIES[config.model_type]
    except KeyError as error:
        raise ValueError(
            f"Streaming GPTQ does not support model_type "
            f"{config.model_type!r}") from error


def _layer_from_tensors(config, index, tensors, prefix):
    decoder_cls, _ = _calibration_types(config)
    layer = decoder_cls(config, index).to(dtype=_CALIBRATION_DTYPE)
    rotation_utils.pad_mlp_modules(layer)
    state = {key[len(prefix):]: value for key, value in tensors.items()}
    # Older Llama checkpoints persist this derived RoPE buffer, while newer
    # Transformers versions compute it from config and no longer register it.
    state.pop("self_attn.rotary_emb.inv_freq", None)
    result = layer.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"streaming GPTQ layer load failed: {result}")
    _install_online_hadamards(layer, config)
    return layer


def _quantize_layer(layer, inps, outs, position_ids, position_embeddings,
                    cache_position, args):
    quantizers = {}
    full = dict(layer.named_modules())
    for names in _GROUPS:
        subset = {name: full[name] for name in names}
        engines = {}
        for name, module in subset.items():
            engine = gptq_utils.GPTQ(module)
            engine.quantizer = gptq_utils.WeightQuantizer()
            engine.quantizer.configure(
                args.w_bits, perchannel=True, sym=not args.w_asym,
                mse=args.w_clip)
            engines[name] = engine

        def add_batch(name):
            def hook(_, inputs, output):
                engines[name].add_batch(inputs[0].data, output.data)
            return hook

        handles = [module.register_forward_hook(add_batch(name))
                   for name, module in subset.items()]
        for sample in range(args.nsamples):
            outs[sample] = layer(
                inps[sample].unsqueeze(0), attention_mask=None,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                cache_position=cache_position)[0]
        for handle in handles:
            handle.remove()
        for name, engine in engines.items():
            engine.fasterquant(
                percdamp=args.percdamp, groupsize=args.w_groupsize,
                actorder=args.act_order, static_groups=False)
            quantizers[name] = engine.quantizer.cpu()
            engine.free()

    for sample in range(args.nsamples):
        outs[sample] = layer(
            inps[sample].unsqueeze(0), attention_mask=None,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            cache_position=cache_position)[0]
    return quantizers


def _preserved_layer_tensors(tensors, prefix):
    return {
        prefix + suffix: tensors[prefix + suffix]
        for suffix in _PRESERVED_LAYER_SUFFIXES
        if prefix + suffix in tensors
    }


def _pack_layer(layer, prefix, quantizers, preserved_tensors=None):
    output = {}
    state = layer.state_dict()
    for key, value in state.items():
        full_key = prefix + key
        if key in {"input_layernorm.weight",
                   "post_attention_layernorm.weight"}:
            continue
        module_name = key[:-len(".weight")] if key.endswith(".weight") else None
        if module_name in quantizers:
            scale = quantizers[module_name].scale
            mapped = stream._remap(full_key)
            output[mapped] = pack_i4(
                (value.cpu() / scale).round().to(torch.int8))
            output[mapped[:-len("weight")] + "weight_scales"] = scale.cpu()
        else:
            output[stream._remap(full_key)] = value.cpu()
    for full_key, value in (preserved_tensors or {}).items():
        output[stream._remap(full_key)] = value.cpu()
    return output


def _load_packed_layer(layer, shard, prefix):
    modules = dict(layer.named_modules())
    with safe_open(shard, framework="pt", device="cpu") as handle:
        for name, module in modules.items():
            if type(module) is not torch.nn.Linear:
                continue
            full_key = stream._remap(prefix + name + ".weight")
            scale_key = full_key[:-len("weight")] + "weight_scales"
            packed = handle.get_tensor(full_key)
            scale = handle.get_tensor(scale_key)
            module.weight.data.copy_(
                (unpack_i4(packed).to(module.weight.dtype) * scale)
                .to(module.weight.device))
            bias_key = stream._remap(prefix + name + ".bias")
            if module.bias is not None and bias_key in handle.keys():
                module.bias.data.copy_(handle.get_tensor(bias_key))
    layer.input_layernorm.weight.data.fill_(1)
    layer.post_attention_layernorm.weight.data.fill_(1)


def _initial_activations(global_path, loader, config, device):
    with safe_open(global_path, framework="pt", device="cpu") as handle:
        embedding = handle.get_tensor("model.embed_tokens.weight").to(
            device=device, dtype=_CALIBRATION_DTYPE)
    inps = torch.empty(
        args_shape := (len(loader), loader[0][0].shape[-1], config.hidden_size),
        dtype=_CALIBRATION_DTYPE, device=device)
    for index, batch in enumerate(loader):
        inps[index] = torch.nn.functional.embedding(
            batch[0].to(device), embedding).squeeze(0)
    del embedding
    return inps, torch.empty(args_shape, dtype=inps.dtype, device=device)


@torch.inference_mode()
def convert(model, output, config, args):
    _, rotary_cls = _calibration_types(config)
    source = stream._SafeTensorSource(model)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    rotation_device = device if args.rotation_device == "cuda" else torch.device("cpu")
    compute_dtype = {"float32": torch.float32,
                     "float64": torch.float64}[args.rotation_dtype]
    q, rotation_signs = stream._seeded_rotation(
        config, args, rotation_device, compute_dtype)
    rotation_metadata = stream._rotation_checkpoint_metadata(
        source, config, rotation_signs, args)
    signature_data = {
        "version": _VERSION, "method": "gptq", "model": str(model),
        "seed": args.seed, "rotation_device": args.rotation_device,
        "rotation_dtype": args.rotation_dtype, "w_bits": args.w_bits,
        "w_groupsize": args.w_groupsize, "w_asym": args.w_asym,
        "w_clip": args.w_clip, "percdamp": args.percdamp,
        "act_order": args.act_order, "dataset": args.cal_dataset,
        "nsamples": args.nsamples, "seqlen": args.seqlen,
    }
    signature = json.dumps(signature_data, sort_keys=True, separators=(",", ":"))
    weight_map = {}

    global_path = output / "model-global.safetensors"
    if not stream._valid_shard(global_path, signature, "model.embed_tokens"):
        tensors = stream._global_tensors(
            source, config, q, rotation_device, compute_dtype)
        stream._atomic_save(
            tensors, global_path,
            {"format": "pt", "quarot_signature": signature})
        del tensors
        gc.collect()
    with safe_open(global_path, framework="pt", device="cpu") as handle:
        weight_map.update({key: global_path.name for key in handle.keys()})

    loader = data_utils.get_loaders(
        args.cal_dataset, nsamples=args.nsamples, seed=args.seed,
        model=args.tokenizer_model or args.model, seqlen=args.seqlen,
        eval_mode=False)
    inps, outs = _initial_activations(global_path, loader, config, device)
    cache_position = torch.arange(args.seqlen, device=device)
    position_ids = cache_position.unsqueeze(0)
    rotary = rotary_cls(config=config).to(device=device)
    position_embeddings = rotary(inps[:1], position_ids)
    config._attn_implementation = "flash_attention_2"

    for index in tqdm.tqdm(
            range(config.num_hidden_layers), desc="Streaming GPTQ", unit="layer"):
        prefix = f"model.layers.{index}."
        shard = output / f"model-layer-{index:05d}.safetensors"
        keys = [key for key in source.keys() if key.startswith(prefix)]
        tensors = source.tensors(keys)
        stream._rotate_layer(
            tensors, prefix, config, q, rotation_device, compute_dtype,
            remove_norms=False)
        preserved_tensors = _preserved_layer_tensors(tensors, prefix)
        layer = _layer_from_tensors(config, index, tensors, prefix)
        del tensors
        if stream._valid_shard(shard, signature, prefix):
            _load_packed_layer(layer, shard, prefix)
            layer = layer.to(device)
            for sample in range(args.nsamples):
                outs[sample] = layer(
                    inps[sample].unsqueeze(0), attention_mask=None,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    cache_position=cache_position)[0]
        else:
            layer = layer.to(device)
            quantizers = _quantize_layer(
                layer, inps, outs, position_ids, position_embeddings,
                cache_position, args)
            packed = _pack_layer(
                layer.cpu(), prefix, quantizers, preserved_tensors)
            stream._atomic_save(
                packed, shard,
                {"format": "pt", "quarot_signature": signature})
            del packed, quantizers
        del preserved_tensors
        inps, outs = outs, inps
        del layer
        gc.collect()
        torch.cuda.empty_cache()
        with safe_open(shard, framework="pt", device="cpu") as handle:
            weight_map.update({key: shard.name for key in handle.keys()})

    total_size = sum(
        path.stat().st_size for path in output.glob("model-*.safetensors"))
    index = {"metadata": {"total_size": total_size},
             "weight_map": weight_map}
    index_path = output / SAFE_WEIGHTS_INDEX_NAME
    temporary = index_path.with_suffix(index_path.suffix + ".tmp")
    temporary.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, index_path)
    return {
        "streaming_signature": signature_data,
        "rotation_metadata": rotation_metadata,
    }
