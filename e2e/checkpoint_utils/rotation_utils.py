import gc
import torch
import typing
import transformers
import tqdm, math
from quarot.functional import random_hadamard_matrix, apply_exact_had_to_linear
from quarot.functional.hadamard import (
    grouped_ffn_physical_width, matmul_grouped_h256)

def _to_compute(tensor, device, dtype):
    return tensor.to(device=device, dtype=dtype)


def fuse_ln_linear(layernorm: torch.nn.Module,
                   linear_layers: typing.Iterable[torch.nn.Linear],
                   device="cpu", dtype=torch.float64) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = _to_compute(linear.weight.data, device, dtype)
        norm_weight = _to_compute(layernorm.weight.data, device, dtype)
        linear.weight.data = (W_ * norm_weight).to(
            device="cpu", dtype=linear_dtype)

        if hasattr(layernorm, 'bias'):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(
                    linear.out_features, dtype=linear_dtype))
            bias = _to_compute(linear.bias.data, device, dtype)
            norm_bias = _to_compute(layernorm.bias.data, device, dtype)
            linear.bias.data = (bias + torch.matmul(W_, norm_bias)).to(
                device="cpu", dtype=linear_dtype)

def fuse_layer_norms(model, device="cpu", dtype=torch.float64):

    # Embedding fusion
    W = model.model.embed_tokens
    weight_dtype = W.weight.data.dtype
    W_ = _to_compute(W.weight.data, device, dtype)
    W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(
        device="cpu", dtype=weight_dtype)

    layers = model.model.layers

    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:
        # fuse the input layernorms into the linear layers
        fuse_ln_linear(layer.post_attention_layernorm,
                       [layer.mlp.up_proj, layer.mlp.gate_proj], device, dtype)
        fuse_ln_linear(layer.input_layernorm,
                       [layer.self_attn.q_proj, layer.self_attn.k_proj,
                        layer.self_attn.v_proj], device, dtype)
        # GPTQ runs calibration forwards after fusion. Neutralize the fused
        # norm scales so calibration sees the same weightless RMSNorm used by
        # the converted runtime instead of applying each scale twice.
        layer.post_attention_layernorm.weight.data.fill_(1)
        layer.input_layernorm.weight.data.fill_(1)


    fuse_ln_linear(model.model.norm, [model.lm_head], device, dtype)
    model.model.norm.weight.data.fill_(1)



def _matmul_to_cpu(left, right, output_dtype, device, compute_dtype):
    left = _to_compute(left, device, compute_dtype)
    right = _to_compute(right, device, compute_dtype)
    return torch.matmul(left, right).to(device="cpu", dtype=output_dtype)


def rotate_embeddings(model, Q: torch.Tensor, device, compute_dtype) -> None:
    # Rotate the embeddings.
    W = model.model.embed_tokens
    dtype = W.weight.data.dtype
    W.weight.data = _matmul_to_cpu(
        W.weight.data, Q, dtype, device, compute_dtype)


def rotate_attention_inputs(layer, Q, device, compute_dtype) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W.weight.data = _matmul_to_cpu(
            W.weight.data, Q, dtype, device, compute_dtype)

def rotate_attention_output(layer, Q, device, compute_dtype) -> None:
    # Rotate output matrix of the self-attention layer.
    W = layer.self_attn.o_proj
    dtype = W.weight.data.dtype
    W.weight.data = _matmul_to_cpu(
        Q.T, W.weight.data, dtype, device, compute_dtype)
    if W.bias is not None:
        W.bias.data = _matmul_to_cpu(
            Q.T, W.bias.data, dtype, device, compute_dtype)

def rotate_mlp_input(layer, Q, device, compute_dtype):
    # Rotate the MLP input weights.
    mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W.weight.data = _matmul_to_cpu(
            W.weight.data, Q, dtype, device, compute_dtype)

def rotate_mlp_output(layer, Q, device, compute_dtype):
    # Rotate the MLP output weights and bias.
    W = layer.mlp.down_proj
    dtype = W.weight.data.dtype
    W.weight.data = _matmul_to_cpu(
        Q.T, W.weight.data, dtype, device, compute_dtype)
    pad_mlp_grouped_h256(layer, device, compute_dtype)
    if W.bias is not None:
        W.bias.data = _matmul_to_cpu(
            Q.T, W.bias.data, dtype, device, compute_dtype)


def pad_mlp_modules(layer):
    """Resize dense FFN modules to the universal physical H256 width."""
    up, gate, down = (layer.mlp.up_proj, layer.mlp.gate_proj,
                      layer.mlp.down_proj)
    logical = up.out_features
    physical = grouped_ffn_physical_width(logical)
    if physical != logical:
        pad_rows = physical - logical
        for module in (up, gate):
            module.weight.data = torch.nn.functional.pad(
                module.weight.data, (0, 0, 0, pad_rows))
            if module.bias is not None:
                module.bias.data = torch.nn.functional.pad(
                    module.bias.data, (0, pad_rows))
            module.out_features = physical
        down.weight.data = torch.nn.functional.pad(
            down.weight.data, (0, pad_rows))
        down.in_features = physical
    return physical


def pad_mlp_grouped_h256(layer, device="cpu", compute_dtype=torch.float64):
    """Pad an FFN and fold block-diagonal normalized H256 into down_proj."""
    pad_mlp_modules(layer)
    down = layer.mlp.down_proj
    down_dtype = down.weight.dtype
    down.weight.data = matmul_grouped_h256(
        down.weight.data.to(device=device, dtype=compute_dtype)).to(
            device="cpu", dtype=down_dtype)

def rotate_head(model, Q: torch.Tensor, device, compute_dtype) -> None:
    # Rotate the head.
    W = model.lm_head
    dtype = W.weight.data.dtype
    W.weight.data = _matmul_to_cpu(
        W.weight.data, Q, dtype, device, compute_dtype)

def rotate_ov_proj(layer, head_num, head_dim):
    v_proj = layer.self_attn.v_proj
    o_proj = layer.self_attn.o_proj

    apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
    apply_exact_had_to_linear(o_proj, had_dim=-1, output=False)


@torch.inference_mode()
def rotate_model(model, device="cpu", dtype=torch.float64):
    Q = random_hadamard_matrix(model.config.hidden_size, device, dtype=dtype)
    # random_hadamard_matrix constructs Q = D H. Persist D for PARD2-TD.
    rotation_signs = torch.sign(Q[:, 0]).to(torch.int8).cpu()
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = getattr(config, "head_dim", model_dim // num_heads)


    rotate_embeddings(model, Q, device, dtype)
    rotate_head(model, Q, device, dtype)
    gc.collect()
    torch.cuda.empty_cache()
    layers = model.model.layers
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layers[idx], Q, device, dtype)
        rotate_attention_output(layers[idx], Q, device, dtype)
        rotate_mlp_input(layers[idx], Q, device, dtype)
        rotate_mlp_output(layers[idx], Q, device, dtype)
        rotate_ov_proj(layers[idx], num_heads, head_dim)
    return rotation_signs
