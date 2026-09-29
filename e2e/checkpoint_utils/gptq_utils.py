import gc
import math
import time
import tqdm
import torch
import torch.nn as nn
import logging
import os
import shutil
from pathlib import Path

from quarot.functional import asym_quant_dequant, sym_quant_dequant

_RTN_CACHE_VERSION = 3

def _rtn_cache_signature(args, layer):
    return {
        "version": _RTN_CACHE_VERSION,
        "model": str(getattr(args, "model", "")),
        "seed": int(getattr(args, "seed", 0)),
        "rotation_device": str(getattr(args, "rotation_device", "")),
        "rotation_dtype": str(getattr(args, "rotation_dtype", "")),
        "w_bits": int(args.w_bits),
        "w_groupsize": int(args.w_groupsize),
        "w_asym": bool(args.w_asym),
        "w_clip": bool(args.w_clip),
        "parameters": {name: (tuple(p.shape), str(p.dtype))
                       for name, p in layer.named_parameters()},
    }

def _serialize_quantizer(q):
    return {
        "bits": q.bits, "perchannel": q.perchannel, "sym": q.sym,
        "mse": q.mse, "norm": q.norm, "grid": q.grid,
        "maxshrink": q.maxshrink, "maxq": q.maxq.detach().cpu(),
        "scale": q.scale.detach().cpu(), "zero": q.zero.detach().cpu(),
    }

def _deserialize_quantizer(state):
    q = WeightQuantizer()
    q.configure(state["bits"], perchannel=state["perchannel"],
                sym=state["sym"], mse=state["mse"], norm=state["norm"],
                grid=state["grid"], maxshrink=state["maxshrink"])
    q.maxq, q.scale, q.zero = state["maxq"], state["scale"], state["zero"]
    return q

def _apply_cached_quantizers(layer, quantizers):
    modules = dict(layer.named_modules())
    for name, quantizer in quantizers.items():
        module = modules[name]
        module.weight.data = quantizer.quantize(module.weight.data)

def _load_rtn_layer_cache(path, signature, layer):
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("signature") != signature:
            logging.warning("Ignoring incompatible RtN cache entry %s", path)
            return None
        quantizers = {name: _deserialize_quantizer(state)
                      for name, state in payload["quantizers"].items()}
        expected_quantizers = {name for name, module in layer.named_modules()
                               if type(module) is torch.nn.Linear}
        valid_quantizers = all(
            quantizer.ready()
            and quantizer.scale.shape[0] == dict(layer.named_modules())[name].out_features
            for name, quantizer in quantizers.items())
        if quantizers.keys() != expected_quantizers or not valid_quantizers:
            logging.warning("Ignoring incomplete RtN cache entry %s", path)
            return None
        _apply_cached_quantizers(layer, quantizers)
        return quantizers
    except Exception as error:
        logging.warning("Ignoring unreadable RtN cache entry %s: %s", path, error)
        return None

def _save_rtn_layer_cache(path, signature, layer, quantizers):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "signature": signature,
        "quantizers": {name: _serialize_quantizer(q)
                       for name, q in quantizers.items()},
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        try:
            torch.save(payload, temporary)
        except (OSError, RuntimeError) as error:
            free = shutil.disk_usage(path.parent).free
            partial = temporary.stat().st_size if temporary.exists() else 0
            raise RuntimeError(
                f"Failed to write RtN cache entry {path} "
                f"({free / 2**30:.2f} GiB free; partial write "
                f"{partial / 2**20:.1f} MiB). Free space or choose a cache "
                "directory on a larger filesystem.") from error
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False



class WeightQuantizer(torch.nn.Module):
    '''From GPTQ Repo'''

    def __init__(self, shape=1):
        super(WeightQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, sym=True,
        mse=False, norm=2.4, grid=100, maxshrink=.8,
    ):
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if sym:
            self.maxq = torch.tensor(2**(bits-1)-1)
        else:
            self.maxq = torch.tensor(2**bits - 1)

    def find_params(self, x):
        if self.bits == 16:
            return
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq)
                else:

                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asym_quant_dequant(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)

                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:

            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)
        return

    # TODO: This should be better refactored into `forward`, which applies quantize and dequantize. A new method `quantize` should be added (if needed) to return the quantized integers and scales, like in ActQuantizer.
    def quantize(self, x):
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            if self.sym:
                return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
            return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
        W = self.layer.weight.data.clone()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        # This ROCm PyTorch build has neither MAGMA nor CPU LAPACK. Use
        # SciPy's LAPACK for the relatively small per-layer Hessian, then
        # return the inverse factor to the weight device for GPTQ updates.
        import numpy as np
        import scipy.linalg
        h_device = H.device
        h_numpy = H.float().cpu().numpy()
        chol = scipy.linalg.cholesky(
            h_numpy, lower=True, overwrite_a=True, check_finite=False)
        inverse = scipy.linalg.cho_solve(
            (chol, True), np.eye(chol.shape[0], dtype=chol.dtype),
            overwrite_b=True, check_finite=False)
        inverse_factor = scipy.linalg.cholesky(
            inverse, lower=False, overwrite_a=True, check_finite=False)
        Hinv = torch.from_numpy(inverse_factor).to(h_device)

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)])
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if actorder:
            Q = Q[:, invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning('NaN in weights')
            import pprint
            pprint.pprint(self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point)
            raise ValueError('NaN in weights')

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()


@torch.no_grad()
def gptq_fwrd(model, dataloader, dev, args):
    '''
    From GPTQ repo
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    logging.info('-----GPTQ Quantization-----')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            cache['position_embeddings'] = kwargs.get('position_embeddings')
            cache['cache_position'] = kwargs.get('cache_position')
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache.get('position_embeddings')
    cache_position = cache.get('cache_position')

    quantizers = {}
    sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]
    for i in range(len(layers)):
        print(f'\nLayer {i}:', flush=True, end=' ')
        layer = layers[i].to(dev)
        full = dict(layer.named_modules())
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                print(f'{name}', end='  ', flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = not(args.w_asym)
                if 'lm_head' in name:
                    layer_weight_bits = 16
                    continue
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits, perchannel=True, sym=layer_weight_sym, mse=args.w_clip
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings, cache_position=cache_position)[0]
            for h in handles:
                h.remove()

            for name in subset:
                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=layer_w_groupsize, actorder=args.act_order, static_groups=False
                )
                quantizers['model.layers.%d.%s' % (i, name)] = (
                    gptq[name].quantizer.cpu())
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings, cache_position=cache_position)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    gc.collect()
    torch.cuda.empty_cache()
    logging.info('-----GPTQ Quantization Done-----\n')
    return quantizers




@torch.no_grad()
def rtn_fwrd(model, dev, args):
    '''
    From GPTQ repo
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    assert args.w_groupsize ==-1, "Groupsize not supported in RTN!"
    layers = model.model.layers
    torch.cuda.empty_cache()

    quantizers = {}
    cache_dir_arg = getattr(args, "rtn_cache_dir", None)
    cache_dir = Path(cache_dir_arg) if cache_dir_arg else None

    for i in tqdm.tqdm(range(len(layers)), desc="(RtN Quant.) Layers"):
        signature = _rtn_cache_signature(args, layers[i])
        cache_path = cache_dir / f"layer_{i:05d}.pt" if cache_dir else None
        cached = (_load_rtn_layer_cache(cache_path, signature, layers[i])
                  if cache_path else None)
        if cached is not None:
            quantizers.update({f"model.layers.{i}.{name}": q
                               for name, q in cached.items()})
            logging.info("Loaded RtN layer %d from %s", i, cache_path)
            continue
        layer = layers[i].to(dev)
        layer_quantizers = {}

        for name, module in layer.named_modules():
            if type(module) != torch.nn.Linear:
                continue
            layer_weight_bits = args.w_bits
            if 'lm_head' in name:
                layer_weight_bits = 16
                continue

            quantizer = WeightQuantizer()
            quantizer.configure(
                layer_weight_bits, perchannel=True, sym=not(args.w_asym), mse=args.w_clip
            )
            W = module.weight.data
            quantizer.find_params(W)
            module.weight.data = quantizer.quantize(W).to(
                next(iter(layer.parameters())).dtype)
            quantizers['model.layers.%d.%s' % (i, name)] = quantizer.cpu()
            layer_quantizers[name] = quantizer
        layers[i] = layer.cpu()
        if cache_path:
            _save_rtn_layer_cache(
                cache_path, signature, layers[i], layer_quantizers)
        torch.cuda.empty_cache()
        del layer

    gc.collect()
    torch.cuda.empty_cache()
    return quantizers
