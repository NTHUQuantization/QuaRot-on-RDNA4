import math
import os
import torch
import quarot
import fast_hadamard_transform


class ShapeHandler:
    def __init__(self, x: torch.Tensor):
        self.size_excl_last = x.numel()//x.shape[-1]
        self.shape_excl_last = tuple(x.shape[:-1])

    # Keep the last dim unchanged, flatten all previous dims
    def flatten(self, x: torch.Tensor):
        return x.view(self.size_excl_last, -1)

    # Recover back to the original shape.
    def unflatten(self, x: torch.Tensor):
        return x.view(self.shape_excl_last + (-1,))

    def unflatten_scale(self, x: torch.Tensor):
        return x.view(self.shape_excl_last)


class Linear4bit(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, dtype=torch.float16):
        '''
        Symmetric 4-bit Linear Layer.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('weight_scales',
                             torch.zeros((self.out_features, 1), requires_grad=False))
        self.register_buffer('weight', (torch.randint(1, 7, (self.out_features, self.in_features // 2),
                                                             # SubByte weight
                                                             dtype=torch.uint8, requires_grad=False)))
        if bias:
            self.register_buffer('bias', torch.zeros((self.out_features), dtype=dtype))
        else:
            self.bias = None

        self._weight_is_prepacked = False
    def _prepack_weight(self):
        if self._weight_is_prepacked or not self.weight.is_cuda:
            return
        prepacked = quarot._HIP.prepack_b(self.weight.contiguous())
        expected = self.out_features * (self.in_features // 2)
        if prepacked.numel() != expected:
            raise RuntimeError(
                "prepacked weight padding is unsupported for this Linear4bit shape")
        # Replace row-packed storage; do not retain a second global-cache copy.
        self.weight = prepacked.view(self.out_features, self.in_features // 2)
        self._weight_is_prepacked = True

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        if self.weight.is_cuda:
            self._prepack_weight()
        return result

    def _load_from_state_dict(self, *args, **kwargs):
        # Existing checkpoints contain conventional row-packed weights.
        self._weight_is_prepacked = False
        return super()._load_from_state_dict(*args, **kwargs)


    def forward(self, x):
        #if torch.cuda.current_device() != x.device:
        #    torch.cuda.set_device(x.device)

        assert type(x) == quarot.PackedQuantizedTensor #Quantized input is given
        x, scales_x = x.quantized_x, x.scales_x
        #shape_handler = ShapeHandler(quantized_x)
        #quantized_x = shape_handler.flatten(quantized_x)
        self._prepack_weight()
        # The grouped-scale GEMM is also the canonical one-scale GEMM. It
        # writes FP16 directly, avoiding an M=1 -> M=16 pad, an INT32 output
        # allocation, and a second dequantization kernel during decode.
        rows = x.numel() // (self.in_features // 2)
        use_legacy_epilogue = (
            os.getenv("QUAROT_GROUPED_SCALE_GEMM", "1") == "0"
            and scales_x.numel() == rows)
        if use_legacy_epilogue:
            accum = quarot.matmul_bpre(
                x, self.weight, self.out_features, self.in_features)
            output = quarot.sym_dequant(
                accum, scales_x, self.weight_scales)
        else:
            output = quarot.matmul_bpre_grouped_scale(
                x, self.weight, scales_x, self.weight_scales,
                self.out_features, self.in_features)
        return output + self.bias if self.bias is not None else output

    @staticmethod
    def fused_forward(x, *projections):
        """Project one packed activation through two or three INT4 weights."""
        if len(projections) not in (2, 3):
            raise ValueError("fused_forward requires two or three projections")
        for projection in projections:
            projection._prepack_weight()
        if (os.getenv("QUAROT_INTERLEAVED_PROJECTIONS", "1") != "0" and
                not all(getattr(projection, "_interleaved_fused", False)
                   for projection in projections)):
            weight_sizes = [projection.weight.numel()
                            for projection in projections]
            scale_sizes = [projection.weight_scales.numel()
                           for projection in projections]
            combined_weight = torch.cat(
                [projection.weight.reshape(-1) for projection in projections])
            combined_scales = torch.cat(
                [projection.weight_scales.reshape(-1)
                 for projection in projections])
            weight_offset = 0
            scale_offset = 0
            for projection, weight_size, scale_size in zip(
                    projections, weight_sizes, scale_sizes):
                projection.weight = combined_weight[
                    weight_offset:weight_offset + weight_size].view_as(
                        projection.weight)
                projection.weight_scales = combined_scales[
                    scale_offset:scale_offset + scale_size].view_as(
                        projection.weight_scales)
                projection._interleaved_fused = True
                weight_offset += weight_size
                scale_offset += scale_size
        packed, scales = x.quantized_x, x.scales_x
        shape = packed.shape[:-1]
        packed = packed.view(-1, packed.shape[-1]).contiguous()
        scales = scales.view(packed.shape[0], -1)
        if scales.shape[1] != 1:
            # The shared-input kernel accumulates once across K. Group-scaled
            # H256 down projections continue through the regular grouped path.
            return tuple(projection(x) for projection in projections)
        p0, p1 = projections[:2]
        p2 = projections[2] if len(projections) == 3 else None
        output = quarot._HIP.matmul_bpre_multi_scale(
            packed, scales.contiguous(),
            p0.weight.contiguous(), p0.weight_scales.view(-1).contiguous(),
            p1.weight.contiguous(), p1.weight_scales.view(-1).contiguous(),
            None if p2 is None else p2.weight.contiguous(),
            None if p2 is None else p2.weight_scales.view(-1).contiguous(),
            p0.out_features, p1.out_features,
            0 if p2 is None else p2.out_features, p0.in_features)
        widths = [p.out_features for p in projections]
        chunks = tuple(chunk.view(*shape, width) for chunk, width in
                       zip(output.split(widths, dim=-1), widths))
        return tuple(chunk if projection.bias is None else chunk + projection.bias
                     for chunk, projection in zip(chunks, projections))

    @staticmethod
    def from_float(module: torch.nn.Linear, weight_scales=None,):
        '''
        Generate a new Linear4bit module from a FP16 Linear module.
        The weight matrix should have the same shape as the weight matrix of the FP16 Linear module and rounded using torch.round()
        routine. We will convert it to subByte representation and save it in the int_weight buffer.
        '''
        weight_matrix = module.weight.data


        int_module = Linear4bit(module.in_features, module.out_features, bias=module.bias is not None, dtype=weight_matrix.dtype).to(weight_matrix.dtype)
        if weight_scales is not None:
            assert weight_scales.shape == (module.out_features, 1), 'weight_scales should have shape (out_features, 1)'
            weight_matrix = weight_matrix.cuda()
            int_module.weight_scales.copy_(weight_scales.to(weight_matrix.dtype))
            int_rounded_weight = (weight_matrix/weight_scales.cuda()).round()
            int_module.weight.copy_(quarot.functional.pack_i4(int_rounded_weight.to(torch.int8)).cpu())

            if module.bias is not None:
                int_module.bias.copy_(module.bias)

        return int_module
