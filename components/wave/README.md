# Wave Hadamard transform

This optional component provides the report's standalone Wave FHT: Small Warp
for H32/H64 and Warp Regs for H128/H256. One wave32 owns each row. Intermediate
values remain in registers, with XOR shuffles for communication between lanes.
Inputs and outputs are FP16 or BF16; the butterfly arithmetic uses FP32.

The selected device helper, butterfly, and wrapper bodies are copied unchanged
from the final component source. Only these four widths are instantiated. The
original eight-wave workgroup layout, normalization, and output conversion are
preserved. [source_manifest.json](source_manifest.json) records their origin and
hashes.

## Build and use

Use the ROCm/PyTorch environment described in the [root README](../../README.md).
From the repository root:

```bash
python -m pip install ./components/wave --no-build-isolation
python -m pytest -q components/wave/test_wave.py
```

The build targets `gfx1201` by default. `QUAROT_HIP_ARCHS` overrides the target.

```python
import torch
from quarot_wave_fht import hadamard_transform

x = torch.randn(17, 256, device="cuda", dtype=torch.float16)
y = hadamard_transform(x, scale=256 ** -0.5)
```

The operation applies the Sylvester Hadamard matrix to the last dimension and
multiplies by `scale`. As in the original binding, `scale=1.0` is the default;
pass `width ** -0.5` for an orthonormal transform. The original implementation
first produces a normalized result in the input dtype, then applies any
remaining scale. Leading dimensions and noncontiguous inputs are supported.

## Relation to the inference runtime

`quarot_wave_fht` is independent of the inference runtime's
`fast_hadamard_transform` dependency. Installing this component leaves the
runtime's dispatch unchanged. The fused grouped FFN and across-head paths use
register/shuffle butterflies within their fused kernels. Quantization-sensitive
H128 query preprocessing retains the deployed FP16 WMMA calculation and safe
eight-row workgroups. This separation preserves the final runtime's numerical
behavior while retaining the report's selected standalone transform.
