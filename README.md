# QuaRot on RDNA 4

**Architecture-Aware Acceleration for Quantized LLM Inference**

[繁體中文](README.zh-TW.md) · [Architecture](docs/architecture.md) · [Checkpoint preparation](docs/checkpoints.md)

This repository contains the implementation accompanying the National Tsing Hua University project *QuaRot on RDNA 4: Architecture-Aware Acceleration for Quantized LLM Inference*. It adapts QuaRot to AMD ROCm/HIP and dense Qwen3 models, combining GPTQ W4A4KV4 inference, RDNA 4 INT4 WMMA kernels, fused operators, and PARD-2 speculative decoding.

The primary platform is the **AMD Radeon AI PRO R9700 (32 GB, `gfx1201`)**. The release retains the final implementation, checkpoint conversion, build sources, and correctness checks. The core kernels, model execution, and conversion algorithms are preserved; the command-line interface, packaging, and documentation are organized for standalone use. Model weights, benchmark datasets, measurement outputs, machine-specific launch scripts, and superseded experiments are excluded.

## Supported inference

| Target | Model profile | Autoregressive | PARD-2 TI | PARD-2 TD |
| --- | --- | --- | --- | --- |
| Qwen3-8B | `qwen3_8b` | Yes | Yes | Yes |
| Qwen3-14B | `qwen3_14b` | Yes | Yes | Yes |
| Qwen3-32B | `qwen3_32b` | Yes | Yes | — |

All targets use GPTQ weights, 4-bit activations, and a paged 4-bit KV cache with native grouped-query attention. Normalization, nonlinearities, and selected other operations remain floating point. The drafter uses BF16.

The supported generation contract is one request at a time, greedy decoding, thinking disabled, and normal EOS termination. PARD-2 drafts 15 tokens in parallel: verification processes 15 positions initially and 16 positions when a pending token is included. TI uses the shared `amd/PARD2-Qwen3-8B` drafter; TD uses the drafter aligned with the target size. See [checkpoint preparation](docs/checkpoints.md) for fixed model revisions and conversion requirements.

## Build

Start in a Linux environment with a working ROCm PyTorch installation, the HIP compiler, and a compatible ROCm FlashAttention installation. The project environment used **ROCm 7.2, PyTorch 2.9.1, Transformers 4.57.6, and a ROCm build of FlashAttention 2.8.3**. PyTorch retains the `torch.cuda` API name on ROCm, so `cuda` in source and commands refers to the AMD GPU in this environment.

```bash
git clone https://github.com/NTHUQuantization/QuaRot-on-RDNA4.git
cd QuaRot-on-RDNA4

python -m pip install -r requirements.txt
python -m pip install -e third-party/hadacore --no-build-isolation
python -m pip install -e . --no-build-isolation
```

The extensions build for `gfx1201` by default. Rebuild both extensions after changing their HIP sources or the ROCm/PyTorch toolchain. The local `fast_hadamard_transform` extension must be built before importing `quarot`.

```bash
python -c "import torch, quarot, fast_hadamard_transform; print('PyTorch:', torch.__version__); print('HIP:', torch.version.hip); print('GPU:', torch.cuda.get_device_name(0))"
```

## Prepare a model

Download the pinned dense target and drafter snapshots, then convert the target with the [checkpoint guide](docs/checkpoints.md). A generic INT4 checkpoint is not interchangeable with this runtime: the grouped Hadamard format, rotation metadata, Qwen3 Q/K norms, and quantization settings must agree.

The following examples assume Qwen3-8B. Replace each placeholder with a local directory:

```bash
SOURCE=/path/to/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
TARGET=/path/to/qwen3-8b-quarot-gptq
DRAFT=/path/to/models--amd--PARD2-Qwen3-8B/snapshots/67a1516c8f6fc145cda99916799a0cbb3a4af135
```

Inference loads these resources locally. It does not download weights or choose a machine-specific checkpoint directory.

## Generate

Run commands from the repository root. The CLI enables the following final runtime settings by default; these explicit exports also override settings inherited from your shell:

```bash
export QUAROT_BATCHED_H128=1
export QUAROT_STATIC_KV_METADATA=1
export QUAROT_VERIFICATION_GRAPH=1
export QUAROT_CHUNK_PREPROCESS=1
```

Autoregressive generation uses the fused quantized target:

```bash
python -m e2e.pard2 \
  --mode ar --profile qwen3_8b \
  --target "$TARGET" --tokenizer "$SOURCE" \
  --compile-mode eager --fused-norm-quant \
  --max-cache-len 4096 --max-new-tokens 256 \
  --prompt 'Explain speculative decoding in one paragraph.'
```

PARD-2 TI adds the shared drafter:

```bash
python -m e2e.pard2 \
  --mode pard2-ti --profile qwen3_8b \
  --target "$TARGET" --tokenizer "$SOURCE" --draft "$DRAFT" \
  --compile-mode eager --fused-norm-quant \
  --max-cache-len 4096 --max-new-tokens 256 \
  --prompt 'Explain speculative decoding in one paragraph.'
```

For Qwen3-8B TD, change `--mode pard2-ti` to `--mode pard2-td`. For Qwen3-14B, use `qwen3_14b`, the matching target/tokenizer, and the 14B drafter for TD; TI continues to use the 8B drafter. Qwen3-32B uses `qwen3_32b` and supports AR or TI.

`--profile` selects the model contract; `--benchmark-profile` remains an alias for existing callers. `--compile-mode eager` controls drafter compilation independently of the target HIP graph switch. Generation prints JSON containing generated text and runtime information. Allow cache capacity for the prompt, generated tokens, and speculative verification positions.

## Source map

| Path | Purpose |
| --- | --- |
| [`quarot/kernels/`](quarot/kernels/) | INT4 WMMA GEMM, fused preprocessing, quantization, and paged attention kernels |
| [`quarot/nn/`](quarot/nn/) | Packed INT4 layers, normalization, and online rotations |
| [`quarot/transformers/`](quarot/transformers/) | Paged KV4 cache and runtime integration |
| [`e2e/quantized_qwen3/`](e2e/quantized_qwen3/) | Quantized Qwen3 model classes |
| [`e2e/checkpoint_utils/`](e2e/checkpoint_utils/) | Streaming GPTQ conversion and rotation metadata |
| [`e2e/speculative.py`](e2e/speculative.py) | PARD-2 drafting, verification, acceptance, and TD feature handling |
| [`e2e/verification_graph.py`](e2e/verification_graph.py) | Fixed-shape target graph capture and replay |
| [`third-party/hadacore/`](third-party/hadacore/) | Integrated HIP Hadamard extension |
| [`components/wave/`](components/wave/) | Standalone final Wave Hadamard component |
| [`tests/`](tests/) | Retained correctness and runtime-contract tests |

The [architecture guide](docs/architecture.md) connects the report's operators to their source files and explains the distinction between the standalone Wave component and the integrated runtime's dispatch.

## Correctness checks

After building the extensions, install the test dependency and run the retained suite:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

GPU checks require the ROCm environment above. The standalone Wave component has separate build and test instructions in its [README](components/wave/README.md).

## Attribution and license

This project builds on [QuaRot](https://github.com/spcl/QuaRot), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [HadaCore](https://arxiv.org/abs/2412.08832), [PARD/PARD-2](https://github.com/AMD-AGI/PARD), and the Qwen3 implementation in Transformers. The repository retains QuaRot's [Apache-2.0 license](LICENSE); see [third-party notices](THIRD_PARTY_NOTICES.md) for component-specific licenses, including the QuIP-sharp-derived Hadamard utilities. Downloaded models and datasets retain their own licenses.
