# Third-party notices

This repository adapts and builds on the projects below. The root [LICENSE](LICENSE) contains the Apache License 2.0 inherited from QuaRot. Original copyright and license notices present in retained source files remain applicable. Model weights and calibration datasets are obtained separately under their own licenses.

## QuaRot

- Upstream: [spcl/QuaRot](https://github.com/spcl/QuaRot)
- Research: [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456)
- Upstream license: [Apache-2.0](https://github.com/spcl/QuaRot/blob/main/LICENSE)

The quantization and rotation utilities, packed INT4 abstractions, model-integration structure, and customized quantized-attention code derive from QuaRot. This project adapts that implementation to HIP/RDNA 4 and dense Qwen3, adding GFX12 GEMM, fused operators, checkpoint metadata, and speculative-decoding integration.

## FlashInfer

- Upstream: [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)
- Upstream license: [Apache-2.0](licenses/FlashInfer-Apache-2.0.txt)

The headers under `quarot/kernels/include_hip/flashinfer/` and their integration implement the HIP adaptation of QuaRot's customized FlashInfer attention and KV-cache code. They are modified project sources, rather than an unmodified installation of the current upstream FlashInfer package.

## QuIP-sharp Hadamard utilities

- Upstream: [Cornell-RelaxML/quip-sharp](https://github.com/Cornell-RelaxML/quip-sharp)
- Original utility: [`lib/utils/matmul_had.py`](https://github.com/Cornell-RelaxML/quip-sharp/blob/main/lib/utils/matmul_had.py)
- Upstream license: [GNU GPL v3](licenses/QuIP-sharp-GPL-3.0.txt)

`quarot/functional/hadamard.py` explicitly attributes its Hadamard utilities to QuIP-sharp. The inherited QuaRot root license does not replace this component's upstream license. Its source attribution and the upstream license text are retained.

## Hadamard transforms

- Research: [HadaCore: Tensor Core Accelerated Hadamard Transform Kernel](https://arxiv.org/abs/2412.08832)
- Related interface: [Dao-AILab/fast-hadamard-transform](https://github.com/Dao-AILab/fast-hadamard-transform)

`third-party/hadacore/` contains the project's HIP Hadamard implementation and exports the `fast_hadamard_transform` module consumed by QuaRot. `components/wave/` retains the final standalone Wave implementation. These directories contain the project's adapted implementations and dispatch choices, rather than a pinned checkout of the upstream CUDA libraries.

## PARD and PARD-2

- Upstream: [AMD-AGI/PARD](https://github.com/AMD-AGI/PARD)
- Source revision used for integration: `6f279bf3f1680e0b5d71c562ca5b91bdeef4c038`
- Upstream license: [MIT](https://github.com/AMD-AGI/PARD/blob/master/LICENSE)

The speculative-decoding integration follows PARD-2's parallel drafting and target-feature contracts. Official drafter snapshots are loaded separately from `amd/PARD2-Qwen3-8B` and `amd/PARD2-Qwen3-14B`; their weights are not redistributed here. The runtime retains its own HIP quantized target, verification, and KV-cache commit logic.

## Transformers and Qwen3

- Model implementation: [Hugging Face Transformers](https://github.com/huggingface/transformers)
- Model family: [Qwen3](https://github.com/QwenLM/Qwen3)

The model adapters subclass Transformers' Qwen3 classes. The original Qwen3 targets, tokenizers, and PARD-2 drafters are external resources. Their pinned revisions and roles are documented in [checkpoint preparation](docs/checkpoints.md).

## Other dependencies

PyTorch, ROCm/HIP, FlashAttention, Hugging Face Hub, Datasets, Safetensors, and the other installed dependencies retain their upstream licenses. Installing this project does not replace those licenses or the terms attached to downloaded models and datasets.
