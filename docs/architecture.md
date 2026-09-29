# Architecture

This guide maps the final system to its implementation. The supported inference contract is dense Qwen3 GPTQ W4A4KV4 on RDNA 4, with single-request greedy generation and optional PARD-2 drafting.

## Quantization and rotations

Checkpoint conversion folds residual-stream RMSNorm scales and compatible orthogonal rotations into model weights before GPTQ calibration. Activations are rotated online before quantization. Qwen3's learned Q/K head-normalization scales remain explicit, and attention retains the model's native KV head count.

Weights use symmetric INT4 with a scale per output channel. Most activations use symmetric quantization per token row with clipping ratio `0.9`. The FFN down-projection input instead uses independent 256-channel Hadamard groups and group scales. Cached keys and values use asymmetric INT4 with FP16 scales and offsets per token and KV head. Floating-point normalization, nonlinearities, and attention arithmetic remain part of the execution path.

Conversion lives in [`e2e/checkpoint_utils/`](../e2e/checkpoint_utils/); Qwen3 model integration lives in [`e2e/quantized_qwen3/`](../e2e/quantized_qwen3/) and [`e2e/quantized_common.py`](../e2e/quantized_common.py).

## Offline BPre INT4 GEMM

[`quarot/kernels/gemm.hip`](../quarot/kernels/gemm.hip) maps a wave32 to a 16-by-16 output tile using native GFX12 INT4 WMMA instructions. The weight operand is prepared in the lane-major layout consumed by those instructions, avoiding repeated rearrangement during inference.

The checkpoint stores ordinary row-packed INT4 weights. [`Linear4bit`](../quarot/nn/linear.py) creates and retains the prepacked representation when moving the layer to the GPU or on first use. This preparation is outside steady-state token generation. The grouped-scale path incorporates activation and weight scaling into its FP16 output, and joint projections share a packed activation input.

## Wave and integrated Hadamard dispatch

The final standalone **Wave** component uses wave shuffles and register-local butterflies. H32 and H64 use the Small Warp path; H128 and H256 use Warp Regs, with four or eight values per lane. This implements the report's standalone optimized FHT component without intermediate LDS traffic between butterfly stages. Its source is retained in [`components/wave/`](../components/wave/).

The fused grouped FFN and across-head transforms also use register/shuffle butterflies. The inference build uses [`third-party/hadacore/`](../third-party/hadacore/) and fused transforms inside [`quarot/kernels/fused_hip.hip`](../quarot/kernels/fused_hip.hip). This integrated extension preserves the final runtime's existing dtype- and shape-dependent dispatch, including the FP16 Hadamard paths used by verification. The standalone Wave component is separate from the default inference build. Substituting it globally would change the retained runtime's numerical execution path.

## Fused decoder operators

The report's fusion boundaries are reflected in these components:

| Report operator | Function | Main source |
| --- | --- | --- |
| KNQ | Row RMSNorm, activation scale, and INT4 packing | `quarot/nn/normalization.py`, `quarot/kernels/fused_hip.hip` |
| KQKV / KGU | Joint Q/K/V or gate/up projection | `quarot/nn/linear.py`, `quarot/kernels/gemm.hip` |
| KSHQ | SwiGLU, grouped H256, and INT4 packing | `quarot/kernels/fused_hip.hip` |
| KQ | Verification Q head norm, RoPE, and within-head H128 | `quarot/kernels/verification_preprocess.hip` |
| KKV | Verification K preprocessing plus direct KV4 cache append | `quarot/kernels/verification_preprocess.hip` |
| KAttn | Attention over the paged native-GQA cache | `quarot/kernels/flashinfer.hip` |
| KHQ | Attention-output head transform and quantization | `quarot/kernels/fused_hip.hip` |

These stages retain intermediates in registers or LDS when their workgroup owns all required data. Gate/up GEMM remains separate from the grouped FFN transform. Query preprocessing and KV preprocessing use separate kernels because their head counts and destinations differ. Prefill, single-token decode, and multi-token verification retain their corresponding dispatch paths.

The attention illustration in the report uses Qwen3-8B's 32 query heads and head dimension 128. The shared model code keeps each supported model's own dimensions; those illustrated sizes are not universal constants for 14B and 32B.

## PARD-2 verification

[`e2e/speculative.py`](../e2e/speculative.py) implements drafting, greedy verification, and cache commit logic:

1. Prefill the target and establish the next target-token prediction.
2. Draft 15 candidates in parallel using the BF16 PARD-2 model.
3. Verify the candidates with the W4A4KV4 target, accepting the longest matching prefix and emitting a correction or bonus token.
4. Keep that emitted token pending until the next target call, which processes it together with the next candidates.

The first verification has 15 rows. Subsequent verification has one pending token plus 15 candidates, giving 16 rows. Both fit within a single 16-row WMMA tile, as does one-token autoregressive decoding. Drafting, attention, and other operations still incur work; this tile relationship alone does not guarantee a generation speedup.

Candidate K/V entries are appended provisionally. Acceptance commits the logical prefix length; rejected positions are overwritten on later appends. Cache contents do not need to be moved. The correctness reference is greedy output from the same quantized target.

TI drafting consumes token context. TD additionally consumes selected target hidden features, restoring the residual basis, normalization, and learned scales expected by the drafter. The metadata needed for this operation is part of the converted checkpoint contract. See [checkpoint preparation](checkpoints.md).

## Metadata reuse and HIP graphs

[`quarot/transformers/kv_cache.py`](../quarot/transformers/kv_cache.py) and [`verification_metadata.hip`](../quarot/kernels/verification_metadata.hip) maintain reusable cache metadata and position buffers. Device-side updates derive append positions and causal lengths from the committed prefix.

[`e2e/verification_graph.py`](../e2e/verification_graph.py) warms and captures fixed 15-row and 16-row target passes using stable buffer addresses. Replaying a graph reduces host dispatch while retaining its individual kernels. Drafting and acceptance decisions remain outside capture, and TD features remain available after replay.

The documented integrated configuration uses:

| Control | Value | Purpose |
| --- | --- | --- |
| `--fused-norm-quant` | Enabled | Exact row normalization fused with INT4 packing |
| `QUAROT_BATCHED_H128` | `1` | Batched head-transform dispatch |
| `QUAROT_STATIC_KV_METADATA` | `1` | Reuse device-side verification metadata |
| `QUAROT_VERIFICATION_GRAPH` | `1` | Capture/replay fixed verification shapes |
| `QUAROT_CHUNK_PREPROCESS` | `1` | Fused multi-token Q/KV preprocessing when eligible |
| `--compile-mode` | `eager` | Leave drafter execution eager |

The generation CLI enables the four environment switches above by default while honoring explicit caller overrides. The single-token preprocessing path retains the implementation's existing configuration-dependent default.

`torch.cuda` graph and device APIs are used through PyTorch's ROCm backend. Shape and dtype guards remain in the original implementation; a requested optimization only runs when its eligibility checks pass.
