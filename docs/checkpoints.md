# Checkpoint preparation

The final target models are dense Qwen3-8B, Qwen3-14B, and Qwen3-32B converted to GPTQ W4A4KV4. Model weights and calibration data are downloaded separately and are not part of this repository.

## Pinned resources

| Resource | Revision | Role |
| --- | --- | --- |
| `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | 8B source target and tokenizer |
| `Qwen/Qwen3-14B` | `40c069824f4251a91eefaf281ebe4c544efd3e18` | 14B source target and tokenizer |
| `Qwen/Qwen3-32B` | `9216db5781bf21249d130ec9da846c4624c16137` | 32B source target and tokenizer |
| `amd/PARD2-Qwen3-8B` | `67a1516c8f6fc145cda99916799a0cbb3a4af135` | TI for all three targets; TD for 8B |
| `amd/PARD2-Qwen3-14B` | `679eff0b65ffaf5abd2dadd21a17909562935798` | TD for 14B |

Preserve the Hugging Face cache layout, including `models--OWNER--NAME/snapshots/REVISION`. The preflight checks use the snapshot revision and checkpoint metadata to validate model identity.

For example, download the 8B resources with the installed Hugging Face Hub library:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

resources = [
    ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
    ("amd/PARD2-Qwen3-8B", "67a1516c8f6fc145cda99916799a0cbb3a4af135"),
]
for model_id, revision in resources:
    path = snapshot_download(model_id, revision=revision)
    print(f"{model_id}: {path}")
PY
```

Use the returned absolute paths for `SOURCE` and `DRAFT`. Download the corresponding source target for 14B or 32B, and the 14B drafter when using 14B TD.

## Streaming GPTQ conversion

Build the extensions as described in the [README](../README.md#build) before conversion. Run from the repository root:

```bash
SOURCE=/path/to/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
TARGET=/path/to/qwen3-8b-quarot-gptq

python -m e2e.checkpoint_utils.quantize_checkpoint \
  --model "$SOURCE" --output "$TARGET" \
  --quant-method gptq --cal-dataset wikitext2 \
  --nsamples 128 --seqlen 2048 --seed 0 \
  --w-groupsize -1 --w-clip --percdamp 0.01 \
  --rotation-device cuda --rotation-dtype float32
```

Use the same conversion settings with the pinned 14B or 32B source. The converter streams source tensors and processes one decoder layer at a time. It still needs GPU memory for calibration activations and the current layer, along with host memory and disk space for the source and output weights. `--rotation-device cpu` only moves offline rotation operations to the CPU; GPTQ calibration still uses a GPU.

The canonical settings are:

| Setting | Value |
| --- | --- |
| Weight precision | Signed symmetric INT4, per output channel |
| GPTQ calibration | WikiText-2 training split, 128 sequences of 2,048 tokens |
| Seed | `0` |
| Weight groupsize | `-1` (per output channel) |
| Weight clipping | Enabled |
| GPTQ damping | `0.01` |
| Activation ordering | Disabled |
| Offline rotation arithmetic | FP32 on the GPU |
| Activation clipping ratio | `0.9` |
| FFN transform and activation scaling | Independent groups of 256 channels |

Do not add `--w-asym` or `--act-order` when preparing this configuration. WikiText-2 is used here for GPTQ calibration; no evaluation dataset or measurement artifacts need to be added to the repository.

## Output contract

The converter writes sharded `safetensors` weights, their index, the tokenizer, a model configuration, and runtime support files. Keep the full output directory together. Relevant configuration fields include:

| Field | Expected value or purpose |
| --- | --- |
| `model_type` | `qwen3_quarot` |
| `quarot_checkpoint_format_version` | `2` |
| `quarot_ffn_format` | `grouped_h256_v1` |
| `quarot_activation_clip_ratio` | `0.9` |
| `quarot_conversion` | GPTQ parameters and conversion signature |
| `quarot_source_model_id`, `quarot_source_revision` | Source-model identity derived from the pinned snapshot |
| Rotation metadata | Residual rotation signs and basis information needed by the runtime and TD feature restoration |

Qwen3's learned Q/K normalization weights are preserved. Residual-stream RMSNorm scales and compatible rotations are folded into projection weights before calibration. TD additionally needs the original final-normalization information recorded by conversion; copying only packed weights is insufficient.

The persisted weights use conventional row-packed INT4 storage. The runtime prepares the Offline BPre layout once when each `Linear4bit` layer is moved to the GPU or first used, then reuses it during inference. Do not save a model after runtime prepacking and treat it as a newly converted checkpoint.

Conversion can reuse completed layer shards when the conversion signature matches. Use a fresh output directory when changing the source model or quantization configuration; keep source paths and parameters stable when resuming an interrupted conversion.

## Draft checkpoints

Pass the original pinned BF16 PARD-2 snapshot through `--draft`. TI only needs token-context drafting. TD also loads the target-feature projection from `warp_model.bin`, so retain the complete drafter snapshot. The target checkpoint and tokenizer must match the selected model profile; the drafter's vocabulary and architecture are checked before the large GPU allocations.

No drafter conversion, adaptation training, or custom TD calibration is required for the documented 8B/14B workflows.
