"""PARD2-TI/TD speculative decoding for the fused dense QuaRot runtime.

The target always remains the W4A4KV4 fused runtime.  The official PARD2
drafter is loaded from a pinned local Hugging Face snapshot; TI deliberately
does not load ``warp_model.bin``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Iterable

import torch


TD_PROXY_QWEN3_14B_ON_QWEN3_32B = "qwen3-14b-on-qwen3-32b"
PARD2_PROFILES = ("qwen3_8b", "qwen3_14b", "qwen3_32b")


def _normalized_hadamard_cpu(
        value: torch.Tensor, *, transpose: bool = False) -> torch.Tensor:
    """Reference normalized HadK transform used by offline QuaRot.

    Unlike a Sylvester-only implementation, this follows ``get_hadK`` for
    non-power-of-two dense widths. Qwen3-32B's width 5120 uses the existing
    K=40 remainder and a 128-wide inner FHT.
    """
    from quarot.functional.hadamard import matmul_hadU

    if value.device.type != "cpu":
        raise ValueError("the reference HadK transform requires a CPU tensor")
    return matmul_hadU(
        value.float().contiguous(), transpose=bool(transpose))


def _inverse_hadamard_cuda(value: torch.Tensor, *, had_k=None,
                            remainder=None) -> torch.Tensor:
    """Apply normalized U^T in gfx1201-safe row chunks.

    ``matmul_had_HIP`` is a pure power-of-two FHT. TD restoration instead has
    to invert the exact generalized matrix used for the target checkpoint.
    The local HadK factory supplies the transposed remainder, while limiting
    each dispatch to eight rows preserves the established gfx1201 contract.
    """
    if not value.is_cuda:
        raise ValueError("the runtime inverse HadK transform requires CUDA/HIP")
    from quarot.functional.hadamard import get_hadK
    from quarot.transformers.kv_cache import matmul_had_HIP

    width = int(value.shape[-1])
    if remainder is None:
        had_k, remainder = get_hadK(width, transpose=True)
    remainder = int(remainder)
    rows = value.contiguous().view(-1, width)
    if remainder == 1:
        restored = matmul_had_HIP(rows, value.dtype)
        return restored.view(value.shape)
    if had_k is None or tuple(had_k.shape) != (remainder, remainder):
        raise ValueError("inverse HadK remainder matrix has an invalid shape")

    inner = width // remainder
    # matmul_hadU_cuda would expose rows * remainder FHT rows to hadacore in
    # one call. That violates gfx1201's <=8-row correctness limit. Reuse the
    # guarded inner FHT, then apply the small transposed remainder matrix.
    restored = matmul_had_HIP(
        rows.view(-1, remainder, inner), value.dtype)
    restored = torch.matmul(
        had_k.to(device=value.device, dtype=value.dtype), restored)
    restored = restored * (remainder ** -0.5)
    restored = restored.reshape(-1, width)
    return restored.view(value.shape)


def fold_td_projection_weight(weight: torch.Tensor, rotation_signs: torch.Tensor,
                              final_norm_weight: torch.Tensor,
                              hidden_size: int) -> torch.Tensor:
    """Fold QuaRot's ``D H`` basis restoration into the four TD weight blocks.

    PARD2 consumes row-vector features.  For rotated target activations
    ``x_rot = x D H``, the equivalent projection weights are
    ``W0' = W0 Gamma D H`` for the final normalized tap and
    ``Wi' = Wi D H`` for the remaining raw taps.  The transform is performed
    blockwise on CPU in FP32 and never constructs a dense Hadamard matrix.
    """
    hidden_size = int(hidden_size)
    if weight.ndim != 2 or weight.shape[1] != 4 * hidden_size:
        raise ValueError(
            "TD projection weight must have shape [output, 4 * hidden_size]")
    signs = rotation_signs.detach().cpu().float().reshape(-1)
    gamma = final_norm_weight.detach().cpu().float().reshape(-1)
    if signs.numel() != hidden_size:
        raise ValueError("rotation_signs width does not match TD hidden size")
    if gamma.numel() != hidden_size:
        raise ValueError("final_norm_weight width does not match TD hidden size")
    source = weight.detach().cpu().float()
    folded = []
    for index, block in enumerate(source.split(hidden_size, dim=1)):
        multiplier = signs * gamma if index == 0 else signs
        folded.append(_normalized_hadamard_cpu(block * multiplier))
    return torch.cat(folded, dim=1)


def _checkpoint_tensor(snapshot, key, rows=None):
    """Read one tensor (or a few leading rows) without materializing a model."""
    from safetensors import safe_open

    snapshot = Path(snapshot)
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        filename = weight_map.get(key)
        if filename is None:
            raise KeyError(f"{key} is absent from {index_path}")
    else:
        filename = "model.safetensors"
    with safe_open(snapshot / filename, framework="pt", device="cpu") as handle:
        tensor = handle.get_slice(key)
        return (tensor[:rows] if rows is not None else tensor[:]).clone()


def _basis_from_rotation_metadata(config):
    """Validate and materialize the self-contained v2 rotation metadata."""
    metadata = {
        name: getattr(config, name, None)
        for name in (
            "quarot_rotation_format", "quarot_rotation_width",
            "quarot_rotation_remainder", "quarot_rotation_inner",
            "quarot_rotation_seed", "quarot_rotation_device",
            "quarot_rotation_dtype", "quarot_rotation_signs",
            "quarot_final_norm_weight",
        )}
    if all(value is None for value in metadata.values()):
        return None
    missing = [name for name, value in metadata.items() if value is None]
    if missing:
        raise ValueError(
            f"incomplete QuaRot TD basis metadata: {', '.join(missing)}")
    saved_signs = metadata["quarot_rotation_signs"]
    saved_norm = metadata["quarot_final_norm_weight"]
    if metadata["quarot_rotation_format"] != "hadk_v1":
        raise ValueError("unsupported QuaRot TD rotation metadata format")

    from quarot.functional.hadamard import get_hadK

    hidden_size = int(config.hidden_size)
    width = int(getattr(config, "quarot_rotation_width", -1))
    remainder = int(getattr(config, "quarot_rotation_remainder", -1))
    inner = int(getattr(config, "quarot_rotation_inner", -1))
    _, expected_remainder = get_hadK(hidden_size)
    if width != hidden_size:
        raise ValueError("QuaRot rotation metadata width does not match target")
    if remainder != expected_remainder or inner != hidden_size // remainder:
        raise ValueError("QuaRot rotation metadata HadK factorization differs")

    signs = torch.tensor(saved_signs, dtype=torch.float32).reshape(-1)
    final_norm = torch.tensor(saved_norm, dtype=torch.float32).reshape(-1)
    if signs.numel() != hidden_size:
        raise ValueError("invalid quarot_rotation_signs checkpoint metadata")
    if final_norm.numel() != hidden_size:
        raise ValueError("invalid quarot_final_norm_weight checkpoint metadata")
    if not torch.all((signs == 1) | (signs == -1)):
        raise ValueError("quarot_rotation_signs must contain only -1 or +1")
    if not torch.isfinite(final_norm).all():
        raise ValueError("quarot_final_norm_weight contains non-finite values")
    return signs, final_norm


def load_td_target_basis(target, source_snapshot):
    """Load v2 TD basis metadata, with source recovery only for legacy 8B."""
    saved = _basis_from_rotation_metadata(target.config)
    if saved is not None:
        return saved
    if source_snapshot is None:
        raise ValueError(
            "legacy TD target lacks rotation metadata and no source was provided")

    source_rows = _checkpoint_tensor(
        source_snapshot, "model.embed_tokens.weight", rows=8).float()
    rotated_rows = target.get_input_embeddings().weight[:8].detach().cpu().float()
    hidden_size = int(target.config.hidden_size)
    if source_rows.shape[-1] != hidden_size:
        raise ValueError("legacy TD source hidden width differs from target")
    restored_columns = _normalized_hadamard_cpu(
        rotated_rows, transpose=True)
    signs = torch.sign((restored_columns * source_rows).sum(dim=0))
    signs[signs == 0] = 1
    reconstructed = _normalized_hadamard_cpu(source_rows * signs)
    rmse = (reconstructed - rotated_rows).square().mean().sqrt().item()
    if rmse > 0.003:
        raise RuntimeError(
            f"cannot recover fused target rotation signs (embedding RMSE {rmse:.5f})")
    final_norm = _checkpoint_tensor(source_snapshot, "model.norm.weight").float()
    return signs, final_norm


def _snapshot_revision(path):
    if path is None:
        return None
    parts = Path(path).resolve().parts
    for index, part in enumerate(parts[:-1]):
        if part == "snapshots":
            return parts[index + 1]
    return None


def _expect_config_fields(config, expected, label):
    for name, wanted in expected.items():
        actual = getattr(config, name, None)
        try:
            matches = int(actual) == int(wanted)
        except (TypeError, ValueError):
            matches = False
        if not matches:
            raise ValueError(
                f"{label} {name} must be {wanted}, got {actual!r}")


@dataclass(frozen=True)
class Pard2Spec:
    model_id: str = "amd/PARD2-Qwen3-8B"
    revision: str = "67a1516c8f6fc145cda99916799a0cbb3a4af135"
    target_model_id: str = "Qwen/Qwen3-8B"
    target_revision: str = "b968826d9c46dd6066d109eabc6255188de91218"
    upstream_commit: str = "6f279bf3f1680e0b5d71c562ca5b91bdeef4c038"
    pard_token: int = 151670
    draft_k: int = 15
    target_layers: tuple[int, ...] = (-1, -8, -16, -24)
    target_dim: int = 16384
    projection_scale: float = 0.02
    benchmark_profile: str | None = None
    td_proxy_profile: str | None = None

    @classmethod
    def for_benchmark_profile(cls, profile=None, td_proxy_profile=None, *,
                              mode=None):
        if profile not in (None, *PARD2_PROFILES):
            raise ValueError(f"unknown PARD2 benchmark profile {profile!r}")
        if td_proxy_profile is not None:
            if profile not in (None, "qwen3_32b"):
                raise ValueError(
                    "the TD proxy profile requires qwen3_32b")
            return cls.for_proxy_profile(td_proxy_profile)
        if profile == "qwen3_14b":
            # The paper's target-independent Qwen3 family evaluation uses one
            # shared drafter. Keep the 8B-aligned checkpoint for TI and reserve
            # the 14B-aligned checkpoint/warp for TD.
            if mode == "pard2-ti":
                return cls(
                    target_model_id="Qwen/Qwen3-14B",
                    target_revision=(
                        "40c069824f4251a91eefaf281ebe4c544efd3e18"),
                    benchmark_profile=profile,
                )
            return cls(
                model_id="amd/PARD2-Qwen3-14B",
                revision="679eff0b65ffaf5abd2dadd21a17909562935798",
                target_model_id="Qwen/Qwen3-14B",
                target_revision="40c069824f4251a91eefaf281ebe4c544efd3e18",
                target_dim=20480,
                benchmark_profile=profile,
            )
        if profile == "qwen3_32b":
            return cls(
                target_model_id="Qwen/Qwen3-32B",
                target_revision="9216db5781bf21249d130ec9da846c4624c16137",
                benchmark_profile=profile,
            )
        return cls(benchmark_profile=profile)

    @classmethod
    def for_proxy_profile(cls, profile=None):
        if profile is None:
            return cls()
        if profile != TD_PROXY_QWEN3_14B_ON_QWEN3_32B:
            raise ValueError(f"unknown TD proxy profile {profile!r}")
        return cls(
            model_id="amd/PARD2-Qwen3-14B",
            revision="679eff0b65ffaf5abd2dadd21a17909562935798",
            target_model_id="Qwen/Qwen3-32B",
            target_revision="9216db5781bf21249d130ec9da846c4624c16137",
            target_dim=20480,
            benchmark_profile="qwen3_32b",
            td_proxy_profile=profile,
        )

    @property
    def target_alignment(self):
        return ("cross_target_proxy"
                if self.td_proxy_profile is not None else "strict")

    def validate(self, target_config, draft_config, *,
                 mode="pard2-td", draft_snapshot=None) -> None:
        if mode not in ("pard2-ti", "pard2-td"):
            raise ValueError("PARD2 spec validation requires TI or TD mode")
        if self.td_proxy_profile is not None and mode != "pard2-td":
            raise ValueError("a TD proxy profile is valid only in pard2-td mode")
        if getattr(target_config, "model_type", None) not in ("qwen3", "qwen3_quarot"):
            raise ValueError("selected PARD2 target must be a dense Qwen3 runtime")
        if int(target_config.vocab_size) != int(draft_config.vocab_size):
            raise ValueError("target and PARD2 vocabularies differ")
        _expect_config_fields(draft_config, {
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
        }, "PARD2 drafter")
        expected = {
            "pard_token": self.pard_token,
            "pard2_target_dim": self.target_dim,
        }
        for name, value in expected.items():
            if int(getattr(draft_config, name, -1)) != value:
                raise ValueError(f"draft {name} does not match pinned PARD2 spec")
        if tuple(draft_config.pard2_target_layers) != self.target_layers:
            raise ValueError("draft hidden taps do not match pinned PARD2 spec")
        if abs(float(draft_config.pard2_scale) - self.projection_scale) > 1e-12:
            raise ValueError("draft projection scale does not match pinned PARD2 spec")

        # Legacy callers did not identify a target profile. Keep their TI
        # path target-independent, while explicit formal profiles validate
        # the actual target architecture before allocating GPU memory.
        if mode == "pard2-ti" and self.benchmark_profile is None:
            return

        if self.benchmark_profile == "qwen3_14b":
            expected_target = {
                "hidden_size": 5120,
                "intermediate_size": 17408,
                "num_hidden_layers": 40,
                "num_attention_heads": 40,
                "num_key_value_heads": 8,
                "head_dim": 128,
            }
        elif (self.benchmark_profile == "qwen3_32b"
              or self.td_proxy_profile is not None):
            expected_target = {
                "hidden_size": 5120,
                "intermediate_size": 25600,
                "num_hidden_layers": 64,
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
                "head_dim": 128,
            }
        else:
            expected_target = {
                "hidden_size": 4096,
                "intermediate_size": 12288,
                "num_hidden_layers": 36,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
            }
        _expect_config_fields(target_config, expected_target, "PARD2 target")
        if (self.td_proxy_profile is not None
                or self.benchmark_profile == "qwen3_14b"):
            draft_revision = _snapshot_revision(draft_snapshot)
            if draft_revision != self.revision:
                draft_name = self.model_id.removeprefix("amd/")
                raise ValueError(
                    f"requires the pinned {draft_name} draft snapshot "
                    f"{self.revision}, got {draft_revision!r}")
            target_source = getattr(
                target_config, "tokenizer_name_or_path", None)
            target_revision = _snapshot_revision(target_source)
            if target_revision != self.target_revision:
                raise ValueError(
                    "PARD2 requires a target converted from the pinned "
                    f"{self.target_model_id} snapshot {self.target_revision}, "
                    f"got {target_revision!r}")

        if mode == "pard2-ti":
            return
        hidden_size = int(target_config.hidden_size)
        if self.target_dim != len(self.target_layers) * hidden_size:
            raise ValueError(
                "draft target projection width does not match tapped target width")
        layer_count = int(target_config.num_hidden_layers)
        if any(not -layer_count <= int(tap) < 0
               for tap in self.target_layers):
            raise ValueError("draft hidden tap is outside the target layer range")


@dataclass
class GenerationResult:
    output_ids: list[int]
    ttft_ms: float
    steady_decode_ms: float
    total_ms: float
    target_forwards: int
    draft_forwards: int
    proposed_draft_tokens: int
    accepted_draft_tokens: int
    verifier_steps: int
    emitted_tokens_per_step: list[int]
    accept_length_by_step: list[int]
    stage_ms: dict[str, float] = field(default_factory=dict)
    peak_vram_bytes: int = 0
    proposal_lengths_by_step: list[int] = field(default_factory=list)
    ar_fallback_tokens: int = 0
    fallback_after_verifier_steps: int | None = None

    def metrics(self) -> dict:
        generated = len(self.output_ids)
        first = self.emitted_tokens_per_step[0] if self.emitted_tokens_per_step else 0
        steady = max(generated - first, 0)
        verified = self.proposal_lengths_by_step
        if not verified and self.verifier_steps:
            average = self.proposed_draft_tokens // self.verifier_steps
            verified = [average] * self.verifier_steps
        accepted = [max(length - 1, 0) for length in self.accept_length_by_step]
        reached = sum(min(count + 1, width)
                      for count, width in zip(accepted, verified))
        conditional_by_position = [
            (sum(count >= position for count in accepted)
             / sum(width >= position and count >= position - 1
                   for count, width in zip(accepted, verified)))
            if any(width >= position and count >= position - 1
                   for count, width in zip(accepted, verified)) else 0.0
            for position in range(1, max(verified, default=0) + 1)]
        return {
            **asdict(self),
            "generated_tokens": generated,
            "steady_tokens_per_s": (
                1000.0 * steady / self.steady_decode_ms
                if steady and self.steady_decode_ms else 0.0),
            "end_to_end_tokens_per_s": (
                1000.0 * generated / self.total_ms if self.total_ms else 0.0),
            "mean_accept_length": (
                sum(self.accept_length_by_step) / len(self.accept_length_by_step)
                if self.accept_length_by_step else 1.0),
            "draft_acceptance": (
                self.accepted_draft_tokens / self.proposed_draft_tokens
                if self.proposed_draft_tokens else 0.0),
            "conditional_acceptance": (
                self.accepted_draft_tokens / reached if reached else 0.0),
            "conditional_acceptance_by_position": conditional_by_position,
        }


def greedy_accept(candidates: torch.Tensor, predictions: torch.Tensor) -> int:
    """Return the longest greedy candidate prefix accepted by the target."""
    if candidates.ndim != 2 or predictions.ndim != 2:
        raise ValueError("candidates and predictions must have shape [batch, tokens]")
    if candidates.shape[0] != 1 or predictions.shape[0] != 1:
        raise ValueError("PARD2 v1 supports batch size one")
    if predictions.shape[1] < candidates.shape[1] + 1:
        raise ValueError("predictions must include the correction/bonus token")
    equal = candidates[0].eq(predictions[0, : candidates.shape[1]])
    mismatch = torch.nonzero(~equal, as_tuple=False)
    return candidates.shape[1] if mismatch.numel() == 0 else int(mismatch[0, 0])


class SelectedHiddenCollector:
    """Collect only the four PARD2 layer outputs, never a full hidden tuple."""

    def __init__(self, target, taps: Iterable[int], rotation_signs=None,
                 final_norm_weight=None, cache_basis=False,
                 folded_basis=False):
        layers = target.model.layers
        count = len(layers)
        self.indices = tuple(count + int(tap) for tap in taps)
        self.values = {}
        self.rotation_signs = rotation_signs
        self.final_norm_weight = final_norm_weight
        self.folded_basis = bool(folded_basis)
        self.basis_cached = False
        self.inverse_had_k = None
        self.inverse_remainder = None
        self.rms_norm_eps = float(
            getattr(getattr(target, "config", None), "rms_norm_eps", 1e-6))
        self.handles = []
        for position, index in enumerate(self.indices):
            if not 0 <= index < count:
                raise ValueError(f"hidden tap {index} is outside {count} layers")
            module = (target.model.norm
                      if self.folded_basis and position == 0 else layers[index])
            self.handles.append(module.register_forward_hook(self._hook(index)))
        if cache_basis and rotation_signs is not None and not self.folded_basis:
            parameter = next(target.parameters())
            self.cache_basis(parameter.device, parameter.dtype)

    def cache_basis(self, device, dtype):
        """Move the TD basis to the target device once, outside decode steps."""
        if self.rotation_signs is None:
            return
        if self.final_norm_weight is None:
            raise ValueError("final_norm_weight is required with rotation_signs")
        self.rotation_signs = self.rotation_signs.to(device=device, dtype=dtype)
        self.final_norm_weight = self.final_norm_weight.to(
            device=device, dtype=dtype)
        from quarot.functional.hadamard import get_hadK
        matrix, self.inverse_remainder = get_hadK(
            int(self.rotation_signs.numel()), transpose=True)
        self.inverse_had_k = (
            matrix.to(device=device, dtype=dtype)
            if matrix is not None else None)
        self.basis_cached = True

    def _hook(self, index):
        def capture(_module, _inputs, output):
            self.values[index] = output[0] if isinstance(output, tuple) else output
        return capture

    def reset(self):
        self.values.clear()

    def features(self, rows=None):
        missing = [index for index in self.indices if index not in self.values]
        if missing:
            raise RuntimeError(f"selected hidden taps were not produced: {missing}")
        values = [self.values[index] for index in self.indices]
        if rows is not None:
            values = [value[:, rows] for value in values]
        if self.folded_basis:
            return torch.cat(values, dim=-1)
        if self.rotation_signs is None:
            return torch.cat(values, dim=-1)
        signs = (self.rotation_signs if self.basis_cached else
                 self.rotation_signs.to(
                     device=values[0].device, dtype=values[0].dtype))
        inverse_kwargs = (
            {"had_k": self.inverse_had_k,
             "remainder": self.inverse_remainder}
            if self.basis_cached else {})
        width = int(values[0].shape[-1])
        remainder = self.inverse_remainder
        if remainder is None:
            from quarot.functional.hadamard import get_hadK
            _, remainder = get_hadK(width, transpose=True)
        if width == 5120 and int(remainder) == 40:
            # The H40 FP16 reduction can overflow on otherwise finite taps.
            # A power-of-two prescale preserves the linear map exactly.
            prescale = 16.0
            safe_signs = signs.float()
            restored = [
                _inverse_hadamard_cuda(
                    value / prescale, **inverse_kwargs).float()
                * prescale * safe_signs for value in values]
            final = restored[0]
            final = final * torch.rsqrt(
                final.square().mean(dim=-1, keepdim=True)
                + self.rms_norm_eps)
            norm_weight = (
                self.final_norm_weight.float() if self.basis_cached else
                self.final_norm_weight.to(
                    device=final.device, dtype=torch.float32))
            features = torch.cat(
                [final * norm_weight, *restored[1:]], dim=-1)
            if not torch.isfinite(features).all():
                raise RuntimeError(
                    "selected-feature restoration produced non-finite values")
            return features

        restored = [
            _inverse_hadamard_cuda(value, **inverse_kwargs) * signs
            for value in values]
        final = restored[0]
        dtype = final.dtype
        final = final.float()
        final = final * torch.rsqrt(
            final.square().mean(dim=-1, keepdim=True) + self.rms_norm_eps)
        norm_weight = (self.final_norm_weight if self.basis_cached else
                       self.final_norm_weight.to(
                           device=final.device, dtype=dtype))
        final = final.to(dtype) * norm_weight
        return torch.cat([final, *restored[1:]], dim=-1)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class TargetFeatEmbedWarp(torch.nn.Module):
    def __init__(self, base_model, target_dim, scale, proj_bias, calibration=None):
        super().__init__()
        self.base_model = base_model
        self.target_proj = torch.nn.Linear(
            target_dim, base_model.config.hidden_size, bias=proj_bias)
        self.scale = float(scale)
        self.config = base_model.config
        calibration = calibration or {}
        for name in ("raw_scale", "raw_bias", "projected_scale", "projected_bias"):
            self.register_buffer(name, calibration.get(name), persistent=False)
        self._td_basis_folded = False

    @property
    def td_basis_folded(self):
        return self._td_basis_folded

    def fold_target_basis_(self, rotation_signs, final_norm_weight):
        """Install an in-memory folded projection; the checkpoint is untouched."""
        if self._td_basis_folded:
            raise RuntimeError("TD target projection basis is already folded")
        if self.raw_scale is not None or self.raw_bias is not None:
            raise ValueError(
                "TD basis folding does not yet support raw feature calibration; "
                "disable --td-basis-fold or omit raw calibration")
        folded = fold_td_projection_weight(
            self.target_proj.weight, rotation_signs, final_norm_weight,
            self.target_proj.in_features // 4)
        with torch.no_grad():
            self.target_proj.weight.copy_(
                folded.to(dtype=self.target_proj.weight.dtype))
        self._td_basis_folded = True
        return self

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    def project_features(self, target_feat):
        """Apply calibration and projection before replicated PARD mask rows."""
        features = target_feat.to(self.target_proj.weight.dtype)
        if self.raw_scale is not None:
            features = features * self.raw_scale.to(features.dtype) + self.raw_bias.to(features.dtype)
        projected = self.target_proj(features)
        if self.projected_scale is not None:
            projected = (projected * self.projected_scale.to(projected.dtype)
                         + self.projected_bias.to(projected.dtype))
        return projected

    def forward(self, input_ids=None, target_feat=None,
                projected_target_feat=None, **kwargs):
        if (target_feat is None) == (projected_target_feat is None):
            raise ValueError(
                "provide exactly one of target_feat or projected_target_feat")
        embeds = self.base_model.get_input_embeddings()(input_ids)
        projected = (self.project_features(target_feat)
                     if projected_target_feat is None
                     else projected_target_feat.to(embeds.dtype))
        return self.base_model(inputs_embeds=embeds + projected * self.scale, **kwargs)


class AdaptiveK:
    """Phase-2 confidence/EMA selector; disabled unless explicitly requested."""

    choices = (8, 12, 15)

    def __init__(self, low=0.55, high=0.80, decay=0.9):
        self.low, self.high, self.decay = float(low), float(high), float(decay)
        self.ema = 1.0

    def choose(self):
        return 8 if self.ema < self.low else (12 if self.ema < self.high else 15)

    def update(self, accepted, proposed):
        observed = accepted / proposed if proposed else 0.0
        self.ema = self.decay * self.ema + (1.0 - self.decay) * observed


class _StageTimer:
    def __init__(self):
        self.events = []

    def record(self, name, fn):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        value = fn()
        end.record()
        self.events.append((name, start, end))
        return value

    def totals(self):
        torch.cuda.synchronize()
        result = {}
        for name, start, end in self.events:
            result[name] = result.get(name, 0.0) + start.elapsed_time(end)
        return result


class _ExactSmallChunkRows(torch.nn.Module):
    """Run a terminal module with the decode (M=1) numerical contract."""

    def __init__(self, module, max_rows):
        super().__init__()
        self.module = module
        self.max_rows = int(max_rows)

    def forward(self, value):
        rows = value.shape[-2]
        if rows <= 1 or rows > self.max_rows:
            return self.module(value)
        outputs = [
            self.module(value[..., index:index + 1, :])
            for index in range(rows)]
        return torch.cat(outputs, dim=-2)


class _RowIndependentRMSNorm(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.mean_dim = module.mean_dim
        self.eps = module.eps

    def forward(self, value):
        if value.is_cuda and value.dtype == torch.float16:
            import quarot
            return quarot._HIP.rms_norm_rows(
                value.contiguous(), self.mean_dim, self.eps)
        return self.module(value)


class _RowIndependentRMSNormQuant(torch.nn.Module):
    """Fuse exact row-independent RMSNorm with symmetric INT4 packing."""

    def __init__(self, module, input_clip_ratio=1.0):
        super().__init__()
        self.module = module
        self.mean_dim = module.mean_dim
        self.eps = module.eps
        self.input_clip_ratio = float(input_clip_ratio)

    def forward(self, value):
        if not value.is_cuda or value.dtype != torch.float16:
            raise RuntimeError(
                "fused RMSNorm+INT4 quantization requires CUDA FP16 input")
        import quarot
        packed, scales = quarot._HIP.rms_norm_quant_i4_rows_clipped(
            value.contiguous(), self.mean_dim, self.eps, self.input_clip_ratio)
        return quarot.PackedQuantizedTensor(
            packed, scales, logical_shape=value.shape)


class FusedPardRuntime:
    def __init__(self, *, mode, target, draft, tokenizer, spec=Pard2Spec(),
                 max_cache_len=4096, page_size=128, compile_mode="max-autotune",
                 ignore_eos=False, adaptive_k=None, native_gqa=True,
                 fused_decode_append=True, exact_row_norm=True,
                 rowwise_lm_head=False, exact_small_chunk=None,
                 td_target_basis=None, td_cache_basis=True,
                 td_lazy_features=False, td_unique_projection=False,
                 td_basis_fold=False, fused_norm_quant=False,
                 ti_zero_accept_fallback=0):
        if mode not in ("ar", "pard2-ti", "pard2-td"):
            raise ValueError("mode must be ar, pard2-ti, or pard2-td")
        if mode == "ar" and draft is not None:
            raise ValueError("AR mode requires draft=None")
        if mode != "ar" and draft is None:
            raise ValueError(f"{mode} requires a PARD2 drafter")
        if spec.td_proxy_profile is not None and mode != "pard2-td":
            raise ValueError(
                "a TD proxy spec is valid only in pard2-td mode")
        self.mode, self.target, self.draft, self.tokenizer = mode, target, draft, tokenizer
        self.spec, self.max_cache_len, self.page_size = spec, max_cache_len, page_size
        self.td_proxy_profile = spec.td_proxy_profile
        self.target_alignment = spec.target_alignment
        self.ignore_eos, self.adaptive_k = ignore_eos, adaptive_k
        if (not isinstance(ti_zero_accept_fallback, int)
                or ti_zero_accept_fallback < 0):
            raise ValueError('ti_zero_accept_fallback must be a nonnegative integer')
        if ti_zero_accept_fallback and mode != 'pard2-ti':
            raise ValueError('zero-accept AR fallback is supported only for PARD2-TI')
        # Opt-in hybrid policy. Zero preserves the pure fixed-k TI benchmark.
        # Check only the initial rounds: later short rejection streaks can
        # recover on otherwise productive requests. State is request-local.
        self.ti_zero_accept_fallback = ti_zero_accept_fallback
        self.draft_prefill_logits_to_keep = 1
        self.native_gqa = bool(native_gqa)
        self.fused_decode_append = bool(fused_decode_append)
        self.td_cache_basis = bool(td_cache_basis)
        self.td_lazy_features = bool(td_lazy_features)
        self.td_unique_projection = bool(td_unique_projection)
        self.td_basis_fold = bool(td_basis_fold)
        if self.td_basis_fold and mode != "pard2-td":
            raise ValueError("TD basis folding is valid only in pard2-td mode")
        # exact_small_chunk is the legacy coupled switch. Keep accepting it
        # when explicitly supplied, but use the independent defaults otherwise.
        # Exact row-wise RMSNorm is required for cache/token parity, while a
        # batched LM-head is parity-safe and substantially cheaper.
        if exact_small_chunk is not None:
            exact_row_norm = bool(exact_small_chunk)
            rowwise_lm_head = bool(exact_small_chunk)
        self.exact_row_norm = bool(exact_row_norm)
        self.rowwise_lm_head = bool(rowwise_lm_head)
        self.fused_norm_quant = bool(fused_norm_quant)
        self.exact_small_chunk = (
            self.exact_row_norm and self.rowwise_lm_head)
        if self.fused_norm_quant and not self.exact_row_norm:
            raise ValueError("fused_norm_quant requires exact_row_norm=True")
        if (self.fused_norm_quant and
                getattr(target, "cache_dtype", None) != "int4"):
            raise ValueError(
                "fused_norm_quant requires a quantized target "
                "(cache_dtype='int4')")
        if (self.exact_row_norm and hasattr(target, "model")
                and hasattr(target.model, "norm")):
            if not isinstance(target.model.norm, _RowIndependentRMSNorm):
                target.model.norm = _RowIndependentRMSNorm(target.model.norm)
            for layer in target.model.layers:
                wrapper = (_RowIndependentRMSNormQuant
                           if self.fused_norm_quant else _RowIndependentRMSNorm)
                if not isinstance(layer.input_layernorm, wrapper):
                    layer.input_layernorm = wrapper(layer.input_layernorm,
                        **({"input_clip_ratio": layer.self_attn.quantizer.input_clip_ratio}
                           if self.fused_norm_quant else {}))
                if not isinstance(layer.post_attention_layernorm, wrapper):
                    layer.post_attention_layernorm = wrapper(
                        layer.post_attention_layernorm,
                        **({"input_clip_ratio": layer.mlp.quantizer.input_clip_ratio}
                           if self.fused_norm_quant else {}))
        if self.rowwise_lm_head and hasattr(target, "lm_head"):
            if not isinstance(target.lm_head, _ExactSmallChunkRows):
                target.lm_head = _ExactSmallChunkRows(
                    target.lm_head, spec.draft_k + 1)
        eos = target.config.eos_token_id
        self.eos_ids = {int(x) for x in (eos if isinstance(eos, list) else [eos]) if x is not None}
        basis = td_target_basis or (None, None)
        self.collector = (SelectedHiddenCollector(
            target, spec.target_layers,
            None if self.td_basis_fold else basis[0],
            None if self.td_basis_fold else basis[1],
            self.td_cache_basis, folded_basis=self.td_basis_fold)
                          if mode == "pard2-td" else None)
        self.draft_forward = draft.forward if draft is not None else None
        self.verification_graph = os.getenv("QUAROT_VERIFICATION_GRAPH", "0") != "0"
        self._verification_cache = None
        self._verification_graphs = {}
        if draft is not None and compile_mode != "eager":
            # Proposal decoding deliberately specializes the drafter for the
            # bounded sequence lengths draft_k..(2 * draft_k - 1). Dynamo's
            # default recompile limit (8) is smaller than PARD2's 15 shapes.
            import torch._dynamo
            required_recompiles = spec.draft_k + 1
            if torch._dynamo.config.recompile_limit < required_recompiles:
                torch._dynamo.config.recompile_limit = required_recompiles
            self.draft_forward = torch.compile(
                draft.forward, mode=compile_mode, fullgraph=True, dynamic=False)

    def close(self):
        if self.collector is not None:
            self.collector.close()
            self.collector.reset()
        self._verification_graphs.clear()
        self._verification_cache = None

    def _target_cache(self, batch_size=1):
        if (self._verification_cache is not None
                and self._verification_cache.batch_size != batch_size):
            # Graphs own the cache storage and fixed batch shape.
            self._verification_graphs.clear()
            self._verification_cache = None
        if self.verification_graph and self._verification_cache is not None:
            cache = self._verification_cache
            cache.length = 0
            cache._needs_init = [True] * cache.n_layers
            cache._active_chunk_metadata = None
            cache._metadata_prepared_key = None
            return cache
        cache = self.target.build_cache(batch_size, self.page_size, self.max_cache_len,
            native_gqa=self.native_gqa,
            fused_decode_append=self.fused_decode_append)
        if batch_size > 1:
            # The persistent flat arange is not the interleaved page order
            # needed by a multi-page batch. Static decode metadata is retained.
            cache._persistent_metadata_enabled = False
        if self.verification_graph:
            self._verification_cache = cache
        return cache


    def _draft_cache(self, batch_size=1):
        if getattr(self.draft.config, "model_type", "").endswith("_quarot"):
            cache = self.draft.build_cache(
                batch_size, self.page_size, self.max_cache_len,
                native_gqa=self.native_gqa,
                fused_decode_append=self.fused_decode_append)
            if batch_size > 1:
                cache._persistent_metadata_enabled = False
            return cache
        from transformers import StaticCache
        return StaticCache(config=self.draft.config, max_batch_size=batch_size,
                           max_cache_len=self.max_cache_len,
                           device=next(self.draft.parameters()).device,
                           dtype=next(self.draft.parameters()).dtype)

    def _target_call(self, ids, cache, positions, materialize_features=True):
        # Internal runtime contract: positions are the contiguous suffix at
        # cache.length. Graph replay derives that suffix on the GPU.
        if self.collector is not None:
            self.collector.reset()
        if (self.verification_graph and ids.shape[1] in (15, 16)
                and getattr(cache, "_static_metadata_enabled", False)
                and not any(cache._needs_init)):
            from e2e.verification_graph import VerificationGraph
            rows = ids.shape[1]
            graph = self._verification_graphs.get((id(cache), rows))
            if graph is None:
                graph = VerificationGraph(self.target, cache, ids, self.collector)
                self._verification_graphs[(id(cache), rows)] = graph
            output = graph.replay(ids)
            features = (self.collector.features()
                if self.collector is not None and materialize_features else None)
            return output, features
        output = self.target(input_ids=ids, past_key_values=cache,
                             cache_position=positions, use_cache=True,
                             attention_mask=None, return_dict=True,
                             output_hidden_states=False)
        features = (self.collector.features()
                    if self.collector is not None and materialize_features else None)
        return output, features

    def _project_td_features(self, features, mask_rows=0):
        """Project unique real-token rows before expanding the final mask row."""
        projected = self.draft.project_features(features)
        if not mask_rows:
            return projected
        padding = projected[:, -1:].expand(-1, mask_rows, -1)
        return torch.cat((projected, padding), dim=1)

    def _accepted_td_features(self, prefill_features, emitted_count,
                              first_round, timer):
        """Materialize exactly the rows consumed by the next draft call."""
        rows = emitted_count - (1 if first_round else 0)
        restored = (timer.record(
            "td_feature_restore",
            lambda: self.collector.features(slice(0, rows)))
            if rows else None)
        if first_round:
            prompt_tail = prefill_features[:, -1:]
            return (prompt_tail if restored is None else
                    torch.cat((prompt_tail, restored), dim=1))
        if restored is None:
            raise RuntimeError(
                "a non-first speculative round must emit at least one token")
        return restored

    def _stop(self, emitted):
        return bool(emitted and not self.ignore_eos and emitted[-1] in self.eos_ids)

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens=256):
        if input_ids.ndim != 2 or min(input_ids.shape) < 1 or max_new_tokens < 1:
            raise ValueError("generation requires nonempty [batch, tokens] and positive output length")
        if input_ids.shape[1] + max_new_tokens + self.spec.draft_k > self.max_cache_len:
            raise ValueError("generation exceeds configured cache capacity")
        if input_ids.shape[0] != 1:
            if getattr(self, 'ti_zero_accept_fallback', 0):
                raise ValueError('zero-accept AR fallback requires batch size one')
            if self.mode == 'ar':
                raise ValueError("Use the batched AR benchmark adapter for batched target generation")
            if not self.ignore_eos or self.adaptive_k is not None:
                raise ValueError("Synchronous PARD2 batches require ignore_eos=True and fixed draft_k")
            from e2e.synchronous_pard import run_synchronous
            return run_synchronous(self, input_ids, max_new_tokens)
        return self._generate_ar(input_ids, max_new_tokens) if self.mode == "ar" else self._generate_spec(input_ids, max_new_tokens)

    def _generate_ar(self, input_ids, max_new_tokens):
        cache, current, generated = self._target_cache(), input_ids, []
        forwards, first_at = 0, None
        timer = _StageTimer()
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.perf_counter()
        while len(generated) < max_new_tokens:
            pos = torch.arange(cache.length, cache.length + current.shape[1], device=current.device)
            output, _ = timer.record("target", lambda: self._target_call(current, cache, pos))
            cache = output.past_key_values
            forwards += 1
            token = int(output.logits[:, -1].argmax(-1))
            generated.append(token)
            if first_at is None:
                torch.cuda.synchronize(); first_at = time.perf_counter()
            if self._stop(generated): break
            current = torch.tensor([[token]], device=input_ids.device, dtype=input_ids.dtype)
        stages = timer.totals(); ended = time.perf_counter(); first_at = first_at or ended
        return GenerationResult(generated, (first_at-started)*1000, (ended-first_at)*1000,
            (ended-started)*1000, forwards, 0, 0, 0, forwards, [1]*len(generated),
            [1]*len(generated), stages, torch.cuda.max_memory_allocated())

    def _generate_spec(self, input_ids, max_new_tokens):
        target_cache, draft_cache = self._target_cache(), self._draft_cache()
        target_input, draft_input = None, input_ids
        draft_cache_len = 0
        draft_features = None
        target_forwards = draft_forwards = proposed = accepted_total = 0
        generated, emitted_steps, accept_lengths, proposal_lengths = [], [], [], []
        fallback_threshold = getattr(self, 'ti_zero_accept_fallback', 0)
        ar_fallback_tokens = 0
        fallback_after_verifier_steps = None
        timer, first_at = _StageTimer(), None
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.perf_counter()

        # Keep target prefill identical to AR.  A combined unquantized
        # "prompt + candidates" prefill can disagree with sequential KV4 AR.
        # The prefill logit supplies the first pending prediction; the first
        # verifier contains k candidates, then later rounds contain pending+k.
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        prefill_out, prefill_features = timer.record(
            "target_feature_prefill" if self.mode == "pard2-td" else "target_prefill",
            lambda: self._target_call(input_ids, target_cache, positions))
        target_cache = prefill_out.past_key_values
        pending_prediction = prefill_out.logits[:, -1:].argmax(-1)
        target_forwards += 1
        first_round = True

        if self.mode == "pard2-td":
            features = prefill_features
            zero = torch.zeros_like(features[:, :1])
            draft_features = torch.cat((zero, features[:, :-1]), dim=1)

        # Keep arbitrary prompt lengths out of the compiled proposal graph.
        # Prefill all but the final prompt token eagerly; the first PARD call
        # is then the reusable fixed M=15 shape (one real token + 14 masks).
        if input_ids.shape[1] > 1:
            prefix = input_ids[:, :-1]
            prefix_pos = torch.arange(prefix.shape[1], device=input_ids.device)
            prefix_kwargs = {}
            if self.mode == "pard2-td":
                prefix_features = draft_features[:, :-1]
                if self.td_unique_projection:
                    prefix_kwargs["projected_target_feat"] = timer.record(
                        "td_feature_project_prefill",
                        lambda: self._project_td_features(prefix_features))
                else:
                    prefix_kwargs["target_feat"] = prefix_features
            timer.record("draft_prefill", lambda: self.draft(
                input_ids=prefix, past_key_values=draft_cache,
                cache_position=prefix_pos, use_cache=True, attention_mask=None,
                return_dict=True, logits_to_keep=getattr(self, 'draft_prefill_logits_to_keep', 1),
                **prefix_kwargs))
            draft_forwards += 1
            draft_cache_len = prefix.shape[1]
            draft_input = input_ids[:, -1:]
            if self.mode == "pard2-td":
                draft_features = draft_features[:, -1:]

        while len(generated) < max_new_tokens:
            if (fallback_threshold and accepted_total == 0
                    and len(accept_lengths) >= fallback_threshold):
                # The committed target cache contains every emitted token
                # except the pending correction. Append that token exactly
                # once, then continue target-only decoding on this cache.
                if fallback_after_verifier_steps is None:
                    fallback_after_verifier_steps = len(accept_lengths)
                target_pos = torch.arange(target_cache.length,
                    target_cache.length + 1, device=input_ids.device)
                target_out, _ = timer.record('target_ar_fallback',
                    lambda: self._target_call(target_input, target_cache, target_pos))
                target_cache = target_out.past_key_values
                token = int(target_out.logits[:, -1].argmax(-1))
                generated.append(token)
                emitted_steps.append(1)
                target_forwards += 1
                ar_fallback_tokens += 1
                if self._stop(generated):
                    break
                target_input = torch.tensor([[token]], device=input_ids.device, dtype=input_ids.dtype)
                continue
            k = self.adaptive_k.choose() if self.adaptive_k is not None else self.spec.draft_k
            masks = torch.full((1, k-1), self.spec.pard_token,
                               device=input_ids.device, dtype=input_ids.dtype)
            draft_ids = torch.cat((draft_input, masks), dim=1)
            draft_pos = torch.arange(draft_cache_len, draft_cache_len + draft_ids.shape[1], device=input_ids.device)
            if hasattr(draft_cache, "length"):
                draft_cache.length = draft_cache_len
            kwargs = {}
            if self.mode == "pard2-td":
                if self.td_unique_projection:
                    kwargs["projected_target_feat"] = timer.record(
                        "td_feature_project",
                        lambda: self._project_td_features(draft_features, k - 1))
                else:
                    pad = draft_features[:, -1:].expand(-1, k-1, -1)
                    kwargs["target_feat"] = torch.cat((draft_features, pad), dim=1)
            draft_out = timer.record("draft", lambda: self.draft_forward(
                input_ids=draft_ids, past_key_values=draft_cache,
                cache_position=draft_pos, use_cache=True, attention_mask=None,
                return_dict=True, **kwargs))
            draft_forwards += 1; draft_cache_len += draft_ids.shape[1]
            candidates = draft_out.logits[:, -k:].argmax(-1)
            proposed += k
            proposal_lengths.append(k)

            verify_ids = (candidates if first_round else
                          torch.cat((target_input, candidates), dim=1))
            # The graph's metadata kernel produces positions from its GPU
            # base-length scalar. Do not launch a redundant arange for replay.
            target_pos = (None if self.verification_graph
                and getattr(target_cache, "_static_metadata_enabled", False)
                and verify_ids.shape[1] in (15, 16) else
                torch.arange(target_cache.length,
                    target_cache.length + verify_ids.shape[1], device=input_ids.device))
            base_length = target_cache.length
            target_out, new_features = timer.record("target_verify",
                lambda: self._target_call(verify_ids, target_cache, target_pos,
                    materialize_features=not self.td_lazy_features))
            if first_round:
                predictions = torch.cat((
                    pending_prediction,
                    target_out.logits[:, -k:].argmax(-1)), dim=1)
            else:
                predictions = target_out.logits[:, -(k+1):].argmax(-1)
            accepted = greedy_accept(candidates, predictions)
            # Transformers may return a shallow cache wrapper instead of
            # mutating the object passed by the caller.  Commit on the returned
            # physical cache; rejected slots remain stale and are overwritten.
            target_cache = target_out.past_key_values
            target_cache.length = base_length + (0 if first_round else 1) + accepted
            target_forwards += 1; accepted_total += accepted
            if self.adaptive_k is not None: self.adaptive_k.update(accepted, k)
            accept_lengths.append(accepted + 1)
            emitted = [int(x) for x in predictions[0, :accepted+1].tolist()]
            emitted = emitted[:max_new_tokens-len(generated)]
            if not self.ignore_eos:
                for i, token in enumerate(emitted):
                    if token in self.eos_ids:
                        emitted = emitted[:i+1]; break
            generated.extend(emitted); emitted_steps.append(len(emitted))
            if self.mode == "pard2-td":
                # Target features are shifted left by one token.  First-round
                # candidate[0] uses the final prompt hidden state; later rounds
                # use the pending correction hidden state.
                if self.td_lazy_features:
                    draft_features = self._accepted_td_features(
                        prefill_features, len(emitted), first_round, timer)
                else:
                    source = (
                        torch.cat((prefill_features[:, -1:], new_features), dim=1)
                        if first_round else new_features)
                    draft_features = source[:, :len(emitted)]
            if first_at is None:
                torch.cuda.synchronize(); first_at = time.perf_counter()
            if self._stop(generated) or len(generated) >= max_new_tokens: break

            target_input = torch.tensor([[generated[-1]]], device=input_ids.device, dtype=input_ids.dtype)
            draft_cache_len = max(0, draft_cache_len - (k-1))
            if hasattr(draft_cache, "length"):
                draft_cache.length = draft_cache_len
            draft_input = torch.tensor([emitted], device=input_ids.device, dtype=input_ids.dtype)
            first_round = False

        stages = timer.totals(); ended = time.perf_counter(); first_at = first_at or ended
        return GenerationResult(generated, (first_at-started)*1000, (ended-first_at)*1000,
            (ended-started)*1000, target_forwards, draft_forwards, proposed,
            accepted_total, len(accept_lengths), emitted_steps, accept_lengths,
            stages, torch.cuda.max_memory_allocated(), proposal_lengths,
            ar_fallback_tokens, fallback_after_verifier_steps)


def _load_td_projection_state(draft_snapshot, spec, draft_config):
    """Validate the pinned CPU warp artifact before allocating the GPU target."""
    warp_path = Path(draft_snapshot) / "warp_model.bin"
    if not warp_path.is_file():
        raise FileNotFoundError(f"missing PARD2 TD warp checkpoint: {warp_path}")
    state = torch.load(warp_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("PARD2 TD warp checkpoint must contain a state dict")
    unrelated = [key for key in state if not key.startswith("target_proj.")]
    if unrelated:
        raise ValueError(
            f"PARD2 TD warp checkpoint has unexpected keys: {unrelated}")
    projection = {
        key.removeprefix("target_proj."): value for key, value in state.items()
    }
    expected_keys = {"weight"}
    if bool(getattr(draft_config, "pard2_proj_bias", False)):
        expected_keys.add("bias")
    if set(projection) != expected_keys:
        raise ValueError(
            "PARD2 TD warp projection keys differ from the config contract")
    expected_shapes = {
        "weight": (int(draft_config.hidden_size), int(spec.target_dim)),
    }
    if "bias" in expected_keys:
        expected_shapes["bias"] = (int(draft_config.hidden_size),)
    for name, shape in expected_shapes.items():
        tensor = projection[name]
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise ValueError(
                f"PARD2 TD warp {name} must have shape {shape}")
        if not tensor.is_floating_point():
            raise ValueError(f"PARD2 TD warp {name} must be floating point")
    return projection


def _preflight_pard2(mode, target_config, draft_config=None, *,
                     draft_snapshot=None, benchmark_profile=None,
                     td_proxy_profile=None):
    """Perform all config and warp checks that do not need a GPU model."""
    if mode not in ("ar", "pard2-ti", "pard2-td"):
        raise ValueError("mode must be ar, pard2-ti, or pard2-td")
    if mode == "ar":
        if draft_config is not None:
            raise ValueError("AR mode requires no draft config")
        if td_proxy_profile is not None:
            raise ValueError("--td-proxy-profile is valid only for pard2-td")
        return Pard2Spec.for_benchmark_profile(
            benchmark_profile, mode=mode), None, None
    if draft_config is None:
        raise ValueError(f"{mode} requires a PARD2 draft config")

    spec = Pard2Spec.for_benchmark_profile(
        benchmark_profile, td_proxy_profile, mode=mode)
    spec.validate(
        target_config, draft_config, mode=mode,
        draft_snapshot=draft_snapshot)
    if mode == "pard2-ti":
        return spec, None, None

    basis = _basis_from_rotation_metadata(target_config)
    if spec.td_proxy_profile is not None and basis is None:
        raise ValueError(
            "Qwen3-32B TD proxy requires self-contained HadK basis metadata")
    projection = _load_td_projection_state(
        draft_snapshot, spec, draft_config)
    return spec, projection, basis


def load_runtime(*, mode, target_checkpoint, draft_snapshot, tokenizer_path,
                 max_cache_len=4096, page_size=128, compile_mode="max-autotune",
                 ignore_eos=False, calibration_path=None, quantized_draft=None,
                 adaptive_k=False, native_gqa=True,
                 fused_decode_append=True, exact_row_norm=True,
                 rowwise_lm_head=False, exact_small_chunk=None,
                 td_cache_basis=True, td_lazy_features=False,
                 td_unique_projection=False, td_basis_fold=False,
                 fused_norm_quant=False, benchmark_profile=None,
                 td_proxy_profile=None, ti_zero_accept_fallback=0):
    """Load a pinned local runtime without network access or checkpoint copies."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from e2e.model_registry import runtime_types

    if mode not in ("ar", "pard2-ti", "pard2-td"):
        raise ValueError("mode must be ar, pard2-ti, or pard2-td")
    if (not isinstance(ti_zero_accept_fallback, int) or ti_zero_accept_fallback < 0
            or (ti_zero_accept_fallback and mode != 'pard2-ti')):
        raise ValueError('zero-accept fallback requires PARD2-TI and a nonnegative integer')
    if td_proxy_profile is not None and mode != "pard2-td":
        raise ValueError("--td-proxy-profile is valid only for pard2-td")

    config_cls, target_cls, _ = runtime_types(target_checkpoint, local_files_only=True)
    target_config = config_cls.from_pretrained(target_checkpoint, local_files_only=True,
                                               attn_implementation="flash_attention_2")
    draft_config = None
    quantized_draft_config = None
    if mode != "ar":
        if not draft_snapshot:
            raise ValueError(f"{mode} requires a PARD2 draft snapshot")
        draft_config = AutoConfig.from_pretrained(draft_snapshot, local_files_only=True)
        if quantized_draft:
            draft_config_cls, draft_cls, _ = runtime_types(
                quantized_draft, local_files_only=True)
            quantized_draft_config = draft_config_cls.from_pretrained(
                quantized_draft, local_files_only=True,
                attn_implementation="flash_attention_2")
            _expect_config_fields(quantized_draft_config, {
                "hidden_size": draft_config.hidden_size,
                "intermediate_size": draft_config.intermediate_size,
                "num_hidden_layers": draft_config.num_hidden_layers,
                "num_attention_heads": draft_config.num_attention_heads,
                "num_key_value_heads": draft_config.num_key_value_heads,
                "vocab_size": draft_config.vocab_size,
            }, "quantized PARD2 drafter")
    spec, projection_state, td_target_basis = _preflight_pard2(
        mode, target_config, draft_config, draft_snapshot=draft_snapshot,
        benchmark_profile=benchmark_profile,
        td_proxy_profile=td_proxy_profile)
    calibration = (
        torch.load(calibration_path, map_location="cpu", weights_only=True)
        if mode == "pard2-td" and calibration_path else None)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True)

    # All cheap CPU contracts have passed before this 17+ GiB allocation.
    target = target_cls.from_pretrained(
        target_checkpoint, config=target_config, torch_dtype=torch.float16,
        local_files_only=True).eval().to("cuda")
    draft = None
    if mode != "ar":
        if quantized_draft:
            draft = draft_cls.from_pretrained(
                quantized_draft, config=quantized_draft_config,
                torch_dtype=torch.float16, local_files_only=True)
            draft_dtype = torch.float16
        else:
            draft = AutoModelForCausalLM.from_pretrained(
                draft_snapshot, torch_dtype=torch.bfloat16, local_files_only=True,
                attn_implementation="eager")
            draft_dtype = torch.bfloat16
        if mode == "pard2-td":
            if td_target_basis is None:
                td_target_basis = load_td_target_basis(target, tokenizer_path)
            draft = TargetFeatEmbedWarp(draft, spec.target_dim, spec.projection_scale,
                                        bool(draft_config.pard2_proj_bias), calibration)
            draft.target_proj.load_state_dict(projection_state)
            if td_basis_fold:
                draft.fold_target_basis_(*td_target_basis)
        draft = draft.eval().to("cuda", dtype=draft_dtype)
    selector = AdaptiveK() if adaptive_k else None
    return FusedPardRuntime(mode=mode, target=target, draft=draft,
        tokenizer=tokenizer, spec=spec, max_cache_len=max_cache_len,
        page_size=page_size, compile_mode=compile_mode, ignore_eos=ignore_eos,
        adaptive_k=selector, native_gqa=native_gqa,
        fused_decode_append=fused_decode_append,
        exact_row_norm=exact_row_norm,
        rowwise_lm_head=rowwise_lm_head,
        exact_small_chunk=exact_small_chunk,
        td_target_basis=td_target_basis,
        td_cache_basis=td_cache_basis,
        td_lazy_features=td_lazy_features,
        td_unique_projection=td_unique_projection,
        td_basis_fold=td_basis_fold,
        fused_norm_quant=fused_norm_quant,
        ti_zero_accept_fallback=ti_zero_accept_fallback)
