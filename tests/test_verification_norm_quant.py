"""Strict checkpoint-quantizer parity for the verification norm kernels."""

import pytest
import torch

quarot = pytest.importorskip("quarot")
if torch.version.hip is None or not torch.cuda.is_available():
    pytest.skip("requires a ROCm/HIP GPU", allow_module_level=True)


def _reference(value, clip_ratio, eps=1e-6):
    normalized = quarot._HIP.rms_norm_rows(value, value.shape[-1], eps)
    result = quarot.nn.Quantizer(clip_ratio)(normalized)
    return result.quantized_x, result.scales_x


@pytest.mark.parametrize("rows", [1, 15, 16])
@pytest.mark.parametrize("width", [128, 4096, 5120, 8192])
@pytest.mark.parametrize("clip_ratio", [1.0, 0.9, 0.73])
def test_clipped_norm_matches_actual_quantizer(rows, width, clip_ratio):
    torch.manual_seed(3400 + rows + width)
    value = torch.randn(1, rows, width, device="cuda", dtype=torch.float16)
    actual = quarot._HIP.rms_norm_quant_i4_rows_clipped(
        value, width, 1e-6, clip_ratio)
    expected = _reference(value, clip_ratio)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    # Neither reduction ordering nor scale calculation may depend on M.
    single_rows = [quarot._HIP.rms_norm_quant_i4_rows_clipped(
        value[:, row:row + 1].contiguous(), width, 1e-6, clip_ratio)
        for row in range(rows)]
    for part in (0, 1):
        assert torch.equal(actual[part], torch.cat(
            [single[part] for single in single_rows], dim=1))


@pytest.mark.parametrize("clip_ratio", [1.0, 0.9, 0.01])
@pytest.mark.parametrize("fill", [0.0, 2 ** -24, 2 ** -14, 1.0, -1.0, 65504.0])
def test_scale_clamp_and_finite_extremes(fill, clip_ratio):
    value = torch.full((1, 16, 4096), fill, device="cuda", dtype=torch.float16)
    actual = quarot._HIP.rms_norm_quant_i4_rows_clipped(
        value, 4096, 1e-6, clip_ratio)
    expected = _reference(value, clip_ratio)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    assert torch.all(actual[1] >= torch.finfo(torch.float16).tiny)


@pytest.mark.parametrize("rows", [1, 15, 16])
@pytest.mark.parametrize("clip_ratio", [1.0, 0.9, 0.73])
def test_residual_preserves_fp16_checkpoint_boundary(rows, clip_ratio):
    torch.manual_seed(4200 + rows)
    value = torch.randn(1, rows, 4096, device="cuda", dtype=torch.float16)
    residual = torch.randn_like(value)
    saved_value, saved_residual = value.clone(), residual.clone()
    expected_residual = residual + value
    expected = _reference(expected_residual, clip_ratio)
    actual = quarot._HIP.residual_rms_norm_quant_i4_rows(
        value, residual, 4096, 1e-6, clip_ratio)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    assert torch.equal(actual[2], expected_residual)
    assert torch.equal(value, saved_value)
    assert torch.equal(residual, saved_residual)


def test_half_ties_and_clipping_thresholds():
    # Exact half integers exercise nearest-even rounding on both signs. The
    # repeated row has a different norm scale, so use the production oracle.
    pattern = torch.tensor(
        [-8, -7.5, -6.5, -3.5, -2.5, -1.5, -0.5, 0,
         0.5, 1.5, 2.5, 3.5, 6.5, 7, 7.5, 8],
        device="cuda", dtype=torch.float16)
    value = pattern.repeat(16, 256).view(1, 16, 4096)
    for ratio in (1.0, 0.9, 0.73):
        actual = quarot._HIP.rms_norm_quant_i4_rows_clipped(
            value, 4096, 1e-6, ratio)
        expected = _reference(value, ratio)
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])


@pytest.mark.parametrize("clip_ratio", [0, -0.1, 1.1, float("nan")])
def test_invalid_checkpoint_clipping_rejected(clip_ratio):
    value = torch.ones(1, 128, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="input_clip_ratio"):
        quarot._HIP.rms_norm_quant_i4_rows_clipped(value, 128, 1e-6, clip_ratio)


def test_residual_shape_and_dtype_contract():
    value = torch.ones(1, 128, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="input shape"):
        quarot._HIP.residual_rms_norm_quant_i4_rows(
            value, value[:, :64].contiguous(), 128, 1e-6, 0.9)
    with pytest.raises(RuntimeError, match="float16"):
        quarot._HIP.residual_rms_norm_quant_i4_rows(
            value, value.float(), 128, 1e-6, 0.9)


def test_current_stream_and_graph_replay():
    # Graph replay must read updated values from the same stable addresses.
    value = torch.randn(1, 16, 4096, device="cuda", dtype=torch.float16)
    residual = torch.randn_like(value)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            quarot._HIP.residual_rms_norm_quant_i4_rows(
                value, residual, 4096, 1e-6, 0.9)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = quarot._HIP.residual_rms_norm_quant_i4_rows(
            value, residual, 4096, 1e-6, 0.9)
    for factor in (1.0, 0.25, -2.0):
        value.mul_(factor)
        graph.replay()
        expected_residual = value + residual
        expected = _reference(expected_residual, 0.9)
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])
        assert torch.equal(actual[2], expected_residual)
