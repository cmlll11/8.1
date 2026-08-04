import torch

from feature_probe.compression import compress_feature_mapping
from feature_probe.fitters import build_feature_mapping


def test_compression_materializes_pruning_and_int4():
    model = build_feature_mapping("mean_shift", (2, 2, 2))
    with torch.no_grad():
        model.delta.copy_(torch.arange(8).reshape(1, 2, 2, 2))

    result = compress_feature_mapping(model, pruning=0.5, quantization="int4")

    assert result.total_values == 8
    assert result.kept_values == 4
    assert torch.count_nonzero(result.model.delta) <= 4
    assert result.model.delta.abs().max() <= model.delta.abs().max()


def test_fp32_without_pruning_preserves_output():
    model = build_feature_mapping("residual_adapter", (3, 4, 4), rank=2).eval()
    images = torch.rand(2, 3, 4, 4)

    result = compress_feature_mapping(model, pruning=0.0, quantization="fp32")

    assert torch.equal(model(images), result.model(images))


def test_compression_preserves_spatial_support_buffer():
    support = torch.zeros(4, 4, dtype=torch.bool)
    support[1:3, 2:4] = True
    model = build_feature_mapping(
        "spatial_gated_fitnets", (3, 4, 4), rank=2, spatial_support=support
    )

    result = compress_feature_mapping(model, pruning=0.5, quantization="int8")

    assert torch.equal(result.model.support_mask.cpu().reshape(4, 4), support)
