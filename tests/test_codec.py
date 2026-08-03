import torch

from feature_probe.codec import count_feature_mapping_bits
from feature_probe.fitters import build_feature_mapping


def encoded_bits(model, *, pruning):
    return count_feature_mapping_bits(
        model,
        layer_id="stem",
        family="residual_adapter",
        rank=2,
        kernel=1,
        quantization="int8",
        pruning=pruning,
    )


def test_sparse_code_uses_actual_nonzero_values():
    model = build_feature_mapping("residual_adapter", (8, 4, 4), rank=2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.adapter.bias[0] = 1.0

    dense = encoded_bits(model, pruning=0.0)
    sparse = encoded_bits(model, pruning=0.9)

    assert sparse.value_bits == 8
    assert sparse.total_bits < dense.total_bits


def test_family_and_shape_are_charged_in_structure_code():
    model = build_feature_mapping("mean_shift", (2, 2, 2))
    bits = count_feature_mapping_bits(
        model,
        layer_id="stem",
        family="mean_shift",
        rank=0,
        kernel=0,
        quantization="fp32",
        pruning=0.0,
    )

    assert bits.structure_bits > 0
    assert bits.value_bits == 8 * 32
    assert bits.total_bits == sum(
        [bits.header_bits, bits.structure_bits, bits.mask_bits, bits.scale_bits, bits.value_bits]
    )
