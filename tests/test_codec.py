import torch

from feature_probe.codec import count_feature_mapping_bits
from feature_probe.fitters import build_feature_mapping


def test_more_pruning_reduces_bits_for_fixed_model():
    model = build_feature_mapping("C2", (8, 4, 4), rank=2)
    dense = count_feature_mapping_bits(model, layer_id="stem", level="C2", rank=2, quantization="int8", pruning=0)
    sparse = count_feature_mapping_bits(model, layer_id="stem", level="C2", rank=2, quantization="int8", pruning=0.9)
    assert sparse.total_bits < dense.total_bits
