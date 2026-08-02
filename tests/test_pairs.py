import torch

from feature_probe.mappings import ConstantPatch
from feature_probe.pairs import apply_mapping


def test_apply_mapping_supports_apply_method():
    images = torch.zeros(2, 3, 32, 32)
    mapping = ConstantPatch(top=28, left=28, size=4)

    mapped = apply_mapping(mapping, images)

    assert torch.all(mapped[:, :, 28:32, 28:32] == 1)


def test_apply_mapping_supports_callable():
    images = torch.zeros(2, 3, 4, 4)

    mapped = apply_mapping(lambda batch: batch + 0.25, images)

    assert torch.all(mapped == 0.25)
