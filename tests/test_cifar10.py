import torch

from feature_probe.cifar10 import fixed_split_indices
from feature_probe.mappings import ConstantPatch


def test_fixed_split_is_disjoint_and_reproducible():
    train_a, validation_a = fixed_split_indices(100, 20, 2026)
    train_b, validation_b = fixed_split_indices(100, 20, 2026)
    assert torch.equal(train_a, train_b)
    assert torch.equal(validation_a, validation_b)
    assert not set(train_a.tolist()).intersection(validation_a.tolist())


def test_badnets_patch_location():
    images = torch.zeros(2, 3, 32, 32)
    mapped = ConstantPatch(top=28, left=28, size=4).apply(images)
    assert mapped[:, :, 28:32, 28:32].eq(1).all()
    assert mapped[:, :, :28, :].eq(0).all()
