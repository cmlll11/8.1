from types import SimpleNamespace

import torch

from feature_probe.artifacts import load_pair_bundle
from feature_probe.mappings import ConstantPatch, apply_mapping
from feature_probe.pairs import build_pair_bundle, save_pair_bundle


def test_apply_mapping_supports_apply_method():
    images = torch.zeros(2, 3, 32, 32)
    mapping = ConstantPatch(top=28, left=28, size=4)

    mapped = apply_mapping(mapping, images)

    assert torch.all(mapped[:, :, 28:32, 28:32] == 1)


def test_apply_mapping_supports_callable():
    images = torch.zeros(2, 3, 4, 4)

    mapped = apply_mapping(lambda batch: batch + 0.25, images)

    assert torch.all(mapped == 0.25)


def test_patch_rejects_out_of_bounds_location():
    images = torch.zeros(1, 3, 4, 4)
    mapping = ConstantPatch(top=3, left=3, size=2)

    try:
        apply_mapping(mapping, images)
    except ValueError as exc:
        assert "boundary" in str(exc)
    else:
        raise AssertionError("Expected an out-of-bounds patch to be rejected")


def test_pair_bundle_round_trip_for_fixed_mapping(tmp_path):
    splits = SimpleNamespace()
    for split in ("train", "validation", "test"):
        setattr(splits, f"{split}_images", torch.zeros(3, 3, 4, 4))
        setattr(splits, f"{split}_labels", torch.tensor([0, 1, 2]))
        setattr(splits, f"{split}_indices", torch.tensor([10, 11, 12]))
    bundle = build_pair_bundle(
        splits,
        ConstantPatch(top=3, left=3, size=1),
        target=0,
        count_per_split=2,
        seed=2026,
        device="cpu",
        metadata={"mapping": "badnets"},
    )
    path = tmp_path / "pairs.pt"
    save_pair_bundle(path, bundle)

    restored = load_pair_bundle(path)

    assert len(restored["clean"]) == 6
    assert restored["metadata"]["protocol"] == "MDL-FEATURE-v1"
    assert torch.equal(restored["clean"][:, :, 3, 3], torch.zeros(6, 3))
    assert torch.equal(restored["mapped"][:, :, 3, 3], torch.ones(6, 3))
