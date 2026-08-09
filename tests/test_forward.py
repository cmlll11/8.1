from types import SimpleNamespace

import pytest
import torch
from torch import nn

from feature_probe.forward import (
    FeaturePairs,
    assert_pair_alignment,
    compare_feature_changes,
    extract_feature_pairs_by_layer,
    fitted_feature_asr,
    subset_pair_bundle_by_split,
)


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 4, 1)
        self.layer1 = nn.ReLU()
        self.head = nn.Linear(4, 2)

    def forward(self, images):
        value = self.layer1(self.stem(images))
        return self.head(value.mean((2, 3)))


def bundle(mapped_offset: float = 0.1):
    clean = torch.linspace(0, 0.8, 24 * 3 * 4 * 4).reshape(24, 3, 4, 4)
    split_codes = torch.tensor([0] * 8 + [1] * 8 + [2] * 8, dtype=torch.uint8)
    return {
        "clean": clean,
        "mapped": clean + mapped_offset,
        "labels": torch.ones(24, dtype=torch.long),
        "indices": torch.arange(24),
        "split_codes": split_codes,
    }


def test_pair_alignment_rejects_different_base_examples():
    first = bundle()
    second = bundle()
    second["indices"] = second["indices"] + 1

    with pytest.raises(ValueError, match="indices"):
        assert_pair_alignment(first, second)


def test_subset_pair_bundle_keeps_aligned_prefix_per_split():
    source = bundle()
    selected = subset_pair_bundle_by_split(source, 3)

    assert len(selected["clean"]) == 9
    assert selected["split_codes"].tolist() == [0] * 3 + [1] * 3 + [2] * 3
    assert selected["indices"].tolist() == [0, 1, 2, 8, 9, 10, 16, 17, 18]


def test_extracts_all_layers_in_one_aligned_bundle():
    source = bundle()
    result = extract_feature_pairs_by_layer(
        TinyClassifier(),
        source,
        ["stem", "layer1"],
        device="cpu",
        batch_size=5,
    )

    assert set(result) == {"stem", "layer1"}
    assert result["stem"].clean.shape == (24, 4, 4, 4)
    assert torch.equal(result["stem"].indices, source["indices"])
    assert len(result["layer1"].select("validation").clean) == 8


def test_feature_tensors_to_keeps_metadata_on_cpu_and_selection_aligned():
    pairs = FeaturePairs(**bundle()).feature_tensors_to("cpu")
    selected = pairs.select("test")

    assert selected.clean.device.type == "cpu"
    assert selected.labels.device.type == "cpu"
    assert torch.equal(selected.indices, torch.arange(16, 24))


def test_change_comparison_uses_train_validation_and_test_splits():
    trigger_bundle = bundle(0.2)
    uap_bundle = bundle(-0.2)
    trigger = FeaturePairs(**trigger_bundle)
    uap = FeaturePairs(**uap_bundle)

    result = compare_feature_changes(trigger, uap)

    assert result["validation_auc"] == 1.0
    assert result["test_auc"] == 1.0
    assert 0 <= result["test_paired_cosine_distance"] <= 2


class MeanTargetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Identity()

    def forward(self, images):
        mean = self.stem(images).mean((1, 2, 3))
        return torch.stack([mean, 1 - mean], dim=1)


class AddOne(nn.Module):
    def forward(self, features):
        return features + 1


def test_fitted_feature_asr_reinjects_mapped_features():
    source = bundle(1.0)
    source["clean"].zero_()
    pairs = FeaturePairs(**source)

    asr = fitted_feature_asr(
        MeanTargetClassifier(),
        AddOne(),
        pairs,
        source["clean"],
        "stem",
        0,
        device="cpu",
        batch_size=4,
    )

    assert asr == 1.0
