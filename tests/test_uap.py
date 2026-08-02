from types import SimpleNamespace

import torch
from torch import nn

from feature_probe.uap import targeted_margin_loss, train_projected_targeted_uap


class MeanThresholdClassifier(nn.Module):
    def forward(self, images):
        mean = images.mean((1, 2, 3))
        return torch.stack([mean, torch.full_like(mean, 0.4)], dim=1)


def test_targeted_margin_loss_rewards_target_logit():
    weak = targeted_margin_loss(torch.tensor([[2.0, 0.0]]), target=1)
    strong = targeted_margin_loss(torch.tensor([[0.0, 2.0]]), target=1)

    assert weak > strong
    assert strong == 0


def test_projected_targeted_uap_learns_fixed_delta(tmp_path):
    images = torch.full((8, 3, 32, 32), 0.5)
    labels = torch.zeros(8, dtype=torch.long)
    splits = SimpleNamespace(
        train_images=images,
        train_labels=labels,
        validation_images=images[:4],
        validation_labels=labels[:4],
        test_images=images[:4],
        test_labels=labels[:4],
    )
    config = {
        "protocol": "MDL-FEATURE-v1",
        "target_label": 1,
        "data": {
            "smoke_train_examples": 8,
            "smoke_validation_examples": 4,
            "smoke_test_examples": 4,
        },
        "qualification": {"minimum_adversarial_asr": 1.0},
        "targeted_uap": {
            "epsilon_candidates": [0.2],
            "restarts": 1,
            "epochs_per_epsilon": 4,
            "smoke_epochs": 4,
            "batch_size": 4,
            "learning_rate": 0.1,
            "confidence": 0.0,
            "cross_entropy_weight": 0.1,
            "success_patience": 1,
        },
    }

    mapping, metrics = train_projected_targeted_uap(
        MeanThresholdClassifier(),
        splits,
        config,
        pair_seed=0,
        device="cpu",
        output_path=tmp_path / "uap.pt",
        smoke=True,
    )

    assert metrics["test_asr"] == 1.0
    assert metrics["linf"] <= 0.200001
    assert torch.equal(mapping.apply(images[:1]), (images[:1] + mapping.delta).clamp(0, 1))
