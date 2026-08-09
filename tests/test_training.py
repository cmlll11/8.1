from types import SimpleNamespace

import torch
from torch import nn

from feature_probe.mappings import ConstantPatch
from feature_probe.training import mapping_asr, train_classifier_pair


class PatchTargetClassifier(nn.Module):
    def forward(self, images):
        target_score = images[:, :, -1, -1].mean(1)
        return torch.stack([target_score, 1 - target_score], dim=1)


class WhiteImageGenerator(nn.Module):
    def forward(self, images):
        return torch.ones_like(images)


def test_mapping_asr_supports_fixed_and_trainable_mappings():
    model = PatchTargetClassifier()
    images = torch.zeros(4, 3, 4, 4)
    labels = torch.ones(4, dtype=torch.long)
    patch = ConstantPatch(top=3, left=3, size=1)
    generator = WhiteImageGenerator().train()

    patch_asr = mapping_asr(
        model,
        patch,
        images,
        labels,
        0,
        batch_size=2,
        device="cpu",
        indices=torch.arange(len(images)),
        split="test",
    )
    generator_asr = mapping_asr(model, generator, images, labels, 0, batch_size=2, device="cpu")

    assert patch_asr == 1.0
    assert generator_asr == 1.0
    assert not generator.training


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.fc = nn.Linear(4, 2)

    def forward(self, images):
        return self.fc(torch.relu(self.conv(images)).mean((2, 3)))


class TinyInputAwareGenerator(nn.Module):
    def __init__(self, output_channels):
        super().__init__()
        self.conv = nn.Conv2d(3, output_channels, 1)

    def forward(self, images):
        return torch.sigmoid(self.conv(images))


class TinyThreshold(nn.Module):
    def forward(self, values):
        return values


def test_classifier_pair_training_pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr("feature_probe.training.CifarResNet18", TinyClassifier)
    generator = torch.Generator().manual_seed(7)
    splits = SimpleNamespace(
        train_images=torch.rand(16, 3, 32, 32, generator=generator),
        train_labels=torch.tensor([0, 1] * 8),
        train_indices=torch.arange(100, 116),
        validation_images=torch.rand(8, 3, 32, 32, generator=generator),
        validation_labels=torch.tensor([0, 1] * 4),
        test_images=torch.rand(8, 3, 32, 32, generator=generator),
        test_labels=torch.tensor([0, 1] * 4),
    )
    config = {
        "protocol": "MDL-FEATURE-v1",
        "target_label": 0,
        "data": {
            "split_seed": 2026,
            "smoke_train_examples": 16,
            "smoke_validation_examples": 8,
            "smoke_test_examples": 8,
        },
        "classifier": {
            "epochs": 1,
            "smoke_epochs": 1,
            "batch_size": 8,
            "learning_rate": 0.01,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "amp": False,
        },
        "backdoor": {
            "trigger_id": "badnets",
            "patch_top": 28,
            "patch_left": 28,
            "patch_size": 4,
            "patch_value": [1.0, 1.0, 1.0],
            "poison_fraction": 0.5,
        },
        "qualification": {
            "minimum_clean_accuracy": 0.0,
            "minimum_backdoor_asr": 0.0,
            "maximum_clean_patch_asr": 1.0,
        },
    }

    controls = train_classifier_pair(
        splits,
        config,
        pair_seed=0,
        device="cpu",
        output_root=tmp_path,
        smoke=True,
    )

    assert controls["all_passed"]
    assert controls["poisoned_examples"] == 4
    assert (tmp_path / "clean/seed0/attack_result.pt").is_file()
    assert (tmp_path / "badnets/seed0/attack_result.pt").is_file()


def test_inputaware_training_and_resume_pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr("feature_probe.training.CifarResNet18", TinyClassifier)
    monkeypatch.setattr(
        "feature_probe.training.build_inputaware_modules",
        lambda _device: SimpleNamespace(
            generator=TinyInputAwareGenerator(3),
            mask=TinyInputAwareGenerator(1),
            threshold=TinyThreshold(),
        ),
    )
    generator = torch.Generator().manual_seed(11)
    splits = SimpleNamespace(
        train_images=torch.rand(16, 3, 32, 32, generator=generator),
        train_labels=torch.tensor([0, 1] * 8),
        train_indices=torch.arange(100, 116),
        validation_images=torch.rand(8, 3, 32, 32, generator=generator),
        validation_labels=torch.tensor([0, 1] * 4),
        validation_indices=torch.arange(200, 208),
        test_images=torch.rand(8, 3, 32, 32, generator=generator),
        test_labels=torch.tensor([0, 1] * 4),
        test_indices=torch.arange(300, 308),
    )
    config = {
        "protocol": "MDL-FEATURE-v1", "target_label": 0,
        "data": {"split_seed": 2026, "smoke_train_examples": 16, "smoke_validation_examples": 8, "smoke_test_examples": 8},
        "classifier": {"epochs": 1, "smoke_epochs": 1, "batch_size": 8, "learning_rate": 0.01, "momentum": 0.0, "weight_decay": 0.0, "amp": False},
        "backdoor": {"poison_fraction": 0.5},
        "triggers": {"inputaware": {"mask_epochs": 1, "lr_G": 0.01, "lr_M": 0.01, "lambda_div": 0.1, "lambda_norm": 0.1, "mask_density": 0.5, "schedulerG_milestones": [2], "schedulerM_milestones": [2]}},
        "qualification": {"minimum_clean_accuracy": 0.0, "minimum_backdoor_asr": 0.0, "maximum_clean_patch_asr": 1.0},
    }

    first = train_classifier_pair(splits, config, pair_seed=0, device="cpu", output_root=tmp_path, smoke=True, trigger_id="inputaware")
    second = train_classifier_pair(splits, config, pair_seed=0, device="cpu", output_root=tmp_path, smoke=True, trigger_id="inputaware")

    assert first["implementation_version"] == 2
    assert second["all_passed"]
    assert (tmp_path / "inputaware/seed0/trigger_state.pt").is_file()
