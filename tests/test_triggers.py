import numpy as np
import pytest
import torch

from feature_probe.backdoorbench_attacks import OfficialInputAwareTrigger, OfficialSSBAArrayTrigger, build_inputaware_modules
from feature_probe.triggers import build_trigger


def config():
    return {
        "target_label": 0,
        "backdoor": {"trigger_id": "badnets", "patch_size": 4, "patch_top": 28, "patch_left": 28, "patch_value": [1.0, 1.0, 1.0]},
        "triggers": {
            "badnets": {"size": 4, "top": 28, "left": 28, "value": [1.0, 1.0, 1.0]},
            "blended": {"alpha": 0.2}, "wanet": {"strength": 0.08},
            "inputaware": {"source": "BackdoorBench/attack/inputaware.py"},
            "low_frequency": {"amplitude": 0.08, "frequency": 2},
            "ssba": {"source": "BackdoorBench/attack/ssba.py"},
        },
    }


def test_fixed_trigger_families_are_deterministic_and_shape_preserving():
    images = torch.rand(8, 3, 32, 32)
    for trigger_id in ("badnets", "blended", "wanet", "low_frequency"):
        trigger = build_trigger(trigger_id, config())
        first = trigger.apply(images)
        second = trigger.apply(images)
        assert first.shape == images.shape
        assert torch.equal(first, second)
        assert float(first.min()) >= 0.0 and float(first.max()) <= 1.0


def test_backdoorbench_inputaware_adapter_is_shape_preserving():
    modules = build_inputaware_modules("cpu")
    trigger = OfficialInputAwareTrigger(modules.generator, modules.mask, modules.threshold)
    images = torch.rand(2, 3, 32, 32)
    output = trigger.apply(images)
    assert output.shape == images.shape
    assert float(output.min()) >= 0.0 and float(output.max()) <= 1.0


def test_ssba_requires_official_arrays(tmp_path):
    train_path = tmp_path / "train.npy"
    test_path = tmp_path / "test.npy"
    array = np.zeros((4, 3, 32, 32), dtype=np.uint8)
    np.save(train_path, array)
    np.save(test_path, array)
    trigger = OfficialSSBAArrayTrigger(train_path, test_path)
    images = torch.rand(2, 3, 32, 32)
    output = trigger.apply(images, indices=torch.tensor([0, 1]), split="train")
    assert output.shape == images.shape


def test_ssba_accepts_official_nhwc_arrays(tmp_path):
    train_path = tmp_path / "train_nhwc.npy"
    test_path = tmp_path / "test_nhwc.npy"
    array = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    array[1, :, :, 0] = 255
    np.save(train_path, array)
    np.save(test_path, array)

    trigger = OfficialSSBAArrayTrigger(train_path, test_path)
    output = trigger.apply(torch.zeros(1, 3, 32, 32), indices=torch.tensor([1]), split="test")

    assert output.shape == (1, 3, 32, 32)
    assert torch.all(output[:, 0] == 1)


def test_inputaware_rejects_incomplete_checkpoint(tmp_path):
    path = tmp_path / "trigger_state.pt"
    torch.save({"generator": {}}, path)

    with pytest.raises(ValueError, match="Invalid Input-Aware"):
        build_trigger("inputaware", config(), checkpoint_path=str(path))
