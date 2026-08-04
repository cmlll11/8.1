import torch

from feature_probe.triggers import build_trigger


def config():
    return {
        "target_label": 0,
        "backdoor": {"trigger_id": "badnets", "patch_size": 4, "patch_top": 28, "patch_left": 28, "patch_value": [1.0, 1.0, 1.0]},
        "triggers": {
            "badnets": {"size": 4, "top": 28, "left": 28, "value": [1.0, 1.0, 1.0]},
            "blended": {"alpha": 0.2}, "wanet": {"strength": 0.08},
            "inputaware": {"amplitude": 0.06}, "low_frequency": {"amplitude": 0.08, "frequency": 2},
            "ssba": {"amplitude": 0.025},
        },
    }


def test_all_trigger_families_are_deterministic_and_shape_preserving():
    images = torch.rand(8, 3, 32, 32)
    for trigger_id in ("badnets", "blended", "wanet", "inputaware", "low_frequency", "ssba"):
        trigger = build_trigger(trigger_id, config())
        first = trigger.apply(images)
        second = trigger.apply(images)
        assert first.shape == images.shape
        assert torch.equal(first, second)
        assert float(first.min()) >= 0.0 and float(first.max()) <= 1.0

