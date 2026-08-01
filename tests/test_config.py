from feature_probe.config import render_asset_path, validate_config


def test_asset_template():
    assert render_asset_path("a/{trigger}/seed{seed}.pt", seed=2, trigger="badnets").as_posix() == "a/badnets/seed2.pt"


def test_minimal_config():
    validate_config(
        {
            "protocol": "MDL-FEATURE-v1",
            "target_label": 0,
            "classifier_seeds": [0],
            "trigger_ids": ["badnets"],
            "layers": {"candidates": ["stem"]},
            "assets": {"models": {}, "pairs": {}},
        }
    )
