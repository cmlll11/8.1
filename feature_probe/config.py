from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from . import PROTOCOL


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    validate_config(payload)
    payload["_config_path"] = str(path.resolve())
    return payload


def validate_config(config: dict[str, Any]) -> None:
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"protocol must be {PROTOCOL!r}")
    required = ("target_label", "classifier_seeds", "trigger_ids", "layers", "assets")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing configuration keys: {missing}")
    if not config["classifier_seeds"] or not config["trigger_ids"]:
        raise ValueError("classifier_seeds and trigger_ids must not be empty")
    if not 0 <= int(config["target_label"]) < 10:
        raise ValueError("target_label must be a CIFAR-10 class in [0, 9]")
    candidates = config["layers"].get("candidates", [])
    if not candidates:
        raise ValueError("layers.candidates must not be empty")
    if len(set(candidates)) != len(candidates):
        raise ValueError("layers.candidates contains duplicates")
    for group in ("models", "pairs"):
        if group not in config["assets"]:
            raise ValueError(f"assets.{group} is required")
    backdoor = config.get("backdoor")
    if backdoor is not None and backdoor.get("trigger_id") not in config["trigger_ids"]:
        raise ValueError("backdoor.trigger_id must be listed in trigger_ids")


def render_asset_path(template: str, *, seed: int, trigger: str | None = None) -> Path:
    values = {"seed": int(seed), "trigger": trigger or ""}
    try:
        return Path(template.format(**values))
    except KeyError as exc:
        raise ValueError(f"Unknown placeholder {exc} in asset path {template!r}") from exc
