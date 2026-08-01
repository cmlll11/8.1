from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch


PAIR_KEYS = ("clean", "mapped", "labels", "indices", "split_codes")


def load_pair_bundle(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Pair bundle must be a dictionary: {path}")
    # Earlier experiments used `oracle`; normalize it without changing the artifact.
    if "mapped" not in payload and "oracle" in payload:
        payload = dict(payload)
        payload["mapped"] = payload["oracle"]
    missing = [key for key in PAIR_KEYS if key not in payload]
    if missing:
        raise ValueError(f"Pair bundle {path} is missing keys: {missing}")
    count = len(payload["clean"])
    if any(len(payload[key]) != count for key in PAIR_KEYS):
        raise ValueError(f"Pair bundle {path} contains misaligned arrays")
    clean, mapped = payload["clean"], payload["mapped"]
    if clean.shape != mapped.shape or clean.ndim != 4:
        raise ValueError(f"Expected aligned NCHW tensors in {path}")
    return payload


def load_classifier(path: str | Path, device: str = "cpu", backdoorbench_root: str | Path = "external/BackdoorBench"):
    path = Path(path)
    bb_root = Path(backdoorbench_root).resolve()
    if bb_root.exists() and str(bb_root) not in sys.path:
        sys.path.insert(0, str(bb_root))
    artifact = torch.load(path, map_location=device, weights_only=False)
    if isinstance(artifact, torch.nn.Module):
        return artifact.to(device).eval()
    if isinstance(artifact, dict):
        for key in ("model", "net", "classifier"):
            if isinstance(artifact.get(key), torch.nn.Module):
                return artifact[key].to(device).eval()
        if "model_name" in artifact and isinstance(artifact.get("model"), dict):
            if not bb_root.exists():
                raise RuntimeError("BackdoorBench is required to rebuild this state dict")
            from utils.aggregate_block.model_trainer_generate import generate_cls_model

            model = generate_cls_model(artifact["model_name"], int(artifact.get("num_classes", 10)))
            state = {name.removeprefix("module."): value for name, value in artifact["model"].items()}
            model.load_state_dict(state)
            return model.to(device).eval()
    raise RuntimeError(f"Unsupported classifier artifact: {path}")
