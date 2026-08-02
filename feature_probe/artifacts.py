from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .models import CifarResNet18


PAIR_KEYS = ("clean", "mapped", "labels", "indices", "split_codes")


def load_pair_bundle(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Pair bundle must be a dictionary: {path}")
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


def load_classifier(path: str | Path, device: str = "cpu"):
    path = Path(path)
    artifact = torch.load(path, map_location=device, weights_only=False)
    if isinstance(artifact, torch.nn.Module):
        return artifact.to(device).eval()
    if isinstance(artifact, dict):
        for key in ("model", "net", "classifier"):
            if isinstance(artifact.get(key), torch.nn.Module):
                return artifact[key].to(device).eval()
        metadata = artifact.get("metadata", {})
        if isinstance(artifact.get("model"), dict) and metadata.get("protocol") == "MDL-FEATURE-v1":
            model = CifarResNet18(classes=10)
            model.load_state_dict(artifact["model"])
            return model.to(device).eval()
    raise RuntimeError(f"Unsupported classifier artifact: {path}")
