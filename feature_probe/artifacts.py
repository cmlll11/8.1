from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from . import PROTOCOL
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
    if count == 0:
        raise ValueError(f"Pair bundle {path} is empty")
    if any(not isinstance(payload[key], torch.Tensor) for key in PAIR_KEYS):
        raise TypeError(f"Pair bundle {path} must store tensors for {PAIR_KEYS}")
    if any(len(payload[key]) != count for key in PAIR_KEYS):
        raise ValueError(f"Pair bundle {path} contains misaligned arrays")
    clean, mapped = payload["clean"], payload["mapped"]
    if clean.shape != mapped.shape or clean.ndim != 4:
        raise ValueError(f"Expected aligned NCHW tensors in {path}")
    if not clean.is_floating_point() or not mapped.is_floating_point():
        raise TypeError(f"Pair images must be floating-point tensors in {path}")
    if not torch.isfinite(clean).all() or not torch.isfinite(mapped).all():
        raise ValueError(f"Pair bundle {path} contains non-finite image values")
    split_codes = payload["split_codes"]
    if not torch.isin(split_codes, torch.tensor([0, 1, 2], dtype=split_codes.dtype)).all():
        raise ValueError(f"Pair bundle {path} contains invalid split codes")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("protocol") != PROTOCOL:
        raise ValueError(f"Pair bundle {path} does not use protocol {PROTOCOL}")
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
        if isinstance(artifact.get("model"), dict) and metadata.get("protocol") == PROTOCOL:
            if metadata.get("architecture") != "cifar_resnet18":
                raise RuntimeError(f"Unsupported classifier architecture in {path}: {metadata.get('architecture')!r}")
            model = CifarResNet18(classes=10)
            model.load_state_dict(artifact["model"])
            return model.to(device).eval()
    raise RuntimeError(f"Unsupported classifier artifact: {path}")
