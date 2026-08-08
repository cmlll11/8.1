Exit code: 0
Wall time: 6.7 seconds
Output:
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class KnownInputMapping(Protocol):
    mapping_id: str

    def apply(self, images: torch.Tensor) -> torch.Tensor: ...


def apply_mapping(mapping, images: torch.Tensor, *, indices=None, split=None) -> torch.Tensor:
    """Apply a neural generator or an object implementing KnownInputMapping."""
    if callable(mapping):
        mapped = mapping(images)
    else:
        apply = getattr(mapping, "apply", None)
        if not callable(apply):
            raise TypeError("mapping must be callable or provide an apply(images) method")
        if indices is None and split is None:
            mapped = apply(images)
        else:
            try:
                mapped = apply(images, indices=indices, split=split)
            except TypeError:
                mapped = apply(images)
    if not isinstance(mapped, torch.Tensor):
        raise TypeError("mapping output must be a torch.Tensor")
    if mapped.shape != images.shape:
        raise ValueError(f"mapping changed image shape from {tuple(images.shape)} to {tuple(mapped.shape)}")
    if mapped.device != images.device:
        raise ValueError("mapping output must remain on the input device")
    return mapped


def set_mapping_eval(mapping):
    """Put trainable mappings in evaluation mode; fixed mappings need no action."""
    evaluate = getattr(mapping, "eval", None)
    if callable(evaluate):
        evaluate()
    return mapping


@dataclass(frozen=True)
class ConstantPatch:
    mapping_id: str = "constant_patch"
    top: int = 29
    left: int = 29
    size: int = 3
    value: tuple[float, ...] = (1.0, 1.0, 1.0)

    def apply(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError("Expected NCHW images")
        if len(self.value) != images.shape[1]:
            raise ValueError("Patch value must contain one value per image channel")
        if self.size <= 0 or self.top < 0 or self.left < 0:
            raise ValueError("Patch size and location are invalid")
        if self.top + self.size > images.shape[2] or self.left + self.size > images.shape[3]:
            raise ValueError("Patch extends beyond the image boundary")
        result = images.clone()
        patch = torch.as_tensor(self.value, dtype=result.dtype, device=result.device).view(1, -1, 1, 1)
        result[:, :, self.top:self.top + self.size, self.left:self.left + self.size] = patch
        return result.clamp(0, 1)


@dataclass(frozen=True)
class UniversalAdditivePerturbation:
    delta: torch.Tensor
    mapping_id: str = "known_uap"

    def __post_init__(self):
        if self.delta.ndim not in (3, 4):
            raise ValueError("delta must have shape CHW or 1CHW")

    def apply(self, images: torch.Tensor) -> torch.Tensor:
        delta = self.delta
        if delta.ndim == 3:
            delta = delta.unsqueeze(0)
        return (images + delta.to(images.device, images.dtype)).clamp(0, 1)

