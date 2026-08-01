from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class KnownInputMapping(Protocol):
    mapping_id: str

    def apply(self, images: torch.Tensor) -> torch.Tensor: ...


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
