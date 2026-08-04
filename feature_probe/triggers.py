from __future__ import annotations

"""Deterministic CIFAR-10 trigger mappings used by the multitype experiment.

The implementations are deliberately stateless: the trigger recipe is part of
the manifest and the mapping can therefore be replayed exactly when pair
bundles are regenerated.  They are small adapters around the public attack
families, not trigger-search procedures.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TriggerSpec:
    trigger_id: str
    target: int
    params: dict


class FamilyTrigger:
    def __init__(self, spec: TriggerSpec):
        self.spec = spec
        self.mapping_id = spec.trigger_id

    def apply(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BlendedTrigger(FamilyTrigger):
    def apply(self, images: torch.Tensor) -> torch.Tensor:
        alpha = float(self.spec.params.get("alpha", 0.20))
        if not 0.0 < alpha <= 1.0:
            raise ValueError("blended alpha must be in (0, 1]")
        n, c, h, w = images.shape
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, h, device=images.device, dtype=images.dtype),
            torch.linspace(0, 1, w, device=images.device, dtype=images.dtype),
            indexing="ij",
        )
        pattern = torch.stack((xx, yy, 1.0 - xx), dim=0).unsqueeze(0)
        if c != 3:
            pattern = pattern[:, :c]
        return ((1.0 - alpha) * images + alpha * pattern).clamp(0, 1)


class WaNetTrigger(FamilyTrigger):
    def apply(self, images: torch.Tensor) -> torch.Tensor:
        strength = float(self.spec.params.get("strength", 0.08))
        n, c, h, w = images.shape
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=images.device, dtype=images.dtype),
            torch.linspace(-1, 1, w, device=images.device, dtype=images.dtype),
            indexing="ij",
        )
        # A fixed smooth displacement is the deterministic inference-time
        # counterpart of the warping trigger family.
        dx = strength * torch.sin(math.pi * y) * torch.cos(math.pi * x)
        dy = strength * torch.cos(math.pi * y) * torch.sin(math.pi * x)
        grid = torch.stack((x + dx, y + dy), dim=-1).unsqueeze(0).expand(n, -1, -1, -1)
        return F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=True).clamp(0, 1)


class LowFrequencyTrigger(FamilyTrigger):
    def apply(self, images: torch.Tensor) -> torch.Tensor:
        amplitude = float(self.spec.params.get("amplitude", 0.08))
        frequency = int(self.spec.params.get("frequency", 2))
        _, c, h, w = images.shape
        y, x = torch.meshgrid(
            torch.arange(h, device=images.device, dtype=images.dtype),
            torch.arange(w, device=images.device, dtype=images.dtype),
            indexing="ij",
        )
        signal = torch.sin(2 * math.pi * frequency * x / max(w, 1))
        signal = signal * torch.cos(2 * math.pi * frequency * y / max(h, 1))
        delta = amplitude * signal.unsqueeze(0).unsqueeze(0)
        if c > 1:
            delta = delta.expand(-1, c, -1, -1)
        return (images + delta).clamp(0, 1)


class InputAwareTrigger(FamilyTrigger):
    def apply(self, images: torch.Tensor) -> torch.Tensor:
        amplitude = float(self.spec.params.get("amplitude", 0.06))
        # The image mean controls phase/amplitude, making the mapping
        # sample-specific while remaining deterministic and differentiable.
        phase = images.mean(dim=(1, 2, 3), keepdim=True) * (2 * math.pi)
        h, w = images.shape[-2:]
        y, x = torch.meshgrid(
            torch.linspace(0, 1, h, device=images.device, dtype=images.dtype),
            torch.linspace(0, 1, w, device=images.device, dtype=images.dtype),
            indexing="ij",
        )
        carrier = torch.sin(2 * math.pi * x + phase) * torch.cos(2 * math.pi * y + phase)
        return (images + amplitude * carrier).clamp(0, 1)


class SSBATrigger(FamilyTrigger):
    def apply(self, images: torch.Tensor) -> torch.Tensor:
        amplitude = float(self.spec.params.get("amplitude", 0.025))
        # A content-dependent high-frequency residual is a lightweight,
        # reproducible stand-in for the sample-specific steganographic family.
        pooled = F.avg_pool2d(images, kernel_size=3, stride=1, padding=1)
        residual = images - pooled
        return (images + amplitude * torch.tanh(residual * 8.0)).clamp(0, 1)


class BadNetsTrigger(FamilyTrigger):
    def apply(self, images: torch.Tensor) -> torch.Tensor:
        top = int(self.spec.params.get("top", 28))
        left = int(self.spec.params.get("left", 28))
        size = int(self.spec.params.get("size", 4))
        value = tuple(float(v) for v in self.spec.params.get("value", [1.0, 1.0, 1.0]))
        if images.ndim != 4 or size <= 0 or top < 0 or left < 0:
            raise ValueError("invalid BadNets trigger configuration")
        if top + size > images.shape[-2] or left + size > images.shape[-1]:
            raise ValueError("BadNets patch exceeds image boundary")
        patch = torch.as_tensor(value, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
        if patch.shape[1] != images.shape[1]:
            raise ValueError("BadNets value must contain one value per channel")
        output = images.clone()
        output[:, :, top:top + size, left:left + size] = patch
        return output.clamp(0, 1)


TRIGGER_CLASSES = {
    "badnets": BadNetsTrigger,
    "blended": BlendedTrigger,
    "wanet": WaNetTrigger,
    "inputaware": InputAwareTrigger,
    "low_frequency": LowFrequencyTrigger,
    "ssba": SSBATrigger,
}


def build_trigger(trigger_id: str, config: dict, *, target: int | None = None):
    trigger_id = str(trigger_id)
    if trigger_id not in TRIGGER_CLASSES:
        raise ValueError(f"Unknown trigger family: {trigger_id}")
    recipes = config.get("triggers", {})
    recipe = dict(recipes.get(trigger_id, {}))
    if trigger_id == "badnets":
        legacy = config.get("backdoor", {})
        recipe = {"top": legacy.get("patch_top", 28), "left": legacy.get("patch_left", 28),
                  "size": legacy.get("patch_size", 4), "value": legacy.get("patch_value", [1.0, 1.0, 1.0]), **recipe}
    return TRIGGER_CLASSES[trigger_id](TriggerSpec(trigger_id, int(config.get("target_label", target or 0)), recipe))
