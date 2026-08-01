from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


def resolve_module(model: nn.Module, name: str) -> nn.Module:
    modules = dict(model.named_modules())
    resolved = name
    if name == "stem" and name not in modules:
        # CIFAR ResNet implementations often use functional ReLU, so `bn1`
        # is the last hookable stem boundary. The same boundary is used for
        # extraction and replacement, preserving exact split consistency.
        for candidate in ("relu", "bn1", "conv1"):
            if candidate in modules:
                resolved = candidate
                break
    if resolved not in modules:
        preview = sorted(key for key in modules if key)[:30]
        raise KeyError(f"Layer {name!r} was not found. First available modules: {preview}")
    return modules[resolved]


class SplitClassifier:
    """Intercept or replace a named module output in a frozen classifier.

    Replacing a whole stage output makes the implementation architecture
    agnostic. `context` is the original image batch used to execute the prefix;
    after the hook replaces the layer output, only the suffix consumes `z`.
    """

    def __init__(self, model: nn.Module):
        self.model = model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def available_layers(self) -> list[str]:
        return [name for name, _ in self.model.named_modules() if name]

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor, layer: str) -> torch.Tensor:
        module = resolve_module(self.model, layer)
        captured: list[torch.Tensor] = []

        def hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"Layer {layer!r} returned {type(output)!r}, expected Tensor")
            captured.append(output.detach())

        handle = module.register_forward_hook(hook)
        try:
            self.model(images)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError(f"Layer {layer!r} executed {len(captured)} times")
        return captured[0]

    def forward_from_features(self, features: torch.Tensor, layer: str, *, context: torch.Tensor) -> torch.Tensor:
        module = resolve_module(self.model, layer)
        used = 0

        def hook(_module, _inputs, output):
            nonlocal used
            used += 1
            if not isinstance(output, torch.Tensor) or output.shape != features.shape:
                raise ValueError(
                    f"Replacement shape {tuple(features.shape)} does not match layer output "
                    f"{tuple(output.shape) if isinstance(output, torch.Tensor) else type(output)!r}"
                )
            return features

        handle = module.register_forward_hook(hook)
        try:
            logits = self.model(context)
        finally:
            handle.remove()
        if used != 1:
            raise RuntimeError(f"Layer {layer!r} executed {used} times")
        return logits

    @torch.no_grad()
    def assert_split_consistency(self, images: torch.Tensor, layers: Iterable[str], atol: float = 1e-5) -> dict[str, float]:
        reference = self.model(images)
        errors = {}
        for layer in layers:
            features = self.extract_features(images, layer)
            rebuilt = self.forward_from_features(features, layer, context=images)
            error = (reference - rebuilt).abs().max().item()
            errors[layer] = error
            if error > atol:
                raise AssertionError(f"Split consistency failed at {layer}: max error {error:g}")
        return errors
