from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch import nn


QUANTIZATION_MAX = {"fp32": None, "int8": 127, "int4": 7}


@dataclass(frozen=True)
class CompressionResult:
    model: nn.Module
    total_values: int
    kept_values: int
    scales: dict[str, float | None]


def compress_feature_mapping(model: nn.Module, *, pruning: float, quantization: str) -> CompressionResult:
    """Materialize the exact pruned and dequantized mapping used for evaluation."""
    if quantization not in QUANTIZATION_MAX:
        raise ValueError(f"Unknown quantization: {quantization}")
    if not 0 <= float(pruning) < 1:
        raise ValueError("pruning must be in [0, 1)")
    compressed = copy.deepcopy(model).cpu().eval()
    state = compressed.state_dict()
    floating = [(name, value.detach().cpu()) for name, value in state.items() if value.is_floating_point()]
    if any(not torch.isfinite(value).all() for _, value in floating):
        raise ValueError("Cannot compress a feature mapping with non-finite parameters")
    total = sum(value.numel() for _, value in floating)
    kept = int(math.ceil(total * (1 - float(pruning))))
    global_values = torch.cat([value.abs().flatten() for _, value in floating]) if floating else torch.empty(0)
    global_mask = torch.zeros(total, dtype=torch.bool)
    if kept:
        selected = torch.topk(global_values, kept, largest=True, sorted=False).indices
        global_mask[selected] = True
    cursor = 0
    scales: dict[str, float | None] = {}
    decoded = dict(state)
    maximum = QUANTIZATION_MAX[quantization]
    encoded_values = 0
    for name, value in floating:
        count = value.numel()
        mask = global_mask[cursor:cursor + count].reshape(value.shape)
        cursor += count
        pruned = torch.where(mask, value, torch.zeros_like(value))
        if maximum is None:
            scales[name] = None
            decoded[name] = pruned
            encoded_values += count if float(pruning) == 0.0 else int(torch.count_nonzero(pruned))
            continue
        selected_values = pruned[mask]
        max_abs = selected_values.abs().max().item() if selected_values.numel() else 0.0
        scale = max_abs / maximum if max_abs > 0 else 1.0
        quantized = torch.round(pruned / scale).clamp(-maximum, maximum)
        decoded[name] = quantized * scale
        scales[name] = float(scale)
        encoded_values += count if float(pruning) == 0.0 else int(torch.count_nonzero(quantized))
    compressed.load_state_dict(decoded)
    return CompressionResult(compressed, total, encoded_values, scales)
