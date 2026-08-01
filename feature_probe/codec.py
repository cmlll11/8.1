from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from . import PROTOCOL


QUANTIZATION_BITS = {"fp32": 32, "int8": 8, "int4": 4}


@dataclass(frozen=True)
class BitBreakdown:
    protocol: str
    header_bits: int
    structure_bits: int
    mask_bits: int
    scale_bits: int
    value_bits: int

    @property
    def total_bits(self) -> int:
        return self.header_bits + self.structure_bits + self.mask_bits + self.scale_bits + self.value_bits

    def as_dict(self):
        return {**self.__dict__, "total_bits": self.total_bits}


def count_feature_mapping_bits(model: nn.Module, *, layer_id: str, level: str, rank: int, quantization: str, pruning: float) -> BitBreakdown:
    if quantization not in QUANTIZATION_BITS:
        raise ValueError(f"Unknown quantization: {quantization}")
    if not 0 <= pruning < 1:
        raise ValueError("pruning must be in [0, 1)")
    tensors = [value.detach().flatten() for value in model.state_dict().values() if value.is_floating_point()]
    values = torch.cat(tensors) if tensors else torch.empty(0)
    total = values.numel()
    kept = int(math.ceil(total * (1 - pruning)))
    bits_per_value = QUANTIZATION_BITS[quantization]
    # Fixed public fields: magic/version, layer string, family, rank, tensor shapes and pruning metadata.
    header_bits = 16 * 8 + len(PROTOCOL.encode("utf-8")) * 8
    structure_bits = (len(layer_id.encode("utf-8")) + len(level.encode("utf-8"))) * 8 + 16 + 32 * len(tensors)
    # Sparse locations are charged by a simple fixed-width index code. Dense models need no mask.
    mask_bits = 0 if kept == total else kept * max(1, math.ceil(math.log2(max(total, 2)))) + 64
    scale_bits = 0 if quantization == "fp32" else 32 * len(tensors)
    value_bits = kept * bits_per_value
    return BitBreakdown(PROTOCOL, header_bits, structure_bits, mask_bits, scale_bits, value_bits)
