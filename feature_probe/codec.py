from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from . import PROTOCOL


QUANTIZATION_BITS = {"fp32": 32, "int8": 8, "int4": 4}
FAMILY_IDS = {"mean_shift": 0, "feature_re": 1, "fitnets": 2, "residual_adapter": 3}


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


def _string_bits(value: str) -> int:
    return 16 + len(value.encode("utf-8")) * 8


def _enumerative_mask_bits(total: int, kept: int) -> int:
    """Bits for kept-count plus a combinatorial code for sparse coordinates."""

    if not 0 <= kept <= total:
        raise ValueError("kept values must lie in [0, total]")
    count_bits = max(1, math.ceil(math.log2(total + 1)))
    if kept in (0, total):
        return 1 + count_bits
    log_combinations = (
        math.lgamma(total + 1) - math.lgamma(kept + 1) - math.lgamma(total - kept + 1)
    ) / math.log(2)
    return 1 + count_bits + math.ceil(log_combinations)


def count_feature_mapping_bits(
    model: nn.Module,
    *,
    layer_id: str,
    family: str,
    rank: int,
    kernel: int,
    quantization: str,
    pruning: float,
) -> BitBreakdown:
    """Count the fixed two-part MDL code for an already compressed mapping."""

    if quantization not in QUANTIZATION_BITS:
        raise ValueError(f"Unknown quantization: {quantization}")
    if family not in FAMILY_IDS:
        raise ValueError(f"Unknown feature mapping family: {family}")
    if not 0 <= float(pruning) < 1:
        raise ValueError("pruning must be in [0, 1)")
    named_tensors = [
        (name, value.detach().cpu())
        for name, value in model.state_dict().items()
        if value.is_floating_point()
    ]
    if any(not torch.isfinite(value).all() for _, value in named_tensors):
        raise ValueError("Cannot encode non-finite feature mapping parameters")
    total = sum(value.numel() for _, value in named_tensors)
    dense = float(pruning) == 0.0
    transmitted = total if dense else sum(int(torch.count_nonzero(value)) for _, value in named_tensors)
    bits_per_value = QUANTIZATION_BITS[quantization]

    # Header fixes magic/version and the public protocol identifier.
    header_bits = 32 + 16 + _string_bits(PROTOCOL)
    # Family and quantizer use fixed public IDs, avoiding arbitrary method-name penalties.
    family_id_bits = max(1, math.ceil(math.log2(len(FAMILY_IDS))))
    quantizer_id_bits = max(1, math.ceil(math.log2(len(QUANTIZATION_BITS))))
    # The decoder needs the layer, architecture rank/kernel, tensor count and tensor shapes.
    # Fit seed, optimizer, regularization and pruning search settings are not part of the mapping.
    structure_bits = _string_bits(layer_id) + family_id_bits + quantizer_id_bits + 32 * 2 + 16
    for _, value in named_tensors:
        structure_bits += 8 + 32 * value.ndim
    mask_bits = 1 if dense else _enumerative_mask_bits(total, transmitted)
    scale_bits = 0
    if quantization != "fp32":
        scale_bits = 32 * sum(
            1
            for _, value in named_tensors
            if dense or torch.count_nonzero(value).item() > 0
        )
    value_bits = transmitted * bits_per_value
    return BitBreakdown(PROTOCOL, header_bits, structure_bits, mask_bits, scale_bits, value_bits)
