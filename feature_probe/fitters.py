from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn


class DeviceTensorBatches:
    """Iterate already-materialized feature pairs without CPU collation."""

    def __init__(self, clean, target, *, batch_size: int, shuffle: bool, seed: int = 0):
        if len(clean) != len(target) or clean.device != target.device:
            raise ValueError("clean and target must be aligned on the same device")
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        self.clean = clean
        self.target = target
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            order = torch.randperm(len(self.clean), generator=generator).to(self.clean.device)
            self.epoch += 1
            for start in range(0, len(order), self.batch_size):
                selected = order[start:start + self.batch_size]
                yield self.clean.index_select(0, selected), self.target.index_select(0, selected)
        else:
            for start in range(0, len(self.clean), self.batch_size):
                yield self.clean[start:start + self.batch_size], self.target[start:start + self.batch_size]


class ChannelAffine(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, z):
        return z * self.scale + self.bias


class FixedResidual(nn.Module):
    def __init__(self, shape: tuple[int, int, int]):
        super().__init__()
        self.delta = nn.Parameter(torch.zeros(1, *shape))

    def forward(self, z):
        return z + self.delta


class BottleneckResidual(nn.Module):
    def __init__(self, channels: int, rank: int, kernel: int, depth: int):
        super().__init__()
        padding = kernel // 2
        blocks = []
        for _ in range(depth):
            blocks.extend(
                [
                    nn.Conv2d(channels, rank, 1, bias=True),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(rank, rank, kernel, padding=padding, bias=True),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(rank, channels, 1, bias=True),
                ]
            )
        self.network = nn.Sequential(*blocks)
        self.depth = depth

    def forward(self, z):
        value = z
        cursor = 0
        for _ in range(self.depth):
            residual = self.network[cursor:cursor + 5](value)
            value = value + residual
            cursor += 5
        return value


def build_feature_mapping(level: str, feature_shape: tuple[int, int, int], rank: int = 1) -> nn.Module:
    channels = int(feature_shape[0])
    if level == "C0":
        return ChannelAffine(channels)
    if level == "C1":
        return FixedResidual(feature_shape)
    if level == "C2":
        return BottleneckResidual(channels, rank, kernel=1, depth=1)
    if level == "C3":
        return BottleneckResidual(channels, rank, kernel=3, depth=1)
    if level == "C4":
        return BottleneckResidual(channels, rank, kernel=3, depth=2)
    raise ValueError(f"Unknown feature mapping level: {level}")


def normalized_rmse(predicted: torch.Tensor, target: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    error = (predicted - target).flatten(1).square().mean(1).sqrt()
    scale = (target - clean).flatten(1).square().mean(1).sqrt().clamp_min(1e-8)
    return (error / scale).mean()


@dataclass
class FitResult:
    model: nn.Module
    best_validation_nrmse: float
    history: list[dict[str, float]]


def fit_feature_mapping(
    model,
    train_batches,
    validation_batches,
    *,
    steps: int,
    learning_rate: float,
    device: str,
    validation_interval: int = 1,
    progress_callback=None,
) -> FitResult:
    if int(steps) < 1 or int(validation_interval) < 1:
        raise ValueError("steps and validation_interval must be positive")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    history = []
    best_score = float("inf")
    best_state = None
    train_iterator = iter(train_batches)
    for step in range(int(steps)):
        try:
            clean, target = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_batches)
            clean, target = next(train_iterator)
        clean, target = clean.to(device), target.to(device)
        predicted = model(clean)
        loss = normalized_rmse(predicted, target, clean)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        should_validate = step == 0 or step + 1 == int(steps) or (step + 1) % int(validation_interval) == 0
        if not should_validate:
            continue
        validation = evaluate_feature_mapping(model, validation_batches, device=device)
        model.train()
        record = {"step": step + 1, "train_nrmse": loss.item(), "validation_nrmse": validation}
        history.append(record)
        if progress_callback is not None:
            progress_callback(record)
        if validation < best_score:
            best_score = validation
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return FitResult(model=model, best_validation_nrmse=best_score, history=history)


@torch.no_grad()
def evaluate_feature_mapping(model, data_batches, *, device: str) -> float:
    model = model.to(device).eval()
    total = 0.0
    examples = 0
    for clean, target in data_batches:
        clean, target = clean.to(device), target.to(device)
        value = normalized_rmse(model(clean), target, clean)
        total += value.item() * len(clean)
        examples += len(clean)
    if examples == 0:
        raise ValueError("Cannot evaluate an empty feature-pair loader")
    return total / examples
