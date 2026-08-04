from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


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


class MeanFeatureShift(nn.Module):
    """Dataset-average trigger-activated change used as a deterministic baseline."""

    closed_form = True

    def __init__(self, shape: tuple[int, int, int]):
        super().__init__()
        self.delta = nn.Parameter(torch.zeros(1, *shape), requires_grad=False)

    def forward(self, z):
        return z + self.delta

    @torch.no_grad()
    def fit_closed_form(self, batches) -> None:
        total = torch.zeros_like(self.delta)
        examples = 0
        for clean, target in batches:
            total += (target - clean).sum(0, keepdim=True)
            examples += len(clean)
        if examples == 0:
            raise ValueError("Cannot fit an empty feature-pair loader")
        self.delta.copy_(total / examples)


class FeatureREMaskPattern(nn.Module):
    """FeatureRE's masked feature replacement: (1-m) * z + m * pattern."""

    closed_form = False

    def __init__(self, shape: tuple[int, int, int], mask_penalty: float):
        super().__init__()
        self.mask = nn.Parameter(torch.full((1, *shape), 0.05))
        self.pattern = nn.Parameter(torch.zeros(1, *shape))
        self.mask_penalty = float(mask_penalty)

    def forward(self, z):
        mask = self.mask.clamp(0.0, 1.0)
        return (1.0 - mask) * z + mask * self.pattern

    def regularization_loss(self) -> torch.Tensor:
        return self.mask_penalty * self.mask.clamp(0.0, 1.0).mean()

    @torch.no_grad()
    def project_parameters(self) -> None:
        self.mask.clamp_(0.0, 1.0)


class FitNetsHintRegressor(nn.Module):
    """FitNets-style convolutional regressor trained with squared hint loss."""

    closed_form = False

    def __init__(self, channels: int, rank: int, kernel: int):
        super().__init__()
        if kernel not in (1, 3):
            raise ValueError("FitNets kernel must be 1 or 3")
        self.regressor = nn.Sequential(
            nn.Conv2d(channels, rank, kernel, padding=kernel // 2, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(rank, channels, 1, bias=True),
        )
        nn.init.zeros_(self.regressor[-1].weight)
        nn.init.zeros_(self.regressor[-1].bias)

    def forward(self, z):
        return z + self.regressor(z)


class ResidualAdapter(nn.Module):
    """Parallel 1x1 residual adapter from Rebuffi et al."""

    closed_form = False

    def __init__(self, channels: int):
        super().__init__()
        self.adapter = nn.Conv2d(channels, channels, 1, bias=True)
        nn.init.zeros_(self.adapter.weight)
        nn.init.zeros_(self.adapter.bias)

    def forward(self, z):
        return z + self.adapter(z)


def derive_spatial_support(
    clean: torch.Tensor,
    mapped: torch.Tensor,
    *,
    relative_threshold: float = 1e-6,
) -> torch.Tensor:
    """Derive a train-only spatial support from average paired feature change."""

    if clean.shape != mapped.shape or clean.ndim != 4:
        raise ValueError("clean and mapped features must have the same NCHW shape")
    if not 0.0 <= float(relative_threshold) < 1.0:
        raise ValueError("relative_threshold must be in [0, 1)")
    energy = (mapped - clean).abs().mean(dim=(0, 1))
    maximum = energy.max()
    if maximum.item() == 0.0:
        raise ValueError("Cannot derive support from an all-zero feature change")
    support = energy > maximum * float(relative_threshold)
    if not support.any():
        raise RuntimeError("Derived spatial support is empty")
    return support


class SpatiallyGatedHintRegressor(nn.Module):
    """FeatureRE spatial gating plus a coordinate-aware FitNets regressor."""

    closed_form = False

    def __init__(self, shape: tuple[int, int, int], rank: int, support: torch.Tensor):
        super().__init__()
        channels, height, width = (int(value) for value in shape)
        if support.shape != (height, width) or support.dtype != torch.bool:
            raise ValueError("support must be a boolean mask matching the feature spatial shape")
        if int(rank) < 1:
            raise ValueError("rank must be positive")
        self.register_buffer("support_mask", support.reshape(1, 1, height, width).clone())
        y = torch.linspace(-1.0, 1.0, height)
        x = torch.linspace(-1.0, 1.0, width)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        self.register_buffer(
            "coordinate_grid",
            torch.stack((grid_y, grid_x)).reshape(1, 2, height, width),
            persistent=False,
        )
        self.spatial_bias = nn.Parameter(torch.zeros(1, channels, height, width))
        self.regressor = nn.Sequential(
            nn.Conv2d(channels + 3, int(rank), 3, padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(int(rank), int(rank), 3, padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(int(rank), channels, 1, bias=True),
        )
        nn.init.zeros_(self.regressor[-1].weight)
        nn.init.zeros_(self.regressor[-1].bias)

    def forward(self, z):
        support = self.support_mask.to(dtype=z.dtype)
        coordinates = self.coordinate_grid.to(dtype=z.dtype).expand(len(z), -1, -1, -1)
        context = torch.cat((z, coordinates, support.expand(len(z), -1, -1, -1)), dim=1)
        residual = self.spatial_bias + self.regressor(context)
        return z + support * residual

    @torch.no_grad()
    def initialize_from_batches(self, batches) -> None:
        total = torch.zeros_like(self.spatial_bias)
        examples = 0
        for clean, target in batches:
            total += (target - clean).sum(0, keepdim=True)
            examples += len(clean)
        if examples == 0:
            raise ValueError("Cannot initialize from an empty feature-pair loader")
        self.spatial_bias.copy_(total / examples * self.support_mask)

    def training_loss(self, predicted, target, clean) -> torch.Tensor:
        support = self.support_mask.to(dtype=predicted.dtype)
        squared_error = (predicted - target).square() * support
        denominator = len(predicted) * predicted.shape[1] * int(self.support_mask.sum())
        return squared_error.sum() / max(denominator, 1)


def build_feature_mapping(
    family: str,
    feature_shape: tuple[int, int, int],
    *,
    rank: int = 1,
    kernel: int = 1,
    mask_penalty: float = 0.0,
    spatial_support: torch.Tensor | None = None,
) -> nn.Module:
    channels = int(feature_shape[0])
    if family == "mean_shift":
        return MeanFeatureShift(feature_shape)
    if family == "feature_re":
        return FeatureREMaskPattern(feature_shape, mask_penalty)
    if family == "fitnets":
        return FitNetsHintRegressor(channels, int(rank), int(kernel))
    if family == "residual_adapter":
        return ResidualAdapter(channels)
    if family == "spatial_gated_fitnets":
        if spatial_support is None:
            raise ValueError("spatial_gated_fitnets requires a spatial support mask")
        return SpatiallyGatedHintRegressor(feature_shape, int(rank), spatial_support)
    raise ValueError(f"Unknown feature mapping family: {family}")


def _squared_error_totals(
    predicted: torch.Tensor,
    target: torch.Tensor,
    clean: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    error = (predicted - target).square().sum(dtype=torch.float64)
    signal = (target - clean).square().sum(dtype=torch.float64)
    return error, signal


def normalized_rmse(predicted: torch.Tensor, target: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Global relative Frobenius error with a finite zero-signal convention."""

    error, signal = _squared_error_totals(predicted, target, clean)
    epsilon = torch.finfo(torch.float64).eps
    return torch.sqrt(error / signal.clamp_min(epsilon))


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
    gradient_clip_norm: float = 5.0,
    progress_callback=None,
) -> FitResult:
    """Fit published feature mappers with MSE; reserve NRMSE for evaluation."""

    if int(steps) < 1 or int(validation_interval) < 1:
        raise ValueError("steps and validation_interval must be positive")
    model = model.to(device)
    if getattr(model, "closed_form", False):
        model.fit_closed_form(train_batches)
        validation = evaluate_feature_mapping(model, validation_batches, device=device)
        record = {"step": 0, "train_mse": 0.0, "regularization": 0.0, "validation_nrmse": validation}
        if progress_callback is not None:
            progress_callback(record)
        return FitResult(model=model, best_validation_nrmse=validation, history=[record])

    if hasattr(model, "initialize_from_batches"):
        model.initialize_from_batches(train_batches)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Trainable feature mapping has no trainable parameters")
    optimizer = torch.optim.Adam(parameters, lr=float(learning_rate))
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
        mse = (
            model.training_loss(predicted, target, clean)
            if hasattr(model, "training_loss")
            else F.mse_loss(predicted, target)
        )
        regularization = (
            model.regularization_loss()
            if hasattr(model, "regularization_loss")
            else torch.zeros((), device=mse.device)
        )
        loss = mse + regularization
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at step {step + 1}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, float(gradient_clip_norm), error_if_nonfinite=True)
        optimizer.step()
        if hasattr(model, "project_parameters"):
            model.project_parameters()
        should_validate = step == 0 or step + 1 == int(steps) or (step + 1) % int(validation_interval) == 0
        if not should_validate:
            continue
        validation = evaluate_feature_mapping(model, validation_batches, device=device)
        model.train()
        record = {
            "step": step + 1,
            "train_mse": float(mse.detach().item()),
            "regularization": float(regularization.detach().item()),
            "validation_nrmse": validation,
        }
        history.append(record)
        if progress_callback is not None:
            progress_callback(record)
        if validation < best_score:
            best_score = validation
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("Feature mapping did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    return FitResult(model=model, best_validation_nrmse=best_score, history=history)


@torch.no_grad()
def evaluate_feature_mapping(model, data_batches, *, device: str) -> float:
    model = model.to(device).eval()
    error = torch.zeros((), dtype=torch.float64, device=device)
    signal = torch.zeros((), dtype=torch.float64, device=device)
    examples = 0
    for clean, target in data_batches:
        clean, target = clean.to(device), target.to(device)
        batch_error, batch_signal = _squared_error_totals(model(clean), target, clean)
        error += batch_error
        signal += batch_signal
        examples += len(clean)
    if examples == 0:
        raise ValueError("Cannot evaluate an empty feature-pair loader")
    epsilon = torch.finfo(torch.float64).eps
    value = torch.sqrt(error / signal.clamp_min(epsilon)).item()
    if not torch.isfinite(torch.tensor(value)):
        raise RuntimeError("Feature mapping evaluation produced a non-finite NRMSE")
    return float(value)
