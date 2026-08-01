from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def flatten_normalize(changes: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    flat = changes.flatten(1)
    return flat / flat.norm(dim=1, keepdim=True).clamp_min(epsilon)


def feature_change_summary(clean: torch.Tensor, mapped: torch.Tensor) -> dict[str, float]:
    if clean.shape != mapped.shape:
        raise ValueError("clean and mapped features must have the same shape")
    delta = mapped - clean
    flat_delta = delta.flatten(1)
    flat_clean = clean.flatten(1)
    energy = flat_delta.square()
    total = energy.sum(1).clamp_min(1e-12)
    sorted_energy = energy.sort(dim=1, descending=True).values
    top_fraction = max(1, math.ceil(energy.shape[1] * 0.01))
    probability = energy / total[:, None]
    participation = total.square() / energy.square().sum(1).clamp_min(1e-12)
    directions = flatten_normalize(delta)
    centroid = F.normalize(directions.mean(0, keepdim=True), dim=1)
    return {
        "relative_l1": (flat_delta.abs().mean(1) / flat_clean.abs().mean(1).clamp_min(1e-12)).mean().item(),
        "relative_l2": (flat_delta.norm(dim=1) / flat_clean.norm(dim=1).clamp_min(1e-12)).mean().item(),
        "mse": flat_delta.square().mean().item(),
        "top1pct_energy": (sorted_energy[:, :top_fraction].sum(1) / total).mean().item(),
        "participation_dimension": participation.mean().item(),
        "direction_consistency": (directions @ centroid.T).mean().item(),
        "nonzero_fraction": (flat_delta.abs() > 1e-8).float().mean().item(),
    }


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.shape[0] != y.shape[0]:
        raise ValueError("CKA inputs must have the same number of examples")
    x = x.flatten(1).double()
    y = y.flatten(1).double()
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    cross = x.T @ y
    numerator = cross.square().sum()
    denominator = ((x.T @ x).square().sum() * (y.T @ y).square().sum()).sqrt().clamp_min(1e-24)
    return (numerator / denominator).item()


def mean_cosine_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    x = flatten_normalize(x)
    y = flatten_normalize(y)
    if x.shape != y.shape:
        raise ValueError("Paired cosine comparison requires matching shapes")
    return (1 - (x * y).sum(1)).mean().item()


def binary_roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.detach().flatten().double()
    labels = labels.detach().flatten().bool()
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.double)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.double, device=scores.device)
    # Exact ties receive their average rank.
    unique, inverse, counts = torch.unique(scores, sorted=True, return_inverse=True, return_counts=True)
    del unique
    if (counts > 1).any():
        for group in torch.nonzero(counts > 1, as_tuple=False).flatten():
            mask = inverse == group
            ranks[mask] = ranks[mask].mean()
    rank_sum = ranks[labels].sum()
    return ((rank_sum - positives * (positives + 1) / 2) / (positives * negatives)).item()


def nearest_centroid_auc(train_trigger: torch.Tensor, train_uap: torch.Tensor, test_trigger: torch.Tensor, test_uap: torch.Tensor) -> float:
    trigger_center = F.normalize(flatten_normalize(train_trigger).mean(0), dim=0)
    uap_center = F.normalize(flatten_normalize(train_uap).mean(0), dim=0)
    test = torch.cat([flatten_normalize(test_trigger), flatten_normalize(test_uap)])
    scores = test @ trigger_center - test @ uap_center
    labels = torch.cat([torch.ones(len(test_trigger)), torch.zeros(len(test_uap))]).to(scores.device)
    return binary_roc_auc(scores, labels)
