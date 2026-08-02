from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn

from .mappings import UniversalAdditivePerturbation
from .training import batches


def targeted_margin_loss(logits: torch.Tensor, target: int, confidence: float = 0.0) -> torch.Tensor:
    target_logits = logits[:, int(target)]
    masked = logits.clone()
    masked[:, int(target)] = -torch.inf
    other_logits = masked.max(1).values
    return torch.relu(other_logits - target_logits + float(confidence)).mean()


@torch.no_grad()
def delta_asr(model, delta, images, labels, target, *, batch_size: int, device: str) -> float:
    successes = examples = 0
    model.eval()
    for batch_images, batch_labels in batches(images, labels, batch_size=batch_size, seed=0, shuffle=False):
        keep = batch_labels != int(target)
        if not keep.any():
            continue
        batch_images = batch_images[keep].to(device)
        mapped = (batch_images + delta.to(device)).clamp(0, 1)
        successes += (model(mapped).argmax(1) == int(target)).sum().item()
        examples += len(batch_images)
    return successes / max(examples, 1)


def train_projected_targeted_uap(
    clean_model,
    splits,
    config,
    *,
    pair_seed: int,
    device: str,
    output_path: str | Path,
    smoke: bool = False,
):
    """Train one fixed targeted UAP with a pre-registered epsilon ladder.

    This is a CIFAR-10 adaptation of the image-agnostic targeted objective used
    by GAP. Directly optimizing the single decoded perturbation avoids adding
    an unnecessary generator parameterization to the experiment asset.
    """
    attack_cfg = config["targeted_uap"]
    target = int(config["target_label"])
    model = clean_model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    train_images = splits.train_images
    train_labels = splits.train_labels
    validation_images = splits.validation_images
    validation_labels = splits.validation_labels
    test_images = splits.test_images
    test_labels = splits.test_labels
    epochs = int(attack_cfg["smoke_epochs"] if smoke else attack_cfg["epochs_per_epsilon"])
    if smoke:
        train_images = train_images[: int(config["data"]["smoke_train_examples"])]
        train_labels = train_labels[: len(train_images)]
        validation_images = validation_images[: int(config["data"]["smoke_validation_examples"])]
        validation_labels = validation_labels[: len(validation_images)]
        test_images = test_images[: int(config["data"]["smoke_test_examples"])]
        test_labels = test_labels[: len(test_images)]
    epsilon_candidates = [float(value) for value in attack_cfg["epsilon_candidates"]]
    if not epsilon_candidates or epsilon_candidates != sorted(epsilon_candidates):
        raise ValueError("targeted_uap.epsilon_candidates must be a non-empty ascending list")
    batch_size = int(attack_cfg["batch_size"])
    restarts = int(attack_cfg.get("restarts", 1))
    if restarts < 1:
        raise ValueError("targeted_uap.restarts must be at least one")
    minimum_asr = float(config["qualification"]["minimum_adversarial_asr"])
    best_validation_asr = -1.0
    best_delta = None
    best_epsilon = None
    history = []
    for epsilon_index, epsilon in enumerate(epsilon_candidates):
        candidate_best_asr = -1.0
        candidate_best_delta = None
        for restart in range(restarts):
            torch.manual_seed(20_000 + int(pair_seed) * 1_000 + epsilon_index * 100 + restart)
            delta = nn.Parameter(torch.empty(1, 3, 32, 32, device=device).uniform_(-epsilon, epsilon))
            optimizer = torch.optim.Adam([delta], lr=float(attack_cfg["learning_rate"]))
            patience = 0
            for epoch in range(epochs):
                for images, labels in batches(
                    train_images,
                    train_labels,
                    batch_size=batch_size,
                    seed=int(pair_seed) * 100_000 + epsilon_index * 10_000 + restart * 1_000 + epoch,
                    shuffle=True,
                ):
                    keep = labels != target
                    if not keep.any():
                        continue
                    images = images[keep].to(device)
                    targets = torch.full((len(images),), target, dtype=torch.long, device=device)
                    logits = model((images + delta).clamp(0, 1))
                    margin = targeted_margin_loss(logits, target, confidence=float(attack_cfg["confidence"]))
                    cross_entropy = nn.functional.cross_entropy(logits, targets)
                    loss = margin + float(attack_cfg["cross_entropy_weight"]) * cross_entropy
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        delta.clamp_(-epsilon, epsilon)
                validation_asr = delta_asr(
                    model,
                    delta,
                    validation_images,
                    validation_labels,
                    target,
                    batch_size=512,
                    device=device,
                )
                history.append(
                    {
                        "epsilon": epsilon,
                        "restart": restart,
                        "epoch": epoch + 1,
                        "validation_asr": validation_asr,
                        "linf": float(delta.detach().abs().max()),
                    }
                )
                print(
                    f"pair={pair_seed} mapping=projected_targeted_uap epsilon={epsilon:.6f} "
                    f"restart={restart + 1}/{restarts} epoch={epoch + 1}/{epochs} "
                    f"validation_asr={validation_asr:.3f}",
                    flush=True,
                )
                if validation_asr > candidate_best_asr:
                    candidate_best_asr = validation_asr
                    candidate_best_delta = delta.detach().cpu().clone()
                if validation_asr >= minimum_asr:
                    patience += 1
                    if patience >= int(attack_cfg["success_patience"]):
                        break
                else:
                    patience = 0
            if candidate_best_asr >= minimum_asr:
                break
        if candidate_best_asr > best_validation_asr:
            best_validation_asr = candidate_best_asr
            best_delta = copy.deepcopy(candidate_best_delta)
            best_epsilon = epsilon
        if candidate_best_asr >= minimum_asr:
            break
    if best_delta is None or best_epsilon is None:
        raise RuntimeError("Targeted UAP optimization produced no candidate")
    mapping = UniversalAdditivePerturbation(best_delta, mapping_id="projected_targeted_uap")
    test_asr = delta_asr(
        model,
        best_delta,
        test_images,
        test_labels,
        target,
        batch_size=512,
        device=device,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "protocol": config["protocol"],
            "mapping_kind": "projected_targeted_uap",
            "method_reference": "Poursaeed et al., CVPR 2018, Generative Adversarial Perturbations",
            "pair_seed": int(pair_seed),
            "target_label": target,
            "selected_epsilon": best_epsilon,
            "delta": best_delta,
            "validation_asr": best_validation_asr,
            "test_asr": test_asr,
            "config": attack_cfg,
            "history": history,
            "smoke": bool(smoke),
        },
        output_path,
    )
    metrics = {
        "method": "projected_targeted_uap",
        "validation_asr": best_validation_asr,
        "test_asr": test_asr,
        "selected_epsilon": best_epsilon,
        "linf": float(best_delta.abs().max()),
        "history": history,
        "path": str(output_path),
    }
    return mapping, metrics
