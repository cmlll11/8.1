from __future__ import annotations

import copy
import time
from pathlib import Path

import torch
from torch import nn

from .cifar10 import augment_batch, select_poison_indices
from .mappings import apply_mapping, set_mapping_eval
from .models import CifarResNet18
from .triggers import build_trigger
from .utils import atomic_write_json


def batches(images, labels, *, batch_size: int, seed: int, shuffle: bool):
    order = torch.arange(len(images))
    if shuffle:
        order = order[torch.randperm(len(order), generator=torch.Generator().manual_seed(int(seed)))]
    for start in range(0, len(order), int(batch_size)):
        selected = order[start:start + int(batch_size)]
        yield images[selected], labels[selected]


@torch.no_grad()
def classifier_metrics(model, images, labels, trigger, target: int, *, batch_size: int, device: str):
    model.eval()
    correct = patch_success = examples = 0
    for batch_images, batch_labels in batches(images, labels, batch_size=batch_size, seed=0, shuffle=False):
        keep = batch_labels != int(target)
        batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
        logits = model(batch_images)
        correct += (logits.argmax(1) == batch_labels).sum().item()
        if keep.any():
            patched = apply_mapping(trigger, batch_images[keep.to(device)])
            patch_success += (model(patched).argmax(1) == int(target)).sum().item()
            examples += int(keep.sum())
    return {
        "clean_accuracy": correct / len(images),
        "patch_asr": patch_success / max(examples, 1),
    }


def train_classifier_pair(
    splits,
    config,
    *,
    pair_seed: int,
    device: str,
    output_root: str | Path,
    smoke: bool = False,
    trigger_id: str | None = None,
):
    classifier_cfg = config["classifier"]
    backdoor_cfg = config.get("backdoor", {})
    target = int(config["target_label"])
    trigger_id = str(trigger_id or backdoor_cfg.get("trigger_id", "badnets"))
    trigger = build_trigger(trigger_id, config, target=target)
    epochs = int(classifier_cfg["smoke_epochs"] if smoke else classifier_cfg["epochs"])
    train_images = splits.train_images[: int(config["data"]["smoke_train_examples"])] if smoke else splits.train_images
    train_labels = splits.train_labels[: len(train_images)]
    validation_images = splits.validation_images[: int(config["data"]["smoke_validation_examples"])] if smoke else splits.validation_images
    validation_labels = splits.validation_labels[: len(validation_images)]
    test_images = splits.test_images[: int(config["data"]["smoke_test_examples"])] if smoke else splits.test_images
    test_labels = splits.test_labels[: len(test_images)]
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    poison_indices = select_poison_indices(
        train_labels,
        target=target,
        fraction=float(backdoor_cfg["poison_fraction"]),
        seed=int(config["data"]["split_seed"]),
    )
    poison_mask = torch.zeros(len(train_labels), dtype=torch.bool)
    poison_mask[poison_indices] = True
    results = {}
    for kind in ("clean", "backdoor"):
        torch.manual_seed(int(pair_seed))
        model = CifarResNet18().to(device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(classifier_cfg["learning_rate"]),
            momentum=float(classifier_cfg["momentum"]),
            weight_decay=float(classifier_cfg["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
        scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda") and classifier_cfg.get("amp", True))
        best_loss = float("inf")
        best_state = None
        best_epoch = None
        history = []
        started = time.time()
        for epoch in range(epochs):
            model.train()
            train_loss = examples = 0
            order = torch.randperm(
                len(train_images), generator=torch.Generator().manual_seed(pair_seed * 10000 + epoch)
            )
            batch_size = int(classifier_cfg["batch_size"])
            for step, start in enumerate(range(0, len(order), batch_size)):
                selected = order[start:start + batch_size]
                images, labels = train_images[selected], train_labels[selected]
                images = augment_batch(images, seed=pair_seed * 1_000_000 + epoch * 10_000 + step)
                if kind == "backdoor":
                    poison = poison_mask[selected]
                    if poison.any():
                        images = images.clone()
                        images[poison] = apply_mapping(trigger, images[poison])
                        labels = labels.clone()
                        labels[poison] = target
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                    loss = nn.functional.cross_entropy(model(images), labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.detach().item() * len(images)
                examples += len(images)
            scheduler.step()
            model.eval()
            validation_loss = validation_examples = 0
            with torch.no_grad():
                for images, labels in batches(validation_images, validation_labels, batch_size=512, seed=0, shuffle=False):
                    images, labels = images.to(device), labels.to(device)
                    loss = nn.functional.cross_entropy(model(images), labels)
                    validation_loss += loss.item() * len(images)
                    validation_examples += len(images)
            validation_loss /= validation_examples
            history.append({"epoch": epoch + 1, "train_loss": train_loss / examples, "validation_loss": validation_loss})
            print(f"pair={pair_seed} model={kind} epoch={epoch + 1}/{epochs} validation_loss={validation_loss:.5f}", flush=True)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
        if best_state is None or best_epoch is None:
            raise RuntimeError(f"No finite checkpoint was produced for {kind}")
        model.load_state_dict(best_state)
        metadata = {
            "protocol": config["protocol"],
            "pair_seed": int(pair_seed),
            "model_kind": kind,
            "architecture": "cifar_resnet18",
            "target_label": target,
            "trigger_id": trigger_id,
            "trigger": config.get("triggers", {}).get(trigger_id, backdoor_cfg),
            "smoke": bool(smoke),
        }
        destination = output_root / ("clean" if kind == "clean" else trigger_id) / f"seed{pair_seed}" / "attack_result.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"metadata": metadata, "model": best_state, "epoch": best_epoch}, destination)
        metrics = classifier_metrics(model, test_images, test_labels, trigger, target, batch_size=512, device=device)
        results[kind] = {**metrics, "checkpoint": str(destination), "history": history, "elapsed_seconds": time.time() - started}
    controls = {
        "pair_seed": int(pair_seed),
        "smoke": bool(smoke),
        "metrics": results,
        "poisoned_examples": int(len(poison_indices)),
        "poison_indices": splits.train_indices[: len(train_images)][poison_indices].tolist(),
        "clean_accuracy_passed": all(
            metrics["clean_accuracy"] >= float(config["qualification"]["minimum_clean_accuracy"])
            for metrics in results.values()
        ),
        "backdoor_asr_passed": results["backdoor"]["patch_asr"] >= float(config["qualification"]["minimum_backdoor_asr"]),
        "clean_trigger_asr_passed": results["clean"]["patch_asr"] <= float(config["qualification"].get("maximum_clean_patch_asr", config["qualification"].get("maximum_clean_trigger_asr", 0.10))),
        "clean_patch_asr_passed": results["clean"]["patch_asr"] <= float(config["qualification"].get("maximum_clean_patch_asr", config["qualification"].get("maximum_clean_trigger_asr", 0.10))),
    }
    controls["all_passed"] = all(value for key, value in controls.items() if key.endswith("_passed"))
    atomic_write_json(output_root / f"controls_seed{pair_seed}.json", controls)
    return controls


@torch.no_grad()
def mapping_asr(model, mapping, images, labels, target, *, batch_size, device):
    successes = examples = 0
    model.eval()
    set_mapping_eval(mapping)
    for batch_images, batch_labels in batches(images, labels, batch_size=batch_size, seed=0, shuffle=False):
        keep = batch_labels != int(target)
        if not keep.any():
            continue
        batch_images = batch_images[keep].to(device)
        mapped = apply_mapping(mapping, batch_images)
        successes += (model(mapped).argmax(1) == int(target)).sum().item()
        examples += len(batch_images)
    return successes / max(examples, 1)
