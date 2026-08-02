from __future__ import annotations

import copy
import time
from pathlib import Path

import torch
from torch import nn

from .cifar10 import augment_batch
from .mappings import ConstantPatch
from .models import AdversarialResidualGenerator, CifarResNet18
from .utils import atomic_write_json


def batches(images, labels, *, batch_size: int, seed: int, shuffle: bool):
    order = torch.arange(len(images))
    if shuffle:
        order = order[torch.randperm(len(order), generator=torch.Generator().manual_seed(int(seed)))]
    for start in range(0, len(order), int(batch_size)):
        selected = order[start:start + int(batch_size)]
        yield images[selected], labels[selected]


@torch.no_grad()
def classifier_metrics(model, images, labels, patch: ConstantPatch, target: int, *, batch_size: int, device: str):
    model.eval()
    correct = patch_success = examples = 0
    for batch_images, batch_labels in batches(images, labels, batch_size=batch_size, seed=0, shuffle=False):
        keep = batch_labels != int(target)
        batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
        logits = model(batch_images)
        correct += (logits.argmax(1) == batch_labels).sum().item()
        if keep.any():
            patched = patch.apply(batch_images[keep.to(device)])
            patch_success += (model(patched).argmax(1) == int(target)).sum().item()
            examples += int(keep.sum())
    return {
        "clean_accuracy": correct / len(images),
        "patch_asr": patch_success / max(examples, 1),
    }


def train_classifier_pair(splits, config, *, pair_seed: int, device: str, output_root: str | Path, smoke: bool = False):
    classifier_cfg = config["classifier"]
    backdoor_cfg = config["backdoor"]
    target = int(config["target_label"])
    patch = ConstantPatch(
        mapping_id="badnets",
        top=int(backdoor_cfg["patch_top"]),
        left=int(backdoor_cfg["patch_left"]),
        size=int(backdoor_cfg["patch_size"]),
        value=tuple(backdoor_cfg["patch_value"]),
    )
    epochs = int(classifier_cfg["smoke_epochs"] if smoke else classifier_cfg["epochs"])
    train_images = splits.train_images[: int(config["data"]["smoke_train_examples"])] if smoke else splits.train_images
    train_labels = splits.train_labels[: len(train_images)]
    validation_images = splits.validation_images[: int(config["data"]["smoke_validation_examples"])] if smoke else splits.validation_images
    validation_labels = splits.validation_labels[: len(validation_images)]
    test_images = splits.test_images[: int(config["data"]["smoke_test_examples"])] if smoke else splits.test_images
    test_labels = splits.test_labels[: len(test_images)]
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
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
        history = []
        started = time.time()
        for epoch in range(epochs):
            model.train()
            train_loss = examples = 0
            iterator = batches(
                train_images,
                train_labels,
                batch_size=int(classifier_cfg["batch_size"]),
                seed=pair_seed * 10000 + epoch,
                shuffle=True,
            )
            for step, (images, labels) in enumerate(iterator):
                images = augment_batch(images, seed=pair_seed * 1_000_000 + epoch * 10_000 + step)
                if kind == "backdoor":
                    eligible = labels != target
                    random = torch.rand(len(labels), generator=torch.Generator().manual_seed(pair_seed * 1_000_000 + epoch * 10_000 + step + 1))
                    poison = eligible & (random < float(backdoor_cfg["poison_fraction"]))
                    if poison.any():
                        images[poison] = patch.apply(images[poison])
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
        model.load_state_dict(best_state)
        metadata = {
            "protocol": config["protocol"],
            "pair_seed": int(pair_seed),
            "model_kind": kind,
            "architecture": "cifar_resnet18",
            "target_label": target,
            "patch": backdoor_cfg,
            "smoke": bool(smoke),
        }
        destination = output_root / ("clean" if kind == "clean" else "badnets") / f"seed{pair_seed}" / "attack_result.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"metadata": metadata, "model": best_state, "epoch": epochs}, destination)
        metrics = classifier_metrics(model, test_images, test_labels, patch, target, batch_size=512, device=device)
        results[kind] = {**metrics, "checkpoint": str(destination), "history": history, "elapsed_seconds": time.time() - started}
    controls = {
        "pair_seed": int(pair_seed),
        "smoke": bool(smoke),
        "metrics": results,
        "clean_accuracy_passed": results["clean"]["clean_accuracy"] >= float(config["qualification"]["minimum_clean_accuracy"]),
        "backdoor_asr_passed": results["backdoor"]["patch_asr"] >= float(config["qualification"]["minimum_backdoor_asr"]),
        "clean_patch_asr_passed": results["clean"]["patch_asr"] <= float(config["qualification"]["maximum_clean_patch_asr"]),
    }
    controls["all_passed"] = all(value for key, value in controls.items() if key.endswith("_passed"))
    atomic_write_json(output_root / f"controls_seed{pair_seed}.json", controls)
    return controls


@torch.no_grad()
def mapping_asr(model, mapping, images, labels, target, *, batch_size, device):
    successes = examples = 0
    model.eval()
    mapping.eval()
    for batch_images, batch_labels in batches(images, labels, batch_size=batch_size, seed=0, shuffle=False):
        keep = batch_labels != int(target)
        if not keep.any():
            continue
        batch_images = batch_images[keep].to(device)
        successes += (model(mapping(batch_images)).argmax(1) == int(target)).sum().item()
        examples += len(batch_images)
    return successes / max(examples, 1)


def train_adversarial_generator(clean_model, splits, config, *, pair_seed: int, device: str, output_path: str | Path, smoke: bool = False):
    generator_cfg = config["adversarial_generator"]
    target = int(config["target_label"])
    model = clean_model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.manual_seed(10_000 + int(pair_seed))
    generator = AdversarialResidualGenerator(
        width=int(generator_cfg["width"]),
        depth=int(generator_cfg["depth"]),
        epsilon=float(generator_cfg["epsilon"]),
    ).to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=float(generator_cfg["learning_rate"]))
    epochs = int(generator_cfg["smoke_epochs"] if smoke else generator_cfg["epochs"])
    train_images = splits.train_images[: int(config["data"]["smoke_train_examples"])] if smoke else splits.train_images
    train_labels = splits.train_labels[: len(train_images)]
    validation_images = splits.validation_images[: int(config["data"]["smoke_validation_examples"])] if smoke else splits.validation_images
    validation_labels = splits.validation_labels[: len(validation_images)]
    test_images = splits.test_images[: int(config["data"]["smoke_test_examples"])] if smoke else splits.test_images
    test_labels = splits.test_labels[: len(test_images)]
    history = []
    best_asr = -1.0
    best_state = None
    for epoch in range(epochs):
        generator.train()
        for images, labels in batches(train_images, train_labels, batch_size=int(generator_cfg["batch_size"]), seed=pair_seed * 1000 + epoch, shuffle=True):
            keep = labels != target
            if not keep.any():
                continue
            images = images[keep].to(device)
            targets = torch.full((len(images),), target, device=device, dtype=torch.long)
            loss = nn.functional.cross_entropy(model(generator(images)), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        validation_asr = mapping_asr(model, generator, validation_images, validation_labels, target, batch_size=512, device=device)
        history.append({"epoch": epoch + 1, "validation_asr": validation_asr})
        print(f"pair={pair_seed} mapping=adversarial_generator epoch={epoch + 1}/{epochs} validation_asr={validation_asr:.3f}", flush=True)
        if validation_asr > best_asr:
            best_asr = validation_asr
            best_state = copy.deepcopy(generator.state_dict())
    generator.load_state_dict(best_state)
    test_asr = mapping_asr(model, generator, test_images, test_labels, target, batch_size=512, device=device)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "protocol": config["protocol"],
            "mapping_kind": "adversarial_residual_generator",
            "pair_seed": int(pair_seed),
            "target_label": target,
            "config": generator_cfg,
            "state_dict": best_state,
            "validation_asr": best_asr,
            "test_asr": test_asr,
            "smoke": bool(smoke),
        },
        output_path,
    )
    return generator, {"validation_asr": best_asr, "test_asr": test_asr, "history": history, "path": str(output_path)}
