from __future__ import annotations

import copy
import time
from pathlib import Path

import torch
from torch import nn

from .cifar10 import augment_batch, select_poison_indices
from .backdoorbench_attacks import build_inputaware_modules, OfficialInputAwareTrigger
from .mappings import apply_mapping, set_mapping_eval
from .models import CifarResNet18
from .triggers import build_trigger
from .utils import atomic_torch_save, atomic_write_json


def batches(images, labels, *, batch_size: int, seed: int, shuffle: bool):
    order = torch.arange(len(images))
    if shuffle:
        order = order[torch.randperm(len(order), generator=torch.Generator().manual_seed(int(seed)))]
    for start in range(0, len(order), int(batch_size)):
        selected = order[start:start + int(batch_size)]
        yield images[selected], labels[selected]


@torch.no_grad()
def classifier_metrics(model, images, labels, trigger, target: int, *, batch_size: int, device: str, indices=None, split="test"):
    model.eval()
    correct = patch_success = examples = 0
    offset = 0
    for batch_images, batch_labels in batches(images, labels, batch_size=batch_size, seed=0, shuffle=False):
        keep = batch_labels != int(target)
        batch_indices = None if indices is None else indices[offset:offset + len(batch_images)]
        batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
        logits = model(batch_images)
        correct += (logits.argmax(1) == batch_labels).sum().item()
        if keep.any():
            patched = apply_mapping(
                trigger,
                batch_images[keep.to(device)],
                indices=None if batch_indices is None else batch_indices[keep],
                split=split,
            )
            patch_success += (model(patched).argmax(1) == int(target)).sum().item()
            examples += int(keep.sum())
        offset += len(batch_images)
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
    if trigger_id == "inputaware":
        return _train_inputaware_classifier_pair(
            splits, config, pair_seed=pair_seed, device=device, output_root=output_root, smoke=smoke
        )
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
    checkpoint_root = output_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    poison_indices = select_poison_indices(
        train_labels,
        target=target,
        fraction=float(backdoor_cfg["poison_fraction"]),
        seed=int(config["data"]["split_seed"]),
    )
    poison_mask = torch.zeros(len(train_labels), dtype=torch.bool)
    poison_mask[poison_indices] = True
    results = {}
    test_indices = getattr(splits, "test_indices", torch.arange(len(test_images)))
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
        start_epoch = 0
        latest_path = checkpoint_root / f"{trigger_id}_seed{pair_seed}_{kind}_latest.pt"
        if latest_path.exists():
            checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
            if checkpoint.get("trigger_id") == trigger_id and checkpoint.get("pair_seed") == int(pair_seed) and checkpoint.get("kind") == kind:
                model.load_state_dict(checkpoint["model_state"])
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                scheduler.load_state_dict(checkpoint["scheduler_state"])
                if checkpoint.get("scaler_state"):
                    scaler.load_state_dict(checkpoint["scaler_state"])
                best_state = checkpoint.get("best_state")
                best_loss = float(checkpoint.get("best_loss", best_loss))
                best_epoch = checkpoint.get("best_epoch")
                history = checkpoint.get("history", [])
                start_epoch = int(checkpoint.get("epoch", 0))
                print(f"pair={pair_seed} model={kind} status=checkpoint_resumed epoch={start_epoch}/{epochs}", flush=True)
        started = time.time()
        for epoch in range(start_epoch, epochs):
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
                        images[poison] = apply_mapping(
                            trigger,
                            images[poison],
                            indices=splits.train_indices[selected[poison]],
                            split="train",
                        )
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
            atomic_torch_save(
                latest_path,
                {
                    "protocol": config["protocol"], "trigger_id": trigger_id,
                    "pair_seed": int(pair_seed), "kind": kind, "epoch": epoch + 1,
                    "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(),
                    "best_state": best_state, "best_loss": best_loss, "best_epoch": best_epoch,
                    "history": history,
                },
            )
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
        metrics = classifier_metrics(
            model, test_images, test_labels, trigger, target, batch_size=512, device=device,
                indices=test_indices[:len(test_images)], split="test",
        )
        results[kind] = {**metrics, "checkpoint": str(destination), "history": history, "elapsed_seconds": time.time() - started}
    controls = {
        "pair_seed": int(pair_seed),
        "smoke": bool(smoke),
        "source": "BackdoorBench/attack/ssba.py" if trigger_id == "ssba" else "project_fixed_trigger_adapter",
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
    atomic_write_json(output_root / f"controls_{trigger_id}_seed{pair_seed}.json", controls)
    if trigger_id == "badnets":
        atomic_write_json(output_root / f"controls_seed{pair_seed}.json", controls)
    return controls


@torch.no_grad()
def mapping_asr(model, mapping, images, labels, target, *, batch_size, device, indices=None, split="test"):
    successes = examples = 0
    model.eval()
    set_mapping_eval(mapping)
    offset = 0
    for batch_images, batch_labels in batches(images, labels, batch_size=batch_size, seed=0, shuffle=False):
        batch_len = len(batch_images)
        batch_indices = None if indices is None else indices[offset:offset + batch_len]
        keep = batch_labels != int(target)
        if not keep.any():
            offset += batch_len
            continue
        batch_images = batch_images[keep].to(device)
        batch_indices = None if batch_indices is None else batch_indices[keep]
        mapped = apply_mapping(
            mapping,
            batch_images,
            indices=batch_indices,
            split=split,
        )
        successes += (model(mapped).argmax(1) == int(target)).sum().item()
        examples += len(batch_images)
        offset += batch_len
    return successes / max(examples, 1)


def _inputaware_validation(model, trigger, images, labels, target, device):
    model.eval()
    trigger.eval()
    with torch.no_grad():
        logits = model(images.to(device))
        clean_accuracy = float((logits.argmax(1).cpu() == labels).float().mean())
        keep = labels != int(target)
        if not keep.any():
            return clean_accuracy, 0.0
        patched = trigger.apply(images[keep].to(device))
        asr = float((model(patched).argmax(1) == int(target)).float().mean())
    return clean_accuracy, asr


def _train_inputaware_classifier_pair(splits, config, *, pair_seed, device, output_root, smoke=False):
    """Train Input-Aware using BackdoorBench's joint C/G/M objective.

    The generator architecture, mask threshold, diversity loss and mask
    density loss are from BackdoorBench ``attack/inputaware.py``.  Only the
    dataset iteration and checkpoint layout are adapted to this project.
    """
    classifier_cfg = config["classifier"]
    attack_cfg = config.get("triggers", {}).get("inputaware", {})
    backdoor_cfg = config.get("backdoor", {})
    target = int(config["target_label"])
    epochs = int(classifier_cfg.get("smoke_epochs", 1) if smoke else classifier_cfg["epochs"])
    mask_epochs = int(attack_cfg.get("mask_epochs", 25 if not smoke else 1))
    train_images = splits.train_images[: int(config["data"].get("smoke_train_examples", len(splits.train_images)))] if smoke else splits.train_images
    train_labels = splits.train_labels[:len(train_images)]
    validation_images = splits.validation_images[: int(config["data"].get("smoke_validation_examples", len(splits.validation_images)))] if smoke else splits.validation_images
    validation_labels = splits.validation_labels[:len(validation_images)]
    test_images = splits.test_images[: int(config["data"].get("smoke_test_examples", len(splits.test_images)))] if smoke else splits.test_images
    test_labels = splits.test_labels[:len(test_images)]
    output_root = Path(output_root)
    checkpoint_root = output_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    poison_indices = select_poison_indices(
        train_labels,
        target=target,
        fraction=float(backdoor_cfg["poison_fraction"]),
        seed=int(config["data"]["split_seed"]),
    )

    def train_clean():
        torch.manual_seed(int(pair_seed))
        model = CifarResNet18().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=float(classifier_cfg["learning_rate"]),
                                    momentum=float(classifier_cfg["momentum"]),
                                    weight_decay=float(classifier_cfg["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
        scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda") and classifier_cfg.get("amp", True))
        latest = checkpoint_root / f"inputaware_seed{pair_seed}_clean_latest.pt"
        best_state, best_loss, best_epoch, history, start = None, float("inf"), None, [], 0
        if latest.exists():
            ckpt = torch.load(latest, map_location=device, weights_only=False)
            if ckpt.get("pair_seed") == int(pair_seed) and ckpt.get("kind") == "clean":
                model.load_state_dict(ckpt["model_state"]); optimizer.load_state_dict(ckpt["optimizer_state"])
                scheduler.load_state_dict(ckpt["scheduler_state"]); best_state = ckpt.get("best_state")
                best_loss = float(ckpt.get("best_loss", best_loss)); best_epoch = ckpt.get("best_epoch")
                history = ckpt.get("history", []); start = int(ckpt.get("epoch", 0))
                print(f"pair={pair_seed} model=clean status=checkpoint_resumed epoch={start}/{epochs}", flush=True)
        for epoch in range(start, epochs):
            model.train(); order = torch.randperm(len(train_images), generator=torch.Generator().manual_seed(pair_seed * 10000 + epoch))
            total = 0.0
            for step, begin in enumerate(range(0, len(order), int(classifier_cfg["batch_size"]))):
                selected = order[begin:begin + int(classifier_cfg["batch_size"])]
                images = augment_batch(train_images[selected], seed=pair_seed * 1_000_000 + epoch * 10_000 + step).to(device)
                labels = train_labels[selected].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                    loss = nn.functional.cross_entropy(model(images), labels)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); total += float(loss.detach()) * len(images)
            scheduler.step()
            model.eval(); val_loss = 0.0; count = 0
            with torch.no_grad():
                for images, labels in batches(validation_images, validation_labels, batch_size=512, seed=0, shuffle=False):
                    val_loss += float(nn.functional.cross_entropy(model(images.to(device)), labels.to(device))) * len(images); count += len(images)
            val_loss /= max(count, 1); history.append({"epoch": epoch + 1, "train_loss": total / len(train_images), "validation_loss": val_loss})
            print(f"pair={pair_seed} model=clean epoch={epoch + 1}/{epochs} validation_loss={val_loss:.5f}", flush=True)
            if val_loss < best_loss:
                best_loss, best_state, best_epoch = val_loss, copy.deepcopy(model.state_dict()), epoch + 1
            atomic_torch_save(latest, {"pair_seed": pair_seed, "kind": "clean", "epoch": epoch + 1, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(), "best_state": best_state, "best_loss": best_loss, "best_epoch": best_epoch, "history": history})
        model.load_state_dict(best_state)
        destination = output_root / "clean" / f"seed{pair_seed}" / "attack_result.pt"; destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"metadata": {"protocol": config["protocol"], "pair_seed": pair_seed, "model_kind": "clean", "trigger_id": "inputaware", "source": "BackdoorBench/inputaware.py"}, "model": best_state, "epoch": best_epoch}, destination)
        return model, history, destination

    clean_model, clean_history, clean_path = train_clean()
    torch.manual_seed(int(pair_seed))
    model = CifarResNet18().to(device)
    modules = build_inputaware_modules(device)
    opt_c = torch.optim.SGD(model.parameters(), lr=float(classifier_cfg["learning_rate"]), momentum=float(classifier_cfg["momentum"]), weight_decay=float(classifier_cfg["weight_decay"]))
    opt_g = torch.optim.Adam(modules.generator.parameters(), lr=float(attack_cfg.get("lr_G", 1e-2)), betas=(0.5, 0.9))
    opt_m = torch.optim.Adam(modules.mask.parameters(), lr=float(attack_cfg.get("lr_M", 1e-2)), betas=(0.5, 0.9))
    scheduler_c = torch.optim.lr_scheduler.CosineAnnealingLR(opt_c, T_max=max(epochs, 1))
    scheduler_g = torch.optim.lr_scheduler.MultiStepLR(opt_g, milestones=list(attack_cfg.get("schedulerG_milestones", [200, 300, 400, 500])), gamma=float(attack_cfg.get("schedulerG_lambda", 0.1)))
    scheduler_m = torch.optim.lr_scheduler.MultiStepLR(opt_m, milestones=list(attack_cfg.get("schedulerM_milestones", [10, 20])), gamma=float(attack_cfg.get("schedulerM_lambda", 0.1)))
    threshold = modules.threshold
    lambda_div = float(attack_cfg.get("lambda_div", 1.0)); lambda_norm = float(attack_cfg.get("lambda_norm", 100.0)); mask_density = float(attack_cfg.get("mask_density", 0.032)); eps = 1e-7
    latest = checkpoint_root / f"inputaware_seed{pair_seed}_backdoor_latest.pt"
    history, best_state, best_trigger, best_asr, start = [], None, None, -1.0, 0
    if latest.exists():
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        if ckpt.get("pair_seed") == int(pair_seed):
            model.load_state_dict(ckpt["model_state"]); modules.generator.load_state_dict(ckpt["generator"]); modules.mask.load_state_dict(ckpt["mask"])
            opt_c.load_state_dict(ckpt["optimizer_c"]); opt_g.load_state_dict(ckpt["optimizer_g"]); opt_m.load_state_dict(ckpt["optimizer_m"])
            scheduler_c.load_state_dict(ckpt["scheduler_c"]); scheduler_g.load_state_dict(ckpt["scheduler_g"]); scheduler_m.load_state_dict(ckpt["scheduler_m"])
            history = ckpt.get("history", []); best_state = ckpt.get("best_state"); best_trigger = ckpt.get("best_trigger"); best_asr = float(ckpt.get("best_asr", best_asr)); start = int(ckpt.get("epoch", 0))
            print(f"pair={pair_seed} model=backdoor status=checkpoint_resumed epoch={start}/{epochs}", flush=True)
    for epoch in range(start, epochs):
        model.train(); modules.generator.train(); modules.mask.eval()
        order1 = torch.randperm(len(train_images), generator=torch.Generator().manual_seed(pair_seed * 20000 + epoch))
        order2 = torch.randperm(len(train_images), generator=torch.Generator().manual_seed(pair_seed * 30000 + epoch))
        train_loss = 0.0; seen = 0; bs = int(classifier_cfg["batch_size"])
        for begin in range(0, len(order1), bs):
            s1, s2 = order1[begin:begin + bs], order2[begin:begin + bs]
            x1 = augment_batch(train_images[s1], seed=pair_seed * 1_000_000 + epoch * 10_000 + begin // bs).to(device)
            x2 = augment_batch(train_images[s2], seed=pair_seed * 1_000_000 + epoch * 10_000 + 5000 + begin // bs).to(device)
            y1 = train_labels[s1].to(device); nbd = min(max(1, int(round(float(backdoor_cfg["poison_fraction"]) * len(x1)))), len(x1) // 2)
            if nbd == 0: continue
            num_cross = nbd
            patterns1 = modules.generator(x1[:nbd]); masks1 = threshold(modules.mask(x1[:nbd])); bd_inputs = x1[:nbd] + (patterns1 - x1[:nbd]) * masks1
            patterns2 = modules.generator(x2[nbd:nbd + num_cross]); masks2 = threshold(modules.mask(x2[nbd:nbd + num_cross])); cross = x1[nbd:nbd + num_cross] + (patterns2 - x1[nbd:nbd + num_cross]) * masks2
            total_inputs = torch.cat((bd_inputs, cross, x1[nbd + num_cross:]), 0); total_targets = torch.cat((torch.full((nbd,), target, device=device, dtype=torch.long), y1[nbd:]), 0)
            opt_c.zero_grad(set_to_none=True); opt_g.zero_grad(set_to_none=True)
            logits = model(total_inputs); loss_ce = nn.functional.cross_entropy(logits, total_targets)
            dist_x = torch.sqrt(F.mse_loss(x1[:nbd], x2[nbd:nbd + num_cross], reduction="none").mean((1, 2, 3)))
            dist_p = torch.sqrt(F.mse_loss(patterns1, patterns2, reduction="none").mean((1, 2, 3)))
            loss = loss_ce + lambda_div * torch.mean(dist_x / (dist_p + eps)); loss.backward(); opt_c.step(); opt_g.step(); train_loss += float(loss.detach()) * len(total_inputs); seen += len(total_inputs)
        scheduler_c.step(); scheduler_g.step()
        modules.mask.train(); opt_m.zero_grad(set_to_none=True)
        x1 = train_images[order1[:bs]].to(device); x2 = train_images[order2[:bs]].to(device)
        masks1, masks2 = threshold(modules.mask(x1)), threshold(modules.mask(x2))
        dist_x = torch.sqrt(F.mse_loss(x1, x2, reduction="none").mean((1, 2, 3))); dist_m = torch.sqrt(F.mse_loss(masks1, masks2, reduction="none").mean((1, 2, 3)))
        mask_loss_div = lambda_div * torch.mean(dist_x / (dist_m + eps)); mask_loss_norm = lambda_norm * F.relu(masks1 - mask_density).mean(); (mask_loss_div + mask_loss_norm).backward(); opt_m.step(); scheduler_m.step(); modules.mask.eval()
        trigger = OfficialInputAwareTrigger(modules.generator, modules.mask, threshold)
        val_acc, val_asr = _inputaware_validation(model, trigger, validation_images, validation_labels, target, device)
        history.append({"epoch": epoch + 1, "train_loss": train_loss / max(seen, 1), "validation_clean_accuracy": val_acc, "validation_asr": val_asr})
        print(f"pair={pair_seed} model=backdoor epoch={epoch + 1}/{epochs} validation_asr={val_asr:.5f}", flush=True)
        if val_asr > best_asr:
            best_asr = val_asr; best_state = copy.deepcopy(model.state_dict()); best_trigger = {"generator": copy.deepcopy(modules.generator.state_dict()), "mask": copy.deepcopy(modules.mask.state_dict())}
        atomic_torch_save(latest, {"pair_seed": pair_seed, "epoch": epoch + 1, "model_state": model.state_dict(), "generator": modules.generator.state_dict(), "mask": modules.mask.state_dict(), "optimizer_c": opt_c.state_dict(), "optimizer_g": opt_g.state_dict(), "optimizer_m": opt_m.state_dict(), "scheduler_c": scheduler_c.state_dict(), "scheduler_g": scheduler_g.state_dict(), "scheduler_m": scheduler_m.state_dict(), "history": history, "best_state": best_state, "best_trigger": best_trigger, "best_asr": best_asr})
    model.load_state_dict(best_state); modules.generator.load_state_dict(best_trigger["generator"]); modules.mask.load_state_dict(best_trigger["mask"]); trigger = OfficialInputAwareTrigger(modules.generator, modules.mask, threshold)
    destination = output_root / "inputaware" / f"seed{pair_seed}" / "attack_result.pt"; destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": {"protocol": config["protocol"], "pair_seed": pair_seed, "model_kind": "backdoor", "trigger_id": "inputaware", "source": "BackdoorBench/inputaware.py"}, "model": best_state, "epoch": max(range(1, len(history) + 1), key=lambda e: history[e - 1]["validation_asr"])}, destination)
    trigger_state = destination.parent / "trigger_state.pt"; atomic_torch_save(trigger_state, best_trigger)
    clean_metrics = classifier_metrics(clean_model, test_images, test_labels, trigger, target, batch_size=512, device=device, indices=splits.test_indices[:len(test_images)], split="test")
    backdoor_metrics = classifier_metrics(model, test_images, test_labels, trigger, target, batch_size=512, device=device, indices=splits.test_indices[:len(test_images)], split="test")
    controls = {"pair_seed": int(pair_seed), "smoke": bool(smoke), "metrics": {"clean": clean_metrics, "backdoor": backdoor_metrics}, "poisoned_examples": int(len(poison_indices)), "source": "BackdoorBench/attack/inputaware.py", "clean_accuracy_passed": clean_metrics["clean_accuracy"] >= float(config["qualification"]["minimum_clean_accuracy"]) and backdoor_metrics["clean_accuracy"] >= float(config["qualification"]["minimum_clean_accuracy"]), "backdoor_asr_passed": backdoor_metrics["patch_asr"] >= float(config["qualification"]["minimum_backdoor_asr"]), "clean_trigger_asr_passed": clean_metrics["patch_asr"] <= float(config["qualification"].get("maximum_clean_patch_asr", 0.10)), "clean_patch_asr_passed": clean_metrics["patch_asr"] <= float(config["qualification"].get("maximum_clean_patch_asr", 0.10))}
    controls["all_passed"] = all(v for k, v in controls.items() if k.endswith("_passed")); atomic_write_json(output_root / f"controls_inputaware_seed{pair_seed}.json", controls)
    return controls
