from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _bootstrap import ROOT  # noqa: F401
from feature_probe.artifacts import load_classifier
from feature_probe.cifar10 import load_cifar10
from feature_probe.config import load_config
from feature_probe.mappings import ConstantPatch
from feature_probe.models import AdversarialResidualGenerator
from feature_probe.pairs import build_pair_bundle, save_pair_bundle
from feature_probe.training import mapping_asr, train_adversarial_generator
from feature_probe.utils import atomic_write_json, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--pair-seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    seed = args.pair_seed
    splits = load_cifar10(config["data"]["root"], split_seed=config["data"]["split_seed"], download=False)
    clean_path = Path(f"artifacts/models/clean/seed{seed}/attack_result.pt")
    backdoor_path = Path(f"artifacts/models/badnets/seed{seed}/attack_result.pt")
    clean_model = load_classifier(clean_path, device=args.device)
    backdoor_model = load_classifier(backdoor_path, device=args.device)
    uap_path = Path(f"artifacts/mappings/uap/seed{seed}.pt")
    generator, mapping_metrics = train_adversarial_generator(
        clean_model,
        splits,
        config,
        pair_seed=seed,
        device=args.device,
        output_path=uap_path,
        smoke=args.smoke,
    )
    pair_count = int(config["data"]["smoke_examples"] if args.smoke else config["data"]["pilot_examples"])
    uap_bundle = build_pair_bundle(
        splits,
        generator.eval(),
        target=config["target_label"],
        count_per_split=pair_count,
        seed=config["data"]["split_seed"],
        device=args.device,
        metadata={"mapping": "uap_generator", "seed": seed, "artifact": str(uap_path)},
    )
    uap_pair_path = Path(f"artifacts/pairs/uap/seed{seed}.pt")
    save_pair_bundle(uap_pair_path, uap_bundle)
    patch_cfg = config["backdoor"]
    patch = ConstantPatch(
        mapping_id="badnets",
        top=patch_cfg["patch_top"],
        left=patch_cfg["patch_left"],
        size=patch_cfg["patch_size"],
        value=tuple(patch_cfg["patch_value"]),
    )
    patch_bundle = build_pair_bundle(
        splits,
        patch,
        target=config["target_label"],
        count_per_split=pair_count,
        seed=config["data"]["split_seed"],
        device=args.device,
        metadata={"mapping": "badnets", "seed": seed, "patch": patch_cfg},
    )
    patch_pair_path = Path(f"artifacts/pairs/badnets/seed{seed}.pt")
    save_pair_bundle(patch_pair_path, patch_bundle)
    patch_clean_asr = mapping_asr(
        clean_model, patch, splits.test_images, splits.test_labels, config["target_label"], batch_size=512, device=args.device
    )
    patch_backdoor_asr = mapping_asr(
        backdoor_model, patch, splits.test_images, splits.test_labels, config["target_label"], batch_size=512, device=args.device
    )
    uap_backdoor_asr = mapping_asr(
        backdoor_model, generator, splits.test_images, splits.test_labels, config["target_label"], batch_size=512, device=args.device
    )
    payload = {
        "pair_seed": seed,
        "smoke": args.smoke,
        "uap": {**mapping_metrics, "sha256": sha256_file(uap_path), "pair_path": str(uap_pair_path)},
        "badnets": {
            "pair_path": str(patch_pair_path),
            "clean_model_asr": patch_clean_asr,
            "backdoor_model_asr": patch_backdoor_asr,
        },
        "uap_backdoor_model_asr": uap_backdoor_asr,
        "minimum_adversarial_asr": config["qualification"]["minimum_adversarial_asr"],
    }
    payload["uap_passed"] = mapping_metrics["test_asr"] >= float(config["qualification"]["minimum_adversarial_asr"])
    atomic_write_json(f"artifacts/mappings/controls_seed{seed}.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not args.smoke and not payload["uap_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
