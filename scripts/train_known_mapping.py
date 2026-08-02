from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
from feature_probe.artifacts import load_classifier
from feature_probe.cifar10 import load_cifar10
from feature_probe.config import load_config, render_asset_path
from feature_probe.mappings import ConstantPatch, set_mapping_eval
from feature_probe.pairs import build_pair_bundle, save_pair_bundle
from feature_probe.training import mapping_asr
from feature_probe.uap import train_projected_targeted_uap
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
    if seed not in [int(value) for value in config["classifier_seeds"]]:
        raise ValueError(f"pair seed {seed} is not registered in classifier_seeds")
    trigger = config["backdoor"]["trigger_id"]
    splits = load_cifar10(config["data"]["root"], split_seed=config["data"]["split_seed"], download=False)
    clean_path = render_asset_path(config["assets"]["models"]["clean"], seed=seed)
    backdoor_path = render_asset_path(config["assets"]["models"]["backdoor"], seed=seed, trigger=trigger)
    clean_model = load_classifier(clean_path, device=args.device)
    backdoor_model = load_classifier(backdoor_path, device=args.device)
    uap_path = Path(f"artifacts/mappings/uap/seed{seed}.pt")
    adversarial_mapping, mapping_metrics = train_projected_targeted_uap(
        clean_model,
        splits,
        config,
        pair_seed=seed,
        device=args.device,
        output_path=uap_path,
        smoke=args.smoke,
    )
    patch_cfg = config["backdoor"]
    patch = ConstantPatch(
        mapping_id=trigger,
        top=patch_cfg["patch_top"],
        left=patch_cfg["patch_left"],
        size=patch_cfg["patch_size"],
        value=tuple(patch_cfg["patch_value"]),
    )
    patch_clean_asr = mapping_asr(
        clean_model, patch, splits.test_images, splits.test_labels, config["target_label"], batch_size=512, device=args.device
    )
    patch_backdoor_asr = mapping_asr(
        backdoor_model, patch, splits.test_images, splits.test_labels, config["target_label"], batch_size=512, device=args.device
    )
    uap_backdoor_asr = mapping_asr(
        backdoor_model, adversarial_mapping, splits.test_images, splits.test_labels, config["target_label"], batch_size=512, device=args.device
    )
    pair_count = int(config["data"]["smoke_examples"] if args.smoke else config["data"]["pilot_examples"])
    uap_pair_path = render_asset_path(config["assets"]["pairs"]["uap"], seed=seed)
    uap_bundle = build_pair_bundle(
        splits,
        set_mapping_eval(adversarial_mapping),
        target=config["target_label"],
        count_per_split=pair_count,
        seed=config["data"]["split_seed"],
        device=args.device,
        metadata={
            "mapping": mapping_metrics["method"],
            "seed": seed,
            "target_label": int(config["target_label"]),
            "source_model": "clean",
            "source_model_sha256": sha256_file(clean_path),
            "source_asr": mapping_metrics["test_asr"],
            "artifact": str(uap_path),
            "artifact_sha256": sha256_file(uap_path),
            "scientific_result": not args.smoke,
        },
    )
    save_pair_bundle(uap_pair_path, uap_bundle)
    patch_pair_path = render_asset_path(config["assets"]["pairs"]["trigger"], seed=seed, trigger=trigger)
    patch_bundle = build_pair_bundle(
        splits,
        patch,
        target=config["target_label"],
        count_per_split=pair_count,
        seed=config["data"]["split_seed"],
        device=args.device,
        metadata={
            "mapping": trigger,
            "seed": seed,
            "target_label": int(config["target_label"]),
            "source_model": "backdoor",
            "source_model_sha256": sha256_file(backdoor_path),
            "source_asr": patch_backdoor_asr,
            "patch": patch_cfg,
            "scientific_result": not args.smoke,
        },
    )
    save_pair_bundle(patch_pair_path, patch_bundle)
    minimum_source_asr = float(config["observation"]["minimum_source_asr"])
    maximum_source_asr_gap = float(config["observation"]["maximum_source_asr_gap"])
    payload = {
        "pair_seed": seed,
        "smoke": args.smoke,
        "uap": {**mapping_metrics, "sha256": sha256_file(uap_path), "pair_path": str(uap_pair_path)},
        trigger: {
            "pair_path": str(patch_pair_path),
            "clean_model_asr": patch_clean_asr,
            "backdoor_model_asr": patch_backdoor_asr,
        },
        "uap_backdoor_model_asr": uap_backdoor_asr,
        "minimum_source_asr": minimum_source_asr,
        "maximum_source_asr_gap": maximum_source_asr_gap,
    }
    payload["uap_passed"] = mapping_metrics["test_asr"] >= float(config["qualification"]["minimum_adversarial_asr"])
    payload["trigger_passed"] = patch_backdoor_asr >= minimum_source_asr
    payload["source_asr_gap_passed"] = abs(mapping_metrics["test_asr"] - patch_backdoor_asr) <= maximum_source_asr_gap
    payload["clean_trigger_control_passed"] = patch_clean_asr <= float(config["qualification"]["maximum_clean_patch_asr"])
    payload["all_passed"] = all(value for key, value in payload.items() if key.endswith("_passed"))
    atomic_write_json(f"artifacts/mappings/controls_seed{seed}.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not args.smoke and not payload["all_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
