from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from _bootstrap import ROOT  # noqa: F401
from feature_probe.artifacts import load_classifier
from feature_probe.cifar10 import load_cifar10
from feature_probe.config import load_config, render_asset_path
from feature_probe.experiment import make_run_id, utc_now, write_run_manifest
from feature_probe.mappings import set_mapping_eval
from feature_probe.pairs import build_pair_bundle, save_pair_bundle
from feature_probe.training import mapping_asr, train_classifier_pair
from feature_probe.triggers import build_trigger
from feature_probe.uap import train_projected_targeted_uap
from feature_probe.utils import atomic_write_json, sha256_file


def main():
    parser = argparse.ArgumentParser(description="Train formal multitype models and known mapping pairs")
    parser.add_argument("config")
    parser.add_argument("--seed", type=int, nargs="+", default=None)
    parser.add_argument("--trigger", nargs="+", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    seeds = [int(v) for v in (args.seed or config["classifier_seeds"])]
    triggers = [str(v) for v in (args.trigger or config["trigger_ids"])]
    unknown = set(triggers) - set(str(v) for v in config["trigger_ids"])
    if unknown:
        raise ValueError(f"Triggers are not registered in config: {sorted(unknown)}")
    splits = load_cifar10(config["data"]["root"], split_seed=config["data"]["split_seed"], download=args.download)
    pair_count = int(config["data"].get("pair_examples", 1024))
    run_id = args.run_id or make_run_id("multitype_assets")
    manifest_path = Path("artifacts/multitype_models") / f"run_{run_id}.json"
    output = {"protocol": config["protocol"], "run_id": run_id, "command": " ".join(sys.argv), "started_at": utc_now(), "seeds": seeds, "triggers": triggers, "runs": []}
    write_run_manifest(manifest_path, output)
    for seed in seeds:
        # The training function writes both clean and the requested backdoor
        # model.  It is intentionally deterministic and records the recipe.
        for trigger_id in triggers:
            clean_path = render_asset_path(config["assets"]["models"]["clean"], seed=seed)
            backdoor_path = render_asset_path(config["assets"]["models"]["backdoor"], seed=seed, trigger=trigger_id)
            controls_path = Path("artifacts/multitype_models") / f"controls_{trigger_id}_seed{seed}.json"
            controls_compatible = False
            if controls_path.exists():
                existing_controls = json.loads(controls_path.read_text(encoding="utf-8"))
                if trigger_id == "inputaware":
                    controls_compatible = (
                        existing_controls.get("source") == "BackdoorBench/attack/inputaware.py"
                        and (backdoor_path.parent / "trigger_state.pt").exists()
                    )
                elif trigger_id == "ssba":
                    controls_compatible = existing_controls.get("source") == "BackdoorBench/attack/ssba.py"
                else:
                    controls_compatible = True
            if clean_path.exists() and backdoor_path.exists() and controls_path.exists() and controls_compatible:
                controls = json.loads(controls_path.read_text(encoding="utf-8"))
                print(f"seed={seed} trigger={trigger_id} status=assets_resumed", flush=True)
            else:
                controls = train_classifier_pair(
                    splits, config, pair_seed=seed, device=args.device,
                    output_root="artifacts/multitype_models", trigger_id=trigger_id, smoke=False,
                )
            clean_model = load_classifier(clean_path, device=args.device)
            backdoor_model = load_classifier(backdoor_path, device=args.device)
            uap_path = Path(config["assets"]["uap_mapping"].format(seed=seed))
            if not uap_path.exists():
                uap_mapping, uap_metrics = train_projected_targeted_uap(
                    clean_model, splits, config, pair_seed=seed, device=args.device,
                    output_path=uap_path, smoke=False,
                )
            else:
                import torch
                artifact = torch.load(uap_path, map_location="cpu", weights_only=False)
                from feature_probe.mappings import UniversalAdditivePerturbation
                uap_mapping = UniversalAdditivePerturbation(artifact["delta"], mapping_id="projected_targeted_uap")
                uap_metrics = {"test_asr": float(artifact.get("test_asr", 0.0)), "validation_asr": float(artifact.get("validation_asr", 0.0))}
            uap_pair_path = render_asset_path(config["assets"]["pairs"]["uap"], seed=seed)
            uap_bundle = build_pair_bundle(
                splits, set_mapping_eval(uap_mapping), target=config["target_label"],
                count_per_split=pair_count, seed=config["data"]["split_seed"], device=args.device,
                metadata={"mapping": "projected_targeted_uap", "source_model": "clean", "pair_seed": seed,
                          "source_asr": uap_metrics.get("test_asr"), "artifact": str(uap_path),
                          "artifact_sha256": sha256_file(uap_path)},
            )
            save_pair_bundle(uap_pair_path, uap_bundle)
            trigger_checkpoint = None
            if trigger_id == "inputaware":
                trigger_checkpoint = str(backdoor_path.parent / "trigger_state.pt")
            trigger = build_trigger(
                trigger_id,
                config,
                target=config["target_label"],
                checkpoint_path=trigger_checkpoint,
                device=args.device,
            )
            trigger_pair_path = render_asset_path(config["assets"]["pairs"]["trigger"], seed=seed, trigger=trigger_id)
            trigger_asr = mapping_asr(
                backdoor_model, trigger, splits.test_images, splits.test_labels,
                config["target_label"], batch_size=512, device=args.device,
                indices=splits.test_indices, split="test",
            )
            clean_trigger_asr = mapping_asr(
                clean_model, trigger, splits.test_images, splits.test_labels,
                config["target_label"], batch_size=512, device=args.device,
                indices=splits.test_indices, split="test",
            )
            trigger_bundle = build_pair_bundle(
                splits, trigger, target=config["target_label"], count_per_split=pair_count,
                seed=config["data"]["split_seed"], device=args.device,
                metadata={"mapping": trigger_id, "source_model": "backdoor", "pair_seed": seed,
                          "source_asr": trigger_asr, "clean_control_asr": clean_trigger_asr,
                          "trigger_recipe": config.get("triggers", {}).get(trigger_id, {})},
            )
            save_pair_bundle(trigger_pair_path, trigger_bundle)
            run = {
                "seed": seed, "trigger": trigger_id, "controls": controls,
                "uap": {"path": str(uap_pair_path), "mapping_path": str(uap_path), "asr": uap_metrics},
                "trigger": {"path": str(trigger_pair_path), "backdoor_asr": trigger_asr, "clean_asr": clean_trigger_asr},
                "qualified": bool(
                    controls.get("all_passed", False)
                    and trigger_asr >= float(config["qualification"]["minimum_backdoor_asr"])
                    and clean_trigger_asr <= float(config["qualification"].get("maximum_clean_patch_asr", 0.10))
                ),
            }
            output["runs"].append(run)
            write_run_manifest(manifest_path, output)
            print(json.dumps(run, indent=2, ensure_ascii=False), flush=True)
    output["finished_at"] = utc_now()
    write_run_manifest(manifest_path, output)
    atomic_write_json("artifacts/multitype_models/manifest.json", output)
    print(json.dumps({"status": "completed", "runs": len(output["runs"]), "manifest": "artifacts/multitype_models/manifest.json"}, indent=2))


if __name__ == "__main__":
    main()
