from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from _bootstrap import ROOT  # noqa: F401
from feature_probe.artifacts import load_classifier, load_pair_bundle
from feature_probe.config import load_config, render_asset_path
from feature_probe.forward import (
    assert_pair_alignment,
    bundle_asr,
    change_maps,
    compare_feature_changes,
    extract_feature_pairs_by_layer,
    subset_pair_bundle_by_split,
)
from feature_probe.metrics import nearest_centroid_auc
from feature_probe.utils import atomic_write_json, environment_record, sha256_file


def bootstrap_auc(train_trigger, train_uap, test_trigger, test_uap, samples: int, seed: int):
    generator = torch.Generator().manual_seed(int(seed))
    values = []
    for _ in range(int(samples)):
        ti = torch.randint(len(train_trigger), (len(train_trigger),), generator=generator)
        ui = torch.randint(len(train_uap), (len(train_uap),), generator=generator)
        tv = torch.randint(len(test_trigger), (len(test_trigger),), generator=generator)
        uv = torch.randint(len(test_uap), (len(test_uap),), generator=generator)
        value = nearest_centroid_auc(train_trigger[ti], train_uap[ui], test_trigger[tv], test_uap[uv])
        if value == value:
            values.append(float(value))
    if not values:
        return {"lower": None, "upper": None, "samples": 0}
    values.sort()
    lower = values[max(0, int(0.025 * len(values)) - 1)]
    upper = values[min(len(values) - 1, int(0.975 * len(values)))]
    return {"lower": lower, "upper": upper, "samples": len(values)}


def observation_is_compatible(payload: dict, *, seed: int, layers: list[str], triggers: list[str], examples_per_split: int) -> bool:
    return bool(
        payload.get("protocol") == "MDL-FEATURE-v1"
        and payload.get("pair_seed") == int(seed)
        and payload.get("layers") == layers
        and payload.get("triggers") == triggers
        and payload.get("observation_examples_per_split") == int(examples_per_split)
    )


def main():
    parser = argparse.ArgumentParser(description="Run all-layer multitype feature observation")
    parser.add_argument("config")
    parser.add_argument("--seed", type=int, nargs="+", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default="outputs/multitype_observation")
    args = parser.parse_args()
    config = load_config(args.config)
    seeds = [int(v) for v in (args.seed or config["classifier_seeds"])]
    layers = list(config["layers"]["candidates"])
    bootstrap_samples = int(config["observation"].get("bootstrap_samples", 200))
    batch_size = int(config["observation"].get("batch_size", 128))
    summary_examples = int(config["observation"].get("summary_examples", 256))
    similarity_examples = int(config["observation"].get("similarity_examples", 128))
    examples_per_split = int(config["observation"].get("examples_per_split", summary_examples))
    trigger_ids = [str(value) for value in config["trigger_ids"]]
    root = Path(args.output_root)
    all_payloads = []
    for seed in seeds:
        clean_path = render_asset_path(config["assets"]["models"]["clean"], seed=seed)
        clean_model = load_classifier(clean_path, device=args.device)
        uap_path = render_asset_path(config["assets"]["pairs"]["uap"], seed=seed)
        uap_bundle = load_pair_bundle(uap_path)
        seed_root = root / f"seed{seed}"
        result_path = seed_root / "observation.json"
        previous = None
        if result_path.exists():
            try:
                candidate = json.loads(result_path.read_text(encoding="utf-8"))
                if observation_is_compatible(candidate, seed=seed, layers=layers, triggers=trigger_ids, examples_per_split=examples_per_split):
                    previous = candidate
            except (OSError, ValueError, TypeError):
                previous = None
        records = dict((previous or {}).get("records", {}))
        source_asr = dict((previous or {}).get("source_asr", {}))
        source_asr["uap_clean"] = bundle_asr(clean_model, uap_bundle, config["target_label"], device=args.device)
        observation_uap_bundle = subset_pair_bundle_by_split(uap_bundle, examples_per_split)
        uap_features = extract_feature_pairs_by_layer(
            clean_model, observation_uap_bundle, layers, device=args.device, batch_size=batch_size,
        )
        artifacts = dict((previous or {}).get("artifacts", {}))
        artifacts.update({
            "clean_model": {"path": str(clean_path), "sha256": sha256_file(clean_path)},
            "uap_pairs": {"path": str(uap_path), "sha256": sha256_file(uap_path)},
        })
        if previous:
            old_artifacts = previous.get("artifacts", {})
            base_assets_match = bool(
                (old_artifacts.get("clean_model") or {}).get("sha256") == artifacts["clean_model"]["sha256"]
                and (old_artifacts.get("uap_pairs") or {}).get("sha256") == artifacts["uap_pairs"]["sha256"]
            )
            if not base_assets_match:
                records, source_asr = {}, {"uap_clean": source_asr["uap_clean"]}
        for trigger_id in trigger_ids:
            backdoor_path = render_asset_path(config["assets"]["models"]["backdoor"], seed=seed, trigger=trigger_id)
            trigger_path = render_asset_path(config["assets"]["pairs"]["trigger"], seed=seed, trigger=trigger_id)
            current_backdoor_sha = sha256_file(backdoor_path)
            current_trigger_sha = sha256_file(trigger_path)
            old_trigger_artifacts = artifacts.get(trigger_id, {})
            trigger_assets_match = bool(
                (old_trigger_artifacts.get("backdoor_model") or {}).get("sha256") == current_backdoor_sha
                and (old_trigger_artifacts.get("trigger_pairs") or {}).get("sha256") == current_trigger_sha
            )
            if trigger_assets_match and trigger_id in records and set(records[trigger_id]) == set(layers) and trigger_id in source_asr:
                print(f"seed={seed} trigger={trigger_id} status=observation_resumed", flush=True)
                continue
            backdoor_model = load_classifier(backdoor_path, device=args.device)
            trigger_bundle = load_pair_bundle(trigger_path)
            assert_pair_alignment(uap_bundle, trigger_bundle)
            observation_trigger_bundle = subset_pair_bundle_by_split(trigger_bundle, examples_per_split)
            assert_pair_alignment(observation_uap_bundle, observation_trigger_bundle)
            backdoor_trigger = extract_feature_pairs_by_layer(backdoor_model, observation_trigger_bundle, layers, device=args.device, batch_size=batch_size)
            clean_trigger = extract_feature_pairs_by_layer(clean_model, observation_trigger_bundle, layers, device=args.device, batch_size=batch_size)
            backdoor_uap = extract_feature_pairs_by_layer(backdoor_model, observation_uap_bundle, layers, device=args.device, batch_size=batch_size)
            source_asr[trigger_id] = {
                "trigger_backdoor": bundle_asr(backdoor_model, trigger_bundle, config["target_label"], device=args.device),
                "uap_backdoor": bundle_asr(backdoor_model, uap_bundle, config["target_label"], device=args.device),
                "trigger_clean": bundle_asr(clean_model, trigger_bundle, config["target_label"], device=args.device),
            }
            records[trigger_id] = {}
            for layer in layers:
                main = compare_feature_changes(
                    backdoor_trigger[layer], uap_features[layer],
                    summary_examples=summary_examples, similarity_examples=similarity_examples,
                )
                cross = compare_feature_changes(
                    clean_trigger[layer], backdoor_uap[layer],
                    summary_examples=summary_examples, similarity_examples=similarity_examples,
                )
                trigger_train = backdoor_trigger[layer].select("train").delta
                uap_train = uap_features[layer].select("train").delta
                trigger_test = backdoor_trigger[layer].select("test").delta
                uap_test = uap_features[layer].select("test").delta
                main["validation_auc_ci"] = bootstrap_auc(
                    backdoor_trigger[layer].select("train").delta,
                    uap_features[layer].select("train").delta,
                    backdoor_trigger[layer].select("validation").delta,
                    uap_features[layer].select("validation").delta,
                    bootstrap_samples, seed * 1000 + len(records[trigger_id]),
                )
                main["test_auc_ci"] = bootstrap_auc(trigger_train, uap_train, trigger_test, uap_test, bootstrap_samples, seed * 2000 + len(records[trigger_id]))
                records[trigger_id][layer] = {"main": main, "cross_control": cross}
            print(json.dumps({"seed": seed, "trigger": trigger_id, "source_asr": source_asr[trigger_id]}, ensure_ascii=False), flush=True)
            artifacts[trigger_id] = {
                "backdoor_model": {"path": str(backdoor_path), "sha256": current_backdoor_sha},
                "trigger_pairs": {"path": str(trigger_path), "sha256": current_trigger_sha},
            }
            running_payload = {
                "protocol": config["protocol"], "status": "running", "pair_seed": seed,
                "layers": layers, "triggers": trigger_ids, "observation_examples_per_split": examples_per_split,
                "source_asr": source_asr, "records": records, "artifacts": artifacts,
                "environment": environment_record(),
            }
            atomic_write_json(result_path, running_payload)
            del backdoor_model, trigger_bundle, observation_trigger_bundle, backdoor_trigger, clean_trigger, backdoor_uap
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        payload = {
            "protocol": config["protocol"],
            "status": "completed",
            "pair_seed": seed,
            "layers": layers,
            "triggers": trigger_ids,
            "observation_examples_per_split": examples_per_split,
            "source_asr": source_asr,
            "records": records,
            "artifacts": artifacts,
            "environment": environment_record(),
        }
        atomic_write_json(result_path, payload)
        all_payloads.append(payload)
    atomic_write_json(root / "all_observations.json", {"status": "completed", "seeds": all_payloads})
    print(json.dumps({"status": "completed", "seeds": seeds, "output": str(root / "all_observations.json")}, indent=2))


if __name__ == "__main__":
    main()
