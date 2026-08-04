from __future__ import annotations

import argparse
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


def cache_pairs(path: Path, by_condition: dict):
    payload = {}
    for condition, by_layer in by_condition.items():
        payload[condition] = {}
        for layer, pairs in by_layer.items():
            payload[condition][layer] = {
                "clean": pairs.clean.cpu(), "mapped": pairs.mapped.cpu(),
                "labels": pairs.labels.cpu(), "indices": pairs.indices.cpu(),
                "split_codes": pairs.split_codes.cpu(),
            }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


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
    root = Path(args.output_root)
    all_payloads = []
    for seed in seeds:
        clean_path = render_asset_path(config["assets"]["models"]["clean"], seed=seed)
        clean_model = load_classifier(clean_path, device=args.device)
        uap_path = render_asset_path(config["assets"]["pairs"]["uap"], seed=seed)
        uap_bundle = load_pair_bundle(uap_path)
        uap_features = extract_feature_pairs_by_layer(clean_model, uap_bundle, layers, device=args.device, batch_size=batch_size)
        seed_root = root / f"seed{seed}"
        records = {}
        feature_cache = {"uap_clean": uap_features}
        source_asr = {"uap_clean": bundle_asr(clean_model, uap_bundle, config["target_label"], device=args.device)}
        for trigger_id in config["trigger_ids"]:
            trigger_id = str(trigger_id)
            backdoor_path = render_asset_path(config["assets"]["models"]["backdoor"], seed=seed, trigger=trigger_id)
            trigger_path = render_asset_path(config["assets"]["pairs"]["trigger"], seed=seed, trigger=trigger_id)
            backdoor_model = load_classifier(backdoor_path, device=args.device)
            trigger_bundle = load_pair_bundle(trigger_path)
            assert_pair_alignment(uap_bundle, trigger_bundle)
            backdoor_trigger = extract_feature_pairs_by_layer(backdoor_model, trigger_bundle, layers, device=args.device, batch_size=batch_size)
            clean_trigger = extract_feature_pairs_by_layer(clean_model, trigger_bundle, layers, device=args.device, batch_size=batch_size)
            backdoor_uap = extract_feature_pairs_by_layer(backdoor_model, uap_bundle, layers, device=args.device, batch_size=batch_size)
            feature_cache[f"trigger_backdoor:{trigger_id}"] = backdoor_trigger
            feature_cache[f"trigger_clean:{trigger_id}"] = clean_trigger
            feature_cache[f"uap_backdoor:{trigger_id}"] = backdoor_uap
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
        cache_path = seed_root / "feature_cache.pt"
        cache_pairs(cache_path, feature_cache)
        payload = {
            "protocol": config["protocol"],
            "status": "completed",
            "pair_seed": seed,
            "layers": layers,
            "triggers": [str(v) for v in config["trigger_ids"]],
            "source_asr": source_asr,
            "records": records,
            "feature_cache": str(cache_path),
            "artifacts": {"clean_model": {"path": str(clean_path), "sha256": sha256_file(clean_path)}, "uap_pairs": {"path": str(uap_path), "sha256": sha256_file(uap_path)}},
            "environment": environment_record(),
        }
        atomic_write_json(seed_root / "observation.json", payload)
        all_payloads.append(payload)
    atomic_write_json(root / "all_observations.json", {"status": "completed", "seeds": all_payloads})
    print(json.dumps({"status": "completed", "seeds": seeds, "output": str(root / "all_observations.json")}, indent=2))


if __name__ == "__main__":
    main()

