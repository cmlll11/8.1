from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from _bootstrap import ROOT  # noqa: F401
from feature_probe import PROTOCOL
from feature_probe.artifacts import load_classifier, load_pair_bundle
from feature_probe.config import load_config, render_asset_path
from feature_probe.forward import (
    assert_pair_alignment,
    bundle_asr,
    change_maps,
    compare_feature_changes,
    extract_feature_pairs_by_layer,
    input_pairs,
)
from feature_probe.split import SplitClassifier
from feature_probe.utils import atomic_write_json, environment_record, sha256_file


def save_heatmaps(path: Path, layer_order: list[str], maps: dict) -> None:
    figure, axes = plt.subplots(len(layer_order), 2, figsize=(8, 3 * len(layer_order)), squeeze=False)
    for row, layer in enumerate(layer_order):
        for column, mapping in enumerate(("trigger", "uap")):
            spatial = maps[layer][mapping]["spatial"].numpy()
            image = axes[row, column].imshow(spatial, cmap="magma")
            axes[row, column].set_title(f"{layer}: {mapping} |delta|")
            axes[row, column].axis("off")
            figure.colorbar(image, ax=axes[row, column], fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_channel_maps(path: Path, layer_order: list[str], maps: dict) -> None:
    figure, axes = plt.subplots(len(layer_order), 2, figsize=(10, 2 * len(layer_order)), squeeze=False)
    for row, layer in enumerate(layer_order):
        for column, mapping in enumerate(("trigger", "uap")):
            channel = maps[layer][mapping]["channel"].numpy()[None, :]
            image = axes[row, column].imshow(channel, cmap="magma", aspect="auto")
            axes[row, column].set_title(f"{layer}: {mapping} channel |delta|")
            axes[row, column].set_yticks([])
            figure.colorbar(image, ax=axes[row, column], fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def summarize_maps(maps: dict) -> dict:
    summary = {}
    for layer, by_mapping in maps.items():
        summary[layer] = {}
        for mapping, values in by_mapping.items():
            channel = values["channel"]
            spatial = values["spatial"]
            count = min(10, len(channel))
            top_values, top_indices = torch.topk(channel, count)
            peak = int(spatial.argmax())
            width = int(spatial.shape[1])
            summary[layer][mapping] = {
                "top_channel_indices": top_indices.tolist(),
                "top_channel_mean_abs_change": top_values.tolist(),
                "spatial_peak_row": peak // width,
                "spatial_peak_column": peak % width,
                "spatial_peak_mean_abs_change": float(spatial.flatten()[peak]),
            }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--pair-seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default="outputs/forward_observation")
    args = parser.parse_args()
    config = load_config(args.config)
    seed = int(args.pair_seed)
    if seed not in [int(value) for value in config["classifier_seeds"]]:
        raise ValueError(f"pair seed {seed} is not registered in classifier_seeds")
    trigger_id = config["backdoor"]["trigger_id"]
    clean_path = render_asset_path(config["assets"]["models"]["clean"], seed=seed)
    backdoor_path = render_asset_path(config["assets"]["models"]["backdoor"], seed=seed, trigger=trigger_id)
    uap_path = render_asset_path(config["assets"]["pairs"]["uap"], seed=seed)
    trigger_path = render_asset_path(config["assets"]["pairs"]["trigger"], seed=seed, trigger=trigger_id)
    clean_model = load_classifier(clean_path, device=args.device)
    backdoor_model = load_classifier(backdoor_path, device=args.device)
    uap_bundle = load_pair_bundle(uap_path)
    trigger_bundle = load_pair_bundle(trigger_path)
    assert_pair_alignment(uap_bundle, trigger_bundle)
    if int(uap_bundle["metadata"]["target_label"]) != int(config["target_label"]):
        raise ValueError("UAP target label does not match the config")
    if int(trigger_bundle["metadata"]["target_label"]) != int(config["target_label"]):
        raise ValueError("Trigger target label does not match the config")
    layers = list(config["layers"]["candidates"])
    probe = uap_bundle["clean"][:4].to(args.device)
    split_errors = {
        "clean": SplitClassifier(clean_model).assert_split_consistency(probe, layers),
        "backdoor": SplitClassifier(backdoor_model).assert_split_consistency(probe, layers),
    }
    batch_size = int(config["observation"].get("batch_size", 128))
    summary_examples = int(config["observation"].get("summary_examples", 256))
    similarity_examples = int(config["observation"].get("similarity_examples", 128))
    comparison_options = {
        "summary_examples": summary_examples,
        "similarity_examples": similarity_examples,
    }
    records = {
        "input": {
            "main": compare_feature_changes(
                input_pairs(trigger_bundle),
                input_pairs(uap_bundle),
                **comparison_options,
            ),
            "cross_control": None,
        }
    }
    maps = {
        "input": {
            "trigger": change_maps(input_pairs(trigger_bundle), max_examples=summary_examples),
            "uap": change_maps(input_pairs(uap_bundle), max_examples=summary_examples),
        }
    }
    for layer in layers:
        clean_uap = extract_feature_pairs_by_layer(
            clean_model, uap_bundle, [layer], device=args.device, batch_size=batch_size
        )[layer]
        backdoor_trigger = extract_feature_pairs_by_layer(
            backdoor_model, trigger_bundle, [layer], device=args.device, batch_size=batch_size
        )[layer]
        main = compare_feature_changes(backdoor_trigger, clean_uap, **comparison_options)
        maps[layer] = {
            "trigger": change_maps(backdoor_trigger, max_examples=summary_examples),
            "uap": change_maps(clean_uap, max_examples=summary_examples),
        }
        del clean_uap, backdoor_trigger
        clean_trigger = extract_feature_pairs_by_layer(
            clean_model, trigger_bundle, [layer], device=args.device, batch_size=batch_size
        )[layer]
        backdoor_uap = extract_feature_pairs_by_layer(
            backdoor_model, uap_bundle, [layer], device=args.device, batch_size=batch_size
        )[layer]
        records[layer] = {
            "main": main,
            "cross_control": compare_feature_changes(clean_trigger, backdoor_uap, **comparison_options),
        }
        del clean_trigger, backdoor_uap
    auc_threshold = float(config["observation"]["auc_threshold"])
    deep_control = config["layers"].get("deep_negative_control")
    eligible = [
        layer for layer in layers
        if layer != deep_control and records[layer]["main"]["validation_auc"] >= auc_threshold
    ]
    selected_layer = eligible[0] if eligible else None
    target = int(config["target_label"])
    source_asr = {
        "uap_clean": bundle_asr(clean_model, uap_bundle, target, device=args.device),
        "trigger_backdoor": bundle_asr(backdoor_model, trigger_bundle, target, device=args.device),
        "uap_backdoor_cross": bundle_asr(backdoor_model, uap_bundle, target, device=args.device),
        "trigger_clean_cross": bundle_asr(clean_model, trigger_bundle, target, device=args.device),
    }
    minimum_asr = float(config["observation"]["minimum_source_asr"])
    maximum_gap = float(config["observation"]["maximum_source_asr_gap"])
    source_gate = (
        source_asr["uap_clean"] >= minimum_asr
        and source_asr["trigger_backdoor"] >= minimum_asr
        and abs(source_asr["uap_clean"] - source_asr["trigger_backdoor"]) <= maximum_gap
    )
    output_root = Path(args.output_root) / f"seed{seed}"
    output_root.mkdir(parents=True, exist_ok=True)
    torch.save(maps, output_root / "change_maps.pt")
    save_heatmaps(output_root / "spatial_heatmaps.png", ["input", *layers], maps)
    save_channel_maps(output_root / "channel_maps.png", ["input", *layers], maps)
    payload = {
        "protocol": PROTOCOL,
        "status": "completed" if source_gate and selected_layer else "failed_gate",
        "scientific_scope": "exploratory_single_seed_single_trigger",
        "pair_seed": seed,
        "trigger_id": trigger_id,
        "target_label": target,
        "selected_layer": selected_layer,
        "selection_rule": "first non-deep layer with validation trigger-vs-UAP AUC at or above threshold",
        "source_gate_passed": source_gate,
        "source_asr": source_asr,
        "split_max_errors": split_errors,
        "layers": records,
        "change_map_summary": summarize_maps(maps),
        "limitations": [
            "Only one trigger type is available, so trigger-to-trigger compression cannot be evaluated.",
            "Only classifier seed 0 is available, so cross-seed consistency cannot be evaluated.",
        ],
        "artifacts": {
            "clean_model": {"path": str(clean_path), "sha256": sha256_file(clean_path)},
            "backdoor_model": {"path": str(backdoor_path), "sha256": sha256_file(backdoor_path)},
            "uap_pairs": {"path": str(uap_path), "sha256": sha256_file(uap_path)},
            "trigger_pairs": {"path": str(trigger_path), "sha256": sha256_file(trigger_path)},
            "change_maps": str(output_root / "change_maps.pt"),
            "spatial_heatmaps": str(output_root / "spatial_heatmaps.png"),
            "channel_maps": str(output_root / "channel_maps.png"),
        },
        "environment": environment_record(),
    }
    atomic_write_json(output_root / "observation.json", payload)
    summary = {
        "status": payload["status"],
        "selected_layer": selected_layer,
        "source_asr": source_asr,
        "validation_auc": {layer: records[layer]["main"]["validation_auc"] for layer in layers},
        "test_auc": {layer: records[layer]["main"]["test_auc"] for layer in layers},
        "output": str(output_root / "observation.json"),
    }
    print(json.dumps(summary, indent=2, allow_nan=False))
    if payload["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
