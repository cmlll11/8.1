from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from _bootstrap import ROOT  # noqa: F401
from feature_probe import PROTOCOL
from feature_probe.artifacts import load_classifier, load_pair_bundle
from feature_probe.codec import count_feature_mapping_bits
from feature_probe.compression import compress_feature_mapping
from feature_probe.config import load_config, render_asset_path
from feature_probe.fitters import build_feature_mapping, evaluate_feature_mapping, fit_feature_mapping
from feature_probe.fitting_experiment import candidate_specs, minimum_bits_by_threshold
from feature_probe.forward import bundle_asr, extract_feature_pairs_by_layer, fitted_feature_asr
from feature_probe.utils import atomic_write_json, environment_record, sha256_file


CONDITIONS = {
    "uap_clean": ("clean", "uap"),
    "trigger_backdoor": ("backdoor", "trigger"),
    "uap_backdoor": ("backdoor", "uap"),
    "trigger_clean": ("clean", "trigger"),
}


def cpu_state_dict(model):
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--pair-seed", type=int, required=True)
    parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--observation", default=None)
    parser.add_argument("--output-root", default="outputs/feature_fitting")
    args = parser.parse_args()
    config = load_config(args.config)
    seed = int(args.pair_seed)
    if seed not in [int(value) for value in config["classifier_seeds"]]:
        raise ValueError(f"pair seed {seed} is not registered in classifier_seeds")
    observation_path = Path(args.observation or f"outputs/forward_observation/seed{seed}/observation.json")
    observation = json.loads(observation_path.read_text(encoding="utf-8"))
    layer = observation.get("selected_layer")
    if observation.get("status") != "completed" or not layer:
        raise ValueError("A completed observation with a selected layer is required")
    trigger_id = config["backdoor"]["trigger_id"]
    model_kind, mapping_kind = CONDITIONS[args.condition]
    if model_kind == "clean":
        model_path = render_asset_path(config["assets"]["models"]["clean"], seed=seed)
    else:
        model_path = render_asset_path(
            config["assets"]["models"]["backdoor"], seed=seed, trigger=trigger_id
        )
    if mapping_kind == "uap":
        pair_path = render_asset_path(config["assets"]["pairs"]["uap"], seed=seed)
    else:
        pair_path = render_asset_path(
            config["assets"]["pairs"]["trigger"], seed=seed, trigger=trigger_id
        )
    model = load_classifier(model_path, device=args.device)
    bundle = load_pair_bundle(pair_path)
    features = extract_feature_pairs_by_layer(
        model,
        bundle,
        [layer],
        device=args.device,
        batch_size=int(config["observation"].get("batch_size", 128)),
    )[layer]
    train = features.select("train")
    validation = features.select("validation")
    test = features.select("test")
    fitting = config["fitting"]
    batch_size = int(fitting["batch_size"])
    validation_loader = DataLoader(
        TensorDataset(validation.clean, validation.mapped), batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(TensorDataset(test.clean, test.mapped), batch_size=batch_size, shuffle=False)
    source_asr = bundle_asr(model, bundle, config["target_label"], device=args.device)
    output_dir = Path(args.output_root) / f"seed{seed}" / args.condition
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.json"
    manifest = {
        "protocol": PROTOCOL,
        "pair_seed": seed,
        "condition": args.condition,
        "model_kind": model_kind,
        "mapping_kind": mapping_kind,
        "layer": layer,
        "target_label": int(config["target_label"]),
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "pair_path": str(pair_path),
        "pair_sha256": sha256_file(pair_path),
        "observation_path": str(observation_path),
        "observation_sha256": sha256_file(observation_path),
        "source_asr": source_asr,
    }
    rows = []
    completed = set()
    if result_path.exists():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        for key in ("condition", "layer", "model_sha256", "pair_sha256"):
            if previous["manifest"][key] != manifest[key]:
                raise ValueError(f"Cannot resume because manifest field {key!r} changed")
        rows = previous.get("rows", [])
        completed = set(previous.get("completed_candidates", []))
    specs = candidate_specs(fitting["levels"], fitting["ranks"], fitting["fit_seeds"])
    total_candidates = len(specs)
    for candidate_index, spec in enumerate(specs):
        if spec.key in completed:
            print(f"condition={args.condition} candidate={spec.key} status=skipped", flush=True)
            continue
        torch.manual_seed(30_000 + seed * 1_000 + spec.fit_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(30_000 + seed * 1_000 + spec.fit_seed)
        train_loader = DataLoader(
            TensorDataset(train.clean, train.mapped),
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(40_000 + seed * 1_000 + spec.fit_seed),
        )
        mapper = build_feature_mapping(spec.level, tuple(train.clean.shape[1:]), rank=spec.rank)
        fit = fit_feature_mapping(
            mapper,
            train_loader,
            validation_loader,
            steps=int(fitting["full_steps"]),
            learning_rate=float(fitting["learning_rate"]),
            device=args.device,
            validation_interval=int(fitting["validation_interval"]),
        )
        checkpoint_path = output_dir / "checkpoints" / f"{spec.key}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "protocol": PROTOCOL,
                "candidate_key": spec.key,
                "level": spec.level,
                "rank": spec.rank,
                "fit_seed": spec.fit_seed,
                "layer": layer,
                "state_dict": cpu_state_dict(fit.model),
                "best_validation_nrmse": fit.best_validation_nrmse,
                "history": fit.history,
            },
            checkpoint_path,
        )
        candidate_rows = []
        for pruning in fitting["pruning"]:
            for quantization in fitting["quantization"]:
                compressed = compress_feature_mapping(
                    fit.model,
                    pruning=float(pruning),
                    quantization=str(quantization),
                )
                validation_nrmse = evaluate_feature_mapping(
                    compressed.model, validation_loader, device=args.device
                )
                test_nrmse = evaluate_feature_mapping(compressed.model, test_loader, device=args.device)
                fitted_asr = fitted_feature_asr(
                    model,
                    compressed.model,
                    features,
                    bundle["clean"],
                    layer,
                    config["target_label"],
                    device=args.device,
                    batch_size=batch_size,
                )
                values = (validation_nrmse, test_nrmse, fitted_asr)
                if not all(math.isfinite(value) for value in values):
                    raise RuntimeError(f"Non-finite metric for {spec.key}")
                bits = count_feature_mapping_bits(
                    compressed.model,
                    layer_id=layer,
                    level=spec.level,
                    rank=spec.rank,
                    quantization=str(quantization),
                    pruning=float(pruning),
                ).as_dict()
                candidate_rows.append(
                    {
                        "candidate_key": spec.key,
                        "level": spec.level,
                        "rank": spec.rank,
                        "fit_seed": spec.fit_seed,
                        "pruning": float(pruning),
                        "quantization": str(quantization),
                        "dense_best_validation_nrmse": fit.best_validation_nrmse,
                        "validation_nrmse": validation_nrmse,
                        "test_nrmse": test_nrmse,
                        "source_asr": source_asr,
                        "fitted_asr": fitted_asr,
                        "asr_gap": abs(fitted_asr - source_asr),
                        "functional_valid": abs(fitted_asr - source_asr)
                        <= float(fitting["maximum_asr_gap"]),
                        "total_values": compressed.total_values,
                        "kept_values": compressed.kept_values,
                        "bits": bits,
                        "checkpoint": str(checkpoint_path),
                    }
                )
                del compressed
        rows.extend(candidate_rows)
        completed.add(spec.key)
        payload = {
            "status": "running",
            "manifest": manifest,
            "environment": environment_record(),
            "completed_candidates": sorted(completed),
            "total_candidates": total_candidates,
            "rows": rows,
            "minimum_bits": minimum_bits_by_threshold(rows, fitting["nrmse_thresholds"]),
        }
        atomic_write_json(result_path, payload)
        best_test = min(row["test_nrmse"] for row in candidate_rows)
        valid_count = sum(row["functional_valid"] for row in candidate_rows)
        print(
            f"condition={args.condition} candidate={spec.key} "
            f"completed={candidate_index + 1}/{total_candidates} "
            f"best_test_nrmse={best_test:.4f} functional_valid={valid_count}/{len(candidate_rows)}",
            flush=True,
        )
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    final_payload = {
        "status": "completed",
        "manifest": manifest,
        "environment": environment_record(),
        "completed_candidates": sorted(completed),
        "total_candidates": total_candidates,
        "rows": rows,
        "minimum_bits": minimum_bits_by_threshold(rows, fitting["nrmse_thresholds"]),
    }
    atomic_write_json(result_path, final_payload)
    print(
        json.dumps(
            {
                "status": "completed",
                "condition": args.condition,
                "layer": layer,
                "source_asr": source_asr,
                "candidates": len(completed),
                "rows": len(rows),
                "minimum_bits": final_payload["minimum_bits"],
                "output": str(result_path),
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
