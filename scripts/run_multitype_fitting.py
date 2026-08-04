from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import torch

from _bootstrap import ROOT  # noqa: F401
from feature_probe import PROTOCOL
from feature_probe.artifacts import load_classifier, load_pair_bundle
from feature_probe.codec import count_feature_mapping_bits
from feature_probe.compression import compress_feature_mapping
from feature_probe.config import load_config, render_asset_path
from feature_probe.fitters import (
    DeviceTensorBatches,
    build_feature_mapping,
    derive_spatial_support,
    evaluate_feature_mapping,
    fit_feature_mapping,
)
from feature_probe.fitting_experiment import candidate_specs, minimum_bits
from feature_probe.forward import bundle_asr, extract_feature_pairs_by_layer, fitted_feature_asr
from feature_probe.utils import atomic_torch_save, atomic_write_json, environment_record, sha256_file


CONDITIONS = {
    "uap_clean": ("clean", "uap"),
    "trigger_backdoor": ("backdoor", "trigger"),
    "uap_backdoor": ("backdoor", "uap"),
    "trigger_clean": ("clean", "trigger"),
}


def cpu_state_dict(model):
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def fit_condition_layer(config, *, seed, trigger_id, condition, layer, device, output_root, steps):
    fitting = config["fitting"]
    model_kind, mapping_kind = CONDITIONS[condition]
    model_path = render_asset_path(config["assets"]["models"]["clean" if model_kind == "clean" else "backdoor"], seed=seed, trigger=trigger_id)
    pair_path = render_asset_path(config["assets"]["pairs"]["uap" if mapping_kind == "uap" else "trigger"], seed=seed, trigger=trigger_id)
    model = load_classifier(model_path, device=device)
    bundle = load_pair_bundle(pair_path)
    features = extract_feature_pairs_by_layer(model, bundle, [layer], device=device, batch_size=int(config["observation"]["batch_size"]))[layer].feature_tensors_to(device)
    train, validation, test = features.select("train"), features.select("validation"), features.select("test")
    support = derive_spatial_support(train.clean, train.mapped) if train.clean.ndim == 4 else None
    source_asr = bundle_asr(model, bundle, int(config["target_label"]), device=device)
    batch_size = int(fitting["batch_size"])
    validation_loader = DeviceTensorBatches(validation.clean, validation.mapped, batch_size=batch_size, shuffle=False)
    test_loader = DeviceTensorBatches(test.clean, test.mapped, batch_size=batch_size, shuffle=False)
    test_contexts = bundle["clean"][bundle["split_codes"] == 2]
    specs = candidate_specs(
        fitting["families"], fitting["ranks"], fitting["fit_seeds"],
        fitnets_kernels=fitting["fitnets_kernels"],
        feature_re_mask_penalties=fitting["feature_re_mask_penalties"],
        spatial_gated_ranks=fitting["spatial_gated_ranks"],
    )
    output_dir = Path(output_root) / f"seed{seed}" / trigger_id / layer / condition
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in specs:
        torch.manual_seed(30_000 + seed * 1_000 + spec.fit_seed)
        mapper = build_feature_mapping(spec.family, tuple(train.clean.shape[1:]), rank=spec.rank, kernel=spec.kernel, mask_penalty=spec.mask_penalty, spatial_support=support)
        checkpoint_path = checkpoint_dir / f"{spec.key}.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            mapper.load_state_dict(checkpoint["state_dict"])
            mapper = mapper.to(device)
            validation_dense = float(checkpoint["validation_nrmse"])
        else:
            train_loader = DeviceTensorBatches(train.clean, train.mapped, batch_size=batch_size, shuffle=True, seed=40_000 + seed)
            fit = fit_feature_mapping(
                mapper, train_loader, validation_loader, steps=int(steps),
                learning_rate=float(fitting["learning_rate"]), device=device,
                validation_interval=int(fitting["validation_interval"]),
                gradient_clip_norm=float(fitting["gradient_clip_norm"]),
            )
            mapper = fit.model
            validation_dense = float(fit.best_validation_nrmse)
            atomic_torch_save(checkpoint_path, {"protocol": PROTOCOL, "candidate_key": spec.key, "layer": layer, "state_dict": cpu_state_dict(mapper), "validation_nrmse": validation_dense})
        for pruning in fitting["pruning"]:
            for quantization in fitting["quantization"]:
                compressed = compress_feature_mapping(mapper, pruning=float(pruning), quantization=str(quantization))
                validation_nrmse = evaluate_feature_mapping(compressed.model, validation_loader, device=device)
                test_nrmse = evaluate_feature_mapping(compressed.model, test_loader, device=device)
                fitted_asr = fitted_feature_asr(model, compressed.model, features, test_contexts, layer, config["target_label"], device=device, batch_size=batch_size)
                if not all(math.isfinite(float(v)) for v in (validation_nrmse, test_nrmse, fitted_asr)):
                    raise RuntimeError(f"Non-finite metric for {condition}/{layer}/{spec.key}")
                bits = count_feature_mapping_bits(compressed.model, layer_id=layer, family=spec.family, rank=spec.rank, kernel=spec.kernel, quantization=str(quantization), pruning=float(pruning)).as_dict()
                source_qualified = source_asr >= float(config["observation"]["minimum_source_asr"])
                activation_valid = bool(
                    source_qualified and fitted_asr >= float(config["qualification"]["minimum_adversarial_asr"])
                    and abs(fitted_asr - source_asr) <= float(fitting["maximum_asr_gap"])
                )
                rows.append({
                    "candidate_key": spec.key, "family": spec.family, "rank": spec.rank, "kernel": spec.kernel,
                    "mask_penalty": spec.mask_penalty, "fit_seed": spec.fit_seed, "pruning": float(pruning),
                    "quantization": str(quantization), "validation_nrmse": float(validation_nrmse),
                    "test_nrmse": float(test_nrmse), "source_asr": float(source_asr), "fitted_asr": float(fitted_asr),
                    "asr_gap": abs(float(fitted_asr) - float(source_asr)), "activation_valid": activation_valid,
                    "bits": bits, "checkpoint": str(checkpoint_path),
                })
        del mapper
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    payload = {
        "status": "completed", "protocol": PROTOCOL, "seed": seed, "trigger_id": trigger_id,
        "condition": condition, "layer": layer, "source_asr": source_asr,
        "nrmse_threshold": float(fitting["nrmse_threshold"]), "rows": rows,
        "minimum_fit": minimum_bits(rows, nrmse_threshold=float(fitting["nrmse_threshold"])),
        "minimum_activation": minimum_bits(rows, nrmse_threshold=float(fitting["nrmse_threshold"]), require_activation=True),
        "model": {"path": str(model_path), "sha256": sha256_file(model_path)},
        "pair": {"path": str(pair_path), "sha256": sha256_file(pair_path)},
    }
    atomic_write_json(output_dir / "results.json", payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description="Fit all candidate feature mappings at every layer")
    parser.add_argument("config")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--trigger", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default="outputs/multitype_fitting")
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.trigger not in [str(v) for v in config["trigger_ids"]]:
        raise ValueError(f"Unknown configured trigger: {args.trigger}")
    steps = int(args.steps or config["fitting"]["steps"])
    layers = list(config["layers"]["candidates"])
    completed = []
    for layer in layers:
        for condition in CONDITIONS:
            result = fit_condition_layer(config, seed=args.seed, trigger_id=args.trigger, condition=condition, layer=layer, device=args.device, output_root=args.output_root, steps=steps)
            completed.append({"layer": layer, "condition": condition, "minimum_fit": result["minimum_fit"], "minimum_activation": result["minimum_activation"]})
            print(json.dumps({"seed": args.seed, "trigger": args.trigger, "layer": layer, "condition": condition, "minimum_fit": result["minimum_fit"], "minimum_activation": result["minimum_activation"]}, ensure_ascii=False), flush=True)
            gc.collect()
    manifest = {"status": "completed", "seed": args.seed, "trigger": args.trigger, "layers": layers, "conditions": list(CONDITIONS), "results": completed, "environment": environment_record()}
    atomic_write_json(Path(args.output_root) / f"seed{args.seed}" / args.trigger / "manifest.json", manifest)
    print(json.dumps({"status": "completed", "seed": args.seed, "trigger": args.trigger, "layers": layers}, indent=2))


if __name__ == "__main__":
    main()

