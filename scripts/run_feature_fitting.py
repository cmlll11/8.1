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
    evaluate_feature_mapping,
    fit_feature_mapping,
)
from feature_probe.fitting_experiment import candidate_specs, minimum_bits_by_threshold
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--pair-seed", type=int, required=True)
    parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--observation", default=None)
    parser.add_argument("--output-root", default="outputs/paper_feature_fitting")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--fit-seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    fitting = config["fitting"]
    optimization_steps = int(args.steps or fitting["full_steps"])
    if optimization_steps < 1:
        raise ValueError("--steps must be positive")
    selected_fit_seeds = (
        [int(value) for value in args.fit_seeds]
        if args.fit_seeds is not None
        else [int(value) for value in fitting["fit_seeds"]]
    )
    if not selected_fit_seeds:
        raise ValueError("At least one fitting seed is required")
    torch.set_num_threads(int(fitting.get("cpu_threads_per_process", 2)))
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
    )[layer].feature_tensors_to(args.device)
    train = features.select("train")
    validation = features.select("validation")
    test = features.select("test")
    del features
    gc.collect()
    batch_size = int(fitting["batch_size"])
    validation_loader = DeviceTensorBatches(
        validation.clean, validation.mapped, batch_size=batch_size, shuffle=False
    )
    test_loader = DeviceTensorBatches(
        test.clean, test.mapped, batch_size=batch_size, shuffle=False
    )
    test_context_mask = bundle["split_codes"] == 2
    test_contexts = bundle["clean"][test_context_mask]
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
        "optimization_steps": optimization_steps,
        "fit_seeds": selected_fit_seeds,
        "fitting_families": fitting["families"],
        "ranks": [int(value) for value in fitting["ranks"]],
        "fitnets_kernels": [int(value) for value in fitting["fitnets_kernels"]],
        "feature_re_mask_penalties": [
            float(value) for value in fitting["feature_re_mask_penalties"]
        ],
        "pruning_grid": [float(value) for value in fitting["pruning"]],
        "quantization_grid": [str(value) for value in fitting["quantization"]],
        "training_objective": "feature_mse",
        "evaluation_metric": "global_relative_frobenius_nrmse",
        "description_length": "MDL-FEATURE-v1-two-part-code",
        "learning_rate": float(fitting["learning_rate"]),
        "gradient_clip_norm": float(fitting["gradient_clip_norm"]),
    }
    rows = []
    completed = set()
    run_environment = environment_record()
    if result_path.exists():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        for key in (
            "condition",
            "layer",
            "model_sha256",
            "pair_sha256",
            "optimization_steps",
            "fit_seeds",
            "fitting_families",
            "ranks",
            "fitnets_kernels",
            "feature_re_mask_penalties",
            "pruning_grid",
            "quantization_grid",
            "training_objective",
            "evaluation_metric",
            "description_length",
            "learning_rate",
            "gradient_clip_norm",
        ):
            if previous["manifest"].get(key) != manifest[key]:
                raise ValueError(f"Cannot resume because manifest field {key!r} changed")
        rows = previous.get("rows", [])
        completed = set(previous.get("completed_candidates", []))
    specs = candidate_specs(
        fitting["families"],
        fitting["ranks"],
        selected_fit_seeds,
        fitnets_kernels=fitting["fitnets_kernels"],
        feature_re_mask_penalties=fitting["feature_re_mask_penalties"],
    )
    total_candidates = len(specs)

    def persist(status: str):
        atomic_write_json(
            result_path,
            {
                "status": status,
                "manifest": manifest,
                "environment": run_environment,
                "completed_candidates": sorted(completed),
                "total_candidates": total_candidates,
                "rows": rows,
                "minimum_bits": minimum_bits_by_threshold(rows, fitting["nrmse_thresholds"]),
            },
        )

    for candidate_index, spec in enumerate(specs):
        if spec.key in completed:
            print(f"condition={args.condition} candidate={spec.key} status=skipped", flush=True)
            continue
        torch.manual_seed(30_000 + seed * 1_000 + spec.fit_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(30_000 + seed * 1_000 + spec.fit_seed)
        train_loader = DeviceTensorBatches(
            train.clean,
            train.mapped,
            batch_size=batch_size,
            shuffle=True,
            seed=40_000 + seed * 1_000 + spec.fit_seed,
        )
        mapper = build_feature_mapping(
            spec.family,
            tuple(train.clean.shape[1:]),
            rank=spec.rank,
            kernel=spec.kernel,
            mask_penalty=spec.mask_penalty,
        )
        checkpoint_path = output_dir / "checkpoints" / f"{spec.key}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if (
                checkpoint.get("candidate_key") != spec.key
                or checkpoint.get("layer") != layer
                or checkpoint.get("structure") != spec.structure()
            ):
                raise ValueError(f"Checkpoint metadata mismatch: {checkpoint_path}")
            mapper.load_state_dict(checkpoint["state_dict"])
            mapper = mapper.to(args.device)
            dense_best_validation_nrmse = float(checkpoint["best_validation_nrmse"])
            fit_history = checkpoint["history"]
            print(
                f"condition={args.condition} candidate={spec.key} status=dense_checkpoint_resumed",
                flush=True,
            )
        else:
            print(
                f"condition={args.condition} candidate={spec.key} "
                f"status=dense_fit_started steps={optimization_steps}",
                flush=True,
            )
            progress_interval = int(fitting.get("progress_interval", 250))

            def report_progress(record):
                step = int(record["step"])
                if step in (0, 1, optimization_steps) or step % progress_interval == 0:
                    print(
                        f"condition={args.condition} candidate={spec.key} step={step}/{optimization_steps} "
                        f"train_mse={record['train_mse']:.6g} "
                        f"validation_nrmse={record['validation_nrmse']:.4f}",
                        flush=True,
                    )

            fit = fit_feature_mapping(
                mapper,
                train_loader,
                validation_loader,
                steps=optimization_steps,
                learning_rate=float(fitting["learning_rate"]),
                device=args.device,
                validation_interval=int(fitting["validation_interval"]),
                gradient_clip_norm=float(fitting["gradient_clip_norm"]),
                progress_callback=report_progress,
            )
            mapper = fit.model
            dense_best_validation_nrmse = fit.best_validation_nrmse
            fit_history = fit.history
            atomic_torch_save(
                checkpoint_path,
                {
                    "protocol": PROTOCOL,
                    "candidate_key": spec.key,
                    "structure": spec.structure(),
                    "fit_seed": spec.fit_seed,
                    "layer": layer,
                    "optimization_steps": optimization_steps,
                    "state_dict": cpu_state_dict(mapper),
                    "best_validation_nrmse": dense_best_validation_nrmse,
                    "history": fit_history,
                },
            )
        existing_variants = {
            (float(row["pruning"]), str(row["quantization"]))
            for row in rows
            if row["candidate_key"] == spec.key
        }
        for pruning in fitting["pruning"]:
            for quantization in fitting["quantization"]:
                variant = (float(pruning), str(quantization))
                if variant in existing_variants:
                    continue
                compressed = compress_feature_mapping(
                    mapper,
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
                    test,
                    test_contexts,
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
                    family=spec.family,
                    rank=spec.rank,
                    kernel=spec.kernel,
                    quantization=str(quantization),
                    pruning=float(pruning),
                ).as_dict()
                row = {
                    "candidate_key": spec.key,
                    "family": spec.family,
                    "rank": spec.rank,
                    "kernel": spec.kernel,
                    "mask_penalty": spec.mask_penalty,
                    "fit_seed": spec.fit_seed,
                    "pruning": float(pruning),
                    "quantization": str(quantization),
                    "dense_best_validation_nrmse": dense_best_validation_nrmse,
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
                rows.append(row)
                existing_variants.add(variant)
                persist("running")
                print(
                    f"condition={args.condition} candidate={spec.key} pruning={float(pruning):.2f} "
                    f"quantization={str(quantization)} test_nrmse={test_nrmse:.4f} "
                    f"fitted_asr={fitted_asr:.3f}",
                    flush=True,
                )
                del compressed
        candidate_rows = [row for row in rows if row["candidate_key"] == spec.key]
        expected_variants = len(fitting["pruning"]) * len(fitting["quantization"])
        if len(candidate_rows) != expected_variants:
            raise RuntimeError(
                f"Candidate {spec.key} has {len(candidate_rows)} variants, expected {expected_variants}"
            )
        completed.add(spec.key)
        persist("running")
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
        "environment": run_environment,
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
