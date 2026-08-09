from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _bootstrap import ROOT  # noqa: F401
from feature_probe.config import load_config
from feature_probe.utils import atomic_write_json


def ratio_ci(values, samples=200, seed=0):
    values = [float(v) for v in values if v is not None and float(v) > 0]
    if not values:
        return {"lower": None, "upper": None, "samples": 0}
    generator = torch.Generator().manual_seed(int(seed))
    draws = []
    source = torch.tensor(values, dtype=torch.float64)
    for _ in range(int(samples)):
        selected = source[torch.randint(len(source), (len(source),), generator=generator)]
        draws.append(float(selected.mean()))
    draws.sort()
    return {"lower": draws[max(0, int(0.025 * len(draws)) - 1)], "upper": draws[min(len(draws) - 1, int(0.975 * len(draws)))], "samples": len(draws)}


def main():
    parser = argparse.ArgumentParser(description="Summarize per-trigger layer complexity and stopping layers")
    parser.add_argument("config")
    parser.add_argument("--observation-root", default="outputs/multitype_observation")
    parser.add_argument("--fitting-root", default="outputs/multitype_fitting")
    parser.add_argument("--output-root", default="reports")
    args = parser.parse_args()
    config = load_config(args.config)
    seeds = [int(v) for v in config["classifier_seeds"]]
    layers = list(config["layers"]["candidates"])
    trigger_ids = [str(v) for v in config["trigger_ids"]]
    threshold = float(config["fitting"]["nrmse_threshold"])
    auc_limit = float(config["observation"]["auc_threshold"])
    consecutive_required = int(config["observation"].get("indistinguishable_consecutive_layers", 2))
    by_trigger = {}
    for trigger in trigger_ids:
        qualification_by_seed = {}
        for seed in seeds:
            controls_path = Path("artifacts/multitype_models") / f"controls_{trigger}_seed{seed}.json"
            obs_path = Path(args.observation_root) / f"seed{seed}" / "observation.json"
            reasons = []
            controls = json.loads(controls_path.read_text(encoding="utf-8")) if controls_path.exists() else None
            observation = json.loads(obs_path.read_text(encoding="utf-8")) if obs_path.exists() else None
            if not controls or not controls.get("all_passed", False):
                reasons.append("model_controls_failed_or_missing")
            if not observation or trigger not in observation.get("source_asr", {}):
                reasons.append("observation_missing")
            else:
                trigger_sources = observation["source_asr"][trigger]
                if float(observation["source_asr"].get("uap_clean", 0.0)) < float(config["qualification"]["minimum_adversarial_asr"]):
                    reasons.append("uap_asr_below_threshold")
                if float(trigger_sources.get("trigger_backdoor", 0.0)) < float(config["qualification"]["minimum_backdoor_asr"]):
                    reasons.append("backdoor_asr_below_threshold")
                if float(trigger_sources.get("trigger_clean", 1.0)) > float(config["qualification"].get("maximum_clean_patch_asr", 0.10)):
                    reasons.append("clean_trigger_asr_above_threshold")
            qualification_by_seed[str(seed)] = {"qualified": not reasons, "reasons": reasons}
        trigger_qualified = all(item["qualified"] for item in qualification_by_seed.values())
        layer_records = {}
        statuses = []
        for layer_index, layer in enumerate(layers):
            ratios = []
            fit_rows = []
            activation_rows = []
            auc_values = []
            auc_upper = []
            cross = []
            for seed in seeds:
                if not qualification_by_seed[str(seed)]["qualified"]:
                    continue
                obs_path = Path(args.observation_root) / f"seed{seed}" / "observation.json"
                fit_base = Path(args.fitting_root) / f"seed{seed}" / trigger / layer
                if not obs_path.exists():
                    continue
                obs = json.loads(obs_path.read_text(encoding="utf-8"))
                record = obs["records"][trigger][layer]["main"]
                auc_values.append(record.get("test_auc"))
                auc_upper.append((record.get("test_auc_ci") or {}).get("upper"))
                uap_path = fit_base / "uap_clean" / "results.json"
                trigger_path = fit_base / "trigger_backdoor" / "results.json"
                if uap_path.exists() and trigger_path.exists():
                    uap = json.loads(uap_path.read_text(encoding="utf-8"))
                    trg = json.loads(trigger_path.read_text(encoding="utf-8"))
                    uap_min = uap.get("minimum_fit")
                    trg_min = trg.get("minimum_fit")
                    uap_act = uap.get("minimum_activation")
                    trg_act = trg.get("minimum_activation")
                    if uap_min and trg_min:
                        ratio = float(uap_min["total_bits"]) / max(float(trg_min["total_bits"]), 1.0)
                        ratios.append(ratio)
                    fit_rows.append({"seed": seed, "uap": uap_min, "trigger": trg_min})
                    activation_rows.append({"seed": seed, "uap": uap_act, "trigger": trg_act})
                for condition in ("uap_backdoor", "trigger_clean"):
                    path = fit_base / condition / "results.json"
                    if path.exists():
                        item = json.loads(path.read_text(encoding="utf-8"))
                        cross.append({"seed": seed, "condition": condition, "minimum_fit": item.get("minimum_fit"), "minimum_activation": item.get("minimum_activation")})
            ci = ratio_ci(ratios, config["observation"].get("bootstrap_samples", 200), layer_index)
            valid_auc_upper = [float(v) for v in auc_upper if v is not None]
            ratio_indistinguishable = ci["lower"] is not None and ci["lower"] <= 1.0 <= ci["upper"]
            auc_indistinguishable = bool(valid_auc_upper) and max(valid_auc_upper) < auc_limit
            complete = len(ratios) == len(seeds) and len(valid_auc_upper) == len(seeds)
            if not trigger_qualified:
                status = "unqualified"
            elif not complete:
                status = "insufficient"
            else:
                status = "indistinguishable" if ratio_indistinguishable and auc_indistinguishable else "distinguishable"
            statuses.append(status)
            layer_records[layer] = {
                "ratio_per_seed": ratios,
                "ratio_ci": ci,
                "test_auc": auc_values,
                "test_auc_ci_upper": valid_auc_upper,
                "ratio_indistinguishable": ratio_indistinguishable,
                "auc_indistinguishable": auc_indistinguishable,
                "status": status,
                "fit_per_seed": fit_rows,
                "activation_per_seed": activation_rows,
                "cross_controls": cross,
            }
        first = confirmation = None
        for index in range(len(layers) - consecutive_required + 1):
            if all(statuses[index + offset] == "indistinguishable" for offset in range(consecutive_required)):
                first = layers[index]
                confirmation = layers[index + consecutive_required - 1]
                break
        by_trigger[trigger] = {
            "qualification_by_seed": qualification_by_seed,
            "qualified": trigger_qualified,
            "layers": layer_records,
            "status_by_layer": dict(zip(layers, statuses)),
            "first_indistinguishable_layer": first,
            "confirmation_layer": confirmation,
            "last_distinguishable_layer": next((layer for layer in reversed(layers) if layer_records[layer]["status"] == "distinguishable"), None),
            "stopping_rule": f"{consecutive_required} consecutive layers with ratio CI containing 1 and AUROC CI upper < {auc_limit}",
        }
    report = {
        "status": "completed",
        "protocol": config["protocol"],
        "seeds": seeds,
        "triggers": trigger_ids,
        "layers": layers,
        "nrmse_threshold": threshold,
        "trigger_results": by_trigger,
    }
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "multitype_feature_complexity.json", report)
    lines = ["# Multitype feature complexity", "", f"- NRMSE threshold: `{threshold}`", "", "| Trigger | First indistinguishable layer | Confirmation layer |", "|---|---|---|"]
    for trigger, result in by_trigger.items():
        first_layer = result["first_indistinguishable_layer"] if result["qualified"] else "unqualified"
        lines.append(f"| {trigger} | {first_layer or 'not reached'} | {result['confirmation_layer'] or 'NA'} |")
    lines.extend(["", "## Per-layer status", ""])
    for trigger, result in by_trigger.items():
        lines.append(f"### {trigger}")
        lines.append("")
        lines.append("| Layer | Status | Ratio CI | AUROC CI upper |")
        lines.append("|---|---|---:|---:|")
        for layer in layers:
            item = result["layers"][layer]
            ci = item["ratio_ci"]
            lines.append(f"| {layer} | {item['status']} | {ci['lower']}–{ci['upper']} | {max(item['test_auc_ci_upper']) if item['test_auc_ci_upper'] else 'NA'} |")
        lines.append("")
    (output_root / "multitype_feature_complexity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "json": str(output_root / "multitype_feature_complexity.json"), "markdown": str(output_root / "multitype_feature_complexity.md")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
