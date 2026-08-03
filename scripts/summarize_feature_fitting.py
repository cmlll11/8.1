from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
from feature_probe.utils import atomic_write_json, sha256_file


CONDITIONS = ("uap_clean", "trigger_backdoor", "uap_backdoor", "trigger_clean")


def load_completed(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise ValueError(f"Fitting result is not completed: {path}")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-seed", type=int, required=True)
    parser.add_argument("--root", default="outputs/feature_fitting")
    args = parser.parse_args()
    seed_root = Path(args.root) / f"seed{args.pair_seed}"
    paths = {condition: seed_root / condition / "results.json" for condition in CONDITIONS}
    results = {condition: load_completed(path) for condition, path in paths.items()}
    layers = {payload["manifest"]["layer"] for payload in results.values()}
    if len(layers) != 1:
        raise ValueError(f"Conditions used different layers: {sorted(layers)}")
    threshold_keys = list(results["uap_clean"]["minimum_bits"])
    comparisons = {}
    for threshold in threshold_keys:
        minima = {
            condition: results[condition]["minimum_bits"].get(threshold)
            for condition in CONDITIONS
        }
        main_uap = minima["uap_clean"]
        main_trigger = minima["trigger_backdoor"]
        ratio = None
        if main_uap is not None and main_trigger is not None:
            ratio = main_uap["total_bits"] / main_trigger["total_bits"]
        comparisons[threshold] = {
            "minimum_bits": {
                condition: value["total_bits"] if value is not None else None
                for condition, value in minima.items()
            },
            "uap_to_trigger_main_ratio": ratio,
            "main_candidates": {
                "uap_clean": main_uap,
                "trigger_backdoor": main_trigger,
            },
            "cross_candidates": {
                "uap_backdoor": minima["uap_backdoor"],
                "trigger_clean": minima["trigger_clean"],
            },
        }
    primary = comparisons.get("0.2")
    if primary is None or primary["uap_to_trigger_main_ratio"] is None:
        verdict = "insufficient: primary threshold unreachable"
    elif primary["uap_to_trigger_main_ratio"] >= 1.25:
        verdict = "supports lower trigger complexity for seed0 BadNets"
    else:
        verdict = "does not support the 1.25 ratio for seed0 BadNets"
    summary = {
        "status": "completed",
        "scientific_scope": "exploratory_single_seed_single_trigger",
        "pair_seed": int(args.pair_seed),
        "layer": next(iter(layers)),
        "verdict": verdict,
        "comparisons": comparisons,
        "inputs": {
            condition: {"path": str(path), "sha256": sha256_file(path)}
            for condition, path in paths.items()
        },
    }
    atomic_write_json(seed_root / "summary.json", summary)
    lines = [
        "# Feature fitting summary",
        "",
        f"- Scope: `{summary['scientific_scope']}`",
        f"- Layer: `{summary['layer']}`",
        f"- Verdict: **{verdict}**",
        "",
        "| NRMSE threshold | UAP clean bits | Trigger backdoor bits | Main ratio | UAP backdoor bits | Trigger clean bits |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for threshold, comparison in comparisons.items():
        bits = comparison["minimum_bits"]
        ratio = comparison["uap_to_trigger_main_ratio"]
        values = [
            threshold,
            bits["uap_clean"],
            bits["trigger_backdoor"],
            f"{ratio:.3f}" if ratio is not None else None,
            bits["uap_backdoor"],
            bits["trigger_clean"],
        ]
        lines.append("| " + " | ".join("NA" if value is None else str(value) for value in values) + " |")
    (seed_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
