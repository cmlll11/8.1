from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
from feature_probe.artifacts import load_pair_bundle
from feature_probe.config import load_config, render_asset_path
from feature_probe.utils import sha256_file


def inspect(path: Path, kind: str) -> dict:
    row = {"kind": kind, "path": str(path), "exists": path.is_file()}
    if not row["exists"]:
        row["status"] = "missing"
        return row
    row["sha256"] = sha256_file(path)
    row["bytes"] = path.stat().st_size
    if kind == "pair_bundle":
        try:
            payload = load_pair_bundle(path)
            row.update(status="valid", examples=len(payload["clean"]), shape=list(payload["clean"].shape))
        except Exception as exc:
            row.update(status="invalid", error=repr(exc))
    else:
        row["status"] = "present_unloaded"
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--json-out")
    args = parser.parse_args()
    config = load_config(args.config)
    rows = []
    assets = config["assets"]
    for seed in config["classifier_seeds"]:
        rows.append(inspect(render_asset_path(assets["models"]["clean"], seed=seed), "model"))
        rows.append(inspect(render_asset_path(assets["pairs"]["uap"], seed=seed), "pair_bundle"))
        for trigger in config["trigger_ids"]:
            rows.append(inspect(render_asset_path(assets["models"]["backdoor"], seed=seed, trigger=trigger), "model"))
            rows.append(inspect(render_asset_path(assets["pairs"]["trigger"], seed=seed, trigger=trigger), "pair_bundle"))
    payload = {
        "config": config["_config_path"],
        "assets": rows,
        "all_present": all(row["status"] not in {"missing", "invalid"} for row in rows),
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(text + "\n", encoding="utf-8")
    if not payload["all_present"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
