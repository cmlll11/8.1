from __future__ import annotations

import argparse
import json

from _bootstrap import ROOT  # noqa: F401
from feature_probe.cifar10 import load_cifar10
from feature_probe.config import load_config
from feature_probe.training import train_classifier_pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--pair-seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    splits = load_cifar10(config["data"]["root"], split_seed=config["data"]["split_seed"], download=args.download)
    controls = train_classifier_pair(
        splits,
        config,
        pair_seed=args.pair_seed,
        device=args.device,
        output_root="artifacts/models",
        smoke=args.smoke,
    )
    print(json.dumps(controls, indent=2, ensure_ascii=False))
    if not args.smoke and not controls["all_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
