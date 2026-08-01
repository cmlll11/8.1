from __future__ import annotations

import argparse
import json

from _bootstrap import ROOT  # noqa: F401
from feature_probe import PROTOCOL
from feature_probe.utils import environment_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    record = environment_record()
    record["protocol"] = PROTOCOL
    print(json.dumps(record, indent=2, ensure_ascii=False))
    if "torch_error" in record:
        raise SystemExit(2)
    if args.require_cuda and not record.get("cuda_available"):
        raise SystemExit("CUDA is required but unavailable")


if __name__ == "__main__":
    main()
