#!/usr/bin/env bash
set -euo pipefail
device="${1:-cuda:0}"
python scripts/train_known_mapping.py configs/forward_smoke.yaml --pair-seed 0 --device "$device" --smoke
