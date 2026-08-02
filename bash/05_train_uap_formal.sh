#!/usr/bin/env bash
set -euo pipefail

DEVICE="${1:-cuda:0}"
PAIR_SEED="${2:-0}"

python scripts/train_known_mapping.py configs/forward_smoke.yaml \
  --pair-seed "$PAIR_SEED" \
  --device "$DEVICE"
