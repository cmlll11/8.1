#!/usr/bin/env bash
set -euo pipefail

DEVICE="${1:-cuda:0}"
CONDITION="${2:-uap_clean}"
PAIR_SEED="${3:-0}"

# Fast single-seed feasibility check: 14 architectures, 250 steps each.
python scripts/run_feature_fitting.py configs/forward_smoke.yaml \
  --pair-seed "$PAIR_SEED" \
  --condition "$CONDITION" \
  --device "$DEVICE" \
  --steps 250 \
  --fit-seeds 0 \
  --output-root outputs/feature_fitting_demo
