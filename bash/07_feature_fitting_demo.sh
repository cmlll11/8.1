#!/usr/bin/env bash
set -euo pipefail

DEVICE="${1:-cuda:0}"
CONDITION="${2:-uap_clean}"
PAIR_SEED="${3:-0}"

# Exploratory experiment-II demo with a spatially gated positive-control mapper.
python scripts/run_feature_fitting.py configs/forward_smoke.yaml \
  --pair-seed "$PAIR_SEED" \
  --condition "$CONDITION" \
  --device "$DEVICE" \
  --output-root outputs/spatial_gated_feature_fitting_demo
