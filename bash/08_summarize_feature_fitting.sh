#!/usr/bin/env bash
set -euo pipefail

PAIR_SEED="${1:-0}"

python scripts/summarize_feature_fitting.py --pair-seed "$PAIR_SEED"
