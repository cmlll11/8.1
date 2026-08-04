#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/multitype_feature_formal.yaml}
DEVICE=${2:-cuda:0}
SEED=${3:?seed is required}
TRIGGER=${4:?trigger is required}
python scripts/run_multitype_fitting.py "$CONFIG" --seed "$SEED" --trigger "$TRIGGER" --device "$DEVICE"
