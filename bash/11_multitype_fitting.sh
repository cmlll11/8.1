#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/multitype_feature_formal.yaml}
DEVICE=${2:-auto}
SEED=${3:?seed is required}
TRIGGER=${4:?trigger is required}
if [[ "$DEVICE" == "auto" ]]; then DEVICE=$(python scripts/select_gpu.py); fi
python scripts/run_multitype_fitting.py "$CONFIG" --seed "$SEED" --trigger "$TRIGGER" --device "$DEVICE"
