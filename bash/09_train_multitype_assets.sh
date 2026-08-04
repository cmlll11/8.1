#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/multitype_feature_formal.yaml}
DEVICE=${2:-auto}
if [[ "$DEVICE" == "auto" ]]; then DEVICE=$(python scripts/select_gpu.py); fi
python scripts/train_multitype_assets.py "$CONFIG" --device "$DEVICE"
