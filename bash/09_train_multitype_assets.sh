#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/multitype_feature_formal.yaml}
DEVICE=${2:-cuda:0}
python scripts/train_multitype_assets.py "$CONFIG" --device "$DEVICE"
