#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/multitype_feature_formal.yaml}
python scripts/summarize_multitype_complexity.py "$CONFIG"
