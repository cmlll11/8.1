#!/usr/bin/env bash
set -euo pipefail
config="${1:-configs/forward_smoke.yaml}"
python scripts/check_assets.py "$config" --json-out outputs/asset_check.json
