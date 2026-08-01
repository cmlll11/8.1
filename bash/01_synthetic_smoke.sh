#!/usr/bin/env bash
set -euo pipefail
device="${1:-cpu}"
python scripts/run_synthetic_smoke.py --device "$device" --steps 2
