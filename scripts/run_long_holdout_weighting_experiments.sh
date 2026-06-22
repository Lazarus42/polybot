#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-reports/long_holdout_weighting}"

"$PYTHON" scripts/long_holdout_weighting_experiments.py \
  --report-dir reports/underdog_optimization_kalshi \
  --data-dir archive/processed/underdog_events \
  --output-dir "$OUTPUT_DIR" \
  "$@"
