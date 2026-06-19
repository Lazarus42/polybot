#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
REPORT_DIR="${REPORT_DIR:-reports/underdog_optimization_kalshi}"
OUTPUT_DIR="${OUTPUT_DIR:-reports/monthly_underdog_experiments}"

"$PYTHON" scripts/monthly_underdog_experiments.py \
  --report-dir "$REPORT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --cut-fractions 0.6 0.7 0.8 \
  --initial-cash 5000 \
  --monthly-budget 5000 \
  "$@"
