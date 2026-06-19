#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
REPORT_DIR="${REPORT_DIR:-reports/underdog_optimization_kalshi}"
OUTPUT_DIR="${OUTPUT_DIR:-reports/online_underdog_allocation}"

"$PYTHON" scripts/online_underdog_allocation.py \
  --report-dir "$REPORT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --cut-fractions 0.5 0.6 0.7 0.8 \
  --weekly-budget 5000 \
  --one-dollar-stake 1 \
  --participation 0.10 \
  "$@"
