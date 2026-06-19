#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
REPORT_DIR="${REPORT_DIR:-reports/underdog_optimization_kalshi}"
OUTPUT_DIR="${OUTPUT_DIR:-reports/realistic_underdog_account}"

"$PYTHON" scripts/realistic_underdog_account.py \
  --report-dir "$REPORT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --cut-fractions 0.6 0.7 0.8 \
  --initial-cash 5000 \
  --weekly-budget 5000 \
  "$@"
