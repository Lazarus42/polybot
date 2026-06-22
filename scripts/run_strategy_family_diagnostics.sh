#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"

"$PYTHON" scripts/strategy_family_diagnostics.py \
  --report-dir reports/underdog_optimization_kalshi \
  --data-dir archive/processed/underdog_events \
  --output-dir reports/strategy_family_diagnostics \
  --test-months 1 2 6 \
  --min-train-months 12 \
  --validation-months 6 \
  --min-entry-fill 10 \
  --min-bucket-trades 50 \
  --shrink-k 200 \
  --edge-thresholds 0.02 0.03 0.05 0.08
