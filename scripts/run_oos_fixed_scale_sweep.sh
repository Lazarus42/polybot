#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"

"$PYTHON" scripts/walk_forward_fixed_scale_sweep.py \
  --report-dir reports/underdog_optimization_kalshi \
  --data-dir archive/processed/underdog_events \
  --output-dir reports/oos_fixed_scale_sweep \
  --fixed-strategy ungated:exclude_high_price:forecast_paced \
  --test-months 6 12 \
  --min-train-months 12 \
  --validation-months 6 \
  --min-fit-trades 10 \
  --reserve-fraction 0.30 \
  --max-category-locked-fraction 0.30 \
  --max-regime-locked-fraction 0.30 \
  --max-stakes 1 2 5 10 25 50 75 100 150 \
  --bankroll-scales 250:5 500:10 1000:15 2500:40 5000:75
