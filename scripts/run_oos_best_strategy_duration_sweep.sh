#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"

"$PYTHON" scripts/walk_forward_oos.py \
  --report-dir reports/underdog_optimization_kalshi \
  --data-dir archive/processed/underdog_events \
  --output-dir reports/oos_best_strategy_duration_sweep \
  --test-months 1 2 6 12 \
  --min-train-months 12 \
  --validation-months 6 \
  --include-partial-final \
  --min-fit-trades 10 \
  --initial-cash 5000 \
  --period-budget 5000 \
  --reserve-fraction 0.30 \
  --max-stake 75 \
  --max-category-locked-fraction 0.30 \
  --max-regime-locked-fraction 0.30 \
  --sizing-policies forecast_paced \
  --gate-profiles exclude_high_price \
  --fixed-strategies ungated:exclude_high_price:forecast_paced
