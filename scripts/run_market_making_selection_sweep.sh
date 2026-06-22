#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"

"$PYTHON" scripts/walk_forward_market_making_oos.py \
  --report-dir reports/underdog_optimization_kalshi \
  --data-dir archive/processed/underdog_events \
  --output-dir reports/market_making_selection_sweep \
  --feature-modes regular advanced \
  --selection-scores mean sharpe lcb capped_mean capped_sharpe \
  --return-cap 2.0 \
  --sigma-floor 0.05 \
  --lcb-z 1.0 \
  --test-months 1 2 6 \
  --min-train-months 12 \
  --validation-months 6 \
  --min-fit-trades 25 \
  --initial-cash 5000 \
  --period-budget 5000 \
  --stake 5 \
  --max-stake 25 \
  --reserve-fraction 0.30
