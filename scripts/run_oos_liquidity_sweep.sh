#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"

"$PYTHON" scripts/walk_forward_oos.py \
  --report-dir reports/underdog_optimization_kalshi \
  --data-dir archive/processed/underdog_events \
  --output-dir reports/oos_liquidity_sweep \
  --test-months 6 12 \
  --min-train-months 12 \
  --validation-months 6 \
  --min-fit-trades 10 \
  --reserve-fraction 0.30 \
  --max-stake 75 \
  --max-category-locked-fraction 0.30 \
  --max-regime-locked-fraction 0.30 \
  --sizing-policies flat_one availability hybrid_floor_lcb forecast_paced \
  --gate-profiles \
    light \
    liquid_entry_5 \
    liquid_entry_10 \
    liquid_entry_25 \
    liquid_entry_50 \
    liquid_entry_100 \
    exit_liquid_10 \
    exit_liquid_25 \
    exit_liquid_50 \
    exit_liquid_100 \
    exit_liquid_250 \
    liquid_entry_exit_25 \
    liquid_entry_exit_50 \
    liquid_entry_exit_100
