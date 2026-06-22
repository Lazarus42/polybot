#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"

"$PYTHON" scripts/summarize_oos_risk.py \
  reports/oos_best_strategy_duration_sweep/oos_period_results.csv \
  reports/oos_ensemble_tests/ensemble_period_results.csv \
  reports/oos_best_strategy_throttle_comparison/combined_oos_period_results.csv \
  reports/market_making_oos/market_making_period_results.csv \
  reports/market_making_selection_sweep/market_making_period_results.csv \
  --output-dir reports/oos_risk_summary \
  --bootstrap-iterations 1000
