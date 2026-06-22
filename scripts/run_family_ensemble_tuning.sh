#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"

"$PYTHON" scripts/tune_family_ensemble.py \
  --signals reports/strategy_family_diagnostics/strategy_family_signals.csv \
  --output-dir reports/family_ensemble_tuning \
  --first-month 2022-11-01 \
  --tune-before 2025-05-01 \
  --test-months 1 2 6 \
  --initial-cash 5000 \
  --period-budget 5000 \
  --budget-period month \
  --reserve-fraction 0.30 \
  --max-trades-per-market 1 \
  --max-components-per-market 1 \
  --participation-fraction 0.10 \
  --min-stake 0.25 \
  --min-stake-fill-fraction 0.02 \
  --max-configs 5000
