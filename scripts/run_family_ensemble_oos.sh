#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"

"$PYTHON" scripts/replay_family_ensemble_oos.py \
  --signals reports/strategy_family_diagnostics/strategy_family_signals.csv \
  --output-dir reports/family_ensemble_oos \
  --first-month 2022-11-01 \
  --test-months 1 2 6 \
  --min-train-months 12 \
  --validation-months 6 \
  --initial-cash 5000 \
  --period-budget 5000 \
  --budget-period month \
  --reserve-fraction 0.30 \
  --max-trades-per-market 1 \
  --max-components-per-market 1
