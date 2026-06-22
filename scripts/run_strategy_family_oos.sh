#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"

"$PYTHON" scripts/replay_strategy_family_oos.py \
  --signals reports/strategy_family_diagnostics/strategy_family_signals.csv \
  --output-dir reports/strategy_family_oos \
  --first-month 2022-11-01 \
  --test-months 1 2 6 \
  --min-train-months 12 \
  --validation-months 6 \
  --initial-cash 5000 \
  --period-budget 5000 \
  --budget-period month \
  --stake 5 \
  --tail-stake 1 \
  --reserve-fraction 0.30 \
  --max-trades-per-market 1
