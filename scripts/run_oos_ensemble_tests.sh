#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"

"$PYTHON" scripts/walk_forward_ensemble_oos.py \
  --report-dir reports/underdog_optimization_kalshi \
  --data-dir archive/processed/underdog_events \
  --output-dir reports/oos_ensemble_tests \
  --test-months 1 2 6 12 \
  --min-train-months 12 \
  --validation-months 6 \
  --include-partial-final \
  --initial-cash 5000 \
  --period-budget 5000 \
  --max-stake 75 \
  --reserve-fraction 0.30 \
  --max-category-locked-fraction 0.30 \
  --max-regime-locked-fraction 0.30
