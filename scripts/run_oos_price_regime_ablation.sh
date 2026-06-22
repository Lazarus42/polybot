#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"

"$PYTHON" scripts/walk_forward_oos.py \
  --report-dir reports/underdog_optimization_kalshi \
  --data-dir archive/processed/underdog_events \
  --output-dir reports/oos_price_regime_ablation \
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
    low_price_1_5c \
    low_mid_price_1_15c \
    exclude_high_price \
    mid_price_6_30c \
    price_16_30c \
    high_price_31_49c
