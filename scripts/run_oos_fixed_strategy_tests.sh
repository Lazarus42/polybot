#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"

run_fixed() {
  local name="$1"
  local fixed="$2"
  local gate_profile="$3"
  local sizing_policy="$4"
  "$PYTHON" scripts/walk_forward_oos.py \
    --report-dir reports/underdog_optimization_kalshi \
    --data-dir archive/processed/underdog_events \
    --output-dir "reports/${name}" \
    --test-months 6 12 \
    --min-train-months 12 \
    --validation-months 6 \
    --min-fit-trades 10 \
    --reserve-fraction 0.30 \
    --max-stake 75 \
    --max-category-locked-fraction 0.30 \
    --max-regime-locked-fraction 0.30 \
    --sizing-policies "$sizing_policy" \
    --gate-profiles "$gate_profile" \
    --fixed-strategies "$fixed"
}

run_fixed oos_fixed_exclude_high_availability \
  ungated:exclude_high_price:availability \
  exclude_high_price \
  availability

run_fixed oos_fixed_exclude_high_forecast_paced \
  ungated:exclude_high_price:forecast_paced \
  exclude_high_price \
  forecast_paced

run_fixed oos_fixed_low_mid_flat_one \
  selected_gate:low_mid_price_1_15c:flat_one \
  low_mid_price_1_15c \
  flat_one

"$PYTHON" scripts/compare_oos_diagnostics.py \
  reports/oos_fixed_exclude_high_availability \
  reports/oos_fixed_exclude_high_forecast_paced \
  reports/oos_fixed_low_mid_flat_one \
  --output-dir reports/oos_fixed_strategy_comparison
