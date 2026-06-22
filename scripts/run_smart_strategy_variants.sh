#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"

run_variant() {
  local name="$1"
  shift
  local out="reports/${name}"
  echo "== $name =="
  "$PYTHON" scripts/long_holdout_weighting_experiments.py \
    --report-dir reports/underdog_optimization_kalshi \
    --data-dir archive/processed/underdog_events \
    --output-dir "$out" \
    "$@"
  "$PYTHON" scripts/visualize_long_holdout_results.py \
    --input "$out/account_summary.csv" \
    --output-dir "$out/visuals"
}

run_variant long_holdout_variant_conservative_pacing \
  --sizing-policies flat_one availability hybrid_floor_lcb forecast_paced \
  --reserve-fraction 0.40 \
  --max-stake 75 \
  --max-category-locked-fraction 0.25 \
  --max-regime-locked-fraction 0.25

run_variant long_holdout_variant_small_stakes \
  --sizing-policies flat_one availability hybrid_floor_lcb forecast_paced \
  --reserve-fraction 0.25 \
  --max-stake 50 \
  --max-category-locked-fraction 0.35 \
  --max-regime-locked-fraction 0.35

run_variant long_holdout_variant_lower_minfit \
  --sizing-policies flat_one availability hybrid_floor_lcb forecast_paced \
  --min-fit-trades 10 \
  --reserve-fraction 0.30 \
  --max-stake 75 \
  --max-category-locked-fraction 0.30 \
  --max-regime-locked-fraction 0.30

run_variant long_holdout_variant_more_time_to_exit \
  --sizing-policies flat_one availability hybrid_floor_lcb forecast_paced \
  --min-minutes-to-close 240 \
  --reserve-fraction 0.30 \
  --max-stake 75
