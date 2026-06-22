#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"

COMMON=(
  --report-dir reports/underdog_optimization_kalshi
  --data-dir archive/processed/underdog_events
  --test-months 6 12
  --min-train-months 12
  --validation-months 6
  --min-fit-trades 10
  --max-category-locked-fraction 0.30
  --max-regime-locked-fraction 0.30
  --sizing-policies availability forecast_paced
  --gate-profiles exclude_high_price low_mid_price_1_15c liquid_low_mid quality_liquid
)

REPORTS=()
for max_stake in 10 25 50 75 150; do
  out="reports/oos_max_stake_${max_stake}"
  REPORTS+=("$out")
  "$PYTHON" scripts/walk_forward_oos.py \
    "${COMMON[@]}" \
    --output-dir "$out" \
    --max-stake "$max_stake" \
    --reserve-fraction 0.30
done

for reserve in 0.10 0.25 0.30 0.40 0.50; do
  label="${reserve/./}"
  out="reports/oos_reserve_${label}"
  REPORTS+=("$out")
  "$PYTHON" scripts/walk_forward_oos.py \
    "${COMMON[@]}" \
    --output-dir "$out" \
    --max-stake 75 \
    --reserve-fraction "$reserve"
done

"$PYTHON" scripts/compare_oos_diagnostics.py "${REPORTS[@]}" \
  --output-dir reports/oos_sizing_bankroll_comparison
