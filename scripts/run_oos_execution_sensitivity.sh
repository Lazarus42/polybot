#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
REPORTS=()

for scenario in neutral conservative very_conservative; do
  out="reports/oos_execution_${scenario}"
  REPORTS+=("$out")
  "$PYTHON" scripts/walk_forward_oos.py \
    --report-dir reports/underdog_optimization_kalshi \
    --data-dir archive/processed/underdog_events \
    --output-dir "$out" \
    --test-months 6 12 \
    --min-train-months 12 \
    --validation-months 6 \
    --min-fit-trades 10 \
    --scenario "$scenario" \
    --reserve-fraction 0.30 \
    --max-stake 75 \
    --max-category-locked-fraction 0.30 \
    --max-regime-locked-fraction 0.30 \
    --sizing-policies availability forecast_paced \
    --gate-profiles exclude_high_price low_mid_price_1_15c liquid_low_mid quality_liquid
done

"$PYTHON" scripts/compare_oos_diagnostics.py "${REPORTS[@]}" \
  --output-dir reports/oos_execution_sensitivity_comparison
