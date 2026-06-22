#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
REPORTS=()

for score in robust raw_profit without_top1 drawdown_heavy min_trades; do
  out="reports/oos_selector_${score}"
  REPORTS+=("$out")
  "$PYTHON" scripts/walk_forward_oos.py \
    --report-dir reports/underdog_optimization_kalshi \
    --data-dir archive/processed/underdog_events \
    --output-dir "$out" \
    --test-months 6 12 \
    --min-train-months 12 \
    --validation-months 6 \
    --selection-score "$score" \
    --min-fit-trades 10 \
    --reserve-fraction 0.30 \
    --max-stake 75 \
    --max-category-locked-fraction 0.30 \
    --max-regime-locked-fraction 0.30 \
    --sizing-policies flat_one availability hybrid_floor_lcb forecast_paced \
    --gate-profiles \
      light \
      exclude_high_price \
      low_mid_price_1_15c \
      liquid_low_mid \
      quality_liquid
done

"$PYTHON" scripts/compare_oos_diagnostics.py "${REPORTS[@]}" \
  --output-dir reports/oos_selector_score_comparison
