#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/bringibson/Desktop/polybot"
PY="$ROOT/.venv/bin/python"
SLUG_DIR="$ROOT/archive/processed/windows"
REPORT_DIR="$ROOT/reports/windows"

mkdir -p "$REPORT_DIR"

for f in "$SLUG_DIR"/window_*_slugs.txt; do
  base="$(basename "$f" _slugs.txt)"
  res="$REPORT_DIR/${base}_resolutions.csv"
  out="$REPORT_DIR/${base}_inverse_90_takeprofit_pnl_fee_2.csv"
  summary="$REPORT_DIR/${base}_inverse_90_takeprofit_summary_fee_2.json"

  if [[ ! -s "$res" ]]; then
    "$PY" "$ROOT/scripts/build_resolutions.py" --slugs "$f" --output "$res"
  fi

  if [[ ! -s "$summary" ]]; then
    "$PY" "$ROOT/scripts/opposite_90_takeprofit_2x.py" \
      --slugs "$f" \
      --resolutions "$res" \
      --markets "$ROOT/archive/markets.csv" \
      --trades "$ROOT/archive/processed/trades.csv" \
      --threshold 0.9 \
      --fee-rate 0.02 \
      --output "$out" \
      --summary "$summary"
  fi
done
