#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/bringibson/Desktop/polybot"
PY="$ROOT/.venv/bin/python"
RES_DIR="$ROOT/reports/windows"
SLUG_DIR="$ROOT/archive/processed/windows"
OUT_DIR="$ROOT/reports/windows_inverse90"

mkdir -p "$OUT_DIR"

for slug_file in "$SLUG_DIR"/window_*_slugs.txt; do
  base="$(basename "$slug_file" _slugs.txt)"
  res="$RES_DIR/${base}_resolutions.csv"
  out="$OUT_DIR/${base}_inverse90_pnl_fee_2.csv"
  summary="$OUT_DIR/${base}_inverse90_summary_fee_2.json"

  if [[ ! -s "$res" ]]; then
    echo "Missing resolutions for $base: $res" >&2
    continue
  fi

  if [[ ! -s "$summary" ]]; then
    "$PY" "$ROOT/scripts/opposite_90_random_slugs.py" \
      --slugs "$slug_file" \
      --resolutions "$res" \
      --markets "$ROOT/archive/markets.csv" \
      --trades "$ROOT/archive/processed/trades.csv" \
      --threshold 0.9 \
      --fee-rate 0.02 \
      --output "$out" \
      --summary "$summary"
  fi
done
