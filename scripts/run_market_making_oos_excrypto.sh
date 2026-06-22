#!/usr/bin/env bash
# Market-making OOS with crypto fenced off and conservative sizing, to test whether a
# liquidity-provision edge survives once the ultra-short-dated crypto markets (which
# flooded the universe in 2025 and carry negative edge) are excluded from fit + replay.
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"

"$PYTHON" scripts/walk_forward_market_making_oos.py \
  --report-dir reports/underdog_optimization_kalshi \
  --data-dir archive/processed/underdog_events \
  --output-dir reports/market_making_oos_excrypto \
  --feature-modes regular advanced \
  --selection-scores capped_sharpe \
  --return-cap 2.0 \
  --sigma-floor 0.05 \
  --test-months 1 2 6 \
  --min-train-months 12 \
  --validation-months 6 \
  --min-fit-trades 25 \
  --initial-cash 5000 \
  --period-budget 5000 \
  --scenario conservative \
  --stake 5 \
  --max-stake 15 \
  --reserve-fraction 0.30 \
  --exclude-categories crypto
# Optional extra knob (not set, to keep this a clean crypto-only comparison):
#   --min-horizon-days 1   # also fences off any sub-daily markets in other categories
