# Polymarket Naive 90% Strategy Backtest

This project backtests a simple strategy: **buy the first option that reaches 90%** on binary Polymarket markets, then hold to resolution.

## Quick start

1) Create a virtual environment and install deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Run the backtest:

```bash
python scripts/run_backtest.py
```

## Configuration

All configuration is via environment variables (defaults match the experiment spec):

- `PM_START_DATE` (default: `2025-07-25`)
- `PM_END_DATE` (default: `2026-01-25`)
- `PM_FEE_RATE` (default: `0.02`) — 2% of gross winnings on a winning trade
- `PM_FIDELITY` (default: `5`) — minutes for price history snapshots
- `PM_OUTPUT_DIR` (default: `reports`)
- `PM_GAMMA_BASE` (default: `https://gamma-api.polymarket.com`)
- `PM_CLOB_BASE` (default: `https://clob.polymarket.com`)
- `PM_US_API_BASE` (default: `https://api.polymarket.us`)

Optional auth (only if your settlement endpoint requires it):

- `PM_US_API_KEY`
- `PM_US_API_SECRET`
- `PM_US_API_PASSPHRASE`

If settlement data is unavailable, the backtester will flag those markets and skip them.

## Outputs

The run generates:

- `reports/summary.json`
- `reports/trades.csv`
- `reports/top_winners.csv`
- `reports/top_losers.csv`
- `reports/by_category.csv`
- `reports/data_issues.json`

## Notes

- Markets are filtered to **binary** (YES/NO) based on outcomes length.
- Resolution date filtering uses the best available timestamps from the Gamma API (closed/ended time); this is logged.
- Price history uses the first snapshot where **any** outcome reaches ≥90%.
- If both outcomes hit 90% at the same timestamp, the market is skipped and noted.
