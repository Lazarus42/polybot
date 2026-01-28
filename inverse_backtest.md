## Inverse 90¢ Backtest Flow (Jake’s Codex)

This document explains the end‑to‑end workflow to reproduce the **inverse 90¢ strategy** experiments.

**Note:** Whenever we say “run an inverse test,” we are referring to the workflow defined in this file.

### What the inverse 90¢ strategy does
- For each market, find the **first trade** where price ≥ **0.90**.
- Instead of buying that 90¢ side, buy the **opposite** side at price **(1 − entry_price)**.
- Hold to resolution and compute P&L on a $1 stake.
- Fees are optional via `--fee-rate` (0.00 or 0.02).

### Required data files (not in git)
You must have these locally:
- `archive/processed/trades.csv` (large trades file)
- `archive/markets.csv` (market metadata: answer1/answer2, slug)
- `polymarket_resolutions.csv` (resolution outcomes by slug)

If you receive a zipped data bundle:
```bash
unzip polymarket_data.zip -d /tmp/polymarket_data
mv /tmp/polymarket_data/archive ./archive
mv /tmp/polymarket_data/polymarket_resolutions.csv ./polymarket_resolutions.csv
```

### Step 1) Generate a random slug list
This samples **resolved binary** markets from `archive/markets.csv`.

```bash
python scripts/generate_random_slugs.py \
  --count 4000 \
  --seed 42 \
  --output archive/processed/random_4000_slugs.txt
```

### Step 2) Fetch resolutions for those slugs
This produces `polymarket_resolutions.csv` by querying Gamma and inferring outcomes from payouts.

```bash
python anakysis.py archive/processed/random_4000_slugs.txt
```

### Step 3) Run inverse 90¢ backtest on that slug list
Uses:
- slug list from step 1
- resolutions from step 2
- metadata + trades to compute P&L

```bash
python scripts/opposite_90_random_slugs.py \
  --slugs archive/processed/random_4000_slugs.txt \
  --resolutions polymarket_resolutions.csv \
  --markets archive/markets.csv \
  --trades archive/processed/trades.csv \
  --threshold 0.9 \
  --fee-rate 0.0 \
  --output reports/opposite_90_random_4000_pnl.csv \
  --summary reports/opposite_90_random_4000_summary.json
```

For **2% fees**, run:
```bash
python scripts/opposite_90_random_slugs.py \
  --slugs archive/processed/random_4000_slugs.txt \
  --resolutions polymarket_resolutions.csv \
  --markets archive/markets.csv \
  --trades archive/processed/trades.csv \
  --threshold 0.9 \
  --fee-rate 0.02 \
  --output reports/opposite_90_random_4000_pnl_fee_2.csv \
  --summary reports/opposite_90_random_4000_summary_fee_2.json
```

### Outputs to inspect
- `reports/opposite_90_random_4000_pnl.csv`
- `reports/opposite_90_random_4000_summary.json`

### Notes
- `trades.csv` is huge; scanning it can take minutes per run.
- `anakysis.py` uses Gamma to infer outcomes from payout=1.0 in `outcomePrices`.
