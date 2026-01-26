## Setup and Reproduce Experiments

This repo contains scripts for backtesting Polymarket strategies. Large data files are excluded from git, so you must place them locally before running.

### 1) Clone and install deps
```bash
git clone <YOUR_REPO_URL>
cd polymarket_exp

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Required data files (not tracked)
Place these files exactly at the paths below:

- `archive/processed/trades.csv`
- `archive/goldsky/orderFilled.csv` (optional for current scripts)
- `archive/markets.csv`
- `polymarket_resolutions.csv`

If you don’t have these files, ask Spencer for a copy.

If Spencer provides a zipped archive of the data, unzip it and move it into this repo so the paths match. Example:
```bash
unzip polymarket_data.zip -d /tmp/polymarket_data
mv /tmp/polymarket_data/archive ./archive
mv /tmp/polymarket_data/polymarket_resolutions.csv ./polymarket_resolutions.csv
```

### 3) Run the core experiments

**A) First-to-90% strategy on labeled subset**
```bash
python scripts/labeled_49_backtest.py --threshold 0.9
```
Outputs:
- `reports/labeled_pnl.csv`
- `reports/labeled_summary.json`

**B) First-to-98% strategy**
```bash
python scripts/labeled_49_backtest.py --threshold 0.98 \
  --output reports/labeled_pnl_98.csv \
  --summary reports/labeled_summary_98.json
```

**C) Resolutions file (Yes/No only)**
```bash
python scripts/resolutions_90_backtest.py
```
Outputs:
- `reports/resolutions_pnl.csv`
- `reports/resolutions_summary.json`

**D) Opposite-side 90% strategy on 4,000 random slugs**
1) Generate random slugs (already in repo if tracked):
```bash
python scripts/generate_random_slugs.py --count 4000 \
  --output archive/processed/random_4000_slugs.txt
```
2) Run the opposite strategy (uses only Yes/No resolutions from `polymarket_resolutions.csv`):
```bash
python scripts/opposite_90_random_slugs.py \
  --slugs archive/processed/random_4000_slugs.txt \
  --output reports/opposite_90_random_4000_pnl.csv \
  --summary reports/opposite_90_random_4000_summary.json
```

### 4) Notes / gotchas
- The `trades.csv` file is very large (~33GB); scripts that scan it can take several minutes.
- The repo ignores `reports/` by default. If you want to keep reports, remove it from `.gitignore`.
- Fee rate defaults to **0** in current scripts.

### 5) Optional: regenerate random slugs
If you need random slugs with a different count, ask Spencer or add a small script to do it.
