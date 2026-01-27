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

### 4) MOST IMPORTANT: Jake’s Codex experiment (repeat 100x on 100 random markets)
This takes priority over all other experiments.

Goal: **100 independent trials**. Each trial:
1) randomly sample **100 resolved binary markets**
2) run **inverse 90¢** strategy on that 100‑market sample
3) collect P&L + distribution stats
4) repeat for **0% fees** and **2% fees**
5) save the **total profit per trial** and analyze the distribution of totals

**Suggested workflow (repeatable, deterministic seeds):**
```bash
python - <<'PY'
import json
from pathlib import Path
import subprocess

TRIALS = 100
N = 100
FEES = [0.0, 0.02]
OUT_DIR = Path("reports/jake_trials")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def run_trial(trial_idx, fee):
    slugs = OUT_DIR / f"slugs_trial_{trial_idx:03d}_fee_{int(fee*100)}.txt"
    summary = OUT_DIR / f"summary_trial_{trial_idx:03d}_fee_{int(fee*100)}.json"
    pnl = OUT_DIR / f"pnl_trial_{trial_idx:03d}_fee_{int(fee*100)}.csv"

    # generate 100 random slugs with a fixed seed per trial
    subprocess.run([
        "python", "scripts/generate_random_slugs.py",
        "--count", str(N),
        "--seed", str(1000 + trial_idx),
        "--output", str(slugs),
    ], check=True)

    # run inverse 90¢ strategy on that slug sample
    subprocess.run([
        "python", "scripts/opposite_90_random_slugs.py",
        "--slugs", str(slugs),
        "--output", str(pnl),
        "--summary", str(summary),
    ], check=True)

    return summary

def collect_totals(fee):
    totals = []
    for trial_idx in range(TRIALS):
        summary_path = OUT_DIR / f"summary_trial_{trial_idx:03d}_fee_{int(fee*100)}.json"
        with open(summary_path) as f:
            s = json.load(f)
        # total profit = mean * count
        if s.get("pnl_mean") is None or s.get("with_pnl") is None:
            continue
        totals.append(s["pnl_mean"] * s["with_pnl"])
    return totals

# Run all trials for each fee
for fee in FEES:
    for trial_idx in range(TRIALS):
        run_trial(trial_idx, fee)

# Aggregate totals
for fee in FEES:
    totals = collect_totals(fee)
    out = OUT_DIR / f"totals_fee_{int(fee*100)}.json"
    out.write_text(json.dumps({
        "fee": fee,
        "trials": len(totals),
        "totals": totals,
        "mean_total": sum(totals)/len(totals) if totals else None,
        "min_total": min(totals) if totals else None,
        "max_total": max(totals) if totals else None,
    }, indent=2))
PY
```

**Outputs to review:**
- `reports/jake_trials/summary_trial_*.json` (per‑trial P&L stats + distribution)
- `reports/jake_trials/totals_fee_0.json` and `totals_fee_2.json` (distribution of total profit across 100 trials)

### 4) Notes / gotchas
- The `trades.csv` file is very large (~33GB); scripts that scan it can take several minutes.
- The repo ignores `reports/` by default. If you want to keep reports, remove it from `.gitignore`.
- Fee rate defaults to **0** in current scripts.

### 5) Optional: regenerate random slugs
If you need random slugs with a different count, ask Spencer or add a small script to do it.
