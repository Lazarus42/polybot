## Jake’s Codex — Take‑Profit 2× Strategy Workflow (Detailed)

This document explains how to run the **inverse 90¢ strategy with a 2× take‑profit exit** using the scripts in this repo. It assumes no prior knowledge of the codebase.

---

## 1) Required inputs (data files)
These files are **not** in git and must exist locally:

- `archive/processed/trades.csv`  
  Large trade history file used to locate entry/exit trades.

- `archive/markets.csv`  
  Market metadata (slug, answer1/answer2, closedTime).

- A **resolutions file** built for the specific slug list you want to test  
  Example path: `reports/last_week_resolutions.csv`

If you receive a zip with data, unzip and move files into the repo:
```bash
unzip polymarket_data.zip -d /tmp/polymarket_data
mv /tmp/polymarket_data/archive ./archive
```

---

## 2) Step‑by‑step workflow (exact order)

### Step A — Create a slug list for the time window
You need a `.txt` file of market slugs, one per line.

Examples:

**Last week of dataset (prebuilt in repo):**
```
archive/processed/last_week_slugs.txt
```

**Last two weeks of dataset (prebuilt in repo):**
```
archive/processed/last_two_weeks_slugs.txt
```

If you want to generate a new random slug list:
```bash
python scripts/generate_random_slugs.py \
  --count 4000 \
  --seed 42 \
  --output archive/processed/random_4000_slugs.txt
```

**Last six months of the dataset (create slug list):**
```bash
python - <<'PY'
import csv
from datetime import datetime, timedelta
from pathlib import Path

markets_path = Path("archive/markets.csv")
out_path = Path("archive/processed/last_six_months_slugs.txt")

# Use latest closedTime in the dataset as anchor
max_dt = None
with markets_path.open("r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if not (row.get("answer1") and row.get("answer2")):
            continue
        closed = row.get("closedTime")
        if not closed:
            continue
        try:
            dt = datetime.fromisoformat(closed.replace("Z", "+00:00"))
        except ValueError:
            continue
        if max_dt is None or dt > max_dt:
            max_dt = dt

start = max_dt - timedelta(days=182)
slugs = []
with markets_path.open("r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if not (row.get("answer1") and row.get("answer2")):
            continue
        closed = row.get("closedTime")
        if not closed:
            continue
        try:
            dt = datetime.fromisoformat(closed.replace("Z", "+00:00"))
        except ValueError:
            continue
        if dt < start or dt > max_dt:
            continue
        slug = row.get("market_slug")
        if slug:
            slugs.append(slug)

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\n".join(slugs))
print("latest_closedTime", max_dt.isoformat())
print("window_start", start.isoformat())
print("slugs", len(slugs))
print("wrote", out_path)
PY
```

---

### Step B — Build a **fresh resolutions file** for the slug list
Use this script to fetch market outcomes from Gamma for the slug list:

```bash
python scripts/build_resolutions.py \
  --slugs archive/processed/last_week_slugs.txt \
  --output reports/last_week_resolutions.csv
```

**What this does:**
- For each slug, it queries Gamma (`/markets/slug/{slug}`).
- It infers the winner by looking for **outcomePrices == 1.0**.
- It writes a CSV with columns:
  `slug, status, resolution, market_title, market_id, closed, endDate, resolutionSource, notes`

**Important:** Always build a new resolutions file per experiment to avoid bias.

---

### Step C — Run the **inverse 90¢ take‑profit 2× strategy**
This uses the script:
```
scripts/opposite_90_takeprofit_2x.py
```

Example (last week, fee = 0):
```bash
python scripts/opposite_90_takeprofit_2x.py \
  --slugs archive/processed/last_week_slugs.txt \
  --resolutions reports/last_week_resolutions.csv \
  --markets archive/markets.csv \
  --trades archive/processed/trades.csv \
  --threshold 0.9 \
  --fee-rate 0.0 \
  --output reports/last_week_inverse_90_takeprofit_pnl.csv \
  --summary reports/last_week_inverse_90_takeprofit_summary.json
```

Example (last two weeks, fee = 2%):
```bash
python scripts/opposite_90_takeprofit_2x.py \
  --slugs archive/processed/last_two_weeks_slugs.txt \
  --resolutions reports/last_two_weeks_resolutions.csv \
  --markets archive/markets.csv \
  --trades archive/processed/trades.csv \
  --threshold 0.9 \
  --fee-rate 0.02 \
  --output reports/last_two_weeks_inverse_90_takeprofit_pnl_fee_2.csv \
  --summary reports/last_two_weeks_inverse_90_takeprofit_summary_fee_2.json
```

---

## 3) Strategy definition (exact logic)

For each market:

1) **Signal:** find the first trade where price ≥ 0.90  
2) **Entry:** buy the **opposite side** at the first **actual opposite‑side trade price** after the signal  
3) **Exit:**  
   - If the opposite side later trades at **≥ 2× entry price**, sell immediately at **exactly 2× entry**  
   - Otherwise, hold to resolution  
4) **P&L (per $1 stake):**  
   - Entry shares = `1 / entry_price`  
   - Take‑profit exit:  
     `gross = shares × (2 × entry_price)`  
     `pnl = gross − 1 − fee`  
   - Resolution exit:  
     If win, `gross = shares × 1`  
     If lose, `pnl = −1`

---

## 4) Outputs and how to present results

Each run creates:

### CSV with per‑market trades
Example:
```
reports/last_week_inverse_90_takeprofit_pnl.csv
```
Columns include: `market_id`, `signal_ts`, `opp_entry_price`, `exit_type`, `pnl`.

### JSON summary
Example:
```
reports/last_week_inverse_90_takeprofit_summary.json
```
This contains:
- counts (target slugs, signals, entries, exits)
- win/lose rate
- mean/median/min/max P&L

---

### Recommended metrics to report
From the summary + CSV:

1) **Trade count**
2) **Win / lose rate**
3) **Mean P&L**
4) **Median P&L**
5) **Total profit** (sum of all P&L)
6) **P&L distribution** (histogram or quantiles)

Example (Python one‑liner):
```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("reports/last_week_inverse_90_takeprofit_pnl.csv")
pnl = df["pnl"].dropna()
print("count", len(pnl))
print("mean", pnl.mean())
print("median", pnl.median())
print("min", pnl.min())
print("max", pnl.max())
print("total_profit", pnl.sum())
print("quantiles", pnl.quantile([0.1,0.25,0.75,0.9]).to_dict())
PY
```

---

## 5) Common pitfalls
- If `build_resolutions.py` runs on the wrong slug list, the outcomes won’t match your test.
- The trades file is huge and scanning it can take minutes.
- The exit is at **exactly 2× entry**, not the trade price; this is by design.
- If a market never hits 2× after entry, it is held to resolution.

---

## 6) Quick checklist
- [ ] Slug list created  
- [ ] Resolutions built for that slug list  
- [ ] Take‑profit backtest run  
- [ ] Results summarized (mean, median, total profit, distribution)

---

## 7) Two‑week windows (100 samples)
If Jake needs **100 different two‑week periods**, he should:

1) Generate **100 slug lists**, one per two‑week window, and store them in a folder (e.g., `archive/processed/windows/`).
   Use the helper script:
   ```bash
   python scripts/generate_two_week_windows.py \
     --out-dir archive/processed/windows \
     --windows 100 \
     --seed 42
   ```
2) Run `build_resolutions.py` for **each** slug list to create a matching resolutions CSV.

**Example folder structure:**
```
archive/processed/windows/
  window_000_slugs.txt
  window_001_slugs.txt
  ...
reports/windows/
  window_000_resolutions.csv
  window_001_resolutions.csv
  ...
```

**Example command to build resolutions for one window:**
```bash
python scripts/build_resolutions.py \
  --slugs archive/processed/windows/window_000_slugs.txt \
  --output reports/windows/window_000_resolutions.csv
```

After resolutions are built, run `scripts/opposite_90_takeprofit_2x.py` for each window.
