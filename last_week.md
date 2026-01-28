## Inverse 90¢ Strategy — Last Week in Dataset

**Window definition (last week of dataset)**  
- Latest `closedTime` in `archive/markets.csv`: **2025-10-07T14:45:40Z**  
- Window start: **2025-09-30T14:45:40Z**  
- Binary closed markets in window: **4,079**

### Status
Slug list generated:
- `archive/processed/last_week_slugs.txt`

### Run steps (requires network access)
```bash
# Build a fresh resolutions file for this window
python scripts/build_resolutions.py \
  --slugs archive/processed/last_week_slugs.txt \
  --output reports/last_week_resolutions.csv

# Run inverse 90¢ backtest with 0% fees
python scripts/opposite_90_random_slugs.py \
  --slugs archive/processed/last_week_slugs.txt \
  --resolutions reports/last_week_resolutions.csv \
  --markets archive/markets.csv \
  --trades archive/processed/trades.csv \
  --threshold 0.9 \
  --fee-rate 0.0 \
  --output reports/last_week_inverse_90_pnl.csv \
  --summary reports/last_week_inverse_90_summary.json
```

### Results
**Completed (fee = 0.0)**  

Counts:
- Target slugs: **4,079**
- Yes/No resolutions in target set: **562**
- Markets with first ≥0.90 hit: **498**
- Trades evaluated: **498**

Win/Lose:
- **Win rate**: **7.83%**
- **Lose rate**: **92.17%**

P&L summary (per $1 stake):
- **Mean**: **+2.1989**
- **Median**: **-1.00**
- **Min / Max**: **-1.00 / +332.33**
- **Std dev**: **18.9728**
- **Total profit (sum of P&L)**: **+1,095.06**

Quantiles:
- p10 **-1.00**, p25 **-1.00**, p75 **-1.00**, p90 **-1.00**

Histogram (counts by bin):
- [-1.0, -0.5]: **459**
- (5, 10]: **4**
- (10, 20]: **17**
- (20, 50]: **12**
- (50, 100]: **5**
- (100, 400]: **1**

Output files:
- `reports/last_week_inverse_90_pnl.csv`
- `reports/last_week_inverse_90_summary.json`

**Completed (fee = 0.02)**  

Counts:
- Target slugs: **4,079**
- Yes/No resolutions in target set: **562**
- Markets with first ≥0.90 hit: **498**
- Trades evaluated: **498**

Win/Lose:
- **Win rate**: **7.83%**
- **Lose rate**: **92.17%**

P&L summary (per $1 stake):
- **Mean**: **+2.1349**
- **Median**: **-1.00**
- **Min / Max**: **-1.00 / +325.67**
- **Std dev**: **18.5933**
- **Total profit (sum of P&L)**: **+1,063.20**

Quantiles:
- p10 **-1.00**, p25 **-1.00**, p75 **-1.00**, p90 **-1.00**

Output files:
- `reports/last_week_inverse_90_pnl_fee_2.csv`
- `reports/last_week_inverse_90_summary_fee_2.json`

---

## Inverse 90¢ using **actual opposite-side trades** (fee = 0.0)
This variant uses the **first opposite-side trade after the signal** (first ≥0.90 trade), rather than a synthetic price of `1 − p`.

Counts:
- Target slugs: **4,079**
- Yes/No resolutions in target set: **562**
- Markets with signal (first ≥0.90): **498**
- Markets with actual opposite trade after signal: **449**
- Trades evaluated: **449**

Win/Lose:
- **Win rate**: **8.69%**
- **Lose rate**: **91.31%**

P&L summary (per $1 stake):
- **Mean**: **+1.3903**
- **Median**: **-1.00**
- **Min / Max**: **-1.00 / +332.33**
- **Std dev**: **17.9213**
- **Total profit (sum of P&L)**: **+624.22**

Quantiles:
- p10 **-1.00**, p25 **-1.00**, p75 **-1.00**, p90 **-1.00**

Histogram (counts by bin):
- [-1.0, -0.5]: **410**
- (0.0, 0.1]: **4**
- (0.1, 0.5]: **4**
- (1, 5]: **3**
- (5, 10]: **6**
- (10, 20]: **11**
- (20, 50]: **8**
- (50, 100]: **2**
- (100, 400]: **1**

Output files:
- `reports/last_week_inverse_90_actual_pnl.csv`
- `reports/last_week_inverse_90_actual_summary.json`
