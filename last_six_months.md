## Inverse 90¢ Strategy — Last Month in Dataset

**Window definition (last month of dataset)**  
- Latest `closedTime` in `archive/markets.csv`: **2025-10-07T14:45:40Z**  
- Window start: **2025-09-07T14:45:40Z**  
- Binary closed markets in window: **25,128**

### Status
The slug list for this window has been generated:
- `archive/processed/last_month_slugs.txt`

The next steps are **long‑running** (≈25k slugs) and require network access to Gamma:

```bash
# Build a fresh resolutions file for this window
python scripts/build_resolutions.py \
  --slugs archive/processed/last_month_slugs.txt \
  --output reports/last_month_resolutions.csv

# Run inverse 90¢ backtest with 0% fees
python scripts/opposite_90_random_slugs.py \
  --slugs archive/processed/last_month_slugs.txt \
  --resolutions reports/last_month_resolutions.csv \
  --markets archive/markets.csv \
  --trades archive/processed/trades.csv \
  --threshold 0.9 \
  --fee-rate 0.0 \
  --output reports/last_month_inverse_90_pnl.csv \
  --summary reports/last_month_inverse_90_summary.json
```

### Results
**Pending** — requires the resolutions file for the full 25,128‑market window.
