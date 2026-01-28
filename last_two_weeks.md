## Inverse 90¢ Strategy — Last Two Weeks in Dataset (Actual Opposite Trades, 2% Fee)

**Window definition (last two weeks of dataset)**  
- Latest `closedTime` in `archive/markets.csv`: **2025-10-07T14:45:40Z**  
- Window start: **2025-09-23T14:45:40Z**  
- Binary closed markets in window: **9,539**  
- Slug list: `archive/processed/last_two_weeks_slugs.txt`

### Results (actual opposite‑side trades, fee = 0.02)

Counts:
- Target slugs: **9,535**
- Yes/No resolutions in target set: **3,298**
- Markets with signal (first ≥0.90): **1,483**
- Markets with opposite trade after signal: **1,215**
- Trades evaluated: **1,215**

Win/Lose:
- **Win rate**: **6.91%**
- **Lose rate**: **93.09%**

P&L summary (per $1 stake):
- **Mean**: **+0.5767**
- **Median**: **-1.00**
- **Min / Max**: **-1.00 / +325.67**
- **Std dev**: **12.3246**
- **Total profit (sum of P&L)**: **+700.65**
- **ROI (profit / total stake)**: **57.67%**

Quantiles:
- p10 **-1.00**, p25 **-1.00**, p75 **-1.00**, p90 **-1.00**

Output files:
- `reports/last_two_weeks_inverse_90_actual_pnl_fee_2.csv`
- `reports/last_two_weeks_inverse_90_actual_summary_fee_2.json`
