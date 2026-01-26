## Inverse 90¢ Strategy Results

**Definition**: When a market’s first trade at ≥$0.90 occurs, buy the *opposite* outcome at the complementary price (e.g., if a trade prints at $0.95, buy the $0.05 side). Fee rate assumed **0**.

### Sample and coverage
- Universe: 4,000 random resolved binary market slugs from `archive/processed/random_4000_slugs.txt`
- Resolutions file: `polymarket_resolutions.csv`
- Eligible Yes/No resolutions within the 4,000: **2,282**
- Markets with a ≥$0.90 first-hit in trades: **1,894**
- Trades evaluated: **1,894**

### P&L summary (per $1 stake)
- **Win rate**: **8.50%** (161 wins / 1,894)
- **Lose rate**: **91.50%** (1,733 losses / 1,894)
- **Mean P&L**: **+2.38199**
- **Median P&L**: **-1.00**
- **Min / Max P&L**: **-1.00 / +999.00**

### Distribution shape (qualitative)
- **Extremely skewed / heavy‑tailed**: most trades lose the full $1, while a small number of wins pay out very large multiples (because the opposite side is purchased at very low prices like 1–5¢).
- This creates a **negative median** with a **positive mean**, indicating that the average is driven by rare, very large wins.

### Key implication
The inverse strategy behaves like a **lottery‑style tail bet**: low win frequency, large upside when correct. The mean is positive in this sample, but the outcome is highly sensitive to rare extreme wins.

### Output source
Summary statistics were produced from `reports/opposite_90_random_4000_summary.json`.
