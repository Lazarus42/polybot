# Realistic Underdog Simulation

## What is being simulated

The source data is historical Polymarket trade execution data for resolved binary
markets. The execution model is a counterfactual application of Kalshi's standard
event-contract fee formula, one-cent price ticks, and 0.01-contract order increments.
It is not a backtest on historical Kalshi order books.

The 1.8 GB deduplicated fill-level Parquet is published as a GitHub Release asset:

```bash
curl -L \
  https://github.com/Lazarus42/polybot/releases/download/underdog-data-2026-06-19/fills_sorted.parquet \
  -o archive/processed/underdog_events/fills_sorted.parquet
```

SHA-256: `2a5dfe36121b3352d8387f68d4892fec879762029eabcc82ce37d411f86fb0a6`.
It is a release asset rather than a Git object because GitHub blocks regular files
over 100 MiB and the repository owner's Git LFS budget is exhausted.

For each market, the candidate underdog is identified causally:

1. Treat the first archived trade as the market opening time.
2. Wait five minutes.
3. The first side trading between 1 and 49 cents becomes the confirmed underdog.
4. Enter at that side's first eligible archived fill.

Only the price, timestamp, side, and triggering fill size visible at that entry are
used for the decision. Final historical volume is not used for sizing or eligibility.

## How bracket parameters are selected

Every chronological training cut is split again:

- The early fit segment chooses a take-profit and stop-loss pair independently for
  each one-cent entry-price level.
- The later calibration segment chooses market-category and scheduled-resolution
  horizon gates.
- All selected parameters are frozen before holdout replay begins.

The tested scheduled horizon gates are 1, 3, 7, 14, 30, 60, 90, 180 days, and no
limit. The horizon is computed from the market's scheduled `end_date`, not its actual
future close. The archive does not retain metadata revisions, so this assumes the
stored scheduled end date was visible and unchanged at entry.

Category candidates include all markets, no crypto, sports plus uncategorized,
training-positive lower-confidence-bound categories, and each individual category.
The calibration score is realized profit minus penalties for dollar-days locked and
capital still open at the calibration boundary. Future outcomes of open positions do
not enter the score.

## How trades and the account are replayed

Holdout entries are processed in timestamp order. Before each new entry, exits whose
timestamps have arrived release their proceeds into cash. A position cannot reuse its
capital before its simulated exit.

The account tracks:

- available cash;
- open positions and locked cost basis;
- realized profit;
- account value;
- fees;
- dollar-days locked;
- drawdown;
- weekly deployment; and
- category and scheduled-horizon attribution.

TP/SL exits use the first later archived fill crossing the selected level. If the
triggering exit fill cannot support the position under the selected participation
limit, the conservative simulator does not claim that fill and holds the position to
the actual market close. Actual close and resolution values affect the account only
when simulated time reaches the close.

Positions open at the holdout boundary are marked at cost because the archive lacks
reliable point-in-time bid/ask snapshots at arbitrary evaluation times. Their eventual
historical outcomes are reported only in an audit field and are excluded from realized
profit and account value.

Fill scenarios are:

| Scenario | Fill participation | Entry slippage | Exit slippage |
|---|---:|---:|---:|
| Optimistic | 100% | 0 ticks | 0 ticks |
| Neutral | 25% | 0 ticks | 0 ticks |
| Conservative | 10% | 1 adverse tick | 1 adverse tick |
| Very conservative | 5% | 2 adverse ticks | 2 adverse ticks |

For conservative TP/SL exits, the observed threshold-crossing fill must support the
position at the 10% participation limit. Otherwise that exit is rejected and the
position is held to actual close. This is stricter than assuming a price touch fills
the complete order, but historical order-book queue and spread snapshots remain
unavailable.

## Market and budget experiments

The simulator runs three chronological holdouts beginning at the 60%, 70%, and 80%
history cuts. For each cut it runs:

- the train-selected category/horizon `$1` baseline under all four fill scenarios;
- an ungated `$1` conservative baseline;
- first-N causal market exposure at 100, 250, 500, 1,000, 2,500, and 5,000 markets;
- availability-adjusted sizing at weekly budget targets from $50 through $5,000.

Availability-adjusted stake is the weekly target divided by the expected eligible
opportunity count learned from calibration. It does not normalize using markets that
arrive later in the holdout week.

## Current results

The train-selected gates were:

| Cut | Calibration-selected gate |
|---|---|
| 60% | Uncategorized, scheduled horizon at most 180 days |
| 70% | Sports plus uncategorized, scheduled horizon at most 7 days |
| 80% | No crypto, scheduled horizon at most 3 days |

Under conservative fills, realized account profit was:

| Cut | Selected gate | Ungated `$1` |
|---|---:|---:|
| 60% | +$68 | -$154 |
| 70% | -$213 | -$863 |
| 80% | -$375 | -$1,357 |

The latest-cut selected strategy made +$564 under optimistic fills, -$68 under neutral
fills, -$375 under conservative fills, and -$388 under very conservative fills. The
edge therefore does not survive the current conservative execution model.

In the latest capacity frontier, realized profit declined from -$69 at a $50 weekly
target to -$1,942 at a $5,000 target. In the latest first-N experiment, the first 100
and 250 positions were modestly positive, but 500 were approximately flat and 5,000
lost $284. The current data does not support the hypothesis that simply adding more
market positions reliably captures a profitable tail-event edge.

## Remaining limitations

The largest unresolved issue is source-universe selection. The compact Parquet was
built from markets with known resolutions, while the full market metadata file is
larger. Consequently, zero open positions at the final boundary is not evidence that a
live bot would have no unresolved exposure. A true resolved-selection-bias audit
requires rebuilding candidates from all markets visible at each historical timestamp.

The archive also lacks a reliable event ID, historical order-book depth, queue
position, spread snapshots, and metadata revision history. The event-count experiments
therefore count market positions, not independent event clusters. Results should not
be described as production-ready until those gaps are addressed.

## Commands

Rebuild the enriched compact strategy cube:

```bash
.venv/bin/python scripts/optimize_underdog_bracket.py \
  --fee-coefficient 0.07 \
  --price-tick 0.01 \
  --contract-step 0.01 \
  --initial-capital 5000 \
  --output-dir reports/underdog_optimization_kalshi
```

Run the capital-account experiments:

```bash
scripts/run_realistic_underdog_account.sh
```

Run independent `$5,000` monthly cohorts, literal `$1`-until-exhausted sizing, market
gates, and the global price-range strategy:

```bash
scripts/run_monthly_underdog_experiments.sh
```

See `MONTHLY_UNDERDOG_RESULTS.md` for the monthly results.
