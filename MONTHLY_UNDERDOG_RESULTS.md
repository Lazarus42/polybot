# Monthly Underdog Experiments

## Setup

Each calendar month is evaluated as an independent `$5,000` account cohort. The
first and last partial months are flagged and excluded from aggregate statistics.
All entries use the conservative fill model:

- at most 10% of the triggering entry and exit fill;
- one adverse cent at entry and exit;
- Kalshi-style fees, one-cent ticks, and 0.01-contract rounding; and
- if the first TP/SL crossing cannot support the exit, hold until actual close.

Two sizing rules are reported:

1. `one_dollar_until_exhausted`: attempt `$1` on every eligible market until the
   `$5,000` monthly debit cap is reached.
2. `availability_target_5000`: size from the training-estimated eligible opportunity
   count in an attempt to deploy `$5,000`, subject to cash and observed liquidity.

## Strategies

- `ungated_per_level`: optimal training TP/SL for every one-cent entry level.
- `market_gated_per_level`: the same per-level model with category and scheduled
  resolution-horizon gates selected on calibration only.
- `market_price_gated_global`: one global training-selected TP/SL pair, plus a
  calibration-selected entry-price range and the market gates.

The latest 80% cut selected no crypto, a scheduled horizon of at most three days,
a 7.5x take-profit, no stop, and entry prices from 1 to 35 cents for the global model.

## `$1` Until Exhausted

Across the five complete months in the latest holdout:

| Strategy | Profit | Deployed | Mean monthly utilization | Positive months |
|---|---:|---:|---:|---:|
| Ungated per-level | -$534.90 | $7,749.21 | 31.0% | 40% |
| Market-gated per-level | -$286.95 | $2,011.41 | 8.0% | 20% |
| Market + price-gated global | -$239.57 | $935.47 | 3.7% | 20% |

The literal `$1` strategy generally did not exhaust `$5,000`. The ungated version
deployed 75.3% in August 2025 and 98.2% in the partial September cohort, but the
gated strategies deployed substantially less. More deployment coincided with worse
returns in the latest regime.

Latest full-month `$1` realized profit:

| Month | Ungated | Market-gated | Global price-gated |
|---|---:|---:|---:|
| 2025-04 | +$14.87 | +$1.61 | +$0.65 |
| 2025-05 | +$32.16 | -$38.35 | -$56.39 |
| 2025-06 | -$84.35 | -$35.08 | -$32.61 |
| 2025-07 | -$145.16 | -$98.66 | -$88.76 |
| 2025-08 | -$352.42 | -$116.47 | -$62.46 |

## Availability-Sized `$5,000` Attempt

Latest-cut complete-month results:

| Strategy | Profit | Deployed | Mean monthly utilization |
|---|---:|---:|---:|
| Ungated per-level | -$775.29 | $12,674.59 | 50.7% |
| Market-gated per-level | -$929.03 | $5,517.30 | 22.1% |
| Market + price-gated global | -$710.65 | $2,458.54 | 9.8% |

The ungated availability strategy exhausted the full monthly debit cap in August
and the partial September cohort, losing `$333.09` and `$913.19` respectively. Full
deployment is therefore not the missing ingredient in the latest holdout.

## Across Training Cuts

The global price-gated availability model earned `$1,600` over twelve complete months
at the 60% cut and `$735` over nine complete months at the 70% cut, but lost `$711`
over five complete months at the latest 80% cut. The global policy improves some
older comparisons but does not solve the recent regime failure.

## Run

```bash
scripts/run_monthly_underdog_experiments.sh
```

Detailed outputs are in `reports/monthly_underdog_experiments`.
