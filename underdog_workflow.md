# Underdog Bracket Backtests

This workflow tests buying the underdog after allowing a newly traded market time to establish a favorite. It uses actual archived fills and zero fees.

## Prepared data

`archive/processed/underdog_events/fills_sorted.parquet`

- One row per deduplicated fill event
- Sorted by market, timestamp, transaction, side, and price
- Columns: timestamp, market ID, side, price, executed dollars, executed shares, transaction hash, and source-row count

`archive/processed/underdog_events/markets.parquet`

- Resolved binary market metadata
- Includes first trade time, scheduled end, actual close, winning side, and historical volume

The first archived trade is treated as the market open. After `confirmation_minutes`, the first side that actually trades between `entry_min` and `entry_max` is the confirmed underdog. For binary markets, this defaults to 1-49 cents.

## Rebuild the data

```bash
source .venv/bin/activate
python scripts/build_underdog_dataset.py
```

The build scans the full 33 GB CSV archive, deduplicates fills, and writes clustered Parquet. It is a one-time operation.

## Optimize parameters

```bash
source .venv/bin/activate
python scripts/optimize_underdog_bracket.py \
  --confirmation-minutes 5 \
  --entry-delay-minutes 0 \
  --entry-min 0.01 \
  --entry-max 0.49 \
  --take-profits 1.1,1.25,1.5,1.75,2,2.5,3,4,5,7.5,10,15,20 \
  --stop-losses 0,0.01,0.025,0.05,0.075,0.1,0.15,0.25,0.4,0.5,0.6,0.75,0.9
```

`0` disables the stop. A position exits at the first subsequent actual fill crossing either threshold. If neither threshold trades before market close, it settles at the final resolution payout.

The optimizer uses the first 70% of entries chronologically to select the best take-profit and stop for each one-cent entry band. It reports performance on the remaining 30%.

Outputs:

- `reports/underdog_optimization/grid_results.csv`: every parameter combination by entry band and split
- `reports/underdog_optimization/best_by_entry_level.csv`: training-selected parameters and holdout results
- `reports/underdog_optimization/best_by_entry_level.png`: holdout ROI and selected parameters by entry band
- `reports/underdog_optimization/holdout_policy_trades.csv`: per-trade outcomes using each band’s training-selected parameters
- `reports/underdog_optimization/holdout_portfolio_summary.json`: $5,000 training-ROI-weighted holdout evaluation and bootstrap distribution
- `reports/underdog_optimization/summary.json`: run configuration and coverage

Changing confirmation time, entry delay, entry bounds, or either multiplier grid does not require rebuilding the Parquet data. A full sweep takes roughly three minutes on the current machine.

## Kalshi-style execution

The following counterfactual applies Kalshi's standard event-market taker fee, cent ticks, 0.01-contract granularity, and current fee/balance rounding to the archived price paths:

```bash
python scripts/optimize_underdog_bracket.py \
  --fee-coefficient 0.07 \
  --price-tick 0.01 \
  --contract-step 0.01 \
  --initial-capital 5000 \
  --output-dir reports/underdog_optimization_kalshi
```

This models each entry and exit as a single taker fill. It does not assume maker status or special-series fee discounts.

## Robustness tests

Run the complete fast test suite against the Kalshi-adjusted holdout:

```bash
scripts/run_all_underdog_tests.sh
```

Results are written to `reports/underdog_tests`. The suite includes specific calendar
weeks, 10,000 sampled seven-day windows, clustered bootstrap intervals, tail-winner
removal, parameter stability, execution stress, liquidity caps, a chronological cash
and concurrency simulation, simple baselines, market filters, and walk-forward tests.

Every test is also available separately:

```bash
# Particular calendar weeks, or an exact date range.
.venv/bin/python scripts/underdog_test_suite.py periods
.venv/bin/python scripts/underdog_test_suite.py periods \
  --start 2025-09-01 --end 2025-09-08

# Sample seven-day operating periods and report confidence intervals.
.venv/bin/python scripts/underdog_test_suite.py weeks \
  --samples 10000 --seed 123

# Additional falsification tests.
.venv/bin/python scripts/underdog_test_suite.py tail
.venv/bin/python scripts/underdog_test_suite.py bootstrap --samples 10000
.venv/bin/python scripts/underdog_test_suite.py stability
.venv/bin/python scripts/underdog_test_suite.py stress
.venv/bin/python scripts/underdog_test_suite.py liquidity
.venv/bin/python scripts/underdog_test_suite.py bankroll
.venv/bin/python scripts/underdog_test_suite.py baselines
.venv/bin/python scripts/underdog_test_suite.py filter --min-volume 10000
.venv/bin/python scripts/underdog_test_suite.py walk-forward
```

Use `--help` on the main command or any subcommand for all parameters. The random-week
test samples the distinct seven-day windows available in the holdout. Those windows
overlap, so 10,000 bootstrap draws do not represent 10,000 independent historical
weeks. The calendar-week and week-clustered bootstrap reports make the smaller
independent sample size explicit.

The walk-forward command reads `strategy_cube.npz`, produced by the optimizer. It
reselects parameters using only earlier weeks in each fold. The liquidity and bankroll
tests cap fills using the observed dollars in the triggering archived fill; this is a
more conservative capacity model than assuming the full target order always fills.

## Slow strategy sweep

Confirmation and entry-delay changes require rescanning the fill history. Preview or
run the sweep from the command line:

```bash
.venv/bin/python scripts/run_underdog_sweep.py --dry-run
.venv/bin/python scripts/run_underdog_sweep.py \
  --confirmation-minutes 1,5,15,30 \
  --entry-delay-minutes 0,1,5
```

Each configuration gets its own report directory. The combined comparison is written
to `reports/underdog_sweep/sweep_summary.csv`.

## Causal online allocation

Run all weekly allocation families and weight constructions across 50%, 60%, 70%,
and 80% chronological training cuts:

```bash
scripts/run_online_underdog_allocation.sh
```

The runner attempts the `$1` baseline and the budgeted policies with a `$5,000`
weekly budget. By default, each entry is capped at 10% of the archived triggering
fill. Override any option by appending it to the command:

```bash
scripts/run_online_underdog_allocation.sh \
  --cut-fractions 0.65 0.75 0.85 \
  --weekly-budget 1000 \
  --participation 0.05
```

The test uses an inner chronological split for every cut. The early training segment
selects the optimal bracket for each entry-price level. The later training segment
calibrates semantic market weights, level ROI proxies, confidence bounds, opportunity
arrival rates, and thresholds. Both are frozen before the holdout begins.

Holdout opportunities are replayed in timestamp order. Sequential normalization uses
only the current opportunity, remaining budget, elapsed week time, and arrival rates
estimated from training. It never normalizes over the final set of markets that later
appeared during the week. Empty calendar weeks are retained in weekly statistics.

Outputs in `reports/online_underdog_allocation`:

- `training_cuts.csv`: exact fit, calibration, and holdout boundaries
- `weekly_results.csv`: every contiguous holdout week and policy
- `aggregate_results.csv`: profit, ROI, utilization, drawdown, and hit-rate summaries
- `category_results.csv`: deployed capital and profit by semantic market type
- `allocation_comparison.png`: ROI and budget utilization across cuts
- `summary.json`: execution assumptions and causal audit description

Use `--write-trades` when per-order output is needed. It is disabled by default because
the complete grid produces several million trade rows.

## Capital-account simulation

Run train-selected category and scheduled-horizon gates with chronological cash,
open-position, fill-sensitivity, event-count, and capacity accounting:

```bash
scripts/run_realistic_underdog_account.sh
```

See `REALISTIC_UNDERDOG_SIMULATION.md` for the exact market-selection, execution,
look-ahead controls, current results, and data limitations.
