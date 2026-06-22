# Strategy expansion & alpha-search plan

Goal: generate many candidate strategies, find which carry alpha out-of-sample, then
combine the survivors for the best risk-adjusted return on <= $5K cash.

## 1. What was added

The candidate pool in `scripts/strategy_family_diagnostics.py` grew from 24 to **72**
families, in two batches.

`EXPANDED_RULE_VARIANTS` (31 families) use only the pre-existing entry features (price
band, horizon-to-deadline, recent 24h move/volume/count, entry fill):

- Momentum (trend continuation after an up-move): bands 06-15c / 16-30c / 31-49c,
  plus strong-move, long-horizon, near-deadline, and high-conviction variants.
- Reversion / capitulation (bounce after a down-move): same bands, plus deep-drop,
  long-horizon, near-deadline, and crashed-longshot (01-05c) variants.
- Late-game (entries close to the deadline): 24h and 48h windows by band, favorite
  drift, and late momentum / late capitulation.
- Attention / flow (activity surge) and structural carry (band/horizon exposure).

`FEATURE_RULE_VARIANTS` (17 families) use the **new richer features** now computed in
`attach_recent_features` (`scripts/walk_forward_market_making_oos.py`):
multi-timescale price moves (1h / 6h / 7d), 24h realized volatility, 24h acceleration
(last-24h move minus prior-24h move), and tick-rule order-flow imbalance (net
buy/sell USD over 24h, in [-1, 1]). Families include multi-timescale momentum and
reversion, acceleration up / deceleration reversion, buy/sell-flow pressure, and
high-vol breakout / low-vol carry. The window-feature math is unit-tested
(`compute_window_features`).

All reuse the regime-default exit rules, so no exit changes are needed. The
feature-based families require signals regenerated WITH recent features attached
(i.e. do not pass `--skip-recent-features`).

## Sizing controls (how much money enters each market)

Three layers, from universe to per-trade:

- `--min-entry-fill` (in `strategy_family_diagnostics.py`, now **default $2**, was $10):
  the minimum triggering-fill USD for a market to enter the signal set at all.
  Lowering it admits thinner tail markets; they are then sized down by the two
  controls below rather than excluded outright.
- `--participation-fraction` (replay/tuner/portfolio, default 0.10): caps each trade
  at this fraction of the market's triggering fill. This is the dominant size
  constraint on most trades (binds on ~56% of pool signals).
- `--min-stake` + `--min-stake-fill-fraction` (replay/tuner/portfolio, defaults
  $0.25 and 2% of fill in the portfolio harness): a low, market-dependent floor.
  The floor scales with the market's own liquidity and is capped so it never exceeds
  the intended stake or the participation cap, so it only filters dust / budget-
  starved trades and lets cents flow into thin markets instead of skipping them.

Note: with the old `--min-entry-fill $10`, the min-stake floor was non-binding (the
participation cap already forced >= ~$1). Lowering the entry-fill to $2 is what makes
the market-dependent minimum actually govern sizing in the thinnest markets.

## 2. Generate signals (your environment, needs the archive + duckdb)

The generator reads `strategy_cube.npz` and the fills archive, so run it where the
`.venv` and `archive/processed/underdog_events/` exist:

```bash
source .venv/bin/activate
python scripts/strategy_family_diagnostics.py \
  --output-dir reports/strategy_family_diagnostics
```

This rewrites `strategy_family_signals.csv` with all 55 families.

## 3. Find which have alpha (no leakage)

```bash
python scripts/rank_components_tune.py \
  --signals reports/strategy_family_diagnostics/strategy_family_signals.csv \
  --tune-before 2025-05-01 --min-signals 40 --max-components 20
```

Ranks every `strategy x exit_rule` on **tune-period data only** by a one-sigma lower
confidence bound on per-dollar return (penalizes small/noisy combos), and writes the
eligible pool to `reports/component_ranking/selected_components.json`.

Critically: a high tune-period rank is necessary but not sufficient. The existing
family showed severe alpha decay (whole-family per-dollar return went +0.227 on the
tune era to -0.069 on the 2025-05+ holdout). So validate any survivor with a
walk-forward / rolling re-fit before trusting it, not just the single tune/holdout
split.

## 4. Combine the survivors

`scripts/walk_forward_residual_portfolio.py` does this as a causal walk-forward. Each
month it uses only the trailing window to (1) estimate each candidate's monthly edge,
volatility, and pairwise correlation; (2) greedily select a low-correlation subset of
positive-edge components; (3) inverse-vol risk-weight them, capped per name; then (4)
allocate the budget by those weights and replay the next month out-of-sample under the
participation cap. It reports the strategy against equal-weight-all and single-best
baselines (Sharpe, worst month, drawdown, deployment).

```bash
python scripts/walk_forward_residual_portfolio.py \
  --pool-file reports/component_ranking/selected_components.json \
  --train-months 12 --corr-threshold 0.6 --max-components 8 --max-weight 0.40 \
  --participation-fraction 0.10
```

Outputs in `reports/residual_portfolio/`: `portfolio_summary.csv` (per-mode
risk-adjusted comparison), `portfolio_period_results.csv` (monthly profit/deploy/
drawdown + chosen weights), `portfolio_weights_by_month.csv`. If `--pool-file` is
absent it falls back to all combos with `>= --min-total-signals` signals, so it can
run before ranking.

Tunable knobs: `--corr-threshold` (lower = more aggressive de-correlation),
`--max-components`, `--max-weight` (concentration cap), `--train-months` (adaptation
speed). Caveats: returns are monthly with a short history (~23 OOS months), so Sharpe
estimates are noisy; the fit uses scale-free per-dollar edge while the replay uses
capped dollars; and reported drawdown is within-month (the account resets each month).

## 5. Two unlocks that need more than the current features

These are required for the strategy *kinds* that the current data can't express:

1. Richer momentum/flow features. DONE — `attach_recent_features` now computes
   multi-timescale moves (1h/6h/7d), 24h volatility, acceleration, and tick-rule
   order-flow imbalance, and 12 `FEATURE_RULE_VARIANTS` exploit them. Possible
   further extensions: longer lookbacks (30d), per-trade-size flow (large vs small
   orders), and bid/ask spread proxies.

2. Cross-market meta-strategies. These are currently blocked: `event_cluster_id` is
   stubbed to `market_id`, and the account code notes the source data "has no
   reliable event_id." But `markets.parquet` carries `slug` and `question`, and
   slugs typically share an event prefix. Recovering event groups from slugs would
   unlock the meta layer (e.g. buy the cheapest leg in a multi-outcome event, or
   fade events whose outcome probabilities sum to more than 1).
