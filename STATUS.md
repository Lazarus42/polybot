# Project status

_Last updated: 2026-06-22_

Research toward an automated Polymarket trading strategy. Two phases so far: an exhaustive
search for edge in the **historical** data (mostly a negative result, but a thorough one), and
a pivot to **live order-book** data + market-making, which is the active thread.

## Phase 1 — historical strategy search (concluded)

Tested every candidate edge on the archived fills/resolutions, each with a walk-forward
backtest and execution-cost modeling. Full detail in `reports/FINDINGS.md`; the one-line
summary: **no scalable, net-of-cost edge exists in this dataset, and we know the specific
reason for each candidate.**

- Objective redesign (Sharpe → profit/risk/idle utility): adopted; Pareto-beat the baseline.
- Idle capital is a *capacity* problem, not a weighting one (~96% of budget idle).
- Slippage model: killed the apparent "free lunch" from sizing up.
- Favorite-longshot bias: real but erased by costs.
- Outcome prediction: at the market-efficiency ceiling (a GBM can't beat the price OOS).
- Structural / basket arb: real edge but penny-capacity; clean partitions are rare.
- Short-horizon / CLV signal: **real** (24h IC ~0.10 on debounced mids, strongest in liquid
  markets) but uncapturable as a *taker* — the edge ≈ the spread. This is what motivated the
  market-making pivot: capture it as a *maker* instead of paying the spread.

## Phase 2 — live order book + market-making (active)

The free Polymarket CLOB serves the live book (WebSocket) but no historical depth, so we
capture it ourselves going forward.

- **Collector** (`scripts/collect_clob_book.py`): subscribes to the CLOB market WebSocket for a
  strategy-tagged set of events (liquid + volatile + multi-outcome, crypto excluded), writes
  gzipped daily-rotated JSONL with a size cap (~3 weeks of full-fidelity L2 in <12 GB on a
  laptop). Writes a readable `manifest_*.json` tagging each token by category, negRisk,
  reward-eligibility, horizon, and the **real maker-reward params** snapshotted from gamma.
- **Book-aware MM backtest** (`scripts/book_mm_backtest.py`): replays captured book+trades,
  quotes at the real touch with queue position, fills on real crossing flow, marks at the real
  mid, and **measures adverse selection directly**. Includes a causal momentum forecast that
  skews quotes one-sided (signal-informed MM) and a **reward term**: `pnl = spread − adverse −
  fees + capture_share × reward_pool`.

### The current thesis

Maker-reward pools are large (~**$2,000/day per market**; quote ≥ `rewardsMinSize` within
`rewardsMaxSpread` of mid). Spread/adverse-selection P&L is cents by comparison, so the MM
question collapses to **"what fraction of the reward pool can we capture while managing
inventory?"** The forecast-skew's job is to stay in the reward band on the safe side. The
backtest reports `breakeven_capture_share` to test this against the real pools.

### Status of the active thread

- Collector redesigned, tagged, tested; **a multi-week capture is running** (or about to be).
- Reward-aware MM backtest is built and unit-tested but **needs the new capture** (the old
  capture is crypto-heavy, calm, ~100 fills, reward-less — not a valid test).

## Phase 3 — durable data collection (EC2 → S3)

The principle: **land raw immutable, transform later.** The collector is dumb and crash-proof
(receive → gzip spool → upload → delete); anything clever (per-market layout, Parquet, dedup)
is a *replayable* batch step over raw — you can't un-lose real-time data.

- **Collector** (`scripts/collect_clob_book.py`): one WebSocket connection, gzip spool files
  rotated every `--rotate-minutes` (closed as `*.jsonl.gz` for the uploader), `--rediscover-minutes`
  re-scans gamma and **dynamically subscribes to newly-listed markets** while **pruning resolved
  ones** (via `market_resolved`) to stay under `--max-subscriptions` (~500, the WS per-connection
  cap). Selection is strategy-bucketed: `rewards / liquid / volatile / basket / longshot`, each
  event tagged with category, negRisk, reward params, horizon. Penny (1-8¢) legs are captured
  even when below the per-event liquidity cap, so **longshot strategies are testable** (with real
  spreads/depth, unlike the historical tape). Recurring ultra-short crypto (`up-or-down`, `-5m-`)
  is always excluded as noise; substantive crypto is kept in raw and filtered at analysis time.
- **Deployment** (`deploy/`): `bootstrap.sh` (EC2 user-data) installs the collector as a systemd
  service + a 15-min cron `upload_to_s3.sh`; `s3-write-policy.json` is the least-privilege IAM
  role; `README.md` has the setup. Free-tier `t3.micro`, same-region S3 (free transfer), ~few $/mo.
- **S3 layout**: `raw/dt=YYYY-MM-DD/*.jsonl.gz` (immutable, market id in every record) +
  `manifests/*.json` (tags + reward params). Per-market / Parquet via later compaction.

### Collect (nearly) all of Polymarket — BUILT

`scripts/collect_all.py` is the full-universe fleet: it paginates gamma for every active market
(~2,100 events → ~13k tokens; gamma caps `limit`=100/page and offset ~2,000, so this is the whole
tradeable universe ranked by volume), shards across ~450-token WebSocket connections (one
`collect_clob_book.py` child per shard, pid-unique spool files). **Zero-gap, no restarts:** it
re-enumerates every 30 min and adds newly-listed markets *live* via a per-shard add-inbox file
(`--add-file`), while each child prunes its own resolved markets on `market_resolved` and reports
its live set back (`--status-file`) so the launcher knows true occupancy; overflow spins up an extra
shard, and a child is only relaunched if it *crashes*. This matters because resolutions/trades are
point events — a periodic full-restart would lose any that fall in the blackout. Same raw→S3
pipeline. Runs via `deploy/polybot-collect-all.service` on a **`t3.large`** (memory headroom for
~30 child processes; gzip+parse CPU is light). ~$30-60 total for a month of *everything* — which
future-proofs every analysis (cross-event arb, longshots, MM) and removes selection bias.
`--min-liquidity` trims the illiquid mid-priced tail *without* dropping penny longshots (kept
regardless of liquidity). The targeted single-collector mode (free `t3.micro`, ~500 tokens) remains
for a lighter run. `python scripts/collect_all.py --dry-run` previews event/token/shard counts.

## Next steps

1. Let the collector accumulate several days of reward-eligible, non-crypto markets.
2. Run `book_mm_backtest.py` with `--manifest` and read `breakeven_capture_share` vs the real
   reward pools; slice results by the manifest tags (category, horizon, neg_risk) to find which
   market types MM works on — that taxonomy becomes the deployment filter.
3. Refine: credit reward time only when quoting inside the band at min size; model capture
   share from observed competing depth instead of sweeping it.

## Key scripts

| file | purpose |
|---|---|
| `scripts/collect_clob_book.py` | live L2 collector (tagged, gzip, capped) |
| `scripts/book_mm_backtest.py` | book-aware, reward-aware, signal-informed MM backtest |
| `scripts/forward_return_predictability.py` | CLV / short-horizon signal IC harness |
| `scripts/clv_strategy_backtest.py` | liquid-market long-short CLV backtest |
| `scripts/build_event_groups.py` | event grouping + basket-arb opportunity surface |
| `scripts/calibrate_market_impact.py` | causal pre/post market-impact calibration |
| `scripts/estimate_effective_spread.py` | Roll effective-spread estimator |
| `scripts/walk_forward_residual_portfolio.py` | utility-objective portfolio with slippage |

Detailed writeups: `reports/FINDINGS.md`, `reports/OBJECTIVE_REDESIGN_RESULTS.md`,
`reports/CAPACITY_STRATEGY_NOTES.md`. Tests: `tests/` (102 passing).
