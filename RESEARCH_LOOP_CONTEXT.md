# Polymarket Research Loop — Master Context

_Entry point for an automated research loop (read this first, then drill into the linked docs).
Last updated 2026-07-08. Purpose: give a fresh model everything it needs to (a) know what is running
now, (b) know what has already been tried and concluded, (c) propose and evaluate new experiments
without repeating mistakes. This file is an INDEX + current-state + operating manual; the detail
lives in the linked docs, not here._

## 0. The one-paragraph orientation

We are searching for a scalable, net-of-cost automated edge on Polymarket with a ≤ $5k bankroll. The
historical-tape search was largely a **negative result** (no durable taker edge; we know why for each
candidate). The live thread is **market-making / reward-harvesting** plus a **directional longshot**
bet, tested forward with a paper simulator that replays the live order book. The current lead is a
**filtered longshot (buy 4–5¢ underdogs only in thin books)**, under out-of-sample evaluation.

## 1. What is RUNNING right now (2026-07-08)

- **EC2** `t3.large` at `184.72.90.164` (user `ec2-user`, key `polybot-collector.pem`, IAM role
  `polybot-collector-role`). Two systemd services, both active:
  - `polybot-collect-all` — full-universe live CLOB order-book collector (raw L2 → S3).
  - `polybot-paper-sim` — paper market-making simulator (virtual orders, no real money).
- **Paper-sim configs currently live** (`--size 0 --capital 5000 --fill-model prorata`, see
  `deploy/polybot-paper-sim.service`): the original ten — `neutral`, `neutral_ctl`, `predict_skew`,
  `predict_skew_ctl`, `take_profit_3c`, `take_profit_3c_ctl`, `take_profit_5c`, `longshot`
  (control), **`longshot_thin`** (the lead: 4–5¢ entry AND in-band depth <10k, walk-the-book
  fills), `tail_favorite` — plus the **2026-07-09 expansion** (16 configs, operator-authorized
  mid-window redeploy; prereg `reports/prereg/paper-sim-expansion-2026-07-09.md`): 9 longshot
  refinements/probes (`longshot_thin_shortdated/_norewards/_expolitics/_tp15/_tp30/_2x/_5x`,
  `longshot_5_7_thin`, `longshot_thin_5k`), the buy-favorites family (`fav_hold`, `fav_politics`,
  `fav_far`, `fav_politics_far`), and category-gated makers (`predict_skew_gencat`,
  `take_profit_3c_gencat`, `neutral_gencat`). Eval dates: 2026-07-30 (longshot+makers),
  2026-08-09 (favorites). `_ctl` = deploy-everywhere/no-cull control for its strategy.
- **S3** `s3://polybot-polymarket-sjgibson` (us-east-1): `raw/dt=…` (order book, ~13–16 GB/day, best
  effort), `paper/dt=…` (per-minute virtual snapshots — the analysis dataset, small), `manifests/…`
  (token tags + reward params), `paper_sim_summary.json`.
- **Uploader** `deploy/upload_to_s3.sh` via 15-min cron: paper-first, non-fatal per file, flock
  against overlap, parallel (`xargs -P`) raw drain. (It broke twice via a lost exec bit — the exec
  bit is now tracked in git.)
- **Scheduled tasks:** `longshot-paper-healthcheck` (daily, **P&L-blind** pipeline check) and
  `longshot-thin-evaluation-2026-07-22` (one-time pre-registered evaluation — see §4).
- **Forward window:** the sim was restarted 2026-07-08 with walk-the-book fills; the clean
  post-restart window starts at epoch **1783468800** (use `--since 1783468800`). Earlier data is the
  old fills-at-touch regime.

## 2. What we have TRIED (with the reason it did/didn't work) — index

- **Phase 1, historical taker strategies — mostly NEGATIVE, thoroughly.** No scalable net-of-cost
  edge in the archived tape; each candidate's failure reason is documented.
  → `reports/FINDINGS.md`, `reports/OBJECTIVE_REDESIGN_RESULTS.md`, `reports/CAPACITY_STRATEGY_NOTES.md`.
  Older penny/underdog/inverse-90 experiments: `MONTHLY_UNDERDOG_RESULTS.md`, `inverse_90_results.md`,
  `inverse_backtest.md`, `underdog_workflow.md`, `last_week/two_weeks/six_months.md` (early, superseded).
  Key results: favorite-longshot bias real but erased by costs; outcome prediction at the
  efficiency ceiling; structural/basket arb real but penny-capacity; **CLV/short-horizon signal is
  real (24h IC ~0.10 on debounced mids) but uncapturable as a taker (edge ≈ spread)** — this
  motivated the market-making pivot (capture it as a maker).
- **Phase 2, market-making + reward harvesting.** Spread-capture MM ≈ efficient under realistic
  pro-rata fills; the durable edge is **harvesting maker-reward pools, but it is capacity-limited**
  (~$384/day general universe at size 200). Sports pools ~19× bigger gross but WORSE per-dollar for a
  small maker (deep books collapse a small clip's share). "Always the share, never the pool."
  → `reports/MM_REWARD_FINDINGS.md`, project brief `PAPER_TRADING_CONTEXT.md`.
- **Phase 3, live capture + paper sim.** Land raw immutable, transform later; replay the live book to
  test MM/longshot forward. → `PAPER_TRADING_CONTEXT.md`.
- **Longshot segmentation (current lead).** Over a 2-week window the longshot bet was net positive but
  the edge is concentrated: **4–5¢ entries in thin (<10k) books = +$14.2k / +21.5% ROC**; deeper
  pennies (1–3¢) and thicker books lose. Liquidity is an independent filter (4–5¢ loses in 10k–100k
  books). Slippage stress: breakeven ~2.1¢/share, realistic thin-book fills roughly halve the edge →
  plan on ~10–15% ROC, not 21.5%. → `LONGSHOT_STATE_OF_PLAY.md` (has the tables, caveats, and the
  methodology fixes: the per-position P&L bug and ROC normalization).
- **Capacity ceiling of the filtered longshot — CONFIRMED capacity-constrained (2026-07-09,
  pre-registered, analysis-only).** On the frozen pre-restart window, even the optimistic upper
  bound on instantaneously buyable cell inventory (median B_t ≈ $4.8k, marginal vs the $5k
  threshold) ≈ the bankroll; the median eligible market offers ~$3 of in-band ask depth at entry;
  the sim's 1-clip sizing only ever tied up ~$2.1k median. Deployment is a FLOW (~$5.2k/day of new
  entries, ~1.8h median hold), not a stock — the binding constraint is per-market clip size.
  Never quote per-cycle ROC as bankroll return. → `reports/eval/longshot-capacity-ceiling-2026-07-09.md`
  (+ prereg in `reports/prereg/`); forward `size_mult` probes measure marginal slippage directly.

## 3. Tooling the loop uses

- **Pull data cheaply:** only `aws s3 sync` the `paper/` and `manifests/` prefixes; NEVER `raw/`
  (100 GB/week egress). Aggregate before feeding any model. → `ANALYSIS_PLAN.md`.
- **`scripts/analyze_paper_sim.py`** — per-config aggregation (DuckDB): reward, trade P&L,
  `net_if_flat`, ROC/day, per-day/per-category. Flags: `--since/--until`, `--config`.
- **`scripts/longshot_market_analysis.py`** — longshot/holder segmentation (pure stdlib, parallel).
  Flags: `--config <name>` (analyze any holder config, e.g. `longshot_thin` vs `longshot`),
  `--since/--until` (window), `--keep-entry/--keep-liq/--keep-horizon` (carve a cell),
  `--entry-slippage-c` (slippage stress). Metrics include capital-normalized `roc_pct`.
- **Add a new strategy:** add a config to `CONFIGS` in `scripts/paper_sim.py`, add its name to
  `--configs` in `deploy/polybot-paper-sim.service`, keep the prior version in as a **control**, then
  redeploy: commit+push, on the box `git pull` + reinstall the unit (`sudo bash deploy/setup_paper_sim.sh`,
  which does `cp` + `daemon-reload` + restart — a plain `systemctl restart` will NOT pick up a changed
  `--configs`). Directional bets subclass `Holder`; makers use `Quoter` (both in `scripts/quoter.py`).
- **Fill realism:** the `Holder` walks the ask ladder (partial fills + worse average price in thin
  books). Don't assume fills at touch.

## 4. Research discipline (guardrails — the loop MUST follow these)

- **Pre-register every evaluation before looking at P&L.** Fix the metric, the minimum resolved-sample
  gate, and the success criterion up front; commit them to a doc; then don't move the goalposts.
  Peeking daily and stopping when it looks good ("optimistic sampling" / optional stopping) is the
  cardinal sin. The live example: `longshot_thin` is evaluated ONCE on **2026-07-22**, needs ≥1,000
  resolved positions, success = post-fill ROC positive and ≥~10% beating the control. Health checks
  between now and then are **P&L-blind**. → `LONGSHOT_STATE_OF_PLAY.md` §7.
- **Always run a control** (`_ctl` / the unfiltered prior version) on the same markets.
- **Model costs honestly:** entry spread + walk-the-book slippage + partial fills. Do NOT compound a
  per-dollar-cycle ROC into a period return — returns are capacity-limited.
- **Manifest coverage caveat:** ~half of longshot positions are `unknown` on manifest-derived axes
  (penny underdogs often aren't reward-eligible, so untagged). Trust snapshot-derived axes (entry
  price, live liquidity); treat category/horizon/pool slices as built on ~half the data.
- **Small-n noise:** ROC amplifies tiny denominators; require hundreds-to-thousands of positions
  before trusting a segment.

### 4a. Pre-registration checklist — MANDATORY before running ANY experiment

The loop may NOT deploy a config, start a run, or draw a conclusion until it has written a filled copy
of this checklist to `reports/prereg/<experiment-id>.md` and confirmed every box. This is a hard gate,
not a suggestion. If any field is blank or any rule is violated, STOP and do not run.

```
Experiment ID:            <kebab-case, unique>
Date registered:          <YYYY-MM-DD, BEFORE any result is seen>
Hypothesis:               <one falsifiable sentence: "X will beat Y on metric M because ...">
Config under test:        <new config name + exact params>
Control:                  <the prior/unfiltered config it is compared against, same markets>
Primary metric:           <e.g. post-fill roc_pct on resolved positions>
Window:                   <--since epoch, and evaluation date>
Minimum-sample gate:      <N resolved positions required before ANY conclusion>
Success criterion:        <pre-set threshold, e.g. roc_pct > 10% AND beats control AND CI lower bound > 0>
Failure / inconclusive:   <what result = FAIL, what = INCONCLUSIVE-and-extend>
Cost model:               <entry spread + walk-the-book slippage assumed; no per-cycle compounding>
No-peek commitment:       <"P&L not examined before the evaluation date"> [ ]
```

Enforcement rules the loop must self-check every cycle:
1. **No P&L before the evaluation date.** Between registration and the eval date, only run P&L-blind
   health checks. Reading `net_if_flat`/heartbeat P&L early invalidates the experiment — start over.
2. **Do not change the criterion after seeing results.** A disappointing number is a result, not a
   reason to move the goalpost.
3. **The gate is on RESOLVED positions**, not elapsed time. Under the gate → extend, never conclude.
4. **Every experiment ships with its control** in the same run on the same markets.
5. **Never touch the in-flight `longshot-thin-evaluation-2026-07-22`** or its data early.
6. **Never place a real trade or move money.** The sim is paper-only and places no real orders; the
   agent must never execute a real order, transfer, or withdrawal.

## 5. Open frontier — candidate experiments for the loop

1. **Out-of-sample validation of 4–5¢ ∧ <10k longshot** (the 2026-07-22 eval). Single most important.
2. ~~**Capacity / concurrent-capital ceiling**~~ — ANSWERED 2026-07-09 (capacity-constrained; see §2
   and `reports/eval/longshot-capacity-ceiling-2026-07-09.md`). Remaining sub-question — realized
   marginal slippage at >1-clip size — handed to the forward `size_mult` probes.
3. **Reward-harvesting MM deployment taxonomy** — which market tags (category/horizon/neg_risk/pool)
   is maker-reward net-positive on? That taxonomy becomes the deployment filter.
4. **Fix manifest coverage** so category/horizon slices aren't half-`unknown`.
5. **Cross-event / basket arb** on neg-risk markets (the full-universe raw capture future-proofs this).
6. **Profit-taking on the longshot tail** (explored + reverted once; revisit under the entry/liquidity
   filter — does banking pops cut variance without gutting total).

_The loop's optimization objective (max risk-adjusted return on ≤$5k, capacity-aware) is set by the
operator; the items above are the current live hypotheses. When proposing a new experiment, state its
hypothesis, its control, its pre-registered success criterion, and its evaluation date BEFORE running._

## 6. Doc map

| doc | what it is |
|---|---|
| `RESEARCH_LOOP_CONTEXT.md` | **this file** — master index + current state + operating rules |
| `PAPER_TRADING_CONTEXT.md` | project brief: sim design, thesis, config defs, snapshot schema |
| `LONGSHOT_STATE_OF_PLAY.md` | current lead: longshot findings, slippage, pre-registered eval |
| `ANALYSIS_PLAN.md` | cheap pull + analyze runbook (egress + token discipline) |
| `reports/FINDINGS.md` | Phase-1 historical strategy search (the negative results) |
| `reports/MM_REWARD_FINDINGS.md` | market-making + reward-harvesting findings |
| `reports/CAPACITY_STRATEGY_NOTES.md`, `reports/OBJECTIVE_REDESIGN_RESULTS.md` | capacity + objective-redesign detail |
| `STATUS.md` | older project status (2026-06-22, pre-paper-sim — partly stale) |
