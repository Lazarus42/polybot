# Polymarket Paper-Trading — Context Pack for Analysis

_Written 2026-07-07 as a self-contained brief for a fresh model (Fable) to analyze one week of
paper-trading results. It assumes no prior knowledge of the project. Everything needed to interpret
the result files is here._

---

## 1. What this project is

Research toward an **automated Polymarket market-making (MM) strategy**. Two eras:

- **Phase 1 — historical strategy search (concluded, mostly negative).** Every candidate edge
  (favorite-longshot bias, outcome prediction, structural/basket arb, short-horizon/CLV signal) was
  walk-forward backtested with execution-cost modeling. Conclusion: **no scalable, net-of-cost taker
  edge exists in the historical tape, and we know why for each candidate.** The one live thread out
  of it: a real short-horizon CLV signal (24h IC ~0.10 on debounced mids, strongest in liquid
  markets) that is **uncapturable as a taker** because the edge ≈ the spread. That motivated the
  pivot: capture it as a **maker** instead of paying the spread.

- **Phase 2/3 — live order book + market-making (active).** Polymarket's free CLOB serves the live
  book over WebSocket but **no historical depth**, so we capture it ourselves and test MM forward.

## 2. The core thesis being tested

Polymarket pays **maker-reward pools** (dYdX-style quadratic scoring) to liquidity providers who
quote at least `rewards_min_size` within `rewards_max_spread` of mid. Prior backtest finding
(see §7): **spread-capture and adverse-selection P&L are ~cents and roughly efficient; the real,
durable edge is harvesting the reward pools, and it is capacity-limited.** So the whole MM question
collapses to: **what fraction of the reward pool can we capture while keeping inventory/tail risk
controlled?** Everything running now is a variation on that question.

## 3. What is running RIGHT NOW (the paper-trading experiment)

A **real-time paper market-making simulator** (`scripts/paper_sim.py`) has been running on the
collector EC2 box since ~2026-06-30 (≈1 week as of 2026-07-07), landing snapshots to S3.

- **It places NO real orders.** It streams the same live CLOB WebSocket the collector uses, runs
  the *exact same `Quoter`* we backtested (no code drift between backtest and paper), maintains
  **virtual** quotes / inventory / P&L, and accrues **modeled in-band reward** each minute.
- It answers: _"what would our strategies have quoted, filled, and been paid on fresh out-of-sample
  data?"_ — closing every backtest idealization **except the two only real capital can settle:**
  (1) our presence would change the book, and (2) real reward payout needs real resting orders.
- **Service:** `deploy/polybot-paper-sim.service` (systemd, `Restart=always`), set up by
  `deploy/setup_paper_sim.sh`. Runs alongside the raw full-universe collector on the same `t3.large`.

### Deployed run parameters (authoritative — from the service file)

```
--size 0 --capital 5000 --inv-cap-mult 5 --fill-model prorata --max-capture-share 0.10
--min-roc 0.0 --max-roc 0.10 --cull-loss 3.0 --cull-cooldown 12
--quote-latency 0.2 --cancel-on-move 0.01 --max-hold-minutes 120
--rotate-minutes 15 --refresh-minutes 5
```

Budget $5k, pro-rata fills, capture share capped at 10% of a pool, quote latency 0.2s (models
stale-order pickoff), cancel if mid moves >1c, adaptive culling of configs losing >$3.

### The 9 strategy configs running head-to-head

Snapshots are written **per config per token per minute**, so all 9 are compared on identical
markets/timestamps. Definitions (from `CONFIGS` in `paper_sim.py`):

| config | type | what it does |
|---|---|---|
| `neutral` | maker baseline | symmetric quotes at touch, standard selection + culling |
| `neutral_ctl` | control | neutral but `roc_ceil=0` (deploy everywhere) + `no_cull` — isolates what selection/culling buys |
| `predict_skew` | predictive maker | leans quotes with debounced short-horizon momentum (CLV) to stop offering the side about to be run over; inv-skew; take-profit 5c; 10-min max-hold; band [0.10,0.90] w/ liquidation |
| `predict_skew_ctl` | control | predict_skew but deploy-everywhere + no-cull |
| `take_profit_3c` | maker + exit | lock winner at +3c, 10-min bail, band [0.10,0.90] w/ liquidation |
| `take_profit_3c_ctl` | control | take_profit_3c deploy-everywhere + no-cull |
| `take_profit_5c` | maker + exit | same as 3c but +5c take-profit |
| `longshot` | directional buy-and-hold (NOT MM) | buy one clip when mid ∈ [0.01, 0.05], hold to resolution — bet underdogs |
| `tail_favorite` | directional buy-and-hold (NOT MM) | buy one clip when mid ∈ [0.90, 0.99], hold — bet favorites |

The `_ctl` controls are the key experimental design: **each strategy vs. the same strategy with
selection + culling turned off**, on the same markets, tells us whether our market-selection and
adaptive culling actually add value or are noise. `longshot`/`tail_favorite` are directional
sanity checks (does simple buy-and-hold beat MM over the week?).

Universe quoted: reward-eligible, non-crypto tokens from the collector's latest manifest
(`reports/clob_capture/manifest_*.json`), refreshed every 5 min. Ultra-short 5-min crypto
"up/down" markets are excluded as noise.

## 4. Where the results live (S3)

Bucket **`s3://polybot-polymarket-sjgibson`** (region us-east-1 — confirm), IAM role
`polybot-collector-role`. The 15-min uploader cron (`deploy/upload_to_s3.sh`) ships:

- **`paper/dt=YYYY-MM-DD/paper_<host>_<pid>_<epoch>.jsonl.gz`** — the per-minute snapshot rows.
  **This is the paper-trading dataset. It is SMALL.**
- `paper/paper_sim_summary.json` — the sim's own running aggregate (latest snapshot only).
- `raw/dt=.../book_*.jsonl.gz` — the full-universe raw order book (~13–16 GB/day). **Do NOT pull
  this for paper analysis; it is the expensive dataset and unrelated to scoring the strategies.**
- `manifests/manifest_*.json` — token tags + real reward params (category, negRisk,
  reward-eligibility, horizon, `reward_daily_est`, `rewards_min_size`, `rewards_max_spread`).

## 5. Snapshot row schema (`paper/dt=.../*.jsonl.gz`)

One JSON object per line. ~340 bytes uncompressed / ~28 bytes gzipped per row. Fields:

```json
{"t": 1782139589.6, "token": "<asset_id>", "config": "neutral", "size": 200.0,
 "mid": 0.1975, "our_bid": null, "our_ask": null, "inv": 0.0, "marked_pnl": 0.0,
 "reward_cum": 0.0, "reward_step": 0.0, "q_bid_book": 17870724.8, "q_ask_book": 910067.8,
 "n_fills": 0}
```

- `t` epoch seconds · `token` asset id (join key to manifest) · `config` strategy name
- `mid` real mid · `our_bid`/`our_ask` intended virtual quotes (null = standing aside)
- `inv` virtual inventory (shares) · `marked_pnl` mark-to-mid trade P&L
- `reward_cum` / `reward_step` cumulative / per-sample modeled reward
- `q_bid_book`/`q_ask_book` competing in-band depth on each side (drives reward share)
- `n_fills` virtual fills so far

### `paper_sim_summary.json` per-config fields

`reward`, `trade_pnl`, `net`, `liq_now` (cost to liquidate current inventory now),
`net_if_flat` (**honest** net after paying to flatten inventory — the number that matters),
`flatten_cost`, `n_flatten`, `roc_on_budget`. Plus top-level `markets_quoted`,
`capital_deployed_est`, `snapshots`, `best_net`.

## 6. How to read a "good" result (metric guidance for the analysis)

- **`net_if_flat` and `roc_on_budget` are the headline metrics**, not gross `reward`. Reward alone
  ignores the inventory you'd have to dump. The whole tail-risk experiment exists because reward can
  be eaten by the snap-to-0/1 at resolution.
- **Compare each strategy to its `_ctl`.** If `neutral` ≈ `neutral_ctl`, selection/culling is noise.
  If the guarded/predictive variants beat controls on `net_if_flat` (not just gross reward), the
  risk controls are earning their keep.
- **Reward is capacity-limited** — check `roc_on_budget` at $5k, and whether it degrades as more
  capital is implied (prior finding: best ROC at small capital, hard ceiling).
- **Slice by manifest tag** (category / horizon / neg_risk / pool size): prior work says MM works on
  **thin, broad general markets** (breadth) and *not* on deep sports pools (share collapses). Verify.
- **Watch inventory variance and `n_flatten`/`flatten_cost`** — the failure mode is a config that
  looks reward-rich but carries toxic inventory into resolution.
- Everything is still a **modeled** ceiling: no market impact from our presence, reward is modeled
  not paid. Treat cross-config *ranking* as more trustworthy than absolute $ levels.

## 7. Prior findings to test the new results against (from backtest, dt=2026-06-22, ~7h)

- **Classic spread-capture MM ≈ efficient** under realistic pro-rata fills; trading P&L small and
  sign-flips shard to shard. The earlier "$6k neutral" number was a FIFO-fill artifact.
- **Reward harvesting is the real edge, capacity-limited.** General-universe capturable ≈ **$384/day**
  at size 200, spread thin across ~1,150 small pools. Best ROC small: $1k→~2.5%/day, $10k→~1.1%,
  $100k→~0.4%.
- **Breadth > depth.** Many small clips across rich thin pools beat big clips on few.
- **Sports/esports pools ~19x bigger gross but WORSE per dollar** for a small maker — deep contested
  books collapse a small clip's Q_min share. Sports rewards **size**; general rewards **breadth**.
  For ~$5–10k, thin general markets win. "The fat pool headline is a mirage — always the share,
  never the pool."
- **Deep 1–5c longshots: dead on costs** intraday (~29% bid/ask spread); needs hold-to-resolution
  data — which is exactly what `longshot`/`tail_favorite` + a full week now provides.

## 8. Open questions for the week-1 analysis to answer

1. Which config has the best **`net_if_flat` / `roc_on_budget`** over the full week?
2. Do the **risk controls beat their `_ctl` controls** on risk-adjusted net, or is selection noise?
3. Does the backtest's **$384/day-ish, ~1–2%/day at $5k** reward story hold out-of-sample?
4. Does **breadth-beats-depth** and **general-beats-sports** replicate on live data?
5. Did any config carry **toxic inventory into resolutions** (big `flatten_cost` / `net_if_flat` gap)?
6. Do the **directional** bets (`longshot`, `tail_favorite`) beat MM over a full week with real
   resolutions — finally testable now?
7. Which manifest **tags** (category/horizon) is MM net-positive on → that taxonomy becomes the
   real-money deployment filter.

## 9. Key files

| path | purpose |
|---|---|
| `scripts/paper_sim.py` | the live paper MM simulator (what's running) |
| `scripts/quoter.py` | the `Quoter` shared by backtest + paper (no drift) |
| `scripts/reward_model.py` | in-band reward scoring (`side_score`, Q_min share) |
| `scripts/book_mm_backtest.py` | book-aware, reward-aware MM backtest (source of §7) |
| `scripts/analyze_paper_sim.py` | **NEW** — aggregates paper snapshots → compact summary (see ANALYSIS_PLAN.md) |
| `deploy/polybot-paper-sim.service` | the systemd unit + exact run params |
| `reports/MM_REWARD_FINDINGS.md`, `reports/FINDINGS.md` | detailed prior writeups |
| `STATUS.md` | project status (last full update 2026-06-22) |

## 10. Cost guardrails (AWS)

AWS Budgets (~$100/mo) + Cost Anomaly Detection are set. `t3.large` on-demand ~$60/mo; S3 ~$10–15/mo.
**S3 egress is $0.09/GB** — the paper dataset is tiny so this is negligible, but the raw book is
13–16 GB/day so never `sync` `raw/` to a laptop. Terminate the instance when the study is done.
