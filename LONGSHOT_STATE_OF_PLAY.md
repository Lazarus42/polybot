# Longshot Strategy — State of Play

_Draft, 2026-07-08. Covers the paper-trading longshot bet: where the data pipeline stands, what the
week-plus of paper results say, the methodology fixes made along the way, the recommended deployment
filter, and the caveats needed to read the numbers honestly. Pairs with `PAPER_TRADING_CONTEXT.md`
(project background) and the generated digests in `reports/longshot_analysis/`._

## TL;DR

The paper sim has been running ~2 weeks. Its results never reached S3 because the uploader silently
lost its execute bit around June 25; that is now fixed and the backlog is backfilled. On the
recovered data (capped at Mon 2026-07-06 00:00 EDT to exclude an anomalous overnight win), the
longshot bet made **+$6,822** overall, but the edge is highly concentrated:

> **The money is in 4–5¢ entries in thin (<10k in-band depth) books: +$14,194 at +21.5% ROC across
> 9,153 positions.** Everything cheaper bleeds (1–3¢ lose $2–2.5k each), and even 4–5¢ loses in
> thicker books. Filtering to that one cell roughly **doubles** dollar profit on a smaller capital
> base and roughly **triples** ROC (7.9% → 21.5%).

## 1. Data pipeline state

- **Paper sim:** running on EC2 `t3.large` (`184.72.90.164`), one continuous process lifetime, 9
  configs including `longshot`. Places no real orders; lands per-minute virtual quote/inventory/P&L
  snapshots. Live and healthy.
- **The break:** `deploy/upload_to_s3.sh` lost its execute permission (`Permission denied` on every
  15-min cron run since ~June 25), so **nothing — paper or raw — reached S3 for ~2 weeks.** The data
  was not lost; unuploaded files accumulated locally on the box (~3.5 GB, 1,200+ paper files).
- **The fix:** `chmod +x` on the box, then a parallel backfill (`xargs -P`) of the ~2 weeks of paper
  snapshots into `s3://polybot-polymarket-sjgibson/paper/dt=.../`. Manifests refreshed too.
- **Durable fix (verify done):** commit the exec bit so a redeploy's `git pull` can't strip it again
  — `git update-index --chmod=+x deploy/upload_to_s3.sh`, commit, push, pull on the box.
- **Raw order book:** broken by the same bug; the raw spool self-caps at `--max-local-gb`, so there
  is a ~2-week gap in the raw book in S3 that is not fully recoverable. Not blocking longshot work.
- **Analysis flow:** pull `paper/` + `manifests/` locally (a few cents egress; never sync `raw/`),
  run the scripts on the laptop.

## 2. Analysis tooling

- `scripts/analyze_paper_sim.py` — per-config aggregation (DuckDB): reward, trade P&L, `net_if_flat`,
  ROC/day, per-day and per-category slices.
- `scripts/longshot_market_analysis.py` — the longshot deep-dive (pure stdlib, multiprocessing,
  progress output). Slices by entry price, liquidity, horizon, neg_risk, competitive, pool, spread,
  and the entry×liquidity and category×liquidity crosses. Supports `--until` (time cutoff),
  `--keep-entry/--keep-liq/--keep-horizon` (stack filters to price a specific cell), and reports
  `roc_pct` (capital-normalized).

### Two methodology fixes made this session

1. **Per-position P&L bug (critical).** Originally a "clip" was one position per **(token, file)**.
   But the sim rotates a new file every 15 min while the same position keeps being held, and
   `marked_pnl` is the position's **cumulative** running P&L — so summing the last value per file
   multiplied a position's P&L by the number of files it spanned (~150×). That produced a nonsense
   **−$228,034** total and 670,644 "clips". Fixed by making the unit one position per **(token,
   process-pid)** lifetime and taking the *last* cumulative `marked_pnl`, never the per-file sum. The
   corrected total (+$6,822, ~19k positions) reconciles with the EC2 heartbeat's positive net. The
   same fix was applied to `analyze_paper_sim.py`.
2. **Capital normalization (ROC).** A 4–5¢ clip ties up ~3× the capital of a 1–2¢ clip, so raw
   dollars-per-clip structurally flatters pricier entries. Added `roc_pct = P&L ÷ capital deployed`
   and rank segments by it — the allocation-relevant metric for a fixed budget.

### Window / cutoff

All figures below use `--until 1783310400` = **Mon 2026-07-06 00:00 EDT**, deliberately excluding an
anomalous overnight longshot win, for a clean read through end of Sunday.

## 3. Findings

**Overall (through cutoff):** 19,150 positions · 6.0% hit vs 5.6% breakeven · **+$6,822** · 7.9% ROC
(per dollar-cycle — see caveats).

**Entry price is the master filter** — the deep pennies destroy capital:

| entry band | positions | hit | ROC | total P&L |
|---|---|---|---|---|
| 4–5¢ | 9,506 | 9.5% | **+20.1%** | **+$14,178** |
| 3–4¢ | 3,041 | 3.6% | −36.2% | −$2,611 |
| 2–3¢ | 3,767 | 2.5% | −44.2% | −$2,540 |
| 1–2¢ | 2,836 | 2.4% | −70.6% | −$2,204 |

**Liquidity is a real second filter, not just where volume sits.** Within the 4–5¢ band, thickness
flips the sign:

| cell | positions | hit | ROC | total P&L | trust |
|---|---|---|---|---|---|
| 4–5¢ / <10k | 9,153 | 9.5% | **+21.5%** | **+$14,194** | high (large n) |
| 4–5¢ / 10k–100k | 321 | 8.6% | −10.8% | −$421 | ok (loses) |
| 4–5¢ / 100k–1M | 30 | 16.7% | +113% | +$414 | **noise (n=30)** |

Same entry band, thicker book, loses money — so thin liquidity carries independent signal.

**Horizon looks like a viable third filter (suggestive).** Short-dated is strong (3–14d +57.6% ROC,
<3d +19.9%), long-dated bleeds (>60d −20.6%, 14–60d −25.7%). Caveat: ~half of positions have an
`unknown` horizon (manifest gap) and those are profitable (+13.7% ROC), so a hard short-horizon
filter would discard real profit for a data-coverage reason — keep `unknown` if filtering on horizon.

## 4. Recommended deployment filter

**Buy 1-clip longshots only where entry is 4–5¢ AND in-band book depth is <10k.** On the window that
is **+$14,194 at +21.5% ROC over 9,153 positions** — ~2× the unfiltered dollar profit on less
capital. Horizon (avoid >14d) is a plausible refinement to test, not yet a committed filter.

Implemented as the `longshot_thin` config (`buy_lo=0.04, buy_hi=0.05, max_book_depth=10000`) running
head-to-head against the unfiltered `longshot` **control**. The `<10k` gate uses the same live
`side_score` depth the snapshot logs, so the deployed gate == the analysis definition.

### 4a. Slippage fragility (important)

The headline +21.5% assumes fills at the touch ask with unlimited depth. A stress test charging extra
entry slippage (the depth-walk the historical fills-at-touch never paid) shows a linear decay with
**breakeven at ~2.1¢/share** (~676k shares bought total):

| extra slippage | P&L | ROC |
|---|---|---|
| 0.0¢/share | $14,194 | 21.5% |
| 0.5¢/share | $10,814 | 16.4% |
| 1.0¢/share | $7,433 | 11.2% |
| 1.5¢/share | $4,053 | 6.1% |
| 2.0¢/share | $673 | 1.0% |

Read: the edge is **real but soft**. It doesn't die at trivial slippage, but a realistic thin-book
fill cost of 0.5–1.5¢ roughly **halves** it, so **~10–15% ROC is the honest planning number**, not
21.5%. And 1–2¢ is 20–40% of a ≤5¢ entry price — a large haircut, and exactly what walking a thin
(<10k) penny book to fill an ~80-share clip can cost. The compounding story is dead: after fill costs
you are not recycling at 21.5%, and partial fills shrink deployable size further.

**The definitive number will come from the forward walk-the-book data, not this parametric knob.** The
Holder now fills by walking the real ask ladder (worse average price + partial fills in thin books;
deep books unchanged), so `longshot_thin`'s live P&L vs the `longshot` control measures the true
post-friction edge with no assumption. Let both run ~a week and compare.

## 5. How to read the numbers (caveats)

- **ROC is per dollar-cycle, not period return on bankroll.** The "$66,100 capital deployed" for the
  4–5¢/<10k cell is the *sum of entry cost across all 9,153 positions over two weeks*, not capital
  held at once. Only ~$2.6k was deployed concurrently (EC2 heartbeat); it recycled ~25× as markets
  resolved. So the period return on standing capital is far higher than 21.5% — but capacity-limited
  by how much thin 4–5¢ inventory exists to buy at any moment. Use ROC to *compare* segments, not as
  the two-week return on the budget.
- **Modeled optimism (historical data).** The entry spread *is* included (the sim debited the real
  ask), and resolved positions settle at exactly 0/1. But the historical fills were **at the touch
  with unlimited depth** — no depth-walk slippage, no partial fills, no own-market impact — which is
  the main optimism and it is worst in the thin books the strategy selects (see §4a). Open positions
  are also marked at mid (slightly generous). `beats_breakeven` is mid-based and lenient — trust
  `roc_pct` and `avg_pnl_per_clip`. **Forward runs fix the fill side:** the Holder now walks the ask
  ladder, so new `longshot`/`longshot_thin` data carries real slippage and partial fills.
- **Manifest coverage gap.** ~50% of positions are `unknown` on every manifest-derived axis
  (category, horizon, pool, spread) — penny underdogs often aren't reward-eligible and so are never
  tagged. **Entry price and liquidity come straight from the snapshot and are fully reliable;** the
  manifest-axis slices are built on ~half the data and are shakier.
- **Small-n noise.** `min-clips` is only 10, which is too permissive for a ROC ranking — tiny cells
  (n<~200) top the list spuriously (e.g. `4–5¢/100k–1M` at +113% on 30 clips, `sp-500` at +747% on
  1). Trust the hundreds-to-thousands-position cells.
- **Single in-sample window.** One ~2-week window through one cutoff. Every stacked filter is chosen
  *because* it looked good on this same data, so apparent edge inflates as filters pile on. The
  two-filter cell (9,153 positions) is the defensible one; treat anything tighter as suggestive.

## 6. Open questions / next steps

1. **Out-of-sample check.** Re-run on fresh weeks as they accumulate — does 4–5¢ ∧ <10k hold up, or
   was it a two-week artifact? This is the single most important validation.
2. **Bankroll-level return + capacity.** ANSWERED 2026-07-09 (pre-registered): CONFIRMED
   capacity-constrained — median $ in-band ask depth across concurrently-open cell markets ≈ $4.8k
   (≈ the bankroll, and that is an optimistic upper bound); sim 1-clip footprint ~$2.1k median;
   deployment is a ~$5.2k/day flow with ~1.8h median holds, bound by per-market clip size (median
   eligible market: ~$3 of in-band ask depth at entry).
   → `reports/eval/longshot-capacity-ceiling-2026-07-09.md`. Marginal slippage at >1-clip size:
   forward `size_mult` probes (2026-07-09 expansion).
3. **Fix manifest coverage** so category/horizon slices aren't half-`unknown` (capture manifests for
   non-reward penny markets, or revisit token matching).
4. **Profit-taking revisited?** Earlier explored and reverted; may still be worth testing whether
   banking pops trims the tail once the entry/liquidity filter is in place.
5. **Hand to Fable:** `PAPER_TRADING_CONTEXT.md` + `reports/longshot_analysis/digest.md` (+ CSVs) for
   independent interpretation and a deployment recommendation.

## 7. Evaluation plan — PRE-REGISTERED 2026-07-08 (do not change after seeing results)

To avoid optimistic sampling (peeking daily and concluding the first time it looks good), the forward
test of `longshot_thin` (real walk-the-book fills) vs the `longshot` control is committed **now**,
before any post-restart results exist.

- **Clean window:** starts 2026-07-08 (post-restart, walk-the-book fills, fresh pid). Earlier data is
  the old no-slippage regime and is excluded via `--since 1783468800`.
- **One evaluation date:** **2026-07-22** (2 weeks). No P&L is examined before then.
- **Minimum-sample gate:** ≥ 1,000 **resolved** `longshot_thin` positions before any conclusion. If
  short on the date, **extend** — never conclude on thin data.
- **Success criterion:** `longshot_thin` post-fill ROC positive and materially so (target ≥ ~10%, the
  honest half of the historical 21.5%), beating the `longshot` control on the same window, with the
  win count high enough to be confident it's above 0. Positive-but-marginal = INCONCLUSIVE → extend.
- **Between now and then:** a **daily P&L-blind health check** only confirms the sim is alive, S3 is
  fresh, and `longshot_thin` is running — it reports no returns, so it can't bias the decision.
- **Automation:** two scheduled tasks — `longshot-paper-healthcheck` (daily) and
  `longshot-thin-evaluation-2026-07-22` (one-time, fires on the eval date, writes
  `reports/EVAL_2026-07-22.md`).
- **Tooling:** `longshot_market_analysis.py` now takes `--since`/`--until` (window) and `--config`
  (analyze `longshot_thin` vs `longshot`), so the eval runs both arms on the identical clean window.
