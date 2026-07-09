# Pre-registration: longshot-capacity-ceiling-2026-07-09

```
Experiment ID:            longshot-capacity-ceiling-2026-07-09
Date registered:          2026-07-09 (BEFORE any capacity metric is computed)
Hypothesis:               The 4-5c AND <10k filtered longshot is capacity-constrained well below the
                          $5k bankroll: the median (over time) dollar in-band ask depth summed across
                          concurrently-open filter-eligible positions is < $5,000, so the bankroll-
                          level return is bounded by cell P&L/day / bankroll and the per-cycle ROC
                          (21.5% headline / ~10-15% post-slippage) cannot be compounded into it.
Config under test:        `longshot` (unfiltered holder), analytically filtered to the cell
                          entry_bucket == 4-5c AND liq_bucket == <10k — the same snapshot-derived
                          definitions as the deployed `longshot_thin` gate. No new config; no deploy;
                          the paper-sim service is NOT touched (protects the in-flight 07-22 eval).
Control:                  The unfiltered `longshot` clip population on the identical window (same
                          metrics reported side-by-side, for scale reference).
Primary metric:           B_t = sum over positions open at time t of (q_ask_book at entry x entry
                          price), i.e. dollar in-band ask depth of concurrently-open eligible
                          markets — an optimistic upper bound on concurrently-buyable inventory.
                          Statistic: median of B_t over the window (evaluated at entry/exit event
                          times, capital-sweep method). Secondary/diagnostic (reported, not gated):
                          (a) A_t = concurrent deployed capital at sim 1-clip sizing (peak/median/
                          mean), (b) daily new-entry capital flow (sum of entry cost per day),
                          (c) holding-time distribution of resolved clips, (d) implied bankroll-level
                          %/day = cell P&L per day / $5,000, headline and with the pre-committed 50%
                          slippage haircut from LONGSHOT_STATE_OF_PLAY.md §4a.
Window:                   All local paper data from dt=2026-06-23 through --until 1783310400
                          (Mon 2026-07-06 00:00 EDT) — the SAME frozen, already-published in-sample
                          window as LONGSHOT_STATE_OF_PLAY.md. No data at/after 1783310400 is read.
                          Evaluation date: 2026-07-09 (immediate — the window is historical and
                          fixed, so there is no optional-stopping degree of freedom).
Minimum-sample gate:      >= 1,000 filter-eligible clips with a valid entry row in the window
                          (prior published run had 9,153; if the gate fails something is wrong with
                          the pipeline — stop and diagnose, do not conclude).
Success criterion:        HYPOTHESIS CONFIRMED if median(B_t) < $5,000.
Failure / inconclusive:   REFUTED if median(B_t) >= $5,000 (the bankroll could in principle be
                          deployed into in-band depth; capacity is not the binding constraint).
                          INCONCLUSIVE if the sample gate fails or >20% of eligible clips lack a
                          usable q_ask_book at entry (proxy too patchy to trust) -> fix data, rerun.
Cost model:               This window is the fills-at-touch regime: all P&L figures are optimistic
                          and are reported with the pre-committed 50% haircut alongside the headline.
                          B_t itself deliberately ignores walk-the-book price impact, which makes it
                          an UPPER bound — this strengthens (never weakens) a CONFIRMED verdict and
                          is the conservative direction for the hypothesis being tested. No
                          per-cycle ROC compounding anywhere.
No-peek commitment:       No snapshot at/after epoch 1783310400 is read; nothing from the post-
                          restart forward window (>= 1783468800) is read; the in-flight
                          longshot-thin-evaluation-2026-07-22 and its data are untouched. The cell's
                          window P&L (+$14,194) is already published in LONGSHOT_STATE_OF_PLAY.md,
                          so no new P&L information is revealed by this analysis. [x]
```

Notes recorded at registration time:

- This is a retrospective **estimation** experiment on an already-examined window. It can NOT count
  as out-of-sample validation of the longshot edge (that is the job of the 07-22 eval). Its output
  is the capacity ceiling and the honest bankroll-level return arithmetic.
- Known caveat on the primary metric's units: `q_ask_book` is the reward-model in-band side score
  (size-weighted shares within the reward band), not raw ladder shares. It is the same quantity the
  <10k liquidity gate is defined on, so the metric is internally consistent with the strategy's own
  filter; treat absolute dollar levels as approximate, order-of-magnitude estimates.
- Measured A_t (1-clip sim sizing) is also an upper bound on deployable capital in one direction
  (walk-the-book partial fills would shrink real fills) and a lower bound in another (a real
  bankroll could buy >1 clip per market). B_t is what bounds the ">1 clip" direction.
