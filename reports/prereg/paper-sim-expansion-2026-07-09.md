# Pre-registration: paper-sim-expansion-2026-07-09 (three experiments, one deploy)

_Registered 2026-07-09, BEFORE deploy and before any forward result exists. Sources of the
hypotheses: `reports/research/mining-insample-2026-07-09.md` (in-sample, frozen window) and
`reports/research/external-evidence-2026-07-09.md` (verified external literature). The operator
explicitly authorized this mid-window redeploy on 2026-07-09 (see restart note at bottom)._

Shared protocol for all three experiments:

```
Date registered:          2026-07-09
Window:                   --since <DEPLOY_EPOCH> (recorded in the addendum below immediately after
                          deploy; all evals use this exact epoch)
Cost model:               Holder entries walk the real ask ladder (partial fills + slippage);
                          maker fills pro-rata; NO per-cycle ROC compounding into period returns;
                          take-profit sells execute at best bid (slightly optimistic — disclosed).
No-peek commitment:       No P&L (net_if_flat, marked_pnl, heartbeat nets, summary) is read for ANY
                          config before that experiment's evaluation date. Daily health checks are
                          P&L-blind (service active, S3 fresh, config names present). [x]
Controls:                 Every arm runs alongside its prior/ungated config on the same markets in
                          the same window: longshot_thin + longshot (exp A), fav_hold +
                          tail_favorite (exp B), predict_skew / take_profit_3c / neutral (exp C).
In-flight eval:           longshot-thin-evaluation-2026-07-22 proceeds unchanged (see note).
```

## Experiment A: longshot gate refinements + capacity probes

```
Experiment ID:            expansion-longshot-2026-07-09
Hypothesis:               Single added gates on the 4-5c AND <10k lead (horizon <=14d-or-unknown;
                          reward pool <=$1/day; exclude politics/geopolitics/elections/crypto;
                          take-profit at 0.15/0.30) each raise post-fill ROC vs longshot_thin,
                          because the in-sample loss mass concentrates in long-dated, maker-paid,
                          politics-cluster pennies; and clip size can rise to 2x with <5pts ROC
                          degradation (capacity), because median in-band ask depth (~$3+) exceeds
                          1-clip cost (~$7) only marginally — 5x should visibly degrade.
Configs under test:       longshot_thin_shortdated, longshot_thin_norewards,
                          longshot_thin_expolitics, longshot_thin_tp15, longshot_thin_tp30,
                          longshot_thin_2x, longshot_thin_5x, longshot_5_7_thin, longshot_thin_5k
                          (exact params in CONFIGS, commit-pinned with this file).
                          AMENDMENT (2026-07-09, same day, BEFORE any forward result was seen):
                          + longshot_thin_0_3d (4-5c, <10k, max_horizon_days=3, unknown passes) —
                          added after the full-tags de-confound showed the ex-ante horizon edge
                          concentrates in <3d (in-cell +39.2% vs 3-14d −5.6%); same metric, gate,
                          control, and 2026-07-30 eval date as the other gate variants. The
                          redeploy re-bases DEPLOY_EPOCH for ALL expansion arms (window was hours
                          old and P&L-unseen; the discarded stub is documented, not analyzed).
Control:                  longshot_thin (and longshot as the family baseline), same window.
Primary metric:           roc_pct on RESOLVED positions (longshot_market_analysis.py --config X
                          --since <DEPLOY_EPOCH>), per config vs longshot_thin.
Evaluation date:          2026-07-30 (one look; no reads before).
Minimum-sample gate:      >= 500 resolved positions per arm (else EXTEND that arm, never conclude).
Success criterion:        Gate variants (shortdated/norewards/expolitics/5k/5_7): PASS if
                          roc_pct > 0 AND >= longshot_thin + 5pts. TP variants: PASS if
                          roc_pct >= longshot_thin (banking pops must not cost total return);
                          variance reduction is reported, not gated. Capacity probes: PASS-capacity
                          if 2x roc_pct >= longshot_thin - 5pts; 5x is an estimand (marginal
                          slippage curve: avg entry premium + fill ratio vs 1x), not pass/fail.
Failure / inconclusive:   FAIL if criterion missed with gate met. INCONCLUSIVE-extend if under
                          sample gate or if longshot_thin itself fails its own 07-22 eval (then
                          the family baseline is dead and variants are moot — record and stop).
```

## Experiment B: buy-favorites family

```
Experiment ID:            expansion-favorites-2026-07-09
Hypothesis:               Buying 55-90c favorites and holding beats buying longshots net of fills
                          (documented favorite-longshot calibration), and the politics-category /
                          >=7d-horizon gated variants beat unfiltered fav_hold, because the
                          documented mispricing concentrates in politics at long horizons.
Configs under test:       fav_hold, fav_politics, fav_far, fav_politics_far (exact params pinned).
Control:                  fav_hold is the family control for the gated variants; tail_favorite
                          (90-99c) is the adjacent-band reference. Cross-family comparison vs
                          longshot_thin reported.
Primary metric:           roc_pct on RESOLVED positions, same tool, same window.
Evaluation date:          2026-08-09 (longer window: long-horizon holds need time to resolve).
Minimum-sample gate:      >= 300 resolved positions per arm (else EXTEND).
Success criterion:        fav_hold PASS if roc_pct > 0 with resolved n over gate. Each gated
                          variant PASS if roc_pct > fav_hold + 3pts. Mark-to-mid CLV of open
                          positions is reported as a secondary diagnostic, never a gate.
Failure / inconclusive:   FAIL if roc_pct <= 0 (fav_hold) or <= fav_hold (variants) at the gate.
                          INCONCLUSIVE-extend if resolutions are too few by 2026-08-09.
```

## Experiment C: maker category taxonomy

```
Experiment ID:            expansion-makers-2026-07-09
Hypothesis:               Gating each core maker to categories {tech, politics, weather, world}
                          flips/raises its net_if_flat vs the ungated original, because maker trade
                          losses concentrate in sports/soccer/untagged books (in-sample: consistent
                          across all three makers).
Configs under test:       predict_skew_gencat, take_profit_3c_gencat, neutral_gencat.
Control:                  predict_skew, take_profit_3c, neutral (identical params, ungated),
                          same window. (_ctl deploy-everywhere controls also still running.)
Primary metric:           net_if_flat per config over the window (analyze_paper_sim.py
                          --since <DEPLOY_EPOCH>), gated vs its ungated control.
Evaluation date:          2026-07-30 (same look as experiment A).
Minimum-sample gate:      >= 14 elapsed config-days AND >= 200 distinct markets quoted per arm.
Success criterion:        PASS per pair if gated net_if_flat > ungated net_if_flat AND > 0.
Failure / inconclusive:   FAIL if gated <= ungated with gates met; INCONCLUSIVE-extend otherwise.
```

## Explicitly NOT deployed (pre-registered negative decisions)

- `tail_favorite_thin` and any tail (90-99c) variant: mining found NO cell clearing breakeven and
  an inverted thin-book effect. tail_favorite keeps running as reference only.
- 3-4c longshot probes: decisively dead in-sample (−38.9% ROC in thin books).
- Spread-bucket gates: in-sample gradient flagged as a fills-at-touch artifact.
- neg_risk gate: promising but n=332 inside the cell — soft split to WATCH in the A eval, not gate.
- Feed-based flow-following: aggressor-side inference documented unreliable (~41% error).

## Restart note (in-flight 07-22 eval)

This deploy restarts polybot-paper-sim mid-window for `longshot-thin-evaluation-2026-07-22`
(operator-authorized 2026-07-09). Effect: open positions at restart are cut and marked at last mid
(median hold is ~2h, so the truncation bite is small); resolved-position counting is unaffected;
the eval window remains --since 1783468800 spanning both pids. The 07-22 eval MUST note this.

## Addendum — deploy record (filled immediately after deploy; no results seen)

- DEPLOY_EPOCH: **1783592987** (2026-07-09 10:29:47 UTC; sim pid 2460829). All expansion evals use
  `--since 1783592987`.
- Deploy verification (P&L-blind, 2026-07-09): service `active`; 0 Tracebacks in the journal at
  +0s and +90s; all 26 configs present in the live snapshot stream (config-name extraction only —
  no P&L fields read). Gated variants show smaller universes than their controls in the expected
  direction (e.g. fav_politics 820 rows vs fav_hold 2,508; longshot_thin_norewards 60 —
  tiny-pool markets are rare), i.e. the gates demonstrably bind.
