# In-sample mining digest — frozen window (2026-07-09)

_Hypothesis GENERATION only: all numbers in-sample (--until 1783310400), fills-at-touch-optimistic.
Raw outputs in `reports/mining/` (untracked). Produced by a subagent under the no-forward-data
discipline; nothing here is a validated edge._

## Maker taxonomy (per-category net_if_flat, 7 core makers summed)

Net-positive: tech +$404 (951 pos), politics +$147 (2,429), weather +$70, world +$45, crypto +$33.
Net-negative: sports −$4,927 (1,176), unknown −$4,542 (5,394), soccer −$4,246 (901),
pop-culture −$300, geopolitics −$204, elections −$8.
Consistent within each of neutral / predict_skew / take_profit_3c individually. Replicates
"general beats sports; always the share, never the pool" out-of-backtest.
`predict_skew` beats its ctl +$3,910; `take_profit_3c` beats its ctl +$1,860. `neutral_ctl` "beating"
`neutral` (+$5,195) is flagged as an artifact (one unknown-category cell riding $59.5k open marks).
**Candidate: category allowlist {tech, politics, weather, world} — flips the best maker positive
(predict_skew-style ≈ +$610 in-sample vs −$294 unfiltered).**

## Tail_favorite (90–99¢): kill — no candidate

−$3,387 / −2.8% ROC on $119k cycled (2,118 clips). NO cell with n≥200 clears breakeven. Thin-book
effect is OPPOSITE the longshot's (<10k is the worst big cell). Win-rate fields unreliable for tails
(resolution detection misses YES-finishers below 0.99) — trust roc_pct/total_pnl only.

## Longshot boundary cells

| cell | clips | win | ROC | P&L |
|---|---|---|---|---|
| 4−5c ∧ <10k ∧ horizon {<3d, 3−14d, unknown} | 6,016 | 11.8% | **+30.2%** | **+$15,759** |
| 4−5c ∧ <10k ∧ horizon {14−60d, >60d} | 3,137 | 5.5% | −11.3% | −$1,565 |
| 3−4c ∧ <10k | 2,885 | 3.5% | −38.9% | −$2,372 |
| 4−5c ∧ 10k−100k | 321 | 8.6% | −10.8% | −$421 |

Horizon ≤14d-or-unknown gate is a strict in-sample improvement (21.5%→30.2% ROC); the excluded
long-horizon mass is genuinely negative. 3−4c is dead even thin → sharp cliff at 4¢ (pocket looks
real, not smooth-artifact). `unknown` horizon carries most dollars (+26.8%, n=5,362) — gates must
keep unknown.

## Other patterns

- **Reward-pool gate (strongest new):** no-pool longshots +26.7% ROC (n=2,597) vs reward-eligible
  −24.6% (n=7,250); holds inside the winning cell (none +51.7% n=299 vs <100 −10.7% n=435).
  Mechanism: pools attract makers → tighter penny asks; unpaid books have stale asks.
  Caveat: overlaps the unknown-manifest coverage gap.
- **neg_risk > binary** for longshots (+19.0% vs +5.6%; +60% inside cell but n=332) — soft split,
  don't gate yet.
- **Longshot categories:** sports +19.1% / soccer +18.1% win; politics −19.2%, geopolitics −33%,
  elections −27.9%, crypto −14.3% lose → negative filter (exclude politics-cluster+crypto)
  deployable at n≈4,800. Flag: sports resolved win rate below avg ask; some profit in open marks.
- **Quoted-spread gradient: artifact, do not deploy** (wide spreads = max fills-at-touch optimism).
- **"unknown is profitable" on every manifest axis is ONE correlated phenomenon** (untagged ≈
  non-reward-eligible ≈ thin lazy books), not several edges. Fixing manifest coverage de-confounds.
