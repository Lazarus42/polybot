# Evaluation: longshot-capacity-ceiling-2026-07-09

_Evaluated 2026-07-09 against `reports/prereg/longshot-capacity-ceiling-2026-07-09.md` (committed
before the run). Window: local paper data dt=2026-06-23 → `--until 1783310400` (Mon 2026-07-06
00:00 EDT), config `longshot`, cell = entry 4–5¢ AND in-band depth <10k. Tool:
`scripts/longshot_capacity_analysis.py`; digest at `reports/longshot_capacity/digest.md`._

## Verdict: **CONFIRMED (capacity-constrained)** — with an honest "marginal" flag

Pre-registered criterion: median over time of B_t (dollar in-band ask depth summed across
concurrently-open filter-eligible positions) < $5,000.

- **median B_t = $4,839 < $5,000 → CONFIRMED** (mean $4,740, p90 $5,985, peak $6,592).
- Minimum-sample gate: 9,054 eligible clips ≥ 1,000. **PASS.**
- Data-validity gate: 0.0% of clips truly lack `q_ask_book` at entry (≤20%). **PASS.** The initially
  alarming 26.9% "missing" were recorded **zeros** — markets whose in-band ask side was genuinely
  empty at entry (the ask sat outside the reward band, common in thin penny books). A recorded zero
  is a real observation contributing $0 to instantaneously buyable depth, not missing data.

### Marginality disclosure (recorded before any doc update)

The margin is thin: the median sits 3.2% under the threshold and the p90 is above it. Two
sensitivities, disclosed rather than acted on (neither was pre-registered, so neither moves the
verdict): (a) treating the 26.9% zero-depth entries as missing and imputing the positive-median
per-market depth (~$3) would add roughly $250 and could flip the median just over $5,000;
(b) B_t ignores own price impact, which biases it UP (toward REFUTED), and is measured in
reward-model score-share units (approximate dollars). The pre-registered reading — B_t is an
**optimistic upper bound** and even that bound ≈ the bankroll — survives both.

## The numbers that matter for allocation

| metric | cell (4–5¢ ∧ <10k) | unfiltered longshot control |
|---|---|---|
| eligible clips (window) | 9,054 (8,200 resolved) | 18,956 (17,962) |
| **B_t $ ask depth, median / p90 / peak** | **$4,839 / $5,985 / $6,592** | $25,528 / $42,201 / $54,640 |
| A_t deployed @1-clip, median / peak | $2,104 / $3,314 | $2,814 / $4,119 |
| per-market in-band ask depth at entry | median ~60 sc-shares (~$3), mean ~308 (~$14); 26.9% exactly 0 | median ~111, mean ~1,020; 13.0% zero |
| new-entry capital flow | ~$5.2k/day (spike $17.3k on 07-03) | ~$6.9k/day |
| holding time (resolved), median / p90 | 1.8h / 38h | 1.7h / 39h |
| window P&L / capital cycled / ROC | $12,000 / $62,260 / 19.3% | $5,359 / $82,214 / 6.5% |
| bankroll-level %/day on $5k, headline → 50% haircut | 23.5% → **11.75%** | 10.5% → 5.25% |

(Cell P&L here is $12,000 on 9,054 clips vs the published +$14,194 on 9,153: this analysis drops
zero-duration clips (l_t ≤ e_t) and its window starts at the first valid entry; same regime, same
direction, slightly tighter clip filter.)

## Interpretation (what CONFIRMED actually means)

1. **The bankroll cannot be deployed as a stock.** Even the *optimistic upper bound* on
   instantaneously buyable cell inventory (~$4.8k median) barely equals the $5k bankroll, and the
   median eligible market offers ~$3 of in-band ask depth at entry. Real walk-the-book deployment
   at size would be a fraction of B_t.
2. **Deployment is a flow, not a stock.** The strategy cycles (~1.8h median hold, ~$5.2k/day of new
   eligible entries at 1-clip sizing), so capital is recycled roughly daily. The binding constraint
   is per-market clip size — scaling beyond a small clip immediately walks the book.
3. **Per-cycle ROC (19–21%) must never be quoted as a bankroll return.** The honest bankroll-level
   arithmetic on this window is cell-P&L/day ÷ $5k ≈ 23.5%/day headline — but this window's fills
   are at-touch with unlimited depth, which is precisely the assumption capacity kills. The
   pre-committed 50% haircut (≈11.8%/day) is still generous for the same reason.
4. **The definitive post-friction number comes from the forward walk-the-book run** (in-flight
   07-22 eval), and the new `size_mult` capacity probes (2×/5× clips) deployed in the strategy
   expansion will measure the marginal slippage curve directly instead of by assumption.

## Status of frontier item 2

Answered on this window: the filter is capacity-constrained (CONFIRMED). Remaining open: the
realized bankroll-level return net of walk-the-book fills at >1-clip size — handed to the forward
capacity probes (`longshot_thin_2x`, `longshot_thin_5x`) in the 2026-07-09 expansion.
