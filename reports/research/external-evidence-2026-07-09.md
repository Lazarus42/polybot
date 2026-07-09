# External evidence on prediction-market edges — deep-research digest (2026-07-09)

_Adversarially-verified web research (104 agents; each claim survived 2/3+ refutation votes against
primary sources) commissioned to ground the 2026-07-09 paper-sim strategy expansion. Full cited
output in the session transcript; this digest keeps what changes our design decisions._

## Verified findings (high confidence unless noted)

1. **Favorite-longshot bias: the tradable side is BUYING FAVORITES / fading longshots, not buying
   cheap longshots.** Kalshi buyers of sub-10¢ contracts lose >60% of their money; ≥70¢ contracts
   earn small positive post-fee returns (Burgi/Deng/Whelan, 313,972 Kalshi transactions 2021–2025).
   Calibration on 64.7M Kalshi + 227.6M Polymarket trades (Le 2026): prices compress toward 50% at
   long horizons (slope 0.99 at 0–1h → ~1.32 beyond a month); Polymarket favorites above 55¢
   systematically underpriced (Gupta, t up to 361); politics is the strongest pocket (~5 points of
   mispricing at 70¢ on Polymarket, ~13 on Kalshi at one week out).
2. **The bias is conditional:** horizon- and domain-dependent; vanishes in the final hours; weather
   deviates; the Kalshi bias weakened in 2025. Studies filter to 5–95¢ — **our 4–5¢ thin-book cell
   is outside their sample**, so those results neither confirm nor refute `longshot_thin`; the
   07-22 forward eval remains its test.
3. **Maker edge is real but thin, high-variance, and professionalized:** Kalshi makers +0.83pp
   volume-weighted vs takers (p=0.0055); +2.6% post-fee on 50¢+ contracts with 33% return std dev;
   the maker aggregate went positive only after professional algo makers arrived (post-Oct-2024) —
   a naive small quoter cannot assume the aggregate transfers. Capacity suits small bankrolls.
4. **Polymarket reward mechanics** (quadratic closeness, Q_min two-sidedness c=3.0, strictly
   double-sided outside [0.10,0.90], $1/day floor, pro-rata vs competitor depth) are documented and
   match what `reward_model.py` already implements.
5. **Arb history:** ~$40M extracted Apr-2024→Apr-2025 across YES/NO parity and neg-risk baskets;
   election-era data likely overstates today's set; current frequency is an open question.
6. **Execution/data realities:** public-feed aggressor-side inference is wrong ~41% of the time
   (use on-chain OrderFilled for flow studies); quote lifecycle is off-chain and unrecoverable —
   **maker strategies can only be validated prospectively** (exactly what the paper sim does);
   Polymarket trading is fee-free, so cost drag is spread/slippage only.

## Claims REFUTED in verification (do not build on these)

- Category-level maker-alpha numbers (Finance +0.29pp … UFC +3.43pp) — 0/3 votes.
- Specific Polymarket spread-magnitude claims (both the "1,300–1,800bps longshot spreads" and the
  "median effective half-spread ≈ 0 in liquid markets") — 0/3. **Measure spreads from our own
  books instead.**
- The 1,218%/800-day cross-platform arb backtest — 1/3.
- "negRisk adapter had zero fills in a 2026 sample week" — 0/3.

## Design implications adopted in the 2026-07-09 expansion

- **NEW favorites family (55–90¢ buy-and-hold)** with politics/long-horizon/depth gates; the
  external evidence predicts this beats longshot-buying. Kept head-to-head with `tail_favorite`
  (90–99¢) and the longshot family as cross-family controls.
- Longshot family stays (our own in-sample lead + live 07-22 eval decides it, not the literature).
- Maker taxonomy configs must derive from OUR data (external category numbers refuted).
- No feed-based flow-following configs (aggressor misclassification).
- Neg-risk/parity arb: future monitoring overlay, not a paper-sim config this round.
