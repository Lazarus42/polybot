# Week-1 Paper Results — Cheap Pull & Analyze Runbook

_Two kinds of "cost" to avoid: **AWS egress** (pulling GBs out of S3 at $0.09/GB) and **LLM tokens**
(feeding raw rows to Fable). The plan kills both by (1) touching only the tiny `paper/` prefix, never
the raw book, and (2) aggregating to a few-KB summary BEFORE any model sees it. The model reasons
over the summary + `PAPER_TRADING_CONTEXT.md`, not the tape._

## The one thing that keeps costs near zero

The raw order book is **13–16 GB/day** (~100 GB/week). The paper snapshots are **~30–60 MB/day**
(~0.3–0.5 GB/week) because they're 9 configs × quoted tokens × 1/min, ~28 bytes gzipped/row. So:

> **Only ever pull `s3://polybot-polymarket-sjgibson/paper/`. Never `sync` `raw/`.**

Egress on the paper prefix ≈ **a few cents**. Confirm the actual size first:

```bash
aws s3 ls --summarize --human-readable --recursive s3://polybot-polymarket-sjgibson/paper/ | tail -3
```

## Recommended path A — aggregate in-region (zero egress) ⭐

Run the aggregation **on the EC2 box** (S3→EC2 transfer is free, same region), then copy out only
the ~few-KB summary. This is the cheapest and what I'd do.

```bash
# on the t3.large (ssh in with polybot-collector.pem):
cd /home/ec2-user/polymarket_exp
aws s3 sync s3://polybot-polymarket-sjgibson/paper/     data/paper/       # free (in-region)
aws s3 sync s3://polybot-polymarket-sjgibson/manifests/ data/manifests/   # free, tiny
.venv/bin/pip install -q duckdb
.venv/bin/python scripts/analyze_paper_sim.py \
    --paper 'data/paper/dt=*/paper_*.jsonl.gz' \
    --manifest-glob 'data/manifests/manifest_*.json' \
    --out reports/paper_analysis
# copy ONLY the summary back out (KBs -> negligible egress):
aws s3 cp reports/paper_analysis/ s3://polybot-polymarket-sjgibson/paper_analysis/ --recursive
```

Then `aws s3 sync s3://.../paper_analysis/ ./reports/paper_analysis/` to your laptop, or scp it.

## Recommended path B — pull the paper prefix to your laptop

If you'd rather work locally, the paper prefix is small enough that egress is still just cents:

```bash
cd ~/Desktop/polymarket_exp
aws s3 sync s3://polybot-polymarket-sjgibson/paper/     data/paper/       # ~0.3-0.5 GB, ~$0.03-0.05
# tags: prefer the merged artifact (7.6 MB, union of ALL manifests — full coverage) over
# syncing manifests/ (~4 GB of 8.5 MB point-in-time snapshots, each covering ~12% of tokens):
aws s3 cp s3://polybot-polymarket-sjgibson/tags/token_tags.json.gz data/tags/token_tags.json.gz
# then pass --manifest-glob 'data/tags/token_tags.json.gz' to the analysis scripts.
pip install duckdb
python scripts/analyze_paper_sim.py \
    --paper 'data/paper/dt=*/paper_*.jsonl.gz' \
    --manifest-glob 'data/manifests/manifest_*.json' \
    --out reports/paper_analysis
```

`scripts/analyze_paper_sim.py` (new, included) collapses millions of snapshot rows into:

- `reports/paper_analysis/digest.md` — compact, ranked per-config table (**this is what you paste
  to Fable**)
- `by_config.csv` — reward / trade_pnl / net / est_flatten_cost / **net_if_flat** / roc_%/day
- `by_config_day.csv` — daily series (spot the resolution-day tail events)
- `by_config_category.csv` — reward by market category (tests breadth-vs-depth, general-vs-sports)

It sums reward from incremental `reward_step` and takes last `marked_pnl` per run-segment, so it's
correct even though the sim restarted during the week (`Restart=always`).

## Longshot deep-dive — which markets is the bet good in

`longshot` was the top earner but inconsistent. `scripts/longshot_market_analysis.py` (new, **pure
stdlib — no pip needed**, runs anywhere including the EC2 box) slices every longshot clip by
`category`, live in-band `liquidity`, `entry_price`, `horizon`, `neg_risk`, `competitive` score,
`reward_pool`, `spread`, and a `category × liquidity` cross:

```bash
python scripts/longshot_market_analysis.py \
    --paper 'data/paper/dt=*/paper_*.jsonl.gz' \
    --manifest-glob 'data/manifests/manifest_*.json' \
    --out reports/longshot_analysis
```

For each segment it reports clips, resolved count, **`win_rate_resolved`**, `avg_entry`, and a
`beats_breakeven` flag — because a clip bought at price *p* is +EV only if its win rate exceeds *p*
(a win pays ~$1). The "Actionable segments" table at the top of `digest.md` ranks the slices that
clear breakeven with enough clips: that's your candidate deployment filter. This is exactly the
5%→10% lever — find the subpopulation whose hit rate clears its entry price with margin. Caveat:
week 1 may have too few *resolutions* for stable rates; re-run as more markets resolve, and lean on
`win_rate_resolved` (not `win_rate_all`, which counts still-open markets and understates the rate).

## Then hand it to Fable — cheaply

Give Fable exactly three things (total a few KB, so token cost is trivial):

1. `PAPER_TRADING_CONTEXT.md` — the full brief (what's running, config defs, metric guidance,
   prior findings, the 7 open questions).
2. `reports/paper_analysis/digest.md` + its CSVs, and `reports/longshot_analysis/digest.md` +
   its `by_*.csv` — the compact results.
3. The prompt: _"Answer the §8 open questions. Rank configs by `net_if_flat` and `roc_%/day`,
   compare each strategy to its `_ctl`, check whether the backtest's ~1–2%/day-at-$5k reward story
   and breadth-beats-depth replicate, and flag any config carrying toxic inventory into resolution
   (big est_flatten_cost / net_if_flat gap). Then, from the longshot_analysis digest: identify the
   market segments (category, liquidity, entry price, horizon, competitive score) where the longshot
   bet's `win_rate_resolved` clears its `avg_entry` with the most margin and enough clips to trust;
   propose a concrete deployment filter for the longshot bet that would raise its hit rate, and
   estimate the expectancy lift. Recommend one MM config for real capital plus the longshot filter."_

**Do not** paste raw snapshot files into the chat — that's the only thing that would run up token
cost. Everything the model needs to reason is in the aggregates.

## Cost summary

| step | cost |
|---|---|
| Aggregate in-region (path A) | ~$0 egress |
| Pull paper prefix to laptop (path B) | ~$0.03–0.05 egress |
| Accidentally syncing `raw/` | **~$9/week egress — don't** |
| Feeding raw rows to Fable | large token bill — **don't; use the digest** |
| Fable over the digest + context | trivial |

## Housekeeping

The paper sim keeps running and costs ~$60/mo for the `t3.large` (collector + paper share the box).
When you've decided the week's verdict, decide whether to keep collecting or **terminate the instance**
to stop the ~$60/mo. Check AWS Budgets after pulling.
