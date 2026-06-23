# EC2 + S3 data collection

An EC2 instance runs the CLOB L2 collector(s) continuously and ships raw data to S3 every
15 minutes. Local disk only ever buffers ~15 min of data.

## Two modes (set `COLLECT_MODE` in bootstrap.sh)

- **`all` — full universe (recommended).** `collect_all.py` paginates gamma for *every* active
  market (~2,100 events → ~13k tokens), shards across N WebSocket connections (~450 each, one
  child process per shard). **Zero-gap:** it re-enumerates every 30 min and adds newly-listed
  markets *live* to a shard's add-inbox — no fleet restart — while each child prunes its own
  resolved markets on `market_resolved`. So nothing is ever dropped to a restart gap (resolutions
  and trades are point events; a 3h full-restart would miss any that fall in the blackout). Captures
  everything (cross-event arb, longshots, MM) so you never re-decide selection.
  **Instance:** the full universe is ~30 child processes → use a **`t3.large`** (2 vCPU / 8 GB) for
  memory headroom; gzip+parse CPU is light, the constraint is RAM across the children. Spot ≈
  $18/mo. (`t3.medium`/4 GB is borderline at 30 shards.) Data ~15-40 GB/day gzipped → ~$15-40/mo S3.
  `--min-liquidity` raises the floor to trim the illiquid mid-priced tail *without* dropping penny
  longshots (those are kept regardless of liquidity), shrinking the shard count to fit a smaller box.
- **`targeted` — single bucketed collector.** `collect_clob_book.py` captures ~500 tokens chosen
  by the strategy buckets (rewards/liquid/volatile/basket/longshot). Fits **free-tier `t3.micro`**.
  Use this if you want it free and don't need the whole universe.

## Design (why it's shaped this way)

- **Land raw, immutable, transform later.** The collector is dumb and crash-proof: receive →
  append to a gzip spool file → (cron) upload to S3 → delete local. Anything clever
  (per-market layout, Parquet, dedup) is a *replayable* batch step over raw — you can't
  un-lose real-time data, so the live component must be as simple as possible.
- **Time-partitioned raw, market id in every record.** `s3://BUCKET/raw/dt=YYYY-MM-DD/*.jsonl.gz`.
  Per-market analysis is a filter on `asset_id`; you can later compact to
  `curated/market=<id>/dt=…/*.parquet` for fast querying. (No per-market split at write time —
  it makes the collector stateful and produces tens of thousands of tiny objects.)
- **Dynamic re-discovery, no restarts.** Both modes add newly-listed markets by *live subscribe*,
  never by restarting: the single collector re-scans gamma every `--rediscover-minutes`; the fleet
  launcher re-enumerates every `--re-enumerate-minutes` and appends new tokens to a shard's
  add-inbox, which the child subscribes live. Resolved markets are pruned per-child on
  `market_resolved`. Net: new markets caught within ~30 min, and no gap that could drop a
  resolution or trade.
- **Cost:** free-tier `t3.micro`/`t4g.micro`; EC2→S3 in the same region is free; S3 storage is
  ~$0.02/GB/mo (gzipped ~1-1.5 GB/day at 500 tokens → a few $/mo). Add an S3 lifecycle rule to
  Glacier `raw/` after 30 days if you want it near-zero long-term.

## S3 layout

```
s3://BUCKET/raw/dt=2026-06-22/book_<host>_<epoch>.jsonl.gz   # immutable raw event stream
s3://BUCKET/manifests/manifest_<stamp>.json                  # per-token tags + reward params
s3://BUCKET/curated/market=<asset_id>/dt=…/part.parquet      # (later) compaction output
```

## One-time setup

1. **Create an S3 bucket** in your region, e.g. `us-east-1`.
2. **IAM role for the instance:** create a role (trusted by EC2) with `s3-write-policy.json`
   attached (replace `YOUR_BUCKET_NAME`). Attach it as the instance profile.
3. **Launch** an instance with Amazon Linux 2023, the IAM role above, and 30 GB gp3:
   `t3.large` for `COLLECT_MODE=all` (full universe, ~30 shards), or free-tier `t3.micro` for
   `targeted`. Paste
   `bootstrap.sh` into **User data** after editing `POLYBOT_BUCKET`, `REPO_URL`, `COLLECT_MODE`
   — or SSH in and run it once as root.

That's it: the collector(s) start under systemd (auto-restart) and the uploader runs from cron.
Verify the fleet before a long run: `python scripts/collect_all.py --dry-run` prints the event
count, token count, and shard plan without capturing.

## Verify / operate

```bash
journalctl -u polybot-collector -f          # collector logs (subscribes, rotations, re-discovery)
tail -f /home/ec2-user/upload.log           # uploader logs
aws s3 ls s3://BUCKET/raw/ --recursive | tail   # data landing in S3
ls -lh /home/ec2-user/polymarket_exp/reports/clob_capture/   # local spool (should stay small)
```

Tunables are in `polybot-collector.service` (`--discover`, `--max-tokens`, `--rotate-minutes`,
`--rediscover-minutes`); `systemctl restart polybot-collector` after editing.

## Paper trading (live forward-test)

`paper_sim.py` runs the SAME `Quoter` as the backtest against the live book — no real orders — and
lands per-minute virtual quote/fill/reward snapshots to S3, so you can watch how the strategy would
actually do on fresh data. It enforces a **capital budget** (default `--capital 5000`, size 200 →
top 25 reward markets by pool), matching a real $5k book.

**One command** (on the already-bootstrapped collector box, as root):

```bash
sudo bash /home/ec2-user/polymarket_exp/deploy/setup_paper_sim.sh
```

That pulls latest code, starts the `polybot-paper-sim` systemd service, and adds the paper spool to
the uploader cron. Snapshots ship to `s3://BUCKET/paper/dt=YYYY-MM-DD/paper_<host>_<epoch>.jsonl.gz`
every 15 min (plus `paper/paper_sim_summary.json`). Operate it like the collector:

```bash
journalctl -u polybot-paper-sim -f                       # virtual quoting/fills/reward
aws s3 ls s3://BUCKET/paper/ --recursive | tail
```

Tunables in `polybot-paper-sim.service`: `--capital`, `--size`, `--config` (clv_full|neutral),
`--fill-model`. It quotes the reward-eligible markets from the collector's latest manifest. Caveats:
paper orders aren't in the book, so modeled reward assumes our presence doesn't dilute the pool and
no *actual* payout accrues — confirm the load-bearing reward-share assumption later with a small
real-capital deployment and `paper_sim.reconcile_rewards()` against the Markets API.

## Pulling data back for analysis

```bash
aws s3 sync s3://BUCKET/raw/dt=2026-06-22/ ./raw_local/
# DuckDB reads gzipped JSONL directly:
#   SELECT event_type, count(*) FROM read_ndjson_auto('raw_local/*.jsonl.gz') GROUP BY 1;
```
The book MM backtest reads these gz files directly (`book_mm_backtest.py --capture <file>.jsonl.gz
--manifest <manifest>.json`).
