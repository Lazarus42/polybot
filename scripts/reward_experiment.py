#!/usr/bin/env python3
"""In-band reward experiment: reconstruct the live book, cache per-minute reward scores, then
sweep our quote size to see how much of the REAL reward pool a two-sided maker captures.

Two stages, so the expensive parse happens ONCE:

  build-cache : parse the raw shard files, maintain each token's full order book (apply
                price_change level updates over book snapshots), and once per minute record a
                compact sample: (t, mid, best_bid, best_ask, q_bids_book, q_asks_book) where the
                q_*_book are the competing book's reward scores (sum of S(v,spread)*size over
                in-band levels >= min_size). Five floats per sample -> tiny, resumable per shard.
  score       : read the cache and, for each quote SIZE in the sweep, credit reward =
                sum over samples of (pool/1440) * capture_share(our touch quotes of that size vs
                the competing book). The size sweep is pure arithmetic on the cache (seconds).

Reward formula lives in reward_model.py (faithful to Polymarket's published scoring, unit-tested
against their worked example). Sampling is tied to activity: a sample is emitted when an event
arrives >= 60s after the last one, so quiet markets are (conservatively) under-credited rather
than earning on a stale book.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reward_model import order_score, side_score  # noqa: E402

SHARD_RE = re.compile(r"_(\d+)_(\d+)\.jsonl\.gz$")
SAMPLE_SECONDS = 60.0


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _ts(raw):
    v = _f(raw)
    if v is None:
        return 0.0
    return v / 1000.0 if v > 1e11 else v


def group_files_by_shard(day_dir: Path) -> dict[str, list[Path]]:
    shards: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for p in sorted(day_dir.glob("*.jsonl.gz")):
        m = SHARD_RE.search(p.name)
        key, epoch = (m.group(1), int(m.group(2))) if m else (p.name, 0)
        shards[key].append((epoch, p))
    return {k: [p for _, p in sorted(v)] for k, v in shards.items()}


def latest_manifest(d: Path) -> Path | None:
    c = sorted(d.glob("manifest_*.json"))
    return c[-1] if c else None


def load_token_meta(manifest: Path | None) -> dict[str, dict]:
    """token_id -> {pool, min_size, v_cents}. v normalized to cents (vals <1 treated as prob)."""
    out: dict[str, dict] = {}
    if manifest and manifest.exists():
        man = json.loads(manifest.read_text())
        for tok, meta in (man.get("token_meta") or {}).items():
            v = float(meta.get("rewards_max_spread") or 0.0)
            v_cents = v if v > 1 else v * 100.0
            out[str(tok)] = {"pool": float(meta.get("reward_daily_est") or 0.0),
                             "min_size": float(meta.get("rewards_min_size") or 0.0),
                             "v_cents": v_cents}
    return out


# ----------------------------- stage 1: build cache -----------------------------

def build_cache(args, shards, token_meta, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    uncached = [pid for pid in shards if not (cache_dir / f"{pid}.jsonl.gz").exists()]
    already = len(shards) - len(uncached)
    todo = uncached[: args.max_shards] if args.max_shards else uncached
    print(f"build-cache: {len(shards)} shards total, {already} already cached, "
          f"building {len(todo)} this run"
          + (f" (capped by --max-shards {args.max_shards})" if args.max_shards else ""), flush=True)
    t_start = time.time()
    for i, pid in enumerate(todo):
        if args.time_budget and (time.time() - t_start) > args.time_budget:
            print(f"time budget hit after {i} shards; re-run to continue.", flush=True)
            break
        bids: dict[str, dict[float, float]] = defaultdict(dict)
        asks: dict[str, dict[float, float]] = defaultdict(dict)
        last_sample: dict[str, float] = {}
        samples: dict[str, list] = defaultdict(list)

        def emit(tok, t):
            meta = token_meta.get(tok)
            if not meta or meta["v_cents"] <= 0:
                return
            bb_levels = [(p, s) for p, s in bids[tok].items() if s > 0]
            ba_levels = [(p, s) for p, s in asks[tok].items() if s > 0]
            mn = [p for p, s in bb_levels if s >= meta["min_size"]]
            mx = [p for p, s in ba_levels if s >= meta["min_size"]]
            if not mn or not mx:
                return
            bb, ba = max(mn), min(mx)
            mid = (bb + ba) / 2.0
            q1 = side_score(bb_levels, mid, meta["v_cents"], meta["min_size"])
            q2 = side_score(ba_levels, mid, meta["v_cents"], meta["min_size"])
            samples[tok].append([round(t, 1), round(mid, 4), round(bb, 4), round(ba, 4),
                                 round(q1, 2), round(q2, 2)])

        for f in shards[pid]:
            with gzip.open(f, "rt") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    et = e.get("event_type") or e.get("type")
                    t = _ts(e.get("timestamp"))
                    if et == "book":
                        tok = str(e.get("asset_id") or "")
                        if tok not in token_meta:
                            continue
                        bids[tok] = {pp: ss for pp, ss in
                                     ((_f(b.get("price")), _f(b.get("size"))) for b in (e.get("bids") or []))
                                     if pp is not None and ss}
                        asks[tok] = {pp: ss for pp, ss in
                                     ((_f(a.get("price")), _f(a.get("size"))) for a in (e.get("asks") or []))
                                     if pp is not None and ss}
                        if t - last_sample.get(tok, -1e9) >= SAMPLE_SECONDS:
                            emit(tok, t); last_sample[tok] = t
                    elif et == "price_change":
                        for pc in (e.get("price_changes") or []):
                            tok = str(pc.get("asset_id") or "")
                            if tok not in token_meta:
                                continue
                            price, size = _f(pc.get("price")), _f(pc.get("size"))
                            side = str(pc.get("side") or "").upper()
                            if price is None or size is None:
                                continue
                            book = bids[tok] if side == "BUY" else asks[tok]
                            if size <= 0:
                                book.pop(price, None)
                            else:
                                book[price] = size
                            if t - last_sample.get(tok, -1e9) >= SAMPLE_SECONDS:
                                emit(tok, t); last_sample[tok] = t
        out = cache_dir / f"{pid}.jsonl.gz"
        with gzip.open(out, "wt") as w:
            for tok, sm in samples.items():
                if sm:
                    w.write(json.dumps({"token": tok, "s": sm}) + "\n")
        print(f"  shard {pid} cached ({i+1}/{len(todo)}): {sum(len(v) for v in samples.values())} "
              f"samples over {len([1 for v in samples.values() if v])} tokens", flush=True)


# ----------------------------- stage 2: score size sweep -----------------------------

def score(args, token_meta, cache_dir: Path) -> None:
    sizes = args.sizes
    # per-size accumulators
    tot_reward = {x: 0.0 for x in sizes}
    tot_share_num = {x: 0.0 for x in sizes}     # share-weighted by sample for a mean
    n_samples = 0
    tot_pool_minutes = 0.0
    per_size_tokens_paid = {x: set() for x in sizes}
    files = sorted(cache_dir.glob("*.jsonl.gz"))
    for f in files:
        with gzip.open(f, "rt") as fh:
            for line in fh:
                rec = json.loads(line)
                tok = rec["token"]; meta = token_meta.get(tok)
                if not meta or meta["pool"] <= 0 or meta["v_cents"] <= 0:
                    continue
                v, mn, pool = meta["v_cents"], meta["min_size"], meta["pool"]
                per_min_pool = pool / 1440.0
                for t, mid, bb, ba, q1b, q2b in rec["s"]:
                    n_samples += 1
                    tot_pool_minutes += per_min_pool
                    for x in sizes:
                        if x < mn:               # below min_incentive_size -> scores nothing
                            continue
                        s_bid = order_score(v, (mid - bb) * 100.0) * x
                        s_ask = order_score(v, (ba - mid) * 100.0) * x
                        # capture share = our Qmin / (book+our) Qmin, two-sided at the touch
                        sh = _share(s_bid, s_ask, q1b, q2b, mid)
                        if sh > 0:
                            tot_reward[x] += per_min_pool * sh
                            tot_share_num[x] += sh
                            per_size_tokens_paid[x].add(tok)
    print(f"\nin-band reward sweep over {n_samples} per-minute samples; "
          f"total pool exposure ${tot_pool_minutes:,.0f}/day-equivalent\n")
    print(f"{'size':>8} {'daily_reward$':>14} {'mean_capture%':>14} {'tokens_paid':>12}")
    for x in sizes:
        mean_sh = (tot_share_num[x] / n_samples * 100) if n_samples else 0.0
        print(f"{x:8.0f} {tot_reward[x]:14.2f} {mean_sh:14.3f} {len(per_size_tokens_paid[x]):12d}")
    out = {"n_samples": n_samples, "pool_day_equiv": round(tot_pool_minutes, 1),
           "by_size": {str(x): {"daily_reward": round(tot_reward[x], 2),
                                "mean_capture_pct": round((tot_share_num[x] / n_samples * 100)
                                                          if n_samples else 0.0, 4),
                                "tokens_paid": len(per_size_tokens_paid[x])} for x in sizes}}
    (cache_dir.parent / "reward_sweep_summary.json").write_text(json.dumps(out, indent=2) + "\n")


def _share(s_bid, s_ask, q1_book, q2_book, mid, c=3.0):
    """Qmin(self)/Qmin(self+book) with our scores already computed (s_bid,s_ask = S*size)."""
    def qm(a, b):
        if 0.10 <= mid <= 0.90:
            return max(min(a, b), max(a / c, b / c))
        return min(a, b)
    our = qm(s_bid, s_ask)
    tot = qm(q1_book + s_bid, q2_book + s_ask)
    return (our / tot) if tot > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["build-cache", "score", "both"])
    ap.add_argument("--day-dir", type=Path, required=True)
    ap.add_argument("--manifest-dir", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--cache-dir", type=Path, default=Path("reports/book_mm/depth_cache"))
    ap.add_argument("--max-shards", type=int, default=0)
    ap.add_argument("--time-budget", type=float, default=0.0)
    ap.add_argument("--sizes", type=float, nargs="+",
                    default=[20, 100, 200, 500, 1000, 2000, 5000])
    args = ap.parse_args()

    manifest = args.manifest or (latest_manifest(args.manifest_dir) if args.manifest_dir else None)
    token_meta = load_token_meta(manifest)
    shards = group_files_by_shard(args.day_dir)
    print(f"manifest: {manifest}; reward-eligible tokens: "
          f"{sum(1 for m in token_meta.values() if m['pool'] > 0)}", flush=True)

    if args.stage in ("build-cache", "both"):
        build_cache(args, shards, token_meta, args.cache_dir)
    if args.stage in ("score", "both"):
        score(args, token_meta, args.cache_dir)


if __name__ == "__main__":
    main()
