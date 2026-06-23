#!/usr/bin/env python3
"""Unified MM backtest: ONE parse -> per config x per resting size, the NET = trading P&L +
in-band reward, both computed at the SAME size with the config's actual quote placement.

This resolves the cross-size inconsistency of running trading P&L at size 20 and reward at size
2000 separately. For each token we, in a single pass:
  * normalize trades+touch quotes (for fills), and
  * reconstruct the live book and sample competing reward-depth once per minute.
Then for each (config, size) we run `simulate_book_mm` at that resting size with reward accrual:
fills (hence spread/adverse) AND reward both reflect the same size and the config's live quoting
(stepping off a side cuts reward via the Q_min rule; skewing away from mid lowers the score).

Inventory cap scales with size (`--inv-cap-mult` clips) so a large resting size isn't throttled
by a tiny fixed cap. Resumable per shard via a checkpoint; aggregate prints the net surface.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from book_mm_backtest import normalize_events, simulate_book_mm  # noqa: E402
from reward_experiment import (SAMPLE_SECONDS, _f, _ts, group_files_by_shard,  # noqa: E402
                               latest_manifest, load_token_meta)
from reward_model import side_score  # noqa: E402


def build_configs(a) -> dict[str, dict]:
    mw, sk = a.momentum_window, a.skew_threshold
    return {
        "neutral": {},
        "raw_momentum": {"momentum_window": mw, "skew_threshold": sk},
        "clv_debounce": {"momentum_window": mw, "skew_threshold": sk, "debounce_trades": a.debounce_trades},
        "clv_full": {"momentum_window": mw, "skew_threshold": sk, "debounce_trades": a.debounce_trades,
                     "inv_skew": a.inv_skew, "vol_window": a.vol_window,
                     "vol_spread_coeff": a.vol_spread_coeff, "tox_threshold": a.tox_threshold,
                     "tox_window": a.tox_window, "tox_cooldown": a.tox_cooldown},
    }


def extract_shard(files, token_meta):
    """One pass over a shard's files -> (events_by_token, samples_by_token).

    events: normalized trade/quote stream (for fills). samples: per-minute reward depth
    [t, mid, bb, ba, q_bid_book, q_ask_book] from the reconstructed book.
    """
    events: dict[str, list] = defaultdict(list)
    bids: dict[str, dict] = defaultdict(dict)
    asks: dict[str, dict] = defaultdict(dict)
    last_sample: dict[str, float] = {}
    samples: dict[str, list] = defaultdict(list)

    def emit(tok, t):
        meta = token_meta.get(tok)
        if not meta or meta["v_cents"] <= 0:
            return
        bb_l = [(p, s) for p, s in bids[tok].items() if s > 0]
        ba_l = [(p, s) for p, s in asks[tok].items() if s > 0]
        mn = [p for p, s in bb_l if s >= meta["min_size"]]
        mx = [p for p, s in ba_l if s >= meta["min_size"]]
        if not mn or not mx:
            return
        bb, ba = max(mn), min(mx)
        mid = (bb + ba) / 2.0
        q1 = side_score(bb_l, mid, meta["v_cents"], meta["min_size"])
        q2 = side_score(ba_l, mid, meta["v_cents"], meta["min_size"])
        samples[tok].append([t, mid, bb, ba, q1, q2])

    for f in files:
        raw = []
        with gzip.open(f, "rt") as fh:
            for line in fh:
                if line.strip():
                    try:
                        raw.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        part = normalize_events(raw)
        for tok, evs in part.items():
            events[tok].extend(evs)
        for e in raw:
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
        del raw
    for tok in events:
        events[tok].sort(key=lambda x: x["t"])
    return events, samples


def load_or_extract(pid, files, token_meta, cache_dir: Path | None):
    """Return (events, samples) for a shard, using a compact on-disk cache if present.

    The cache stores the config/size-INDEPENDENT parsed intermediate (normalized trade/quote
    events + per-minute depth samples) so re-runs that only change sizes/configs/caps never
    re-parse the 4 GB of raw. Cache file: <cache_dir>/<pid>.jsonl.gz, one JSON line per token.
    """
    if cache_dir is not None:
        cf = cache_dir / f"{pid}.jsonl.gz"
        if cf.exists():
            events: dict[str, list] = defaultdict(list)
            samples: dict[str, list] = {}
            with gzip.open(cf, "rt") as fh:
                for line in fh:
                    rec = json.loads(line)
                    events[rec["token"]] = rec["ev"]
                    if rec["s"]:
                        samples[rec["token"]] = rec["s"]
            return events, samples
    events, samples = extract_shard(files, token_meta)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache_dir / f"{pid}.jsonl.gz.tmp"
        with gzip.open(tmp, "wt") as w:
            for tok in set(events) | set(samples):
                w.write(json.dumps({"token": tok, "ev": events.get(tok, []),
                                    "s": samples.get(tok, [])}) + "\n")
        tmp.replace(cache_dir / f"{pid}.jsonl.gz")   # atomic: no half-written cache on crash
    return events, samples


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day-dir", type=Path, required=True)
    ap.add_argument("--manifest-dir", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=Path("reports/book_mm"))
    ap.add_argument("--parse-cache-dir", type=Path, default=Path("reports/book_mm/parse_cache"),
                    help="reusable cache of parsed events+depth (config/size-independent)")
    ap.add_argument("--no-parse-cache", action="store_true", help="disable the parse cache")
    ap.add_argument("--sizes", type=float, nargs="+", default=[200, 1000, 5000])
    ap.add_argument("--inv-cap-mult", type=float, default=5.0, help="inventory cap = mult * size")
    ap.add_argument("--fill-model", choices=["prorata", "fifo"], default="prorata",
                    help="prorata shares crossing flow with competing depth (realistic); fifo = legacy")
    ap.add_argument("--capture-mult", type=float, default=1.0,
                    help="extra fill haircut for queue-jumping/latency (e.g. 0.5)")
    ap.add_argument("--mark-delay-s", type=float, default=60.0)
    ap.add_argument("--max-shards", type=int, default=0)
    ap.add_argument("--time-budget", type=float, default=0.0)
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--per-token-dump", action="store_true",
                    help="write per-market net/trade/reward for every config at --dump-size (reads cache)")
    ap.add_argument("--dump-size", type=float, default=200.0)
    ap.add_argument("--momentum-window", type=float, default=300.0)
    ap.add_argument("--skew-threshold", type=float, default=0.005)
    ap.add_argument("--debounce-trades", type=int, default=10)
    ap.add_argument("--inv-skew", type=float, default=0.01)
    ap.add_argument("--vol-window", type=float, default=300.0)
    ap.add_argument("--vol-spread-coeff", type=float, default=0.5)
    ap.add_argument("--tox-threshold", type=float, default=0.01)
    ap.add_argument("--tox-window", type=float, default=60.0)
    ap.add_argument("--tox-cooldown", type=float, default=60.0)
    args = ap.parse_args()

    manifest = args.manifest or (latest_manifest(args.manifest_dir) if args.manifest_dir else None)
    token_meta = load_token_meta(manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.output_dir / "unified_checkpoint.jsonl"
    cfgs = build_configs(args)
    sizes = args.sizes

    shards = group_files_by_shard(args.day_dir)
    shard_ids = list(shards)
    if args.max_shards:
        shard_ids = shard_ids[: args.max_shards]

    if args.per_token_dump:
        dump_sizes = args.sizes          # write every size in --sizes (for size-vs-breadth analysis)
        cache_dir = None if args.no_parse_cache else args.parse_cache_dir
        out = args.output_dir / "per_token_dump.jsonl"
        n = 0
        with out.open("w") as w:
            for i, pid in enumerate(shard_ids):
                events, samples = load_or_extract(pid, shards[pid], token_meta, cache_dir)
                for tok, evs in events.items():
                    meta = token_meta.get(tok)
                    if not meta:
                        continue
                    n_tr = sum(1 for e in evs if e["type"] == "trade")
                    if n_tr < 5 and not (meta["pool"] > 0 and samples.get(tok)):
                        continue
                    res = {cname: {} for cname in cfgs}
                    for X in dump_sizes:
                        for cname, kw in cfgs.items():
                            r = simulate_book_mm(
                                evs, our_size=X, inventory_cap=X * args.inv_cap_mult,
                                mark_delay_s=args.mark_delay_s, depth_samples=samples.get(tok),
                                reward_pool=meta["pool"], reward_min_size=meta["min_size"],
                                reward_v_cents=meta["v_cents"], fill_model=args.fill_model,
                                capture_mult=args.capture_mult, **kw)
                            res[cname][str(int(X))] = [round(r["pnl"] + r["reward"], 3),
                                                       round(r["pnl"], 3), round(r["reward"], 3)]
                    w.write(json.dumps({"token": tok, "pool": meta["pool"],
                                       "sizes": [int(x) for x in dump_sizes], "results": res}) + "\n")
                    w.flush()
                    n += 1
                    if n % 25 == 0:
                        print(f"  ...{n} markets dumped (shard {pid})", flush=True)
                print(f"  dumped shard {pid} ({i+1}/{len(shard_ids)}), {n} markets so far", flush=True)
        print(f"\nwrote {n} markets to {out} at sizes {[int(x) for x in dump_sizes]}")
        return

    done = set()
    if ckpt.exists():
        done = {json.loads(l)["shard"] for l in ckpt.read_text().splitlines() if l.strip()}
    todo = [s for s in shard_ids if s not in done]

    if not args.aggregate_only:
        print(f"shards {len(shard_ids)} selected, {len(done)} done, {len(todo)} to go; "
              f"configs={list(cfgs)}; sizes={sizes}", flush=True)
        t0 = time.time()
        with ckpt.open("a") as ck:
            for i, pid in enumerate(todo):
                if args.time_budget and time.time() - t0 > args.time_budget:
                    print(f"time budget hit after {i} shards; re-run to continue.", flush=True)
                    break
                cache_dir = None if args.no_parse_cache else args.parse_cache_dir
                events, samples = load_or_extract(pid, shards[pid], token_meta, cache_dir)
                rec = {"shard": pid, "results": {}}
                for cname, kw in cfgs.items():
                    for X in sizes:
                        acc = defaultdict(float)
                        for tok, evs in events.items():
                            meta = token_meta.get(tok)
                            if not meta:
                                continue
                            # process if it has tradeable flow OR is reward-eligible with depth
                            # (reward accrues on resting quotes even with little/no trade flow)
                            n_tr = sum(1 for e in evs if e["type"] == "trade")
                            reward_eligible = meta["pool"] > 0 and samples.get(tok)
                            if n_tr < 5 and not reward_eligible:
                                continue
                            r = simulate_book_mm(
                                evs, our_size=X, inventory_cap=X * args.inv_cap_mult,
                                mark_delay_s=args.mark_delay_s, depth_samples=samples.get(tok),
                                reward_pool=meta["pool"], reward_min_size=meta["min_size"],
                                reward_v_cents=meta["v_cents"],
                                fill_model=args.fill_model, capture_mult=args.capture_mult, **kw)
                            acc["pnl"] += r["pnl"]; acc["reward"] += r["reward"]
                            acc["adverse"] += r["adverse_selection"]; acc["n_fills"] += r["n_fills"]
                            acc["tokens"] += 1
                        rec["results"][f"{cname}@{int(X)}"] = dict(acc)
                ck.write(json.dumps(rec) + "\n"); ck.flush()
                print(f"  shard {pid} ({i+1}/{len(todo)}): "
                      + ", ".join(f"{c}@{int(X)} net={rec['results'][f'{c}@{int(X)}']['pnl']+rec['results'][f'{c}@{int(X)}']['reward']:.0f}"
                                  for c in cfgs for X in sizes[-1:]), flush=True)

    # aggregate
    sums: dict[str, defaultdict] = defaultdict(lambda: defaultdict(float))
    n_done = 0
    for l in ckpt.read_text().splitlines():
        if not l.strip():
            continue
        rec = json.loads(l); n_done += 1
        for key, v in rec["results"].items():
            for k, val in v.items():
                sums[key][k] += val
    surface = {}
    for key, v in sums.items():
        net = v["pnl"] + v["reward"]
        surface[key] = {"trading_pnl": round(v["pnl"], 1), "reward": round(v["reward"], 1),
                        "net": round(net, 1), "adverse_selection": round(v["adverse"], 1),
                        "n_fills": int(v["n_fills"]), "tokens": int(v["tokens"])}
    ranking = sorted(surface, key=lambda k: surface[k]["net"], reverse=True)
    out = {"day_dir": str(args.day_dir), "manifest": str(manifest), "shards_processed": n_done,
           "sizes": sizes, "configs": list(cfgs), "surface": surface,
           "ranking_by_net": ranking, "best": ranking[0] if ranking else None}
    (args.output_dir / "unified_summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\nNET SURFACE (trading P&L + in-band reward), summed over processed shards:")
    print(f"{'config@size':22}{'trading$':>11}{'reward$':>11}{'net$':>11}{'adverse$':>11}{'fills':>9}")
    for key in ranking:
        s = surface[key]
        print(f"{key:22}{s['trading_pnl']:11.0f}{s['reward']:11.0f}{s['net']:11.0f}"
              f"{s['adverse_selection']:11.0f}{s['n_fills']:9d}")
    print(f"\nbest by net: {out['best']}  (shards processed: {n_done})")


if __name__ == "__main__":
    main()
