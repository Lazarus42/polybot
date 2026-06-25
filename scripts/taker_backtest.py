#!/usr/bin/env python3
"""Taker / pickoff backtest: the OTHER side of the maker's adverse selection.

The maker bleeds because informed flow runs it over. The taker tries to BE that flow: when an
aggressive trade sweeps the book (a signal that fair value just moved), fire a marketable order in
the SAME direction, paying the spread to cross, and bank the continuation. If the maker's loss is
real, the taker's gain should be too — net of the crossing cost (Polymarket taker fees are 0 in
the captured tape, so the only cost is the half-spread paid on entry, plus whatever latency erodes).

Signal: follow a trade whose size >= follow_size in the aggressor's direction (a SELL pushes price
down -> we SELL; a BUY pushes up -> we BUY). One position per asset at a time (cooldown = hold).

Entry: cross the touch after `latency` seconds (buy at best_ask / sell at best_bid). Modeling
latency as a delay means a fast move can leave us a worse entry — the realistic pickoff tax.
Exit:  mark out at the mid `hold` seconds later (the taker holds briefly then is flat).
PnL/share = sign*(mid(t_entry+hold) - entry_cross_price).  We also report the gross drift vs the
mid AT signal time, to separate "did price move" from "did we overcome the spread".

Pure core (`taker_pnl`) is unit-tested; the tape loader streams via `gzip -dc` like the fitter.

USAGE
  python3 scripts/taker_backtest.py 'data/pull/dt=2026-06-22/*.jsonl.gz' \
      --follow 500 1000 5000 --hold 30 --latency 0.2
"""
from __future__ import annotations

import argparse
import bisect
import glob
import json
import os
import re
import subprocess
from collections import defaultdict, deque


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _is_ephemeral(q):
    q = (q or "").lower()
    return "up or down" in q or "updown" in q


def iter_lines(paths):
    gz = [p for p in paths if p.endswith(".gz")]
    plain = [p for p in paths if not p.endswith(".gz")]
    if gz:
        proc = subprocess.Popen(["gzip", "-dc", *gz], stdout=subprocess.PIPE, bufsize=1 << 20)
        for line in proc.stdout:
            yield line
        proc.stdout.close(); proc.wait()
    for p in plain:
        with open(p, "rb") as fh:
            for line in fh:
                yield line


def taker_pnl(side: str, entry_cross: float, mid_future: float, size: float) -> float:
    """Net taker PnL (in $) holding to the markout. BUY: bought at ask, profits if mid rises.
    SELL: sold at bid, profits if mid falls. The crossing cost is already in entry_cross."""
    sign = 1.0 if side == "BUY" else -1.0
    return sign * (mid_future - entry_cross) * size


# bucket trade size (shares) for the report
SIZE_BUCKETS = [0, 100, 500, 1000, 5000, 1e12]


def run(paths, follow_sizes, hold, latency, size, band):
    """Stream the tape; for each aggressive trade >= the smallest follow size, open a taker position
    and resolve its markout `hold`s later. Accumulate net PnL per (follow-size bucket, category)."""
    pidgroups = defaultdict(list)
    for p in paths:
        m = re.search(r"_(\d+)_\d+\.jsonl", os.path.basename(p))
        pidgroups[m.group(1) if m else "raw"].append(p)
    min_follow = min(follow_sizes)
    agg = defaultdict(lambda: {"n": 0, "net": 0.0, "drift": 0.0, "win": 0, "notional": 0.0})
    meta = {}

    for _g, gp in pidgroups.items():
        gp.sort(key=lambda p: int(re.search(r"_(\d+)\.jsonl", os.path.basename(p)).group(1))
                if re.search(r"_(\d+)\.jsonl", os.path.basename(p)) else 0)
        bb_cur, ba_cur = {}, {}
        hist = defaultdict(deque)          # asset -> (t, mid) pruned to hold+buffer
        pend = defaultdict(list)           # asset -> open taker positions awaiting markout
        cooloff = {}                       # asset -> time until which we won't re-enter

        def resolve(a, now, force=False):
            h = hist[a]
            if not pend[a]:
                return
            ts = [p[0] for p in h]
            keep = []
            for pos in pend[a]:
                if not force and now < pos["t"] + hold:
                    keep.append(pos); continue
                i = bisect.bisect_left(ts, pos["t"] + hold)
                midf = (h[-1][1] if h else None) if i >= len(h) else h[i][1]
                if midf is None:
                    continue
                net = taker_pnl(pos["side"], pos["entry"], midf, size)
                drift = (1.0 if pos["side"] == "BUY" else -1.0) * (midf - pos["mid0"]) * size
                for fs in follow_sizes:
                    if pos["sz"] >= fs:
                        d = agg[(fs, pos["cat"])]
                        d["n"] += 1; d["net"] += net; d["drift"] += drift
                        d["win"] += 1 if net > 0 else 0
                        d["notional"] += pos["entry"] * size
            pend[a] = keep

        for line in iter_lines(gp):
            try:
                r = json.loads(line)
            except Exception:
                continue
            et = r.get("event_type") or r.get("type")
            if et == "new_market":
                if r.get("market"):
                    meta[r["market"]] = (r.get("slug", ""), r.get("question", ""))
            elif et == "book":
                a = r.get("asset_id")
                bb = max((_f(x["price"]) for x in r.get("bids", [])), default=None)
                ba = min((_f(x["price"]) for x in r.get("asks", [])), default=None)
                t = int(r.get("timestamp", 0)) / 1000.0
                if a and bb is not None and ba is not None:
                    bb_cur[a], ba_cur[a] = bb, ba
                    hist[a].append((t, (bb + ba) / 2))
                    while hist[a] and hist[a][0][0] < t - hold - 5:
                        hist[a].popleft()
                    resolve(a, t)
            elif et == "price_change":
                t = int(r.get("timestamp", 0)) / 1000.0
                for ch in r.get("price_changes", []):
                    a = ch.get("asset_id")
                    bb = _f(ch.get("best_bid")); ba = _f(ch.get("best_ask"))
                    if a and bb is not None and ba is not None and 0 < bb < ba < 1:
                        bb_cur[a], ba_cur[a] = bb, ba
                        hist[a].append((t, (bb + ba) / 2))
                        while hist[a] and hist[a][0][0] < t - hold - 5:
                            hist[a].popleft()
                        resolve(a, t)
            elif et == "last_trade_price":
                a = r.get("asset_id"); mk = r.get("market")
                price = _f(r.get("price")); sz = _f(r.get("size")); side = r.get("side")
                t = int(r.get("timestamp", 0)) / 1000.0
                if not (a and price and sz and side):
                    continue
                bb, ba = bb_cur.get(a), ba_cur.get(a)
                if bb is None or ba is None:
                    continue
                mid0 = (bb + ba) / 2
                if not (band[0] <= mid0 <= band[1]):
                    continue
                if sz < min_follow or t < cooloff.get(a, 0):
                    continue
                # follow the aggressor: BUY lifts the ask, SELL hits the bid. Enter crossing.
                entry = ba if side == "BUY" else bb            # we pay the touch to cross now
                slug, q = meta.get(mk, ("", ""))
                cat = "crypto" if _is_ephemeral(q) else "other"
                pend[a].append({"t": t + latency, "side": side, "entry": entry, "mid0": mid0,
                                "sz": sz, "cat": cat})
                cooloff[a] = t + hold
        for a in list(pend.keys()):
            resolve(a, 10**18, force=True)

    # ---- report ----
    print(f"\n==== TAKER (follow aggressive trades) hold={hold}s latency={latency}s size={size} ====")
    hdr = f"{'follow>=':>9} {'cat':8} {'ntrade':>7} {'net$':>12} {'net¢/sh':>9} {'drift¢/sh':>10} {'win%':>6}"
    print(hdr); print("-" * len(hdr))
    for fs in follow_sizes:
        for cat in ("other", "crypto"):
            d = agg[(fs, cat)]
            if d["n"] == 0:
                continue
            print(f"{fs:>9.0f} {cat:8} {d['n']:7d} {d['net']:12.2f} "
                  f"{100*d['net']/d['n']/size:9.3f} {100*d['drift']/d['n']/size:10.3f} "
                  f"{100*d['win']/d['n']:6.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("globs", nargs="+", help="tape file(s)/glob(s)")
    ap.add_argument("--follow", type=float, nargs="+", default=[100, 500, 1000, 5000],
                    help="follow trades with size >= each of these (shares)")
    ap.add_argument("--hold", type=float, default=30.0, help="markout horizon seconds")
    ap.add_argument("--latency", type=float, default=0.2, help="seconds before our cross lands")
    ap.add_argument("--size", type=float, default=100.0, help="our taker clip (shares)")
    ap.add_argument("--band", type=float, nargs=2, default=[0.05, 0.95], help="only trade mid in band")
    args = ap.parse_args()
    paths = []
    for g in args.globs:
        paths.extend(glob.glob(g))
    if not paths:
        print("no input files"); return
    run(paths, sorted(args.follow), args.hold, args.latency, args.size, tuple(args.band))


if __name__ == "__main__":
    main()
