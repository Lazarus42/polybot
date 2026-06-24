#!/usr/bin/env python3
"""Fit lambda(s) (fill rate vs depth) and eta(s) (adverse selection per fill vs depth) from tape.

These are the two empirical inputs the depth_optimal strategy needs. Both fall out of one pass
over trades, bucketed by the trade's DEPTH s = |trade_price - pre_trade_mid| (cents):

  lambda(s): a resting order at depth s is taken when a trade reaches >= s. So the rate of trades
             at depth >= s, divided by market-active-minutes, is the fill rate. The decay constant
             k of that survival curve is what most moves s*.
  eta(s):    for trades at depth s, the signed post-fill mid drift = sign*(mid(t+H) - pre_mid) in
             cents = exactly the adverse selection a maker resting at that depth suffers.

USAGE
  # full run (one process; fine off the sandbox where there's no 45s limit):
  python3 scripts/fit_lambda_eta.py --out /tmp/le.json 'data/pull/dt=2026-06-22/*.jsonl.gz'
  # report a saved aggregate:
  python3 scripts/fit_lambda_eta.py --out /tmp/le.json --report
  # combine several worker aggregates (parallel mode) and report:
  python3 scripts/fit_lambda_eta.py --merge /tmp/le_w*.json

RESUMABLE: aggregates are saved per PID-group, so a killed run resumes (skips done groups) by
re-issuing the same command. gz files stream through `gzip -dc` (far faster than Python's gzip).

Sign: trade `side` is the aggressor (a SELL lifts the bid). BUY taker profits when mid rises.
"""
from __future__ import annotations

import bisect
import glob
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict, deque

H_ETA = 60.0          # markout horizon for eta (matches Quoter.mark_delay_s)
PRUNE_S = H_ETA + 5
BUCKETS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 99.0]   # cent edges


def iter_lines(paths):
    """Yield raw bytes lines for a group of files. gz files stream through the system `gzip -dc`
    (much faster than Python's gzip module, and portable across Linux/macOS); order preserved."""
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


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def classify(slug, q):
    s = (slug or "").lower(); ql = (q or "").lower()
    if "updown" in s or "up or down" in ql:
        return "crypto_updown"
    if any(k in s for k in ("nba", "nfl", "mlb", "nhl", "ufc", "epl", "soccer")) or " vs " in ql:
        return "sports"
    return "other"


def bucket_of(depth_c):
    i = bisect.bisect_right(BUCKETS, depth_c) - 1
    return max(0, min(i, len(BUCKETS) - 2))


def pidkey(p):
    m = re.search(r"_(\d+)_\d+\.jsonl", os.path.basename(p))
    return m.group(1) if m else "raw"


def blank_agg():
    return {"cat": defaultdict(lambda: {"n": 0.0, "mk": 0.0, "net": 0.0, "sz": 0.0}),
            "span": defaultdict(lambda: [None, None]),
            "acat": {}}


def process_group(paths, agg, seen_tx, meta):
    hist = defaultdict(deque)          # asset -> (t_s, mid)  [for markout]
    bb_cur = {}; ba_cur = {}           # asset -> current best bid / best ask (for touch-relative depth)
    pend = defaultdict(list)           # meta (market->slug/question) is shared across groups

    def resolve(a, now, force=False):
        plist = pend.get(a)
        if not plist:                  # hot path: no trade waiting on this asset -> skip
            return
        h = hist[a]; ts = [p[0] for p in h]; keep = []
        for tr in plist:
            if not force and now < tr["t"] + PRUNE_S:
                keep.append(tr); continue
            i = bisect.bisect_left(ts, tr["t"] + H_ETA)
            midH = (h[-1][1] if h else None) if i >= len(h) else h[i][1]
            if midH is None:
                continue
            sign = 1.0 if tr["side"] == "BUY" else -1.0
            mk = sign * (midH - tr["midp"]) * 100.0       # eta contribution (cents/share)
            net = sign * (midH - tr["price"]) * 100.0
            b = bucket_of(tr["depth_c"])                  # depth PAST the touch (cents)
            d = agg["cat"][(tr["cat"], b)]
            d["n"] += 1; d["mk"] += mk; d["net"] += net; d["sz"] += tr["size"]
        pend[a] = keep

    for line in iter_lines(paths):
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
                bb_cur[a] = bb; ba_cur[a] = ba
                hist[a].append((t, (bb + ba) / 2))
                while hist[a] and hist[a][0][0] < t - PRUNE_S:
                    hist[a].popleft()
                resolve(a, t)
        elif et == "price_change":
            t = int(r.get("timestamp", 0)) / 1000.0
            for ch in r.get("price_changes", []):
                a = ch.get("asset_id")
                bb = _f(ch.get("best_bid")); ba = _f(ch.get("best_ask"))
                if a and bb is not None and ba is not None and 0 < bb < ba < 1:
                    bb_cur[a] = bb; ba_cur[a] = ba
                    hist[a].append((t, (bb + ba) / 2))
                    while hist[a] and hist[a][0][0] < t - PRUNE_S:
                        hist[a].popleft()
                    resolve(a, t)
        elif et == "last_trade_price":
            a = r.get("asset_id"); tx = r.get("transaction_hash")
            price = _f(r.get("price")); size = _f(r.get("size")); side = r.get("side")
            t = int(r.get("timestamp", 0)) / 1000.0
            if not (a and price and size and side) or tx in seen_tx:
                continue
            seen_tx.add(tx)
            midp = hist[a][-1][1] if hist[a] else None
            if midp is None or not (0.02 <= midp <= 0.98):
                continue
            # depth = how far PAST the touch the trade reached (cents). A resting order sitting that
            # far behind the front of the line would have been the one filled. SELL hits the bid,
            # BUY lifts the ask.
            if side == "SELL":
                touch = bb_cur.get(a)
                depth_c = (touch - price) * 100.0 if touch is not None else None
            else:
                touch = ba_cur.get(a)
                depth_c = (price - touch) * 100.0 if touch is not None else None
            if depth_c is None:
                continue
            depth_c = max(0.0, depth_c)
            slug, q = meta.get(r.get("market"), ("", ""))   # trade carries its own market id
            cat = classify(slug, q)
            agg["acat"][a] = cat
            sp = agg["span"][a]
            sp[0] = t if sp[0] is None else min(sp[0], t)
            sp[1] = t if sp[1] is None else max(sp[1], t)
            pend[a].append({"t": t, "price": price, "size": size, "side": side,
                            "cat": cat, "midp": midp, "depth_c": depth_c})
    for a in list(pend.keys()):
        resolve(a, 10**18, force=True)


def merge_into(out_path, agg, group_id=None, meta=None):
    base = {"cat": {}, "span": {}, "acat": {}, "pids": [], "meta": {}}
    if os.path.exists(out_path):
        base = json.load(open(out_path)); base.setdefault("pids", []); base.setdefault("meta", {})
    if meta:
        base["meta"].update(meta)
    for (cat, b), d in agg["cat"].items():
        k = f"{cat}|{b}"
        o = base["cat"].setdefault(k, {"n": 0.0, "mk": 0.0, "net": 0.0, "sz": 0.0})
        for f in ("n", "mk", "net", "sz"):
            o[f] += d[f]
    for a, sp in agg["span"].items():
        o = base["span"].get(a)
        base["span"][a] = [sp[0], sp[1]] if o is None else [min(o[0], sp[0]), max(o[1], sp[1])]
    base["acat"].update(agg["acat"])
    if group_id is not None and group_id not in base["pids"]:
        base["pids"].append(group_id)
    json.dump(base, open(out_path, "w"))
    return base


def merge_files(paths):
    base = {"cat": {}, "span": {}, "acat": {}, "pids": []}
    for p in paths:
        if not os.path.exists(p):
            continue
        b = json.load(open(p))
        for k, d in b.get("cat", {}).items():
            o = base["cat"].setdefault(k, {"n": 0.0, "mk": 0.0, "net": 0.0, "sz": 0.0})
            for f in ("n", "mk", "net", "sz"):
                o[f] += d[f]
        for a, sp in b.get("span", {}).items():
            o = base["span"].get(a)
            base["span"][a] = [sp[0], sp[1]] if o is None else [min(o[0], sp[0]), max(o[1], sp[1])]
        base["acat"].update(b.get("acat", {}))
        base["pids"].extend(b.get("pids", []))
    return base


def report(base):
    mm = defaultdict(float)
    for a, sp in base["span"].items():
        if sp[0] is not None and sp[1] is not None and sp[1] > sp[0]:
            mm[base["acat"].get(a, "other")] += (sp[1] - sp[0]) / 60.0
    cats = sorted({k.split("|")[0] for k in base["cat"]})
    bnames = [f"[{BUCKETS[i]:.1f},{BUCKETS[i+1]:.1f})" for i in range(len(BUCKETS) - 1)]
    for cat in cats:
        print(f"\n==== {cat}   (market-minutes={mm[cat]:.0f}) ====")
        print(f"{'depth(c)':12} {'ntr':>6} {'lambda/min':>11} {'eta c/sh':>9} {'net c/sh':>9}")
        rows = []
        for b in range(len(BUCKETS) - 1):
            d = base["cat"].get(f"{cat}|{b}")
            if not d or d["n"] == 0:
                continue
            lam = d["n"] / mm[cat] if mm[cat] > 0 else 0.0
            rows.append((b, d["n"], lam, d["mk"] / d["n"], d["net"] / d["n"]))
            print(f"{bnames[b]:12} {int(d['n']):6d} {lam:11.4f} {d['mk']/d['n']:9.3f} "
                  f"{d['net']/d['n']:9.3f}")
        if len(rows) >= 2 and mm[cat] > 0:
            mids = [(BUCKETS[b] + BUCKETS[b + 1]) / 2 for b, *_ in rows]
            lams = [r[2] for r in rows]
            xs = [(mids[i], math.log(lams[i])) for i in range(len(rows)) if lams[i] > 0]
            if len(xs) >= 2:
                n = len(xs); sx = sum(x for x, _ in xs); sy = sum(y for _, y in xs)
                sxx = sum(x * x for x, _ in xs); sxy = sum(x * y for x, y in xs)
                denom = n * sxx - sx * sx
                k = -(n * sxy - sx * sy) / denom if denom else 0.0
                a0 = math.exp((sy + k * sx) / n)
                print(f"  FIT: a(touch)~{a0:.3f}/min  k~{k:.3f}  eta0~{rows[0][3]:.3f}c  "
                      f"(DEPTH_PARAMS candidate)")


def main():
    args = sys.argv[1:]
    out = "lambda_eta_agg.json"
    if "--out" in args:
        i = args.index("--out"); out = args[i + 1]; del args[i:i + 2]
    if "--merge" in args:
        args.remove("--merge")
        mp = []
        for a in args:
            mp.extend(glob.glob(a))
        report(merge_files(mp)); return
    if "--report" in args:
        args.remove("--report")
        report(json.load(open(out))); return
    paths = []
    for a in args:
        paths.extend(glob.glob(a))
    groups = defaultdict(list)
    for p in paths:
        groups[pidkey(p)].append(p)
    done = set(); meta = {}
    if os.path.exists(out):
        prev = json.load(open(out)); done = set(prev.get("pids", [])); meta = prev.get("meta", {})
    for g in sorted(groups):
        if g in done:
            continue
        gp = sorted(groups[g], key=lambda p: int(re.search(r"_(\d+)\.jsonl", os.path.basename(p)).group(1))
                    if re.search(r"_(\d+)\.jsonl", os.path.basename(p)) else 0)
        agg = blank_agg(); seen_tx = set()
        process_group(gp, agg, seen_tx, meta)
        merge_into(out, agg, group_id=g, meta=meta)   # persist after EACH group (resumable)
        print(f"  done group {g}: {len(gp)} files", file=sys.stderr)
    report(json.load(open(out)))


if __name__ == "__main__":
    main()
