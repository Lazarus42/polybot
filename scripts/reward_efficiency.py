#!/usr/bin/env python3
"""Per-market REWARD-CAPTURE EFFICIENCY report from a per_token_dump.jsonl + manifest.

The aggregate "$X/day on $5k" number hides which markets actually pay. This surfaces, per market,
the *capture fraction* = reward we capture / the market's daily pool — high fraction = the pool is
handed out inefficiently (decent pool, thin/uncontested book), which is the sweet spot for a small
maker. Bins by pool size and lists the most inefficiently-rewarded markets (by capture fraction,
among those with a non-trivial pool), with the market question so you can see the league/game.

Usage:
  python scripts/reward_efficiency.py --dump reports/sports_book_mm/per_token_dump.jsonl \
      --manifest-dir reports/sports_capture --size 1000 --config clv_full
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def load_dump(p: Path):
    rows = []
    for line in p.open():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(r.get("results", {}).get("clv_full"), dict):
            rows.append(r)
    return rows


def load_meta(manifest: Path):
    m = json.loads(manifest.read_text())
    return m.get("token_meta", {})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--manifest-dir", type=Path, default=None)
    ap.add_argument("--size", type=int, default=1000)
    ap.add_argument("--config", default="clv_full")
    ap.add_argument("--min-pool", type=float, default=20.0,
                    help="ignore markets with pool below this when ranking by capture fraction")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    manifest = args.manifest or (sorted(Path(args.manifest_dir).glob("manifest_*.json"))[-1]
                                 if args.manifest_dir else None)
    meta = load_meta(manifest)
    rows = load_dump(args.dump)
    S, C = str(args.size), args.config

    recs = []
    for r in rows:
        tok = r["token"]; pool = r["pool"]
        if S not in r["results"].get(C, {}):
            continue
        net, trade, reward = r["results"][C][S]
        q = (meta.get(tok, {}) or {}).get("question", "")[:46]
        recs.append({"tok": tok, "pool": pool, "reward": reward, "net": net,
                     "frac": (reward / pool) if pool > 0 else 0.0, "q": q})

    print(f"capture-efficiency @ size {args.size}, config {C}  ({len(recs)} markets; manifest {manifest.name})\n")

    bands = [(0, 50), (50, 200), (200, 500), (500, 2000), (2000, 1e12)]
    print(f"{'pool band $/d':16}{'#mkts':>6}{'avg pool':>9}{'tot reward$':>12}{'avg capt%':>10}{'rew/$cap%':>10}")
    for lo, hi in bands:
        g = [r for r in recs if lo <= r["pool"] < hi]
        if not g:
            continue
        npool = sum(r["pool"] for r in g) or 1
        rw = sum(r["reward"] for r in g)
        fr = sum(r["frac"] for r in g) / len(g)
        cap = len(g) * args.size
        print(f"{f'{lo:.0f}-{hi:.0f}':16}{len(g):>6}{npool/len(g):>9.0f}{rw:>12.1f}{fr:>9.1%}{rw/cap:>10.3%}")

    print(f"\nmost inefficiently-rewarded markets (pool >= ${args.min_pool:.0f}, by capture fraction):")
    print(f"{'pool$/d':>8}{'reward$':>9}{'capt%':>7}{'rew/$cap%':>10}  question")
    elig = sorted([r for r in recs if r["pool"] >= args.min_pool], key=lambda r: r["frac"], reverse=True)
    for r in elig[: args.top]:
        print(f"{r['pool']:>8.0f}{r['reward']:>9.1f}{r['frac']:>7.1%}{r['reward']/args.size:>10.3%}  {r['q']}")


if __name__ == "__main__":
    main()
