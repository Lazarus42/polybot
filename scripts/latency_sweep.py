#!/usr/bin/env python3
"""Latency sweep for the paper MM: replay a raw collector capture through the SAME PaperSim at
several quote latencies (and cancel-on-move settings) to find the BREAKEVEN latency — how fast
your infra must be before reward beats stale-order pickoff.

Also reports the MEASURED feed latency (server->you) straight from the capture: every record has
the server `timestamp` and our `_recv_ts`, so `_recv_ts - timestamp` is the real one-way feed
delay you actually experience. Your true pickoff latency is roughly that PLUS your order/cancel
round-trip, so the measured feed delay is a hard floor on the latency column to take seriously.

Usage:
  python scripts/latency_sweep.py --capture 'reports/clob_capture/book_*.jsonl.gz' \
      --manifest-dir reports/clob_capture --capital 5000 \
      --latencies 0 0.05 0.1 0.2 0.5 --cancels 0 0.005
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import statistics as st
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_sim as ps  # noqa: E402


def _recv_epoch(s):
    try:
        return datetime.fromisoformat(s).timestamp()
    except (ValueError, TypeError):
        return None


def load_events(patterns, max_events):
    files = []
    for p in patterns:
        files += glob.glob(p)
    files.sort()
    events, feed_lat = [], []
    for f in files:
        with gzip.open(f, "rt") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                events.append(e)
                ts = ps._ts(e.get("timestamp")); rv = _recv_epoch(e.get("_recv_ts"))
                if ts and rv:
                    feed_lat.append(rv - ts)
                if max_events and len(events) >= max_events:
                    return events, feed_lat
    return events, feed_lat


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", nargs="+", required=True, help="raw collector jsonl(.gz) file(s)/glob(s)")
    ap.add_argument("--manifest", type=str, default=None)
    ap.add_argument("--manifest-dir", type=str, default=None)
    ap.add_argument("--configs", nargs="+", default=list(ps.CONFIGS))
    ap.add_argument("--capital", type=float, default=5000.0)
    ap.add_argument("--size", type=float, default=200.0)
    ap.add_argument("--latencies", type=float, nargs="+", default=[0, 0.05, 0.1, 0.2, 0.5])
    ap.add_argument("--cancels", type=float, nargs="+", default=[0.0, 0.005])
    ap.add_argument("--max-hold-minutes", type=float, default=120.0)
    ap.add_argument("--max-capture-share", type=float, default=0.10)
    ap.add_argument("--max-events", type=int, default=0, help="cap events loaded (0 = all)")
    args = ap.parse_args()

    import pathlib
    manifest = (pathlib.Path(args.manifest) if args.manifest
                else sorted(pathlib.Path(args.manifest_dir).glob("manifest_*.json"))[-1])
    token_meta = ps.load_token_meta(manifest)
    events, feed_lat = load_events(args.capture, args.max_events)
    print(f"loaded {len(events)} events; manifest {manifest.name}\n")

    if feed_lat:
        feed_lat.sort()
        print("MEASURED feed latency (server -> you), from _recv_ts - timestamp:")
        print(f"  median {1000*st.median(feed_lat):.0f}ms  p90 {1000*feed_lat[int(.9*len(feed_lat))]:.0f}ms  "
              f"p99 {1000*feed_lat[int(.99*len(feed_lat))]:.0f}ms  (n={len(feed_lat)})")
        print("  -> your true pickoff latency is THIS + your order/cancel round-trip.\n")

    import tempfile
    tmp = pathlib.Path(tempfile.mkdtemp())

    class _Null:                              # swallow snapshot writes during the sweep
        def write(self, *a): pass
        def flush(self): pass
    null = _Null()

    print("net_if_flat ($, mark-to-liquidate) by quote latency x config:")
    for cancel in args.cancels:
        print(f"\n--- cancel_on_move = {cancel} ---")
        print(f"{'latency':>9}" + "".join(f"{c:>14}" for c in args.configs))
        for lat in args.latencies:
            sim = ps.PaperSim(token_meta, args.size, 1.0, args.configs, "prorata", 1.0,
                              tmp, 99999.0, capital=args.capital,
                              max_capture_share=args.max_capture_share,
                              quote_latency=lat, cancel_on_move=cancel,
                              max_hold_seconds=args.max_hold_minutes * 60.0)
            sim._writer = lambda t: null   # don't write snapshot files during sweep
            for e in events:
                sim.process_message(e)
            row = f"{lat*1000:>7.0f}ms"
            for c in args.configs:
                a = sim._agg(c)
                row += f"{a['net_if_flat']:>14.2f}"
            print(row)
    print("\nread: the highest latency with POSITIVE net_if_flat is your breakeven — you must be "
          "faster than that. Compare to the measured feed latency above (your floor).")


if __name__ == "__main__":
    main()
