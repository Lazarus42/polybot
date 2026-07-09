#!/usr/bin/env python3
"""Which markets is the LONGSHOT bet actually good in?

The longshot config buys one clip of a 1-5c underdog and holds to resolution. Most resolve to 0;
the P&L lives in the rare markets that pop/resolve YES. The economics: a clip bought at price p
breaks even at a win rate of ~p (a win pays ~$1). So the lever is finding a *subpopulation* of
markets whose realised win rate clears its entry price with room to spare -- e.g. lifting the hit
rate from ~5% to ~10% on 3c clips roughly doubles expectancy.

This slices every longshot clip by the market attributes we have (category, horizon, neg_risk,
competitive score, reward pool, quoted spread, live in-band book depth, and entry price) and reports,
per segment: clips, resolved, WIN RATE, avg entry, avg P&L per clip (= realised expectancy), total
P&L, and a breakeven check (win_rate vs avg entry). Segments that beat breakeven with enough clips
are the deployment filter.

Pure stdlib (no duckdb/pandas) so it runs anywhere, including in-region on the EC2 box with no pip.
Reads the same gzipped snapshot tape + manifests documented in PAPER_TRADING_CONTEXT.md.

Unit of analysis = one CLIP = (token, source-file) segment. The sim restarts (Restart=always) create
a fresh clip per market per run; that's fine -- each clip is one bet, which is exactly what a hit
rate should count.

Usage:
    python scripts/longshot_market_analysis.py \
        --paper 'data/paper/dt=*/paper_*.jsonl.gz' \
        --manifest-glob 'data/manifests/manifest_*.json' \
        --out reports/longshot_analysis
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import multiprocessing as mp
import os
import statistics as st
from collections import defaultdict
from pathlib import Path

try:                              # orjson parses several times faster than stdlib json, if present
    import orjson as _json
    _loads = _json.loads
except Exception:
    _loads = json.loads


def load_manifest_tags(manifest_glob: str) -> dict[str, dict]:
    """Load token tags from raw manifest_*.json files AND/OR a merged token_tags.json.gz artifact
    (built by build_token_tags.py — the full-coverage union of every manifest; prefer it)."""
    tags: dict[str, dict] = {}
    for p in sorted(glob.glob(manifest_glob)):        # later manifests win (tags are stable)
        try:
            if p.endswith(".gz"):
                with gzip.open(p, "rt") as fh:
                    tm = json.load(fh).get("token_meta") or {}
            else:
                tm = json.loads(Path(p).read_text()).get("token_meta") or {}
        except Exception:
            continue
        for tok, m in tm.items():
            tags[str(tok)] = m
    return tags


def bucket(x, edges, labels):
    if x is None:
        return "unknown"
    for e, lab in zip(edges, labels):
        if x < e:
            return lab
    return labels[-1]


LIQ_EDGES = [1e4, 1e5, 1e6];    LIQ_LABELS = ["<10k", "10k-100k", "100k-1M", ">1M"]
POOL_EDGES = [1, 100, 1000];    POOL_LABELS = ["none", "<100", "100-1k", ">1k"]
HORIZON_EDGES = [3, 14, 60];    HORIZON_LABELS = ["<3d", "3-14d", "14-60d", ">60d"]
ENTRY_EDGES = [0.02, 0.03, 0.04]; ENTRY_LABELS = ["1-2c", "2-3c", "3-4c", "4-5c"]
COMP_EDGES = [0.3, 0.6, 0.85];  COMP_LABELS = ["<.3", ".3-.6", ".6-.85", ">.85"]
SPREAD_EDGES = [0.005, 0.02, 0.05]; SPREAD_LABELS = ["<0.5c", "0.5-2c", "2-5c", ">5c"]


# --- worker globals (set once per process via the pool initializer under spawn) ---
_TAGS: dict = {}
_UNTIL = None
_SINCE = None
_LO = 0.10
_HI = 0.90
_CONFIG = "longshot"
_NEEDLE = '"config": "longshot"'


def _worker_init(tags, until, since, lo, hi, config):
    global _TAGS, _UNTIL, _SINCE, _LO, _HI, _CONFIG, _NEEDLE
    _TAGS, _UNTIL, _SINCE, _LO, _HI = tags, until, since, lo, hi
    _CONFIG, _NEEDLE = config, f'"config": "{config}"'


def _pid_of(path: str) -> str:
    """Extract the sim process id from a paper_<host>_<pid>_<epoch>.jsonl.gz filename. A single pid =
    one continuous Holder lifetime (marked_pnl is cumulative within it); a new pid = a restart."""
    parts = os.path.basename(path).split("_")
    return parts[-2] if len(parts) >= 3 else os.path.basename(path)


def _process_file(path: str):
    """Reduce ONE file to per-token PARTIAL aggregates (earliest entry row + latest row). A position
    spans many 15-min files within a pid, so files are NOT independent clips — the parent merges
    these partials per (token, pid) and takes the LAST marked_pnl (cumulative), never the sum.
    Returns (status, pid, {token: partial})."""
    pid = _pid_of(path)
    part: dict = {}
    try:
        with gzip.open(path, "rt") as fh:
            for line in fh:
                if _NEEDLE not in line:             # fast path: skip json-parsing non-matching rows
                    continue
                try:
                    r = _loads(line)
                except Exception:
                    continue
                if r.get("config") != _CONFIG:
                    continue
                t = r.get("t") or 0.0
                if _UNTIL is not None and t >= _UNTIL:
                    continue
                if _SINCE is not None and t < _SINCE:
                    continue
                tok = r["token"]
                p = part.get(tok)
                if p is None:
                    p = part[tok] = {"e_t": None, "e_mid": None, "e_liq": None, "e_cost": None,
                                     "e_ask": None, "l_t": -1.0, "l_pnl": 0.0, "l_mid": None}
                inv = r.get("inv") or 0.0
                mid = r.get("mid")
                if p["e_t"] is None and inv > 0 and mid is not None:          # first buy in this file
                    p["e_t"] = t; p["e_mid"] = mid
                    p["e_liq"] = (r.get("q_bid_book") or 0.0) + (r.get("q_ask_book") or 0.0)
                    p["e_ask"] = r.get("q_ask_book")   # ask-side in-band depth at entry (capacity)
                    p["e_cost"] = inv * mid          # capital deployed on the clip (shares x price)
                if t >= p["l_t"]:                                             # latest row in this file
                    p["l_t"] = t; p["l_pnl"] = r.get("marked_pnl") or 0.0; p["l_mid"] = mid
    except Exception:
        return ("BAD", pid, {})
    return ("OK", pid, part)


def _merge(dst: dict, src: dict):
    """Merge one file's partial into the accumulator: keep the earliest entry and the latest row."""
    if src["e_t"] is not None and (dst["e_t"] is None or src["e_t"] < dst["e_t"]):
        dst["e_t"], dst["e_mid"], dst["e_liq"], dst["e_cost"], dst["e_ask"] = \
            src["e_t"], src["e_mid"], src["e_liq"], src["e_cost"], src["e_ask"]
    if src["l_t"] > dst["l_t"]:
        dst["l_t"], dst["l_pnl"], dst["l_mid"] = src["l_t"], src["l_pnl"], src["l_mid"]


def collect_clips(paper_glob: str, tags: dict, resolved_lo: float, resolved_hi: float,
                  until=None, since=None, config="longshot"):
    """Process files in parallel, then reduce to ONE clip per (token, pid) — a real position/lifetime,
    not a per-file segment. Prints progress so a long run doesn't look like a hang."""
    files = sorted(glob.glob(paper_glob))
    if not files:
        return []
    n_proc = max(1, (os.cpu_count() or 2))
    print(f"scanning {len(files)} files for config '{config}' across {n_proc} processes...", flush=True)
    acc: dict = {}          # (token, pid) -> merged partial
    skipped = done = 0
    with mp.Pool(n_proc, initializer=_worker_init,
                 initargs=(tags, until, since, resolved_lo, resolved_hi, config)) as pool:
        for status, pid, part in pool.imap_unordered(_process_file, files, chunksize=4):
            done += 1
            if status == "BAD":
                skipped += 1
            else:
                for tok, p in part.items():
                    key = (tok, pid)
                    d = acc.get(key)
                    if d is None:
                        acc[key] = p
                    else:
                        _merge(d, p)
            if done % 100 == 0 or done == len(files):
                print(f"  {done}/{len(files)} files | {len(acc)} positions so far", flush=True)
    if skipped:
        print(f"skipped {skipped} unreadable/partial gzip file(s)")

    clips = []
    for (tok, _pid), p in acc.items():
        if p["e_t"] is None:                      # never bought -> not a position
            continue
        m = tags.get(tok, {})
        last_mid = p["l_mid"]
        resolved = last_mid is not None and (last_mid <= resolved_lo or last_mid >= resolved_hi)
        clips.append({
            "token": tok, "entry": p["e_mid"], "pnl": p["l_pnl"], "cost": p["e_cost"] or 0.0,
            "e_t": p["e_t"], "l_t": p["l_t"], "e_ask": p["e_ask"],
            "last_mid": last_mid, "resolved": resolved, "win": p["l_pnl"] > 0,
            "category": m.get("category", "unknown"), "neg_risk": bool(m.get("neg_risk")),
            "horizon_bucket": bucket(m.get("horizon_days"), HORIZON_EDGES, HORIZON_LABELS),
            "competitive_bucket": bucket(m.get("competitive"), COMP_EDGES, COMP_LABELS),
            "pool_bucket": bucket(m.get("reward_daily_est"), POOL_EDGES, POOL_LABELS),
            "spread_bucket": bucket(m.get("spread"), SPREAD_EDGES, SPREAD_LABELS),
            "entry_bucket": bucket(p["e_mid"], ENTRY_EDGES, ENTRY_LABELS),
            "liq_bucket": bucket(p["e_liq"], LIQ_EDGES, LIQ_LABELS),
        })
    return clips


def summarize(clips, key_fn):
    groups = defaultdict(list)
    for c in clips:
        groups[key_fn(c)].append(c)
    rows = []
    for seg, cs in groups.items():
        res = [c for c in cs if c["resolved"]]
        wins_all = sum(c["win"] for c in cs)
        wins_res = sum(c["win"] for c in res)
        avg_entry = st.mean(c["entry"] for c in cs)
        avg_pnl = st.mean(c["pnl"] for c in cs)
        total_pnl = sum(c["pnl"] for c in cs)
        total_cost = sum(c["cost"] for c in cs)                # capital deployed across the segment
        wr_res = (wins_res / len(res)) if res else None      # hit rate over RESOLVED clips
        rows.append({
            "segment": seg, "clips": len(cs), "resolved": len(res),
            "win_rate_resolved": round(wr_res, 4) if wr_res is not None else None,
            "win_rate_all": round(wins_all / len(cs), 4),
            "avg_entry": round(avg_entry, 4),
            "beats_breakeven": (wr_res is not None and wr_res > avg_entry),  # win pays ~$1
            "avg_pnl_per_clip": round(avg_pnl, 3),
            "capital": round(total_cost, 2),
            # capital-weighted return: total P&L per dollar deployed — the allocation-relevant metric
            "roc_pct": round(100.0 * total_pnl / total_cost, 2) if total_cost > 0 else None,
            "total_pnl": round(total_pnl, 2),
        })
    rows.sort(key=lambda r: (r["roc_pct"] if r["roc_pct"] is not None else -1e9), reverse=True)
    return rows


CSV_COLS = ["segment", "clips", "resolved", "win_rate_resolved", "win_rate_all", "avg_entry",
            "beats_breakeven", "avg_pnl_per_clip", "capital", "roc_pct", "total_pnl"]


def write_csv(path: Path, rows: list[dict]):
    lines = [",".join(CSV_COLS)]
    for r in rows:
        lines.append(",".join("" if r.get(c) is None else str(r[c]) for c in CSV_COLS))
    path.write_text("\n".join(lines) + "\n")


def md_table(rows, top=None):
    rows = rows[:top] if top else rows
    if not rows:
        return "_(no clips)_"
    cols = ["segment", "clips", "resolved", "win_rate_resolved", "avg_entry",
            "avg_pnl_per_clip", "roc_pct", "total_pnl"]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if r.get(c) is None else str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper", required=True, help="glob for paper_*.jsonl.gz snapshots")
    ap.add_argument("--manifest-glob", required=True, help="glob for manifest_*.json")
    ap.add_argument("--config", default="longshot",
                    help="which holder config to analyze, e.g. longshot (control) or longshot_thin")
    ap.add_argument("--resolved-lo", type=float, default=0.10, help="last_mid <= this = resolved NO")
    ap.add_argument("--resolved-hi", type=float, default=0.90, help="last_mid >= this = resolved YES")
    ap.add_argument("--min-clips", type=int, default=10, help="min clips for a segment to be 'actionable'")
    # stack filters to price a specific cell, e.g. --keep-entry 4-5c --keep-liq '<10k'. Comma-separate
    # to keep several buckets; omit to keep all. 'unknown' is a valid value to keep untagged clips.
    ap.add_argument("--keep-entry", default=None, help="only these entry buckets, e.g. 4-5c")
    ap.add_argument("--keep-liq", default=None, help="only these liquidity buckets, e.g. <10k")
    ap.add_argument("--keep-horizon", default=None, help="only these horizon buckets, e.g. <3d,3-14d,unknown")
    ap.add_argument("--entry-slippage-c", type=float, default=0.0,
                    help="stress test: extra entry slippage in CENTS/share beyond the touch-ask the "
                         "sim already paid (the historical snapshots recorded fills at touch, no "
                         "depth-walk). Subtracts slippage x shares from each position's P&L.")
    ap.add_argument("--until", default=None,
                    help="exclude snapshots at/after this cutoff (epoch seconds or ISO-8601 with "
                         "offset). e.g. 1783310400 = Mon 2026-07-06 00:00 EDT")
    ap.add_argument("--since", default=None,
                    help="exclude snapshots BEFORE this (epoch or ISO-8601). Use to restrict to the "
                         "clean post-restart window for the walk-the-book forward test.")
    ap.add_argument("--out", type=Path, default=Path("reports/longshot_analysis"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    def _epoch(spec):
        if not spec:
            return None
        s = spec.strip()
        if s.replace(".", "", 1).isdigit():
            return float(s)
        from datetime import datetime
        return datetime.fromisoformat(s).timestamp()

    until, since = _epoch(args.until), _epoch(args.since)
    if until is not None:
        print(f"cutoff: excluding snapshots at/after epoch {until:.0f}")
    if since is not None:
        print(f"window start: excluding snapshots before epoch {since:.0f}")

    tags = load_manifest_tags(args.manifest_glob)
    clips = collect_clips(args.paper, tags, args.resolved_lo, args.resolved_hi,
                          until=until, since=since, config=args.config)

    # slippage stress test: charge extra cents/share of entry slippage the historical fills-at-touch
    # never paid. shares = cost / entry_mid; extra = slippage_c/100 * shares. Repriced P&L + win flag.
    if args.entry_slippage_c and clips:
        sc = args.entry_slippage_c / 100.0
        for c in clips:
            shares = (c["cost"] / c["entry"]) if c["entry"] else 0.0
            c["pnl"] -= sc * shares
            c["win"] = c["pnl"] > 0
        print(f"slippage stress: charged {args.entry_slippage_c}c/share extra entry cost", flush=True)

    # optional stacked filter: carve to a specific cell and price it (whole digest becomes that cell)
    def _keep(spec, field):
        if not spec:
            return None
        wanted = {x.strip() for x in spec.split(",")}
        return [c for c in clips if c[field] in wanted]
    for spec, field in ((args.keep_entry, "entry_bucket"), (args.keep_liq, "liq_bucket"),
                        (args.keep_horizon, "horizon_bucket")):
        sub = _keep(spec, field)
        if sub is not None:
            clips = sub
    if any((args.keep_entry, args.keep_liq, args.keep_horizon)) and clips:
        cap = sum(c["cost"] for c in clips)
        print(f"FILTER kept {len(clips)} clips | capital ${cap:,.0f} | "
              f"P&L ${sum(c['pnl'] for c in clips):,.2f} | "
              f"ROC {100*sum(c['pnl'] for c in clips)/max(cap,1e-9):.1f}%", flush=True)

    if not clips:
        print("No longshot clips found (or the filter removed everything).")
        return

    axes = {
        "category": lambda c: c["category"],
        "liquidity": lambda c: c["liq_bucket"],
        "entry_price": lambda c: c["entry_bucket"],
        "horizon": lambda c: c["horizon_bucket"],
        "neg_risk": lambda c: "neg_risk" if c["neg_risk"] else "binary",
        "competitive": lambda c: c["competitive_bucket"],
        "reward_pool": lambda c: c["pool_bucket"],
        "spread": lambda c: c["spread_bucket"],
        "category_x_liquidity": lambda c: f"{c['category']} / {c['liq_bucket']}",
        # entry x liquidity: BOTH axes come from the snapshot (not the patchy manifest), so this is
        # the trustworthy 2-D cut. It disentangles whether 4-5c wins regardless of book depth (entry
        # is the real edge) or only when the book is also thin (liquidity is a second, real filter).
        "entry_x_liquidity": lambda c: f"{c['entry_bucket']} / {c['liq_bucket']}",
    }
    summaries = {name: summarize(clips, fn) for name, fn in axes.items()}
    for name, rows in summaries.items():
        write_csv(args.out / f"by_{name}.csv", rows)

    # overall baseline
    res = [c for c in clips if c["resolved"]]
    base_wr = (sum(c["win"] for c in res) / len(res)) if res else None
    base_entry = st.mean(c["entry"] for c in clips)
    total = sum(c["pnl"] for c in clips)

    # actionable segments: enough clips AND beats breakeven, ranked by avg_pnl_per_clip
    actionable = []
    for name in ("category", "liquidity", "entry_price", "horizon", "neg_risk",
                 "competitive", "reward_pool", "spread", "category_x_liquidity", "entry_x_liquidity"):
        for r in summaries[name]:
            if r["clips"] >= args.min_clips and r["beats_breakeven"]:
                actionable.append({**r, "axis": name})
    actionable.sort(key=lambda r: (r["roc_pct"] if r["roc_pct"] is not None else -1e9), reverse=True)

    lines = [
        "# Longshot market-segmentation — which markets the 1-5c bet works in",
        "",
        f"- Clips (markets entered): **{len(clips)}**  ·  resolved: **{len(res)}**",
        f"- Baseline hit rate (resolved): **{base_wr:.1%}**" if base_wr is not None else "- Baseline hit rate: n/a (nothing resolved yet)",
        f"- Baseline avg entry: **{base_entry:.3f}**  ·  breakeven hit rate ≈ entry ≈ **{base_entry:.1%}**",
        f"- Total longshot P&L over window: **${total:,.2f}**",
        "",
        f"- Capital deployed: **${sum(c['cost'] for c in clips):,.0f}**  ·  "
        f"overall ROC: **{100*total/max(sum(c['cost'] for c in clips),1e-9):.1f}%**",
        "",
        "Economics: a clip bought at price *p* needs a win rate > *p* to be +EV (a win pays ~$1). "
        "`roc_pct` = segment P&L per dollar of capital deployed (position price x shares) — the "
        "allocation-relevant metric, since a 4-5c clip ties up ~3x the capital of a 1-2c clip. "
        "Segments are ranked by ROC; `beats_breakeven` (mid-based) is a lenient secondary flag.",
        "",
        "## Actionable segments (≥ min-clips AND beat breakeven, best expectancy first)",
        "",
        md_table(actionable) if actionable else "_None cleared breakeven with enough clips yet — "
        "likely too few resolutions in week 1. Re-run as more markets resolve._",
        "",
    ]
    for name in axes:
        lines += [f"## By {name.replace('_', ' ')}", "", md_table(summaries[name], top=15), ""]
    lines += ["_Per-segment CSVs (all rows) written alongside this digest. "
              "`win_rate_resolved` uses only clips whose last mid left [lo,hi] (actually resolved); "
              "`win_rate_all` includes still-open markets and understates the rate._"]
    (args.out / "digest.md").write_text("\n".join(lines) + "\n")

    print(f"wrote {args.out}/digest.md (+ {len(axes)} by_*.csv).")
    print(f"clips={len(clips)} resolved={len(res)} "
          f"baseline_hit={'%.1f%%' % (base_wr*100) if base_wr is not None else 'n/a'} "
          f"avg_entry={base_entry:.3f} total_pnl=${total:,.2f}")
    if actionable:
        print("top actionable segment:", actionable[0]["axis"], "->", actionable[0]["segment"],
              f"(hit {actionable[0]['win_rate_resolved']}, ${actionable[0]['avg_pnl_per_clip']}/clip)")


if __name__ == "__main__":
    main()
