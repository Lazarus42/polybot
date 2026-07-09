#!/usr/bin/env python3
"""Capacity / concurrent-capital ceiling of a longshot cell (prereg longshot-capacity-ceiling-2026-07-09).

The per-cycle ROC of the filtered longshot (4-5c AND <10k) is meaningless for a bankroll unless the
strategy can actually deploy the bankroll. This measures, over a frozen historical window:

  A_t  concurrent deployed capital at the sim's 1-clip-per-market sizing
       (sum of entry cost over positions open at t)
  B_t  dollar in-band ask depth of the concurrently-open eligible markets
       (sum of q_ask_book-at-entry x entry price over positions open at t) — an optimistic upper
       bound on how many dollars COULD be deployed at once into the cell (primary prereg metric)

plus daily new-entry capital flow, holding-time distribution, and the implied bankroll-level %/day
(cell P&L per day / bankroll), headline and with the pre-committed 50% fills-at-touch haircut.

Both time series are computed with an event sweep over (entry_t, +x) / (exit_t, -x) deltas and
summarized time-weighted. A position is "open" from its first buy row to its last snapshot row
(after resolution the market leaves the universe, so l_t ~= resolution time).

Pure stdlib; reuses collect_clips from longshot_market_analysis (same clip and bucket definitions
as the deployed longshot_thin gate, so the measured cell == the deployable filter).

Usage:
    python scripts/longshot_capacity_analysis.py \
        --paper 'data/paper/dt=*/paper_*.jsonl.gz' \
        --manifest-glob 'data/manifests/manifest_*.json' \
        --until 1783310400 --keep-entry 4-5c --keep-liq '<10k' \
        --bankroll 5000 --out reports/longshot_capacity
"""
from __future__ import annotations

import argparse
import statistics as st
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from longshot_market_analysis import collect_clips, load_manifest_tags


def sweep_series(intervals):
    """Event-sweep a list of (start_t, end_t, value) into a step series [(t, level), ...].

    Level at t = sum of value over intervals with start_t <= t < end_t. Zero-length or inverted
    intervals contribute nothing. Returned series starts at the first event.
    """
    deltas = defaultdict(float)
    for s, e, v in intervals:
        if s is None or e is None or e <= s or not v:
            continue
        deltas[s] += v
        deltas[e] -= v
    series = []
    level = 0.0
    for t in sorted(deltas):
        level += deltas[t]
        series.append((t, max(level, 0.0)))  # clamp fp dust
    return series


def time_weighted_stats(series, until=None):
    """Time-weighted mean/median/p90/peak of a step series. Each level holds from its event time to
    the next event (the final level holds until `until` if given, else contributes no duration)."""
    if not series:
        return {"peak": 0.0, "mean": 0.0, "median": 0.0, "p90": 0.0, "span_days": 0.0}
    spans = []  # (level, duration)
    for (t, lvl), (t2, _) in zip(series, series[1:]):
        spans.append((lvl, t2 - t))
    if until is not None and until > series[-1][0]:
        spans.append((series[-1][1], until - series[-1][0]))
    total = sum(d for _, d in spans)
    peak = max(lvl for _, lvl in series)
    if total <= 0:
        return {"peak": peak, "mean": 0.0, "median": 0.0, "p90": 0.0, "span_days": 0.0}
    mean = sum(lvl * d for lvl, d in spans) / total

    def _quantile(q):
        target = q * total
        acc = 0.0
        for lvl, d in sorted(spans):
            acc += d
            if acc >= target:
                return lvl
        return spans and sorted(spans)[-1][0] or 0.0

    return {"peak": peak, "mean": mean, "median": _quantile(0.5), "p90": _quantile(0.9),
            "span_days": total / 86400.0}


def daily_flow(clips, key):
    """Sum of clip[key] per UTC day of entry."""
    out = defaultdict(float)
    for c in clips:
        day = datetime.fromtimestamp(c["e_t"], tz=timezone.utc).strftime("%Y-%m-%d")
        out[day] += c[key] or 0.0
    return dict(sorted(out.items()))


def analyze_cell(clips, bankroll, until=None, label="cell"):
    """All prereg metrics for one clip population."""
    valid = [c for c in clips if c.get("e_t") is not None and c.get("l_t", -1) > c["e_t"]]
    n_missing_ask = sum(1 for c in valid if not c.get("e_ask"))
    a_series = sweep_series([(c["e_t"], c["l_t"], c["cost"]) for c in valid])
    b_series = sweep_series([(c["e_t"], c["l_t"], (c["e_ask"] or 0.0) * (c["entry"] or 0.0))
                             for c in valid])
    a_stats = time_weighted_stats(a_series, until=until)
    b_stats = time_weighted_stats(b_series, until=until)

    resolved = [c for c in valid if c["resolved"]]
    hold_h = sorted((c["l_t"] - c["e_t"]) / 3600.0 for c in resolved)
    total_pnl = sum(c["pnl"] for c in valid)
    total_cost = sum(c["cost"] for c in valid)
    t0 = min(c["e_t"] for c in valid) if valid else 0.0
    t1 = until if until is not None else (max(c["l_t"] for c in valid) if valid else 0.0)
    days = max((t1 - t0) / 86400.0, 1e-9)

    return {
        "label": label,
        "clips": len(valid), "resolved": len(resolved),
        "missing_ask_frac": (n_missing_ask / len(valid)) if valid else 1.0,
        "window_days": days,
        "total_pnl": total_pnl, "total_cost": total_cost,
        "roc_pct": 100.0 * total_pnl / total_cost if total_cost else None,
        "A_deployed": a_stats, "B_ask_depth": b_stats,
        "daily_entry_flow": daily_flow(valid, "cost"),
        "hold_hours_median": st.median(hold_h) if hold_h else None,
        "hold_hours_p90": hold_h[int(0.9 * (len(hold_h) - 1))] if hold_h else None,
        "pnl_per_day": total_pnl / days,
        "bankroll_pct_per_day": 100.0 * total_pnl / days / bankroll,
        "bankroll_pct_per_day_haircut50": 50.0 * total_pnl / days / bankroll,
    }


def fmt_money(x):
    return f"${x:,.0f}"


def render(res, bankroll, threshold):
    m = res["B_ask_depth"]["median"]
    verdict = "CONFIRMED (capacity-constrained)" if m < threshold else "REFUTED (not capacity-constrained)"
    lines = [
        f"## {res['label']}",
        "",
        f"- clips: **{res['clips']}** (resolved {res['resolved']}) over {res['window_days']:.1f} days"
        f" · missing q_ask_book at entry: {res['missing_ask_frac']:.1%}",
        f"- **B_t — $ in-band ask depth of concurrently-open markets (PRIMARY):** "
        f"median **{fmt_money(res['B_ask_depth']['median'])}**, mean {fmt_money(res['B_ask_depth']['mean'])}, "
        f"p90 {fmt_money(res['B_ask_depth']['p90'])}, peak {fmt_money(res['B_ask_depth']['peak'])}",
        f"- A_t — concurrent deployed capital at sim 1-clip sizing: "
        f"median {fmt_money(res['A_deployed']['median'])}, mean {fmt_money(res['A_deployed']['mean'])}, "
        f"p90 {fmt_money(res['A_deployed']['p90'])}, peak {fmt_money(res['A_deployed']['peak'])}",
        f"- entry flow: mean {fmt_money(st.mean(res['daily_entry_flow'].values()) if res['daily_entry_flow'] else 0)}/day"
        f" of new clips · holding time median {res['hold_hours_median'] and round(res['hold_hours_median'],1)}h,"
        f" p90 {res['hold_hours_p90'] and round(res['hold_hours_p90'],1)}h (resolved clips)",
        f"- window P&L {fmt_money(res['total_pnl'])} on {fmt_money(res['total_cost'])} cycled"
        f" (ROC {res['roc_pct'] and round(res['roc_pct'],1)}%) → **bankroll-level "
        f"{res['bankroll_pct_per_day']:.2f}%/day on {fmt_money(bankroll)}** headline, "
        f"**{res['bankroll_pct_per_day_haircut50']:.2f}%/day** with the pre-committed 50% fills-at-touch haircut",
        "",
        f"**Verdict vs prereg threshold (median B_t < {fmt_money(threshold)}): {verdict}**",
        "",
    ]
    return lines, verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper", required=True)
    ap.add_argument("--manifest-glob", required=True)
    ap.add_argument("--config", default="longshot")
    ap.add_argument("--keep-entry", default="4-5c", help="entry bucket(s) for the cell under test")
    ap.add_argument("--keep-liq", default="<10k", help="liquidity bucket(s) for the cell under test")
    ap.add_argument("--bankroll", type=float, default=5000.0)
    ap.add_argument("--threshold", type=float, default=5000.0,
                    help="prereg success threshold on median B_t (dollars)")
    ap.add_argument("--resolved-lo", type=float, default=0.10)
    ap.add_argument("--resolved-hi", type=float, default=0.90)
    ap.add_argument("--until", default=None, help="epoch cutoff (prereg: 1783310400)")
    ap.add_argument("--since", default=None)
    ap.add_argument("--out", type=Path, default=Path("reports/longshot_capacity"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    until = float(args.until) if args.until else None
    since = float(args.since) if args.since else None

    tags = load_manifest_tags(args.manifest_glob)
    clips = collect_clips(args.paper, tags, args.resolved_lo, args.resolved_hi,
                          until=until, since=since, config=args.config)
    ent = {x.strip() for x in args.keep_entry.split(",")}
    liq = {x.strip() for x in args.keep_liq.split(",")}
    cell = [c for c in clips if c["entry_bucket"] in ent and c["liq_bucket"] in liq]

    res_cell = analyze_cell(cell, args.bankroll, until=until,
                            label=f"cell: entry {args.keep_entry} AND liq {args.keep_liq}")
    res_all = analyze_cell(clips, args.bankroll, until=until,
                           label=f"control: unfiltered {args.config}")

    lines = [
        "# Longshot capacity / concurrent-capital ceiling",
        "",
        f"_Prereg: `reports/prereg/longshot-capacity-ceiling-2026-07-09.md` · config `{args.config}`"
        f" · window until {args.until or 'end of data'} · bankroll {fmt_money(args.bankroll)}_",
        "",
        "B_t sums q_ask_book-at-entry x entry over OPEN positions — an optimistic upper bound on "
        "concurrently-buyable dollars (in-band score-shares, no price impact). A_t is what the sim's "
        "1-clip sizing actually tied up. All stats are time-weighted over the window.",
        "",
    ]
    cl, verdict = render(res_cell, args.bankroll, args.threshold)
    lines += cl
    al, _ = render(res_all, args.bankroll, args.threshold)
    lines += al
    lines += ["### Daily new-entry capital flow (cell)", "", "| day | new entry $ |", "|---|---|"]
    lines += [f"| {d} | {fmt_money(v)} |" for d, v in res_cell["daily_entry_flow"].items()]

    (args.out / "digest.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.out}/digest.md")
    return verdict


if __name__ == "__main__":
    main()
