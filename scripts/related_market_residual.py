#!/usr/bin/env python3
"""Axis 1: related-market residual / multi-outcome basket arbitrage.

The directional cross-event version of "trade the residual" (e.g. nomination price vs
presidency price) needs semantic links between events that our metadata does not carry. The
testable form on this data is the WITHIN-event basket constraint: in a genuinely
mutually-exclusive event, the YES prices partition probability and should sum to ~1. When
they sum to < 1 (underround) you can buy the complete set for < $1 and collect $1 at
resolution; when > 1 (overround) the set is rich. This backtests that, net of costs, over
the gated clean partitions from build_event_groups, using realized winners for settlement.

Pure core (`basket_arb_trade`) is unit-tested; the tape/resolution loaders need the venv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def basket_arb_trade(yes_prices: list[float], winner_in_set: bool,
                     fee: float = 0.0, edge_buffer: float = 0.02) -> dict[str, Any]:
    """One event. If the priced YES outcomes sum to < 1 - edge_buffer (after fee), buy the
    complete set: pay sum*(1+fee), and collect $1 iff the realized winner is one of the
    priced outcomes. Returns the trade cost (capital deployed) and realized pnl.

    `winner_in_set` = did the actual winning outcome belong to the priced set (vs a missing
    outcome / "none of these"). Buying an incomplete set that misses the winner pays $0 —
    this is the honest risk of partition incompleteness, so we pass coverage info through.
    """
    ys = [p for p in yes_prices if p is not None and np.isfinite(p)]
    s = sum(ys)
    cost = s * (1.0 + fee)
    if s >= 1.0 - edge_buffer:                    # not cheap enough after costs -> skip
        return {"traded": 0, "cost": 0.0, "pnl": 0.0, "sum": s}
    payoff = 1.0 if winner_in_set else 0.0
    return {"traded": 1, "cost": float(cost), "pnl": float(payoff - cost), "sum": s}


def backtest_basket_arb(events: list[dict[str, Any]], fee: float = 0.0,
                        edge_buffer: float = 0.02) -> dict[str, Any]:
    """events: list of {yes_prices, winner_in_set}. Aggregate the complete-set arb."""
    rows = [basket_arb_trade(e["yes_prices"], e["winner_in_set"], fee, edge_buffer) for e in events]
    traded = [r for r in rows if r["traded"]]
    pnl = np.array([r["pnl"] for r in traded], float)
    cost = np.array([r["cost"] for r in traded], float)
    return {
        "events_seen": len(events),
        "events_traded": len(traded),
        "total_pnl": float(pnl.sum()) if len(pnl) else 0.0,
        "total_deployed": float(cost.sum()) if len(cost) else 0.0,
        "roi": float(pnl.sum() / cost.sum()) if len(cost) and cost.sum() > 0 else float("nan"),
        "hit_rate": float((pnl > 0).mean()) if len(pnl) else float("nan"),
        "mean_pnl_per_trade": float(pnl.mean()) if len(pnl) else 0.0,
    }


def load_events_for_backtest(groups: pd.DataFrame, yes_prices: dict[str, float],
                             winners: dict[str, bool], min_coverage: float = 0.9) -> list[dict]:
    """Assemble per-event {yes_prices, winner_in_set} for categorical events meeting coverage.
    `winners[market_id]` = True iff that outcome won (from resolutions)."""
    g = groups.copy()
    g["yes"] = g["id"].map(yes_prices)
    events = []
    for ev, grp in g.groupby("event_id"):
        if grp["event_type"].iloc[0] != "categorical":
            continue
        priced = grp.dropna(subset=["yes"])
        if len(priced) < 2 or len(priced) / len(grp) < min_coverage:
            continue
        winner_in_set = bool(priced["id"].map(lambda m: winners.get(str(m), False)).any())
        events.append({"yes_prices": priced["yes"].tolist(), "winner_in_set": winner_in_set,
                       "event_id": ev})
    return events


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markets", type=Path, default=Path("archive/markets.csv"))
    ap.add_argument("--fills", type=Path, default=Path("archive/processed/underdog_events/fills_sorted.parquet"))
    ap.add_argument("--resolutions", type=Path, default=Path("polymarket_resolutions.csv"))
    ap.add_argument("--output-dir", type=Path, default=Path("reports/related_market_residual"))
    ap.add_argument("--fee", type=float, default=0.0)
    ap.add_argument("--edge-buffer", type=float, default=0.02, help="Required underround after costs.")
    ap.add_argument("--min-coverage", type=float, default=0.9)
    ap.add_argument("--max-stale-hours", type=float, default=48.0)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_event_groups import build_event_groups, load_yes_prices_asof_duckdb

    markets = pd.read_csv(args.markets, low_memory=False)
    groups = build_event_groups(markets)
    snaps = groups[["id", "snapshot_ts"]].rename(columns={"id": "market_id"})
    yes = load_yes_prices_asof_duckdb(args.fills, snaps, args.max_stale_hours)

    res = pd.read_csv(args.resolutions, low_memory=False)
    res["market_id"] = res["market_id"].astype(str)
    winners = {r.market_id: str(r.resolution).strip().lower() in ("yes", "1", "true")
               for r in res.itertuples()}

    events = load_events_for_backtest(groups, yes, winners, args.min_coverage)
    summary = backtest_basket_arb(events, args.fee, args.edge_buffer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "basket_arb_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("\nread: positive total_pnl with sane ROI and hit_rate => the basket arb is real;"
          " near-zero events_traded => clean underround partitions are rare (capacity wall).")


if __name__ == "__main__":
    main()
