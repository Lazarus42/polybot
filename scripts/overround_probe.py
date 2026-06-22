#!/usr/bin/env python3
"""Probe cross-market overround on clean mutually-exclusive events.

Foundational measurement before building a cross-market backtest. For the largest
clean events (>= min-legs, exactly one winning leg) from the cluster map, we
reconstruct each leg's price shortly before close from the fills archive and sum the
leg prices.

Two purposes:
  1. Self-validate the price mapping. If we read leg win-probabilities correctly, the
     per-event sum of leg prices should cluster a little ABOVE 1 (the bookmaker
     overround / vig). If sums cluster near 0 or near the leg count, the side mapping
     is inverted and we flip it.
  2. Measure the edge. sum > 1 => the basket is collectively overpriced (fade); the
     theoretical buy-all-legs return is (1 - sum) per $1 of basket cost (you pay the
     sum, exactly one leg pays 1 at resolution).

Side mapping assumption: we take each market's token1 ("answer1") price as P(outcome).
The sum-distribution output tells us whether that assumption holds; pass --flip-side
to use token2 instead.

Runs in the project venv (needs duckdb + fills archive):
    .venv/bin/python scripts/overround_probe.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import duckdb
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("duckdb required to read parquet; run inside the project venv") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("archive/processed/underdog_events"))
    parser.add_argument("--cluster-map", type=Path, default=Path("reports/event_clusters/market_event_map.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/cross_market_probe"))
    parser.add_argument("--min-legs", type=int, default=3)
    parser.add_argument("--max-events", type=int, default=300, help="Probe the N highest-volume clean events.")
    parser.add_argument("--lead-hours", type=float, nargs="+", default=[2.0, 24.0, 72.0, 168.0],
                        help="Sweep: use the last leg price at least this long before close (term structure).")
    parser.add_argument("--min-coverage", type=float, default=0.9,
                        help="For the 'complete' view, fraction of an event's legs that must have a price.")
    parser.add_argument("--max-staleness-hours", type=float, default=48.0,
                        help="A leg's price must have traded within this window before the lead point (0 disables).")
    parser.add_argument("--side", default="token1", help="Which side's price = P(outcome) (token1 or token2).")
    parser.add_argument("--flip-side", action="store_true", help="Use the opposite side (shortcut for testing the mapping).")
    args = parser.parse_args()

    side = "token2" if args.flip_side else args.side
    con = duckdb.connect()
    mpath = str((args.data_dir / "markets.parquet").resolve()).replace("'", "''")
    markets = con.execute(
        f"SELECT market_id, end_date, closed_time, winner_side, historical_volume FROM read_parquet('{mpath}')"
    ).fetchdf()

    cmap = pd.read_csv(args.cluster_map)
    cmap = cmap.merge(markets, on="market_id", how="left")

    # Clean events: >= min_legs and exactly one winning leg.
    grp = cmap.groupby("event_id")
    sizes = grp.size()
    wins = grp["won"].sum() if "won" in cmap.columns else None
    if wins is None:
        raise SystemExit("cluster map lacks 'won'; rerun explore_event_clusters.py")
    clean_ids = sizes[(sizes >= args.min_legs)].index.intersection(wins[wins == 1].index)
    clean = cmap[cmap["event_id"].isin(clean_ids)].copy()
    # rank events by total volume
    ev_vol = clean.groupby("event_id")["historical_volume"].sum().sort_values(ascending=False)
    chosen_ids = list(ev_vol.head(args.max_events).index)
    chosen = clean[clean["event_id"].isin(chosen_ids)].copy()
    print(f"clean events: {len(clean_ids)}; probing top {len(chosen_ids)} by volume "
          f"({chosen['market_id'].nunique()} legs)")

    market_ids = chosen["market_id"].astype(int).tolist()
    fpath = str((args.data_dir / "fills_sorted.parquet").resolve()).replace("'", "''")
    con.execute("CREATE TEMP TABLE sel(market_id BIGINT)")
    con.executemany("INSERT INTO sel VALUES (?)", [(m,) for m in market_ids])
    fills = con.execute(
        f"""
        SELECT f.market_id, f.timestamp, f.side, f.price
        FROM read_parquet('{fpath}') AS f JOIN sel USING (market_id)
        ORDER BY f.market_id, f.timestamp
        """
    ).fetchdf()
    fills["timestamp"] = pd.to_datetime(fills["timestamp"], utc=True)

    # close time per market
    chosen["close"] = pd.to_datetime(chosen["closed_time"].fillna(chosen["end_date"]), utc=True, errors="coerce")
    close_by = dict(zip(chosen["market_id"], chosen["close"]))
    fills_by_market = {int(mid): g for mid, g in fills.groupby("market_id")}

    def overround_at(lead_hours: float) -> pd.DataFrame:
        lead = pd.Timedelta(hours=lead_hours)
        stale = pd.Timedelta(hours=args.max_staleness_hours) if args.max_staleness_hours > 0 else None
        leg_price = {}
        for mid, g in fills_by_market.items():
            c = close_by.get(mid)
            if pd.isna(c):
                continue
            cutoff = c - lead
            gg = g[(g["side"].astype(str) == side) & (g["timestamp"] <= cutoff)]
            if stale is not None:
                gg = gg[gg["timestamp"] >= cutoff - stale]  # price must be FRESH at the lead point
            if gg.empty:
                continue  # no fresh price at this lead (do NOT fall back)
            leg_price[int(mid)] = float(gg.iloc[-1]["price"])
        rows = []
        for ev, g in chosen.groupby("event_id"):
            legs = g["market_id"].astype(int).tolist()
            total_legs = len(legs)
            prices = [leg_price[m] for m in legs if m in leg_price]
            if len(prices) < args.min_legs:
                continue
            s = float(np.sum(prices))
            winner_mid = int(g[g["won"] == True]["market_id"].iloc[0]) if (g["won"] == True).any() else None
            rows.append({
                "lead_hours": lead_hours, "event_id": ev,
                "total_legs": total_legs, "legs_priced": len(prices),
                "coverage": len(prices) / total_legs,
                "sum_legs": s, "fade_all_return": s - 1.0,
                "winner_price": leg_price.get(winner_mid) if winner_mid else None,
            })
        return pd.DataFrame(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_res = []
    summary = []
    for lh in args.lead_hours:
        res = overround_at(lh)
        if res.empty:
            continue
        all_res.append(res)
        comp = res[res["coverage"] >= args.min_coverage]
        def block(d):
            if d.empty:
                return {"n": 0}
            return {
                "n": int(len(d)),
                "median_sum": float(d["sum_legs"].median()),
                "mean_sum": float(d["sum_legs"].mean()),
                "p90_sum": float(d["sum_legs"].quantile(0.90)),
                "frac_underround": float((d["sum_legs"] < 1).mean()),
                "mean_fade_return": float(d["fade_all_return"].mean()),
                "median_fade_return": float(d["fade_all_return"].median()),
            }
        summary.append({
            "lead_hours": lh,
            "mean_winner_price": float(res["winner_price"].dropna().mean()),
            **{f"all_{k}": v for k, v in block(res).items()},
            **{f"complete_{k}": v for k, v in block(comp).items()},
        })
    if all_res:
        pd.concat(all_res, ignore_index=True).to_csv(args.output_dir / "overround_by_event.csv", index=False)
    sdf = pd.DataFrame(summary)
    sdf.to_csv(args.output_dir / "overround_term_structure.csv", index=False)
    cols = ["lead_hours", "mean_winner_price", "all_n", "all_frac_underround",
            "complete_n", "complete_median_sum", "complete_mean_sum", "complete_p90_sum",
            "complete_frac_underround", "complete_mean_fade_return", "complete_median_fade_return"]
    cols = [c for c in cols if c in sdf.columns]
    print("\n=== overround term structure (side=%s) — 'complete' = events with >=%.0f%% legs priced ==="
          % (side, args.min_coverage * 100))
    print("(if complete_frac_underround stays low while all_frac_underround rises, the underround was a missing-leg artifact)")
    print(sdf[cols].round(3).to_string(index=False))
    print("\nwrote:", args.output_dir / "overround_term_structure.csv")


if __name__ == "__main__":
    main()
