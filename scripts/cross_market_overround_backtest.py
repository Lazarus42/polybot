#!/usr/bin/env python3
"""Backtest cross-market overround fades on mutually-exclusive event clusters.

For each clean event cluster, the strategy reconstructs leg prices at a fixed lead
time before close. If the priced outcome legs sum above 1 + threshold, it fades the
event by buying the opposite side ("No") on every priced leg in equal contract size.

With exactly one winning leg among the priced legs:

    cost per basket unit = n_priced - sum(outcome_prices)
    payoff per basket unit = n_priced - 1
    profit per basket unit = sum(outcome_prices) - 1

The script sizes each basket by the thinnest fade-side leg liquidity, reports return
on capital, writes event-level signals, and summarizes monthly period performance so
the sleeve can be fed into the residual portfolio harness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from realistic_underdog_account import write_csv


def parse_thresholds(values: list[str]) -> list[float]:
    out: list[float] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                out.append(float(part))
    return sorted(set(out))


def parse_int_sweep(values: list[str]) -> list[int]:
    out: list[int] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                out.append(int(part))
    return sorted(set(out))


def month_key(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m")


def synthetic_market_id(event_id: str, lead_hours: float) -> int:
    digest = hashlib.blake2b(f"{event_id}|{lead_hours:g}".encode("utf-8"), digest_size=6).hexdigest()
    return 9_000_000_000 + int(digest, 16) % 900_000_000


def load_inputs(data_dir: Path, cluster_map: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect()
    mpath = str((data_dir / "markets.parquet").resolve()).replace("'", "''")
    markets = con.execute(
        f"""
        SELECT market_id, slug, question, answer1, answer2, winner_side,
               end_date, closed_time, historical_volume
        FROM read_parquet('{mpath}')
        """
    ).fetchdf()
    con.close()
    cmap = pd.read_csv(cluster_map)
    df = cmap.merge(markets, on="market_id", how="left", suffixes=("", "_market"))
    if "won" not in df.columns:
        df["won"] = df["winner_side"].astype(str).str.lower().eq("token1")
    df["close"] = pd.to_datetime(df["closed_time"].fillna(df["end_date"]), utc=True, errors="coerce")
    return df, markets


def selected_clean_events(cmap: pd.DataFrame, min_legs: int, max_events: int | None) -> pd.DataFrame:
    g = cmap.dropna(subset=["close"]).copy()
    sizes = g.groupby("event_id").size()
    wins = g.groupby("event_id")["won"].sum()
    clean_ids = sizes[(sizes >= min_legs)].index.intersection(wins[wins == 1].index)
    clean = g[g["event_id"].isin(clean_ids)].copy()
    if max_events:
        volume = clean.groupby("event_id")["historical_volume"].sum().sort_values(ascending=False)
        clean = clean[clean["event_id"].isin(volume.head(max_events).index)].copy()
    return clean


def load_fills(data_dir: Path, market_ids: list[int]) -> pd.DataFrame:
    con = duckdb.connect()
    con.execute("CREATE TEMP TABLE sel(market_id BIGINT)")
    con.executemany("INSERT INTO sel VALUES (?)", [(int(m),) for m in market_ids])
    fpath = str((data_dir / "fills_sorted.parquet").resolve()).replace("'", "''")
    fills = con.execute(
        f"""
        SELECT f.market_id, f.timestamp, f.side, f.price, f.usd_amount
        FROM read_parquet('{fpath}') AS f JOIN sel USING (market_id)
        ORDER BY f.market_id, f.timestamp
        """
    ).fetchdf()
    con.close()
    fills["timestamp"] = pd.to_datetime(fills["timestamp"], utc=True)
    fills["side"] = fills["side"].astype(str)
    return fills


def build_event_candidates(
    clean: pd.DataFrame,
    fills: pd.DataFrame,
    lead_hours: list[float],
    outcome_side: str,
    max_staleness_hours: float,
    min_legs: int,
    min_coverage: float,
    participation_fraction: float,
    max_leg_price: float,
    require_fade_side_fill: bool,
) -> pd.DataFrame:
    opposite_side = "token2" if outcome_side == "token1" else "token1"
    stale = pd.Timedelta(hours=max_staleness_hours) if max_staleness_hours > 0 else None
    fills_by_market = {int(mid): g for mid, g in fills.groupby("market_id", sort=False)}

    rows: list[dict[str, Any]] = []
    for lead_hour in lead_hours:
        lead = pd.Timedelta(hours=lead_hour)
        for event_id, event in clean.groupby("event_id", sort=False):
            event_close = event["close"].min()
            if pd.isna(event_close):
                continue
            cutoff = event_close - lead
            leg_data: dict[int, dict[str, float | pd.Timestamp]] = {}
            legs = event["market_id"].astype(int).tolist()
            for mid in legs:
                g = fills_by_market.get(mid)
                if g is None:
                    continue
                window = g[g["timestamp"] <= cutoff]
                if stale is not None:
                    window = window[window["timestamp"] >= cutoff - stale]
                outcome = window[window["side"] == outcome_side]
                if outcome.empty:
                    continue
                last_outcome = outcome.iloc[-1]
                outcome_price = float(last_outcome["price"])
                if not math.isfinite(outcome_price) or outcome_price <= 0.0 or outcome_price >= 1.0:
                    continue
                fade_price = max(0.0, min(1.0, 1.0 - outcome_price))
                if fade_price <= 0.0 or fade_price > max_leg_price:
                    continue
                fade_side = window[window["side"] == opposite_side]
                fade_usd = float(fade_side.iloc[-1]["usd_amount"]) if not fade_side.empty else float("nan")
                if require_fade_side_fill and (not math.isfinite(fade_usd) or fade_usd <= 0.0):
                    continue
                leg_data[mid] = {
                    "timestamp": cutoff,
                    "outcome_price": outcome_price,
                    "fade_price": fade_price,
                    "fade_usd": fade_usd if math.isfinite(fade_usd) else 0.0,
                    "last_trade": pd.Timestamp(last_outcome["timestamp"]),
                }
            priced = [mid for mid in legs if mid in leg_data]
            if len(priced) < min_legs:
                continue
            coverage = len(priced) / len(legs)
            if coverage < min_coverage:
                continue
            winner_rows = event[event["won"] == True]
            if winner_rows.empty:
                continue
            winner_mid = int(winner_rows["market_id"].iloc[0])
            if winner_mid not in priced:
                continue

            prices = np.asarray([float(leg_data[mid]["outcome_price"]) for mid in priced], dtype=float)
            fade_prices = np.asarray([float(leg_data[mid]["fade_price"]) for mid in priced], dtype=float)
            fade_usd = np.asarray([float(leg_data[mid]["fade_usd"]) for mid in priced], dtype=float)
            basket_cost = float(fade_prices.sum())
            gross_payoff = float(len(priced) - 1)
            profit_per_unit = float(prices.sum() - 1.0)
            if basket_cost <= 0.0:
                continue
            # Equal-contract basket size. Each leg's dollar spend is q * fade_price.
            positive_cap = np.where(fade_prices > 0, participation_fraction * fade_usd / fade_prices, np.inf)
            unit_capacity = float(np.nanmin(positive_cap)) if len(positive_cap) else 0.0
            if not math.isfinite(unit_capacity) or unit_capacity <= 0.0:
                continue
            entry_time = pd.Timestamp(cutoff)
            rows.append({
                "timestamp": entry_time,
                "close_time": event_close,
                "event_id": event_id,
                "lead_hours": lead_hour,
                "total_legs": int(len(legs)),
                "legs_priced": int(len(priced)),
                "coverage": float(coverage),
                "sum_legs": float(prices.sum()),
                "overround": profit_per_unit,
                "basket_cost_per_unit": basket_cost,
                "gross_payoff_per_unit": gross_payoff,
                "unit_return": profit_per_unit / basket_cost,
                "unit_capacity": unit_capacity,
                "max_capacity_capital": unit_capacity * basket_cost,
                "min_fade_leg_usd": float(np.nanmin(fade_usd)),
                "mean_fade_leg_usd": float(np.nanmean(fade_usd)),
                "winner_market_id": winner_mid,
                "winner_price": float(leg_data[winner_mid]["outcome_price"]),
                "market_ids": " ".join(str(mid) for mid in priced),
            })
    return pd.DataFrame(rows)


def replay_threshold(
    candidates: pd.DataFrame,
    threshold: float,
    max_legs: int | None,
    initial_cash: float,
    period_budget: float,
    reserve_fraction: float,
    min_event_capital: float,
    max_event_capital: float,
    max_events_per_month: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame()
    selected = candidates[candidates["overround"] > threshold].copy()
    if max_legs is not None:
        selected = selected[selected["legs_priced"] <= max_legs]
    selected = selected.sort_values(["timestamp", "overround"], ascending=[True, False], kind="stable")

    month_spend: dict[str, float] = defaultdict(float)
    month_counts: dict[str, int] = defaultdict(int)
    cash = initial_cash
    reserve_floor = initial_cash * reserve_fraction
    signal_rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = defaultdict(int)

    for row in selected.itertuples(index=False):
        ts = pd.Timestamp(row.timestamp)
        key = month_key(ts)
        if month_counts[key] >= max_events_per_month:
            skipped["month_event_cap"] += 1
            continue
        available = min(
            max(0.0, period_budget - month_spend[key]),
            max(0.0, cash - reserve_floor),
            float(row.max_capacity_capital),
            max_event_capital,
        )
        if available < min_event_capital:
            skipped["below_min_capital_or_budget"] += 1
            continue
        profit = available * float(row.unit_return)
        cash += profit
        month_spend[key] += available
        month_counts[key] += 1
        leg_cap_name = "all" if max_legs is None else str(max_legs)
        signal_rows.append({
            **row._asdict(),
            "market_id": synthetic_market_id(str(row.event_id), float(row.lead_hours)),
            "strategy": f"cross_market_overround_fade_{int(threshold * 10000):04d}bp_maxlegs_{leg_cap_name}",
            "exit_rule": f"lead_{float(row.lead_hours):g}h_no_basket",
            "sleeve": "cross_market",
            "category": "cross_market",
            "horizon_days": float(row.lead_hours) / 24.0,
            "entry_fill_usd": float(row.max_capacity_capital),
            "deployed": available,
            "realized_profit": profit,
            "account_cash_after": cash,
            "threshold": threshold,
            "max_legs_cap": 0 if max_legs is None else max_legs,
        })

    sig = pd.DataFrame(signal_rows)
    period_rows: list[dict[str, Any]] = []
    if not sig.empty:
        for month, g in sig.groupby(sig["timestamp"].apply(lambda x: month_key(pd.Timestamp(x)))):
            profits = g["realized_profit"].to_numpy(dtype=float)
            deployed = g["deployed"].to_numpy(dtype=float)
            period_rows.append({
                "strategy": "cross_market_overround_fade",
                "threshold": threshold,
                "max_legs_cap": 0 if max_legs is None else max_legs,
                "month": month,
                "entries": int(len(g)),
                "deployed": float(deployed.sum()),
                "realized_profit": float(profits.sum()),
                "return_on_capital": float(profits.sum() / deployed.sum()) if deployed.sum() else 0.0,
                "median_event_roc": float(g["unit_return"].median()),
                "mean_event_roc": float(g["unit_return"].mean()),
                "median_overround": float(g["overround"].median()),
                "mean_legs": float(g["legs_priced"].mean()),
                "worst_event_profit": float(profits.min()) if len(profits) else 0.0,
                "best_event_profit": float(profits.max()) if len(profits) else 0.0,
            })
    if skipped:
        period_rows.append({
            "strategy": "cross_market_overround_fade",
            "threshold": threshold,
            "max_legs_cap": 0 if max_legs is None else max_legs,
            "month": "_skipped",
            "entries": 0,
            "deployed": 0.0,
            "realized_profit": 0.0,
            "return_on_capital": 0.0,
            "median_event_roc": 0.0,
            "mean_event_roc": 0.0,
            "median_overround": 0.0,
            "mean_legs": 0.0,
            "worst_event_profit": 0.0,
            "best_event_profit": 0.0,
            "skipped": json.dumps(dict(skipped), sort_keys=True),
        })
    return sig, pd.DataFrame(period_rows)


def summarize(periods: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    usable = periods[periods["month"] != "_skipped"].copy() if not periods.empty else periods
    if usable.empty:
        return out
    for (threshold, max_legs_cap), g in usable.groupby(["threshold", "max_legs_cap"], dropna=False):
        p = g["realized_profit"].to_numpy(dtype=float)
        d = g["deployed"].to_numpy(dtype=float)
        out.append({
            "strategy": "cross_market_overround_fade",
            "threshold": float(threshold),
            "max_legs_cap": int(max_legs_cap),
            "months": int(len(g)),
            "total_entries": int(g["entries"].sum()),
            "total_deployed": float(d.sum()),
            "total_profit": float(p.sum()),
            "return_on_capital": float(p.sum() / d.sum()) if d.sum() else 0.0,
            "mean_monthly_profit": float(p.mean()),
            "median_monthly_profit": float(np.median(p)),
            "positive_rate": float(np.mean(p > 0)),
            "worst_month": float(p.min()),
            "best_month": float(p.max()),
            "mean_monthly_deployed": float(d.mean()),
            "mean_entries": float(g["entries"].mean()),
            "median_event_roc": float(g["median_event_roc"].median()),
            "mean_legs": float(g["mean_legs"].mean()),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("archive/processed/underdog_events"))
    parser.add_argument("--cluster-map", type=Path, default=Path("reports/event_clusters/market_event_map.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/cross_market_overround"))
    parser.add_argument("--min-legs", type=int, default=3)
    parser.add_argument("--max-legs", nargs="+", default=["20", "50", "80"],
                        help="Sweep leg-count caps; use 0 to disable a cap.")
    parser.add_argument("--max-events", type=int, default=0, help="Limit to top-volume clean events; 0 = all.")
    parser.add_argument("--lead-hours", type=float, nargs="+", default=[24.0, 72.0, 168.0])
    parser.add_argument("--thresholds", nargs="+", default=["0.02", "0.05", "0.10", "0.20"])
    parser.add_argument("--min-coverage", type=float, default=0.90)
    parser.add_argument("--max-staleness-hours", type=float, default=48.0)
    parser.add_argument("--outcome-side", choices=["token1", "token2"], default="token1")
    parser.add_argument("--participation-fraction", type=float, default=0.10)
    parser.add_argument("--max-leg-price", type=float, default=0.99)
    parser.add_argument("--allow-missing-fade-fill", action="store_true",
                        help="Use zero capacity instead of requiring recent opposite-side fill liquidity.")
    parser.add_argument("--initial-cash", type=float, default=5000.0)
    parser.add_argument("--period-budget", type=float, default=5000.0)
    parser.add_argument("--reserve-fraction", type=float, default=0.30)
    parser.add_argument("--min-event-capital", type=float, default=0.25)
    parser.add_argument("--max-event-capital", type=float, default=250.0)
    parser.add_argument("--max-events-per-month", type=int, default=1000)
    args = parser.parse_args()

    thresholds = parse_thresholds(args.thresholds)
    max_legs_values = parse_int_sweep(args.max_legs)
    cmap, _ = load_inputs(args.data_dir, args.cluster_map)
    clean = selected_clean_events(cmap, args.min_legs, args.max_events or None)
    if clean.empty:
        raise SystemExit("No clean event clusters matched.")
    fills = load_fills(args.data_dir, clean["market_id"].astype(int).unique().tolist())
    candidates = build_event_candidates(
        clean=clean,
        fills=fills,
        lead_hours=args.lead_hours,
        outcome_side=args.outcome_side,
        max_staleness_hours=args.max_staleness_hours,
        min_legs=args.min_legs,
        min_coverage=args.min_coverage,
        participation_fraction=args.participation_fraction,
        max_leg_price=args.max_leg_price,
        require_fade_side_fill=not args.allow_missing_fade_fill,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.output_dir / "candidate_events.csv", index=False)

    signal_frames = []
    period_frames = []
    for threshold in thresholds:
        for max_legs_raw in max_legs_values:
            max_legs = None if max_legs_raw <= 0 else max_legs_raw
            sig, periods = replay_threshold(
                candidates,
                threshold,
                max_legs,
                args.initial_cash,
                args.period_budget,
                args.reserve_fraction,
                args.min_event_capital,
                args.max_event_capital,
                args.max_events_per_month,
            )
            if not sig.empty:
                signal_frames.append(sig)
            if not periods.empty:
                period_frames.append(periods)
    signals = pd.concat(signal_frames, ignore_index=True, sort=False) if signal_frames else pd.DataFrame()
    period_results = pd.concat(period_frames, ignore_index=True, sort=False) if period_frames else pd.DataFrame()
    signals.to_csv(args.output_dir / "cross_market_signals.csv", index=False)
    write_csv(args.output_dir / "cross_market_period_results.csv", period_results.to_dict("records"))
    summary = summarize(period_results)
    write_csv(args.output_dir / "cross_market_summary.csv", summary)
    (args.output_dir / "summary.json").write_text(json.dumps({
        "clean_events": int(clean["event_id"].nunique()),
        "candidate_events": int(len(candidates)),
        "thresholds": thresholds,
        "max_legs": max_legs_values,
        "lead_hours": args.lead_hours,
        "files": [
            "candidate_events.csv",
            "cross_market_signals.csv",
            "cross_market_period_results.csv",
            "cross_market_summary.csv",
        ],
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "clean_events": int(clean["event_id"].nunique()),
        "candidate_events": int(len(candidates)),
        "signals": int(len(signals)),
        "summary_rows": len(summary),
    }, indent=2))


if __name__ == "__main__":
    main()
