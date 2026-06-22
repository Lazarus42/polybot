#!/usr/bin/env python3
"""Axis 2, step 4: does the short-horizon signal make money NET of costs in liquid markets?

Walk-forward-predict the 24h price move, restrict to liquid markets, go long the top-decile
predicted-up and short (buy the opposite token) the bottom-decile predicted-down, exit at
24h. Charge a round-trip cost per name and report net monthly P&L, Sharpe, positive-month
rate and a capacity ($/mo) estimate. The point is the cost sweep: if net P&L is only positive
at ~0 cost, it isn't worth the buck.

Core (`clv_long_short_backtest`) is pure and unit-tested; tape loading needs the venv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def clv_long_short_backtest(df: pd.DataFrame, roundtrip_cost: float,
                            top_frac: float = 0.1, participation: float = 0.1) -> dict[str, Any]:
    """df rows need: period (month key), pred (OOS prediction), fwd_move (realized 24h move
    in price), fill (entry_fill_usd for capacity). Each period: long the top `top_frac` by
    pred, short the bottom `top_frac`; per-name pnl = side*fwd_move - roundtrip_cost. Returns
    monthly net return (per $ deployed) and a capacity-weighted net $ series."""
    d = df.dropna(subset=["pred", "fwd_move", "period"]).copy()
    monthly_ret, monthly_dollars, monthly_dep = [], [], []
    for _, g in d.groupby("period"):
        if len(g) < 10:
            continue
        k = max(1, int(len(g) * top_frac))
        order = g["pred"].to_numpy().argsort()
        longs = g.iloc[order[-k:]]
        shorts = g.iloc[order[:k]]
        # per-name net return in price units (cost charged round trip)
        long_ret = longs["fwd_move"].to_numpy() - roundtrip_cost
        short_ret = -shorts["fwd_move"].to_numpy() - roundtrip_cost
        rets = np.concatenate([long_ret, short_ret])
        monthly_ret.append(float(rets.mean()))
        # capacity: dollars deployable per name = participation * fill, net $ = ret * dollars
        ld = (participation * longs["fill"].to_numpy()); sd = (participation * shorts["fill"].to_numpy())
        net_dollars = float((long_ret * ld).sum() + (short_ret * sd).sum())
        monthly_dollars.append(net_dollars)
        monthly_dep.append(float(ld.sum() + sd.sum()))
    r = np.array(monthly_ret, float)
    dollars = np.array(monthly_dollars, float)
    dep = np.array(monthly_dep, float)
    std = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    return {
        "months": int(len(r)),
        "mean_monthly_return": float(r.mean()) if len(r) else float("nan"),
        "ann_sharpe": float(r.mean() / std * np.sqrt(12)) if std > 0 else 0.0,
        "positive_month_rate": float((r > 0).mean()) if len(r) else float("nan"),
        "net_dollars_total": float(dollars.sum()),
        "net_dollars_per_month": float(dollars.mean()) if len(dollars) else 0.0,
        "mean_deployed_per_month": float(dep.mean()) if len(dep) else 0.0,
    }


def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from forward_return_predictability import (
        FEATURES, load_forward_moves_duckdb, walk_forward_predictions,
    )

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signals", type=Path, default=Path("reports/strategy_family_diagnostics/strategy_family_signals.csv"))
    ap.add_argument("--fills", type=Path, default=Path("archive/processed/underdog_events/fills_sorted.parquet"))
    ap.add_argument("--output-dir", type=Path, default=Path("reports/clv_strategy"))
    ap.add_argument("--horizon", type=float, default=24.0)
    ap.add_argument("--liquidity-quantile", type=float, default=0.66,
                    help="Only trade markets with entry_fill_usd above this quantile (liquid subset).")
    ap.add_argument("--top-frac", type=float, default=0.1)
    ap.add_argument("--participation", type=float, default=0.1)
    ap.add_argument("--max-entries", type=int, default=80000)
    ap.add_argument("--smooth-trades", type=int, default=0,
                    help="Debounce price with a trailing VWAP over the last N token1 trades (try 15).")
    args = ap.parse_args()

    df = pd.read_csv(args.signals, low_memory=False)
    ts = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
    df["entry_ts"] = ts.astype("int64") / 1e9
    df["period"] = ts.dt.strftime("%Y-%m")
    for c in FEATURES + ["entry_fill_usd"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["market_id", "entry_ts", "entry_fill_usd"])
    if len(df) > args.max_entries:
        df = df.sample(args.max_entries, random_state=0)
    df = df.reset_index(drop=True)
    df["row_id"] = np.arange(len(df))

    moves = load_forward_moves_duckdb(args.fills, df, [args.horizon], args.smooth_trades)
    moves["fwd_move"] = moves[f"fwd_move_{args.horizon}h"]
    moves["fill"] = moves["entry_fill_usd"]
    feats = [c for c in FEATURES if c in moves.columns]
    moves = moves.dropna(subset=feats + ["fwd_move", "period"])

    # Restrict to the LIQUID subset FIRST, then train the model within it — the signal is
    # liquidity-regime-specific (illiquid IC is ~0/negative), so a pooled model dilutes or
    # inverts it. Train where we trade.
    liq_cut = moves["fill"].quantile(args.liquidity_quantile)
    liquid = moves[moves["fill"] >= liq_cut].sort_values("entry_ts").reset_index(drop=True)
    liquid["pred"] = walk_forward_predictions(liquid, feats, "fwd_move")
    print(f"liquid subset: {len(liquid)} of {len(moves)} (fill >= ${liq_cut:.0f})")

    results = {}
    for cost in (0.0, 0.005, 0.01, 0.015, 0.02, 0.03):
        res = clv_long_short_backtest(liquid, cost, args.top_frac, args.participation)
        results[f"cost_{cost}"] = res
        print(json.dumps({"roundtrip_cost": cost,
                          "mean_monthly_ret": round(res["mean_monthly_return"], 4),
                          "ann_sharpe": round(res["ann_sharpe"], 2),
                          "pos_months": round(res["positive_month_rate"], 2),
                          "net_$/mo": round(res["net_dollars_per_month"], 1),
                          "deployed/mo": round(res["mean_deployed_per_month"], 0)}))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "clv_backtest.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {args.output_dir}/clv_backtest.json")
    print("read: find the round-trip cost where net_$/mo crosses 0 — that's the spread you must beat.")


if __name__ == "__main__":
    main()
