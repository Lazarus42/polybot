#!/usr/bin/env python3
"""Axis 2: are SHORT-HORIZON price moves predictable from our features?

The efficiency test (GBM vs price) showed we cannot predict *resolution outcomes* better
than the market. That does not settle whether intra-life price *moves* are predictable —
which is the relevant question for a ~1-day-holding, closing-line-value strategy. This
harness tests it directly: for each entry, pull the price now and the price H hours later
from the tape, and ask whether our features have out-of-sample skill at the forward move.

Pipeline (tape loader needs the project venv):
  1. For each signal entry (market_id, entry_ts, features): ASOF the entry price and the
     price H hours later from the fill tape; forward_move = p(t+H) - p(t) in probability.
  2. Walk-forward: fit a (standardized) linear model of forward_move on the features using
     only past entries; predict the held-out future block; accumulate OOS predictions.
  3. Score skill with the information coefficient (Spearman corr of prediction vs realized)
     and a decile long-short return (information ratio). IC ~ 0 => no exploitable path edge;
     IC reliably > 0 => a tradeable short-horizon signal.

Pure core (`information_coefficient`, `long_short_return`, `walk_forward_ic`) is unit-tested
on synthetic data with a planted signal in `tests/test_forward_predictability.py`.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FEATURES = [
    "recent_price_move_24h", "recent_price_move_1h", "recent_price_move_6h",
    "recent_price_move_7d", "recent_volatility_24h", "recent_accel_24h",
    "recent_flow_imbalance_24h", "horizon_days", "limit_price",
]


def information_coefficient(pred: np.ndarray, realized: np.ndarray) -> float:
    """Spearman rank correlation between prediction and realized forward move."""
    pred, realized = np.asarray(pred, float), np.asarray(realized, float)
    ok = np.isfinite(pred) & np.isfinite(realized)
    if ok.sum() < 10:
        return float("nan")
    pr = pd.Series(pred[ok]).rank().to_numpy()
    rr = pd.Series(realized[ok]).rank().to_numpy()
    if pr.std() == 0 or rr.std() == 0:
        return 0.0
    return float(np.corrcoef(pr, rr)[0, 1])


def long_short_return(pred: np.ndarray, realized: np.ndarray, frac: float = 0.2) -> float:
    """Mean realized move of the top-`frac` predictions minus the bottom-`frac`
    (a costless decile long-short — the gross edge a signal would capture)."""
    pred, realized = np.asarray(pred, float), np.asarray(realized, float)
    ok = np.isfinite(pred) & np.isfinite(realized)
    pred, realized = pred[ok], realized[ok]
    if len(pred) < 20:
        return float("nan")
    k = max(1, int(len(pred) * frac))
    order = np.argsort(pred)
    return float(realized[order[-k:]].mean() - realized[order[:k]].mean())


def _fit_predict_ols(Xtr, ytr, Xte, l2=1e-3):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr = np.column_stack([np.ones(len(Xtr)), (Xtr - mu) / sd])
    Xte = np.column_stack([np.ones(len(Xte)), (Xte - mu) / sd])
    A = Xtr.T @ Xtr + l2 * np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, Xtr.T @ ytr)
    return Xte @ w


def walk_forward_predictions(df: pd.DataFrame, feature_cols: list[str], target: str,
                             n_folds: int = 5) -> np.ndarray:
    """Time-ordered walk-forward OOS predictions aligned to df's (sorted) rows; NaN where
    not scored. df must be pre-sorted by entry_ts."""
    n = len(df)
    preds = np.full(n, np.nan)
    if n < n_folds * 40:
        return preds
    X = df[feature_cols].to_numpy(float)
    y = df[target].to_numpy(float)
    bounds = np.linspace(0, n, n_folds + 1).astype(int)
    for i in range(1, n_folds):
        tr = slice(0, bounds[i])
        te = slice(bounds[i], bounds[i + 1])
        if te.stop - te.start < 10:
            continue
        preds[te] = _fit_predict_ols(X[tr], y[tr], X[te])
    return preds


def walk_forward_ic(df: pd.DataFrame, feature_cols: list[str], target: str,
                    n_folds: int = 5) -> dict[str, Any]:
    """Time-ordered walk-forward; returns OOS information coefficient + long-short edge."""
    df = df.dropna(subset=feature_cols + [target]).sort_values("entry_ts")
    n = len(df)
    if n < n_folds * 40:
        return {"n": n, "oos_ic": float("nan"), "oos_long_short": float("nan")}
    y = df[target].to_numpy(float)
    preds = walk_forward_predictions(df, feature_cols, target, n_folds)
    mask = np.isfinite(preds)
    return {
        "n": int(n),
        "n_scored": int(mask.sum()),
        "oos_ic": information_coefficient(preds[mask], y[mask]),
        "oos_long_short": long_short_return(preds[mask], y[mask]),
    }


def feature_ic_breakdown(df: pd.DataFrame, feature_cols: list[str], target: str,
                         n_folds: int = 5) -> dict[str, float]:
    """Univariate OOS IC for each feature alone — identifies what drives the signal
    (e.g. is it just momentum?)."""
    return {f: walk_forward_ic(df, [f], target, n_folds)["oos_ic"] for f in feature_cols}


def bucketed_ic(df: pd.DataFrame, feature_cols: list[str], target: str,
                bucket_col: str, n_buckets: int = 3, n_folds: int = 5) -> dict[str, Any]:
    """OOS IC within terciles of `bucket_col` (e.g. entry_fill_usd) — is the edge in the
    LIQUID, tradeable markets or only the illiquid ones?"""
    d = df.dropna(subset=[bucket_col]).copy()
    try:
        d["_b"] = pd.qcut(d[bucket_col], n_buckets, labels=False, duplicates="drop")
    except ValueError:
        return {}
    out = {}
    for b, g in d.groupby("_b"):
        r = walk_forward_ic(g, feature_cols, target, n_folds)
        out[f"bucket_{int(b)}"] = {"oos_ic": r["oos_ic"], "n": r["n"],
                                   "median_liq": float(g[bucket_col].median())}
    return out


def net_decile_long_short(df: pd.DataFrame, feature_cols: list[str], target: str,
                          cost: float, n_folds: int = 5, frac: float = 0.1) -> float:
    """Gross decile long-short minus a per-leg transaction cost (round-trip = 2*cost),
    so the user can see whether the edge survives realistic spreads/slippage."""
    df = df.dropna(subset=feature_cols + [target]).sort_values("entry_ts")
    n = len(df)
    if n < n_folds * 40:
        return float("nan")
    X = df[feature_cols].to_numpy(float); y = df[target].to_numpy(float)
    bounds = np.linspace(0, n, n_folds + 1).astype(int)
    preds = np.full(n, np.nan)
    for i in range(1, n_folds):
        tr = slice(0, bounds[i]); te = slice(bounds[i], bounds[i + 1])
        if te.stop - te.start < 10:
            continue
        preds[te] = _fit_predict_ols(X[tr], y[tr], X[te])
    mask = np.isfinite(preds)
    gross = long_short_return(preds[mask], y[mask], frac)
    return float(gross - 2.0 * cost)


def load_forward_moves_duckdb(fills_path: Path, entries: pd.DataFrame,
                              horizons_hours: list[float], smooth_trades: int = 0) -> pd.DataFrame:
    """For each entry (market_id, entry_ts), ASOF the token1 price now and at t+H hours.

    `smooth_trades` > 0 replaces the raw last-trade price with a causal trailing usd-weighted
    VWAP over the last N token1 trades (a debounced mid proxy that strips bid-ask bounce, which
    is the thing suspected of manufacturing the short-horizon signal). 0 = raw last trade.
    """
    import duckdb  # noqa: PLC0415

    con = duckdb.connect()
    con.execute("SET threads = 4")
    p = str(fills_path.resolve()).replace("'", "''")
    if smooth_trades and smooth_trades > 1:
        price_expr = (f"sum(price*usd_amount) OVER w / nullif(sum(usd_amount) OVER w, 0)")
        window = (f"WINDOW w AS (PARTITION BY market_id ORDER BY ts "
                  f"ROWS BETWEEN {smooth_trades - 1} PRECEDING AND CURRENT ROW)")
    else:
        price_expr, window = "price", ""
    con.execute(f"""
        CREATE TEMP TABLE tape AS
        SELECT market_id, ts, {price_expr} AS price FROM (
            SELECT market_id, epoch(timestamp) AS ts, price, usd_amount
            FROM read_parquet('{p}')
            WHERE price > 0 AND price < 1 AND side = 'token1'
        ) {window}
    """)
    e = entries.copy()
    e["market_id"] = e["market_id"].astype("int64")
    con.register("entries", e[["row_id", "market_id", "entry_ts"]])
    out = entries.copy()
    base = con.execute("""
        SELECT s.row_id, t.price AS p0
        FROM entries s ASOF LEFT JOIN tape t
        ON s.market_id = t.market_id AND t.ts <= s.entry_ts
    """).fetch_df()
    out = out.merge(base, on="row_id", how="left")
    for h in horizons_hours:
        sec = h * 3600.0
        con.execute(f"CREATE OR REPLACE TEMP VIEW q AS SELECT row_id, market_id, entry_ts + {sec} AS tt FROM entries")
        fwd = con.execute("""
            SELECT q.row_id, t.price AS pH
            FROM q ASOF LEFT JOIN tape t
            ON q.market_id = t.market_id AND t.ts <= q.tt
        """).fetch_df()
        out = out.merge(fwd.rename(columns={"pH": f"p_{h}h"}), on="row_id", how="left")
        out[f"fwd_move_{h}h"] = out[f"p_{h}h"] - out["p0"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signals", type=Path, default=Path("reports/strategy_family_diagnostics/strategy_family_signals.csv"))
    ap.add_argument("--fills", type=Path, default=Path("archive/processed/underdog_events/fills_sorted.parquet"))
    ap.add_argument("--output-dir", type=Path, default=Path("reports/forward_predictability"))
    ap.add_argument("--horizons", type=float, nargs="+", default=[1.0, 6.0, 24.0])
    ap.add_argument("--max-entries", type=int, default=60000, help="Subsample entries for speed.")
    ap.add_argument("--smooth-trades", type=int, default=0,
                    help="Debounce: use a trailing VWAP over the last N token1 trades instead of the "
                         "raw last trade (strips bid-ask bounce). 0 = raw. Try 10-20.")
    args = ap.parse_args()

    df = pd.read_csv(args.signals, low_memory=False)
    df["entry_ts"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed").astype("int64") / 1e9
    for c in FEATURES + ["entry_fill_usd"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["market_id", "entry_ts"])
    if len(df) > args.max_entries:
        df = df.sample(args.max_entries, random_state=0)
    df = df.reset_index(drop=True)
    df["row_id"] = np.arange(len(df))

    moves = load_forward_moves_duckdb(args.fills, df, args.horizons, args.smooth_trades)
    feats = [c for c in FEATURES if c in moves.columns]
    results = {}
    for h in args.horizons:
        tgt = f"fwd_move_{h}h"
        res = walk_forward_ic(moves, feats, tgt)
        results[f"{h}h"] = res
        print(json.dumps({"horizon": f"{h}h", **{k: (round(v, 4) if isinstance(v, float) else v)
                                                 for k, v in res.items()}}))
    # breakdown on the longest horizon (where the signal is strongest)
    h = max(args.horizons); tgt = f"fwd_move_{h}h"
    breakdown = {
        "horizon": f"{h}h",
        "feature_ic": feature_ic_breakdown(moves, feats, tgt),
        "liquidity_buckets": bucketed_ic(moves, feats, tgt, "entry_fill_usd")
        if "entry_fill_usd" in moves.columns else {},
        "net_decile_long_short": {f"cost_{c}": net_decile_long_short(moves, feats, tgt, c)
                                  for c in (0.0, 0.01, 0.02, 0.03)},
    }
    results["breakdown"] = breakdown
    print("\n=== 24h breakdown ===")
    print("per-feature OOS IC:", json.dumps({k: round(v, 4) for k, v in breakdown["feature_ic"].items()}))
    print("IC by liquidity tercile:", json.dumps({k: round(v["oos_ic"], 4)
                                                  for k, v in breakdown["liquidity_buckets"].items()}))
    print("net decile long-short vs per-leg cost:",
          json.dumps({k: round(v, 4) for k, v in breakdown["net_decile_long_short"].items()}))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "forward_ic.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {args.output_dir}/forward_ic.json")
    print("read: which feature carries the IC (momentum?), whether it lives in liquid markets"
          " (tradeable) vs illiquid, and whether the decile edge survives spread/slippage.")


if __name__ == "__main__":
    main()
