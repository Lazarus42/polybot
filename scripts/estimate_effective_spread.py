#!/usr/bin/env python3
"""Estimate the real effective round-trip spread per market from the trade tape.

The CLV strategy (clv_strategy_backtest.py) breaks even at a round-trip cost of ~1.1c. We
have been *assuming* the cost; this measures it. With no order book, we use the Roll (1984)
estimator: bid-ask bounce makes consecutive trade-price changes negatively serially
correlated, and the implied spread is

    spread = 2 * sqrt(-cov(dp_t, dp_{t-1}))     (when the covariance is negative)

That spread is the round-trip cost (cross half on entry, half on exit), directly comparable
to the strategy's breakeven. We report it by liquidity bucket so we can read the cost in the
exact markets the strategy trades (and deeper).

`roll_spread` / `roll_spread_from_cov` are pure and unit-tested; the per-market covariance is
computed in DuckDB (needs the venv).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def roll_spread_from_cov(cov: float) -> float:
    """Roll effective spread from the lag-1 serial covariance of price changes."""
    if cov is None or not np.isfinite(cov) or cov >= 0:
        return 0.0
    return 2.0 * math.sqrt(-cov)


def roll_spread(prices: np.ndarray) -> float:
    """Roll effective spread from a single market's trade-price series."""
    prices = np.asarray(prices, float)
    dp = np.diff(prices)
    if len(dp) < 3:
        return float("nan")
    cov = float(np.cov(dp[1:], dp[:-1])[0, 1])
    return roll_spread_from_cov(cov)


def load_market_spread_duckdb(fills_path: Path, min_trades: int = 30) -> pd.DataFrame:
    """Per-market Roll covariance + liquidity, computed in-DB. Returns one row per market
    with cov, n_trades, median_trade_usd, total_usd."""
    import duckdb  # noqa: PLC0415

    con = duckdb.connect()
    con.execute("SET threads = 4")
    p = str(fills_path.resolve()).replace("'", "''")
    df = con.execute(f"""
        WITH base AS (
            -- ONE token's price series only: mixing token1 (price p) and token2 (price 1-p)
            -- would manufacture huge fake serial covariance from outcome-switching.
            SELECT market_id, epoch(timestamp) AS ts, price, usd_amount
            FROM read_parquet('{p}') WHERE price > 0 AND price < 1 AND side = 'token1'
        ),
        d AS (
            SELECT market_id, ts, usd_amount, price,
                   price - lag(price) OVER (PARTITION BY market_id ORDER BY ts) AS dp
            FROM base
        ),
        d2 AS (
            SELECT market_id, usd_amount,
                   dp, lag(dp) OVER (PARTITION BY market_id ORDER BY ts) AS dp_prev
            FROM d
        )
        SELECT market_id,
               covar_samp(dp, dp_prev)      AS cov,
               count(*)                      AS n_trades,
               median(usd_amount)            AS median_trade_usd,
               sum(usd_amount)               AS total_usd
        FROM d2
        GROUP BY market_id
        HAVING count(*) >= {min_trades}
    """).fetch_df()
    return df


def summarize_spread(df: pd.DataFrame, liquidity_col: str = "median_trade_usd",
                     breakeven: float = 0.011) -> dict[str, Any]:
    """Roll spread by liquidity bucket vs the strategy breakeven."""
    d = df.copy()
    d["roll_spread"] = d["cov"].map(roll_spread_from_cov)
    out: dict[str, Any] = {
        "markets": int(len(d)),
        "median_roll_spread_all": float(d["roll_spread"].median()),
        "breakeven": breakeven,
    }
    try:
        d["liq_bucket"] = pd.qcut(d[liquidity_col], 5, labels=False, duplicates="drop")
        buckets = {}
        for b, g in d.groupby("liq_bucket"):
            buckets[f"q{int(b)}"] = {
                "median_liq_usd": round(float(g[liquidity_col].median()), 1),
                "median_roll_spread": round(float(g["roll_spread"].median()), 4),
                "frac_below_breakeven": round(float((g["roll_spread"] < breakeven).mean()), 3),
                "n": int(len(g)),
            }
        out["by_liquidity_quintile"] = buckets
    except ValueError:
        pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fills", type=Path, default=Path("archive/processed/underdog_events/fills_sorted.parquet"))
    ap.add_argument("--output-dir", type=Path, default=Path("reports/effective_spread"))
    ap.add_argument("--min-trades", type=int, default=30)
    ap.add_argument("--breakeven", type=float, default=0.011, help="CLV strategy breakeven round-trip cost.")
    args = ap.parse_args()

    df = load_market_spread_duckdb(args.fills, args.min_trades)
    summary = summarize_spread(df, breakeven=args.breakeven)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "effective_spread.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("\nread: in the liquidity buckets the CLV strategy trades, is median_roll_spread"
          f" below the {args.breakeven*100:.1f}c breakeven? If yes -> tradeable; if no -> spread eats it.")


if __name__ == "__main__":
    main()
