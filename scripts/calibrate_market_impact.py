#!/usr/bin/env python3
"""Calibrate the replay engine's market-impact coefficient from realized fills.

The replay engine (`replay_family_ensemble_oos.replay_ensemble`) models slippage as a
worse effective entry price:

    s     = coef * (debit / fill) ** alpha          # alpha = 1 linear, 0.5 sqrt
    r_eff = (1 + unit_return) / (1 + s) - 1

`coef` and `alpha` are free parameters. This script estimates them from the archived
trade tape (`fills_sorted.parquet`, columns: timestamp, market_id, side, price,
usd_amount, token_amount) with NO order-book depth required, via a causal pre/post
event study:

  1. For each trade, measure the permanent price displacement across the trade:
     pre_vwap = usd-weighted VWAP of the `--window` trades BEFORE it, post_vwap = the same
     for the `--window` trades AFTER it (same market), impact i = |post_vwap/pre_vwap - 1|.
     Averaging on both sides cancels bid-ask bounce, so a noise-driven small trade reads
     i ~ 0 and only trades that actually move the market register. This fixes the earlier
     trade-vs-VWAP estimator, which was dominated by small-trade bounce and gave a
     non-physical negative slope (impact falling with size). Pre/post measures *permanent*
     impact, a clean lower bound on execution slippage.
  2. relative size x = usd_amount / L, L = mean trade size over the pre window (this order
     vs a typical local fill, the empirical analog of debit/fill in the engine).
  3. Trades are bucketed into `--buckets` quantiles of x; per bucket we take the median
     x and median i (medians = robust to the fat tails of trade-level noise).
  4. Fit log(i) = log(coef) + alpha * log(x) across buckets (power law), and separately
     report the linear-model (alpha=1) coefficient median(i / x). Output is the
     recommended `--slippage-model`/`--slippage-coef` to pass to the backtests. A
     non-physical fit (alpha <= 0) is flagged rather than silently recommended.

KNOWN LIMITATION (read before trusting the point estimate): on a dense tape where every
trade moves price, the pre/post window also captures *neighbouring* trades' moves, which
biases the fitted exponent DOWNWARD (synthetic tests recover the sign and monotonicity but
only a fraction of a planted alpha). So treat the fitted (coef, alpha) as an
order-of-magnitude LOWER BOUND that confirms impact is positive and rising in size — and
make the actual sizing decision from the slippage SWEEP (run the participation ladder at
coef in {0.2, 0.5, 1.0}), not from this single point. Unbiased calibration needs order-book
depth, which this archive does not contain.

Engines: `--engine duckdb` (default; streams + samples in-DB, for the 64M-row file) or
`--engine pandas` (needs pyarrow; for small files / testing). Both feed the SAME
estimator (`fit_impact`), which is unit-tested on synthetic data in
`tests/test_impact_calibration.py`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def fit_impact(trades: pd.DataFrame, buckets: int = 20, agg: str = "mean") -> dict[str, Any]:
    """Estimate impact law from a per-trade frame with columns 'relsize','impact'.

    `impact` may be signed (pre/post causal estimator) or a positive magnitude. Trades are
    bucketed into quantiles of relsize; per bucket the central impact is aggregated by
    `agg` ('mean' isolates directional/permanent impact when impact is signed, because
    neighbour moves and bid-ask bounce average to zero; 'median' suits positive
    magnitudes). The law log(impact) = log(coef) + alpha*log(relsize) is fit across the
    buckets with positive aggregate impact. Returns power-law (coef, alpha), an implied
    linear (alpha=1) coef, per-bucket diagnostics, and R^2.
    """
    df = trades[["relsize", "impact"]].copy()
    df = df[(df["relsize"] > 0) & np.isfinite(df["relsize"]) & np.isfinite(df["impact"])]
    if len(df) < buckets * 5:
        raise ValueError(f"Too few valid trades ({len(df)}) for {buckets} buckets.")

    # robustness: trim extreme tails of size and of impact magnitude
    lo, hi = df["relsize"].quantile([0.005, 0.995])
    df = df[(df["relsize"] >= lo) & (df["relsize"] <= hi)]
    ilo, ihi = df["impact"].quantile([0.005, 0.995])
    df = df[(df["impact"] >= ilo) & (df["impact"] <= ihi)]

    df["bucket"] = pd.qcut(df["relsize"], q=buckets, labels=False, duplicates="drop")
    grp = df.groupby("bucket")
    agg_impact = grp["impact"].mean() if agg == "mean" else grp["impact"].median()
    diag = pd.DataFrame({
        "median_relsize": grp["relsize"].median(),
        "agg_impact": agg_impact,
        "n": grp.size(),
    }).reset_index(drop=True)

    fit_rows = diag[diag["agg_impact"] > 0]
    if len(fit_rows) < 5:
        raise ValueError(f"Only {len(fit_rows)} buckets with positive impact; cannot fit.")
    lx = np.log(fit_rows["median_relsize"].to_numpy())
    ly = np.log(fit_rows["agg_impact"].to_numpy())
    alpha, intercept = np.polyfit(lx, ly, 1)
    coef_power = float(np.exp(intercept))
    ss_res = float(np.sum((ly - (alpha * lx + intercept)) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # implied linear (alpha=1) coef: median bucket slope agg_impact / relsize
    coef_linear = float((fit_rows["agg_impact"] / fit_rows["median_relsize"]).median())

    return {
        "n_trades_used": int(len(df)),
        "n_buckets_fit": int(len(fit_rows)),
        "alpha": float(alpha),
        "coef_power": coef_power,
        "r2_loglog": float(r2),
        "coef_linear_alpha1": coef_linear,
        "diagnostics": diag,
    }


def load_trades_duckdb(path: Path, window: int, min_window: int,
                       sample: float, max_markets: int | None) -> pd.DataFrame:
    """Compute per-trade (relsize, impact) in DuckDB and return a sampled frame."""
    import duckdb  # noqa: PLC0415 — only needed for this engine

    con = duckdb.connect()
    con.execute("SET threads = 4")
    p = str(path.resolve()).replace("'", "''")
    market_filter = ""
    if max_markets is not None:
        market_filter = (
            f"WHERE market_id IN (SELECT market_id FROM "
            f"(SELECT DISTINCT market_id FROM read_parquet('{p}')) USING SAMPLE {max_markets} ROWS)"
        )
    # Sampling must happen AFTER windowing (sampling raw rows would corrupt the trailing
    # VWAP), so it is isolated in a clean outer SELECT over the filtered CTE.
    sample_clause = f"USING SAMPLE {sample * 100:.4f} PERCENT (bernoulli)" if 0 < sample < 1 else ""
    query = f"""
        WITH base AS (
            SELECT market_id, epoch(timestamp) AS ts, price, usd_amount
            FROM read_parquet('{p}')
            {market_filter}
        ),
        windowed AS (
            SELECT
                price,
                usd_amount,
                sum(price * usd_amount) OVER w_pre
                    / nullif(sum(usd_amount) OVER w_pre, 0)        AS pre_vwap,
                sum(price * usd_amount) OVER w_post
                    / nullif(sum(usd_amount) OVER w_post, 0)       AS post_vwap,
                avg(usd_amount) OVER w_pre                          AS l_scale,
                count(*) OVER w_pre                                 AS n_pre,
                count(*) OVER w_post                                AS n_post
            FROM base
            WINDOW
                w_pre AS (PARTITION BY market_id ORDER BY ts
                          ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING),
                w_post AS (PARTITION BY market_id ORDER BY ts
                           ROWS BETWEEN 1 FOLLOWING AND {window} FOLLOWING)
        ),
        valid AS (
            SELECT
                usd_amount / l_scale                                          AS relsize,
                CASE WHEN price >= pre_vwap THEN 1.0 ELSE -1.0 END
                    * (post_vwap / pre_vwap - 1.0)                            AS impact
            FROM windowed
            WHERE n_pre >= {min_window} AND n_post >= {min_window}
              AND pre_vwap > 0 AND post_vwap > 0 AND l_scale > 0
              AND usd_amount > 0
        )
        SELECT relsize, impact FROM valid
        {sample_clause}
    """
    return con.execute(query).fetch_df()


def compute_prepost_impact(raw: pd.DataFrame, window: int, min_window: int) -> pd.DataFrame:
    """Per-trade (relsize, impact) via the causal pre/post event study.

    `raw` needs columns market_id, timestamp, price, usd_amount. Pure (no I/O) so it can
    be unit-tested directly on synthetic tapes. The DuckDB loader implements the same
    definitions in SQL for the full 64M-row file.
    """
    raw = raw.sort_values(["market_id", "timestamp"])
    out = []
    for _, g in raw.groupby("market_id", sort=False):
        price = g["price"].to_numpy(float)
        usd = g["usd_amount"].to_numpy(float)
        n = len(g)
        for i in range(n):
            pre_lo = max(0, i - window)
            post_hi = min(n, i + 1 + window)
            pre_p, pre_u = price[pre_lo:i], usd[pre_lo:i]
            post_p, post_u = price[i + 1:post_hi], usd[i + 1:post_hi]
            if len(pre_p) < min_window or len(post_p) < min_window:
                continue
            pre_denom, post_denom = pre_u.sum(), post_u.sum()
            if pre_denom <= 0 or post_denom <= 0 or usd[i] <= 0:
                continue
            pre_vwap = float((pre_p * pre_u).sum() / pre_denom)
            post_vwap = float((post_p * post_u).sum() / post_denom)
            l_scale = float(pre_u.mean())
            if pre_vwap <= 0 or post_vwap <= 0 or l_scale <= 0:
                continue
            # infer aggressor direction from the trade's own print vs the pre-VWAP, then
            # measure the permanent move in that direction. Averaged per size bucket,
            # neighbour moves and bid-ask bounce cancel, leaving the trade's own impact.
            sign = 1.0 if price[i] >= pre_vwap else -1.0
            signed_impact = sign * (post_vwap / pre_vwap - 1.0)
            out.append((usd[i] / l_scale, signed_impact))
    return pd.DataFrame(out, columns=["relsize", "impact"])


def load_trades_pandas(path: Path, window: int, min_window: int) -> pd.DataFrame:
    """Reference loader for small files (needs pyarrow). Same definitions as the SQL."""
    raw = pd.read_parquet(path, columns=["market_id", "timestamp", "price", "usd_amount"])
    return compute_prepost_impact(raw, window, min_window)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fills", type=Path,
                    default=Path("archive/processed/underdog_events/fills_sorted.parquet"))
    ap.add_argument("--output-dir", type=Path, default=Path("reports/impact_calibration"))
    ap.add_argument("--engine", choices=["duckdb", "pandas"], default="duckdb")
    ap.add_argument("--window", type=int, default=10,
                    help="Trades each side for pre/post VWAP. Smaller isolates the trade better "
                         "(less neighbour contamination) but is noisier; 8-12 is a good range.")
    ap.add_argument("--min-window", type=int, default=5, help="Require at least this many trades each side.")
    ap.add_argument("--buckets", type=int, default=20)
    ap.add_argument("--sample", type=float, default=0.05,
                    help="duckdb engine: Bernoulli row sample fraction (1.0 = all rows).")
    ap.add_argument("--max-markets", type=int, default=None,
                    help="duckdb engine: restrict to a random N markets for a fast first pass.")
    args = ap.parse_args()

    if args.engine == "duckdb":
        trades = load_trades_duckdb(args.fills, args.window, args.min_window,
                                    args.sample, args.max_markets)
    else:
        trades = load_trades_pandas(args.fills, args.window, args.min_window)

    res = fit_impact(trades, args.buckets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    res["diagnostics"].to_csv(args.output_dir / "impact_buckets.csv", index=False)

    warning = None
    if res["alpha"] <= 0:
        warning = (f"NON-PHYSICAL FIT: alpha={res['alpha']:.3f} <= 0 means impact falls with "
                   f"size. Do not use this coef; inspect impact_buckets.csv and the estimator.")
        recommend_model, recommend_coef = "linear", res["coef_linear_alpha1"]
    elif res["alpha"] < 0.75:
        recommend_model, recommend_coef = "sqrt", res["coef_power"]
    else:
        recommend_model, recommend_coef = "linear", res["coef_linear_alpha1"]
    summary = {
        "fills": str(args.fills),
        "engine": args.engine,
        "n_trades_used": res["n_trades_used"],
        "fitted_power_law": {"coef": round(res["coef_power"], 4),
                             "alpha": round(res["alpha"], 4),
                             "r2_loglog": round(res["r2_loglog"], 4)},
        "linear_model_coef_alpha1": round(res["coef_linear_alpha1"], 4),
        "warning": warning,
        "recommended": {
            "slippage_model": recommend_model,
            "slippage_coef": round(recommend_coef, 4),
            "note": "Pass these to walk_forward_residual_portfolio.py --slippage-model/--slippage-coef.",
        },
    }
    (args.output_dir / "impact_calibration.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
