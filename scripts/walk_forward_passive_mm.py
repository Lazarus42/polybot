#!/usr/bin/env python3
"""Passive (spread-earning) market-making backtest from the trade tape.

Unlike `walk_forward_market_making_oos.py` (which enters at an *adverse* price — it TAKES
liquidity and pays the spread, then trades a directional bracket), this models REAL passive
market-making: rest two-sided quotes and get paid the half-spread when others cross to you.

Fill-when-crossed model, per market, causal (quotes set from past trades only):
  - reference mid = trailing median of the last `ref_window` trade prices (>= `min_ref`).
  - rest bid = mid - h, ask = mid + h  (h = half_spread, in price units).
  - an incoming trade printing at P <= bid lifts our bid: we BUY quote_size (capped by the
    trade's own size and the inventory cap) at our bid; P >= ask: we SELL at our ask.
  - trades printing inside (bid, ask) do not fill us (someone quoted tighter).
  - inventory is capped at +/- inventory_cap contracts; at the end we flatten the residual
    paying the spread (exit at mid -/+ h), so leftover inventory is marked conservatively.
  - fees charged per fill on notional; adverse selection falls out naturally — if our fills
    precede an adverse drift, the inventory we carry is marked into the loss.

The realized pnl already contains spread capture, inventory risk, adverse selection and
fees. `spread_captured` is reported separately as the gross theoretical edge for diagnosis.

The simulator (`simulate_passive_mm`) is pure and unit-tested on synthetic order flow
(`tests/test_passive_mm.py`). The DuckDB loader streams the real 64M-row tape and must run
in the project venv. Sizes in the tape are USD; contracts = usd_amount / price.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def simulate_passive_mm(
    prices: np.ndarray,
    contracts: np.ndarray,
    half_spread: float,
    quote_size: float,
    inventory_cap: float,
    fee_rate: float = 0.0,
    ref_window: int = 20,
    min_ref: int = 5,
    vol_gate: float = float("inf"),
    months: np.ndarray | None = None,
) -> dict[str, Any]:
    """Simulate passive two-sided quoting over one market's trade sequence (time order).

    Returns realized pnl (net of fees, residual inventory flattened paying the spread),
    gross spread captured, fees, deployed (buy notional), fill counts, and — if `months`
    is given (a per-trade integer month key) — a per-month pnl dict for a monthly series.
    """
    inv = 0.0            # signed inventory in contracts (+ long)
    cash = 0.0           # signed cash flow (+ received)
    fees = 0.0
    spread_captured = 0.0
    deployed = 0.0       # gross buy notional (capital put to work)
    n_buy = n_sell = 0
    ref_prices: deque[float] = deque(maxlen=ref_window)
    monthly: dict[int, float] = defaultdict(float)
    last_ref = float("nan")

    n = len(prices)
    for i in range(n):
        P = float(prices[i])
        sz = float(contracts[i])
        if len(ref_prices) >= min_ref:
            ref = float(np.median(ref_prices))
            last_ref = ref
            # Volatility/trend gate: don't provide liquidity when the recent price range is
            # wide (fast/trending/resolving market) — that is where stale quotes get picked
            # off. Only quote in calm, mean-reverting regimes.
            if (max(ref_prices) - min(ref_prices)) > vol_gate:
                ref_prices.append(P)
                continue
            bid = max(0.001, ref - half_spread)
            ask = min(0.999, ref + half_spread)
            q = min(quote_size, sz) if sz > 0 else quote_size
            mkey = int(months[i]) if months is not None else 0
            if P <= bid and inv < inventory_cap:
                fill = min(q, inventory_cap - inv)
                if fill > 0:
                    cash -= fill * bid
                    fee = fee_rate * fill * bid
                    fees += fee
                    inv += fill
                    deployed += fill * bid
                    spread_captured += fill * (ref - bid)
                    monthly[mkey] += -fill * bid - fee
                    n_buy += 1
            elif P >= ask and inv > -inventory_cap:
                fill = min(q, inventory_cap + inv)
                if fill > 0:
                    cash += fill * ask
                    fee = fee_rate * fill * ask
                    fees += fee
                    inv -= fill
                    spread_captured += fill * (ask - ref)
                    monthly[mkey] += fill * ask - fee
                    n_sell += 1
        ref_prices.append(P)

    # flatten residual inventory at the last reference, paying the spread to exit
    if inv != 0.0 and math.isfinite(last_ref):
        exit_price = last_ref - math.copysign(half_spread, inv)
        cash += inv * exit_price
        if months is not None and n:
            monthly[int(months[-1])] += inv * exit_price
        inv = 0.0

    pnl = cash - fees
    return {
        "pnl": float(pnl),
        "gross_cash": float(cash),
        "fees": float(fees),
        "spread_captured": float(spread_captured),
        "deployed": float(deployed),
        "n_buy": int(n_buy),
        "n_sell": int(n_sell),
        "monthly_pnl": {int(k): float(v) for k, v in monthly.items()},
    }


def load_market_fills_duckdb(path: Path, sample: float, max_markets: int | None) -> pd.DataFrame:
    """Stream per-market fills (market_id, ts, price, contracts, month) from the tape."""
    import duckdb  # noqa: PLC0415

    con = duckdb.connect()
    con.execute("SET threads = 4")
    p = str(path.resolve()).replace("'", "''")
    # Build a single WHERE from all conditions. Market sampling keeps whole markets so each
    # kept market's tape stays intact (sampling rows would corrupt the trailing reference).
    conditions = ["price > 0", "usd_amount > 0"]
    if max_markets is not None:
        conditions.append(
            f"market_id IN (SELECT market_id FROM "
            f"(SELECT DISTINCT market_id FROM read_parquet('{p}')) USING SAMPLE {max_markets} ROWS)"
        )
    elif 0 < sample < 1:
        conditions.append(
            f"market_id IN (SELECT DISTINCT market_id FROM read_parquet('{p}') "
            f"USING SAMPLE {sample*100:.4f} PERCENT (bernoulli))"
        )
    where = " AND ".join(conditions)
    query = f"""
        SELECT market_id,
               epoch(timestamp) AS ts,
               price,
               usd_amount / nullif(price, 0) AS contracts,
               date_diff('month', DATE '2022-01-01', timestamp::DATE) AS month
        FROM read_parquet('{p}')
        WHERE {where}
        ORDER BY market_id, ts
    """
    return con.execute(query).fetch_df()


def run_over_markets(df: pd.DataFrame, resolution_gate_frac: float = 0.0, **kw) -> dict[str, Any]:
    """Run the simulator per market and aggregate.

    resolution_gate_frac: drop the last fraction of each market's trades (the endgame drift
    to 0/1 is the dominant adverse-selection regime; stop quoting before it).
    """
    total = defaultdict(float)
    monthly: dict[int, float] = defaultdict(float)
    n_markets = 0
    for _, g in df.groupby("market_id", sort=False):
        if 0 < resolution_gate_frac < 1 and len(g) > 10:
            g = g.iloc[: int(len(g) * (1.0 - resolution_gate_frac))]
        if len(g) < kw.get("min_ref", 5) + 1:
            continue
        res = simulate_passive_mm(
            g["price"].to_numpy(float), g["contracts"].to_numpy(float),
            months=g["month"].to_numpy(int), **kw,
        )
        n_markets += 1
        for k in ("pnl", "fees", "spread_captured", "deployed", "n_buy", "n_sell"):
            total[k] += res[k]
        for m, v in res["monthly_pnl"].items():
            monthly[m] += v
    mp = np.array([monthly[m] for m in sorted(monthly)], dtype=float)
    std = float(np.std(mp, ddof=1)) if len(mp) > 1 else 0.0
    return {
        "n_markets": n_markets,
        "total_pnl": float(total["pnl"]),
        "total_deployed": float(total["deployed"]),
        "total_fees": float(total["fees"]),
        "gross_spread_captured": float(total["spread_captured"]),
        "n_fills": int(total["n_buy"] + total["n_sell"]),
        "months": int(len(mp)),
        "monthly_mean": float(mp.mean()) if len(mp) else 0.0,
        "monthly_sharpe_ann": float(mp.mean() / std * np.sqrt(12)) if std > 0 else 0.0,
        "worst_month": float(mp.min()) if len(mp) else 0.0,
        "positive_months": float((mp > 0).mean()) if len(mp) else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fills", type=Path,
                    default=Path("archive/processed/underdog_events/fills_sorted.parquet"))
    ap.add_argument("--output-dir", type=Path, default=Path("reports/passive_mm"))
    ap.add_argument("--half-spreads", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.03])
    ap.add_argument("--quote-size", type=float, default=50.0, help="Contracts quoted per side per event.")
    ap.add_argument("--inventory-cap", type=float, default=200.0, help="Max abs inventory in contracts.")
    ap.add_argument("--fee-rate", type=float, default=0.0, help="Per-fill fee as fraction of notional.")
    ap.add_argument("--ref-window", type=int, default=20)
    ap.add_argument("--min-ref", type=int, default=5)
    ap.add_argument("--vol-gate", type=float, default=float("inf"),
                    help="Skip quoting when the ref-window price range exceeds this (only quote calm "
                         "regimes; e.g. 0.03). Defends against stale-quote pickoff in fast markets.")
    ap.add_argument("--resolution-gate-frac", type=float, default=0.0,
                    help="Drop the last fraction of each market's trades (stop quoting near "
                         "resolution, where price drifts to 0/1); e.g. 0.2.")
    ap.add_argument("--sample", type=float, default=0.1, help="Fraction of markets to sample.")
    ap.add_argument("--max-markets", type=int, default=None)
    args = ap.parse_args()

    df = load_market_fills_duckdb(args.fills, args.sample, args.max_markets)
    rows = []
    for h in args.half_spreads:
        res = run_over_markets(
            df, resolution_gate_frac=args.resolution_gate_frac,
            half_spread=h, quote_size=args.quote_size, inventory_cap=args.inventory_cap,
            fee_rate=args.fee_rate, ref_window=args.ref_window, min_ref=args.min_ref,
            vol_gate=args.vol_gate,
        )
        res["half_spread"] = h
        rows.append(res)
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items()}))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "passive_mm_summary.csv", index=False)
    print(f"\nwrote {args.output_dir}/passive_mm_summary.csv")


if __name__ == "__main__":
    main()
