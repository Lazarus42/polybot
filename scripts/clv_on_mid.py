#!/usr/bin/env python3
"""Re-test the short-horizon (CLV) signal on TRUE MIDS instead of last-trade prices.

The forward-return signal (forward_return_predictability.py) was measured on the fill tape,
i.e. last-trade prices full of bid-ask bounce — which is exactly what can manufacture fake
mean-reversion. The free CLOB `prices-history` endpoint returns the midpoint (the displayed
implied probability; it falls back to last-trade only when the spread > $0.10), so it is a far
cleaner price. If the 24h information coefficient SURVIVES on mids, the signal is real path
alpha; if it collapses, that confirms it was a microstructure-bounce artifact.

Pulls one price history per token (cached to parquet), looks up the mid at entry and entry+H,
then reuses the walk-forward IC harness. Network + venv. The forward-move core is pure/tested.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def asof_mid(t_arr: np.ndarray, mid_arr: np.ndarray, t: float) -> float:
    """Last mid at or before time t (epoch seconds)."""
    if len(t_arr) == 0:
        return float("nan")
    i = int(np.searchsorted(t_arr, t, side="right")) - 1
    return float(mid_arr[i]) if i >= 0 else float("nan")


def forward_moves_from_mid(entries: pd.DataFrame, histories: dict[str, tuple[np.ndarray, np.ndarray]],
                           horizons_hours: list[float]) -> pd.DataFrame:
    """For each entry (token, entry_ts), compute mid now and mid at t+H -> forward move on mid.
    `histories[token] = (sorted_t_array, mid_array)`."""
    out = entries.copy()
    p0 = np.full(len(out), np.nan)
    fwd = {h: np.full(len(out), np.nan) for h in horizons_hours}
    for i, row in enumerate(out.itertuples()):
        hist = histories.get(str(row.token))
        if hist is None:
            continue
        t_arr, mid_arr = hist
        m0 = asof_mid(t_arr, mid_arr, row.entry_ts)
        p0[i] = m0
        for h in horizons_hours:
            mh = asof_mid(t_arr, mid_arr, row.entry_ts + h * 3600.0)
            fwd[h][i] = mh - m0
    out["mid0"] = p0
    for h in horizons_hours:
        out[f"fwd_move_{h}h"] = fwd[h]
    return out


def fetch_price_history(token_id: str, fidelity: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Pull a token's full mid history from the CLOB. Returns (epoch_ts[], mid[])."""
    import requests  # noqa: PLC0415

    url = "https://clob.polymarket.com/prices-history"
    r = requests.get(url, params={"market": token_id, "interval": "max", "fidelity": fidelity},
                     timeout=30)
    r.raise_for_status()
    pts = r.json().get("history", [])
    if not pts:
        return np.array([]), np.array([])
    t = np.array([float(p["t"]) for p in pts])
    m = np.array([float(p["p"]) for p in pts])
    order = np.argsort(t)
    return t[order], m[order]


def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from forward_return_predictability import (
        FEATURES, walk_forward_ic, feature_ic_breakdown, bucketed_ic,
    )

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signals", type=Path, default=Path("reports/strategy_family_diagnostics/strategy_family_signals.csv"))
    ap.add_argument("--markets", type=Path, default=Path("archive/markets.csv"))
    ap.add_argument("--cache", type=Path, default=Path("reports/clv_on_mid/price_history_cache.parquet"))
    ap.add_argument("--output-dir", type=Path, default=Path("reports/clv_on_mid"))
    ap.add_argument("--horizons", type=float, nargs="+", default=[1.0, 6.0, 24.0])
    ap.add_argument("--max-tokens", type=int, default=1500, help="Cap unique tokens pulled (rate limits).")
    ap.add_argument("--sleep", type=float, default=0.2, help="Seconds between API calls (throttle).")
    args = ap.parse_args()

    df = pd.read_csv(args.signals, low_memory=False)
    df["entry_ts"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed").astype("int64") / 1e9
    for c in FEATURES + ["entry_fill_usd"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # market_id -> token1 (the YES token), consistent with the fills-tape token1 used before
    mk = pd.read_csv(args.markets, low_memory=False)
    mk["id"] = mk["id"].astype(str)
    tok_map = dict(zip(mk["id"], mk["token1"].astype(str)))
    df["token"] = df["market_id"].astype(str).map(tok_map)
    df = df.dropna(subset=["token", "entry_ts"]).reset_index(drop=True)

    tokens = list(dict.fromkeys(df["token"]))[: args.max_tokens]
    df = df[df["token"].isin(set(tokens))].reset_index(drop=True)

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    cached = {}
    if args.cache.exists():
        cdf = pd.read_parquet(args.cache)
        for tok, g in cdf.groupby("token"):
            cached[str(tok)] = (g["t"].to_numpy(float), g["mid"].to_numpy(float))
    rows = []
    for i, tok in enumerate(tokens):
        if tok in cached:
            continue
        try:
            t_arr, m_arr = fetch_price_history(tok)
        except Exception as exc:  # keep going on individual failures
            print("fetch failed", tok, exc); continue
        cached[tok] = (t_arr, m_arr)
        rows.extend({"token": tok, "t": t, "mid": m} for t, m in zip(t_arr, m_arr))
        if i % 50 == 0:
            print(f"pulled {i}/{len(tokens)} tokens")
        time.sleep(args.sleep)
    if rows:
        new = pd.DataFrame(rows)
        combined = pd.concat([pd.read_parquet(args.cache), new]) if args.cache.exists() else new
        combined.to_parquet(args.cache)

    moves = forward_moves_from_mid(df, cached, args.horizons)
    feats = [c for c in FEATURES if c in moves.columns]
    results = {}
    for h in args.horizons:
        tgt = f"fwd_move_{h}h"
        results[f"{h}h"] = walk_forward_ic(moves, feats, tgt)
        print(json.dumps({"horizon": f"{h}h",
                          **{k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in results[f"{h}h"].items()}}))
    h = max(args.horizons); tgt = f"fwd_move_{h}h"
    results["breakdown"] = {
        "feature_ic": feature_ic_breakdown(moves, feats, tgt),
        "liquidity_buckets": bucketed_ic(moves, feats, tgt, "entry_fill_usd")
        if "entry_fill_usd" in moves.columns else {},
    }
    print("\n24h IC by liquidity tercile (on MID):",
          json.dumps({k: round(v["oos_ic"], 4) for k, v in results["breakdown"]["liquidity_buckets"].items()}))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "clv_on_mid.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(f"\nwrote {args.output_dir}/clv_on_mid.json")
    print("compare to the last-trade run: if 24h oos_ic stays ~0.05 on mids -> real path signal;"
          " if it collapses toward 0 -> it was bid-ask bounce.")


if __name__ == "__main__":
    main()
