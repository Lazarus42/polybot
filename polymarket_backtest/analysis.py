from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from .backtest import TradeRecord


def summarize_trades(trades: Iterable[TradeRecord], counters: Dict[str, Any]) -> Dict[str, Any]:
    trade_list = list(trades)
    pnl = np.array([t.pnl for t in trade_list], dtype=float)
    summary: Dict[str, Any] = {
        "markets_seen": counters.get("markets_seen", 0),
        "binary_markets": counters.get("binary_markets", 0),
        "markets_in_window": counters.get("in_window", 0),
        "trades_triggered": counters.get("triggered", 0),
        "trades_no_trigger": counters.get("no_trigger", 0),
        "skipped_missing_resolution": counters.get("skipped_missing_resolution", 0),
        "skipped_missing_history": counters.get("skipped_missing_history", 0),
        "skipped_missing_outcome": counters.get("skipped_missing_outcome", 0),
        "skipped_tie": counters.get("skipped_tie", 0),
    }

    if len(pnl) == 0:
        summary.update(
            {
                "win_rate": None,
                "total_pnl": 0.0,
                "avg_pnl": None,
                "median_pnl": None,
                "stdev_pnl": None,
                "pnl_histogram": None,
            }
        )
        return summary

    wins = [t for t in trade_list if t.pnl > 0]
    summary.update(
        {
            "win_rate": len(wins) / len(trade_list),
            "total_pnl": float(pnl.sum()),
            "avg_pnl": float(pnl.mean()),
            "median_pnl": float(np.median(pnl)),
            "stdev_pnl": float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0,
        }
    )

    hist_counts, hist_bins = np.histogram(pnl, bins=10)
    summary["pnl_histogram"] = {
        "bins": hist_bins.tolist(),
        "counts": hist_counts.tolist(),
    }
    return summary


def trades_to_dataframe(trades: Iterable[TradeRecord]) -> pd.DataFrame:
    rows = [asdict(t) for t in trades]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def summarize_by_category(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty or "category" not in trades_df.columns:
        return pd.DataFrame()
    grouped = trades_df.groupby("category", dropna=False).agg(
        trades=("pnl", "count"),
        win_rate=("pnl", lambda s: (s > 0).mean()),
        total_pnl=("pnl", "sum"),
        avg_pnl=("pnl", "mean"),
        median_pnl=("pnl", "median"),
    )
    return grouped.reset_index().sort_values(by="total_pnl", ascending=False)

