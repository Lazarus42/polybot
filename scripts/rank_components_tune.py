#!/usr/bin/env python3
"""Rank strategy-family components using tune-period data only (no holdout leakage).

For each strategy x exit_rule we compute edge statistics on signals strictly before
``--tune-before`` and rank by a small-sample-penalized lower bound so noisy combos
do not float to the top. The output is the candidate pool the ensemble tuner should
choose from, with a sleeve, default stake/cap, and priority assigned by rank.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def assign_sleeve(exit_rule: str) -> str:
    # The 1-5c band is the high-variance tail sleeve; everything else is core.
    return "tail" if exit_rule.startswith("01-05c") else "core"


def rank_components(df: pd.DataFrame) -> pd.DataFrame:
    """Edge ranking for each strategy x exit_rule over the supplied (time-filtered) rows.

    Ranks by a one-sigma lower confidence bound on the mean per-dollar return so that
    small/noisy combos are penalized rather than floating to the top.
    """
    df = df[df["unit_return"].notna()]
    rows = []
    for (strategy, exit_rule), g in df.groupby(["strategy", "exit_rule"]):
        r = g["unit_return"].to_numpy(dtype=float)
        n = len(r)
        if n == 0 or not strategy:
            continue
        mean = float(np.mean(r))
        std = float(np.std(r, ddof=1)) if n > 1 else float("inf")
        lcb = mean - std / np.sqrt(n) if np.isfinite(std) else -1e9
        without_top1 = float(np.sort(r)[::-1][1:].sum()) / n if n > 1 else mean
        rows.append({
            "strategy": strategy,
            "exit_rule": exit_rule,
            "sleeve": assign_sleeve(exit_rule),
            "tune_signals": n,
            "mean_unit_return": mean,
            "hit_rate": float(np.mean(r > 0)),
            "lcb_mean": lcb,
            "without_top1_per_sig": without_top1,
            "mean_entry_fill_usd": float(g["entry_fill_usd"].mean()),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("lcb_mean", ascending=False).reset_index(drop=True)


def select_pool(
    rank_df: pd.DataFrame,
    min_signals: int,
    max_components: int,
    core_stake: float,
    core_cap: float,
    tail_stake: float,
    tail_cap: float,
) -> list[dict]:
    """Return component dicts (replay-compatible) for combos that clear the edge filter."""
    if rank_df.empty:
        return []
    eligible = rank_df[
        (rank_df["tune_signals"] >= min_signals)
        & (rank_df["lcb_mean"] > 0)
        & (rank_df["without_top1_per_sig"] > 0)
    ].head(max_components).reset_index(drop=True)
    pool = []
    n_sel = len(eligible)
    for i, row in eligible.iterrows():
        sleeve = row["sleeve"]
        stake = tail_stake if sleeve == "tail" else core_stake
        cap = tail_cap if sleeve == "tail" else core_cap
        pool.append({
            "strategy": row["strategy"],
            "exit_rule": row["exit_rule"],
            "sleeve": sleeve,
            "stake": stake,
            "monthly_cap": cap,
            "priority": n_sel - i,
            "component": f"{row['strategy']}:{row['exit_rule']}",
        })
    return pool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", type=Path, default=Path("reports/strategy_family_diagnostics/strategy_family_signals.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/component_ranking"))
    parser.add_argument("--tune-before", default="2025-05-01")
    parser.add_argument("--min-signals", type=int, default=40, help="Minimum tune-period signals to be eligible.")
    parser.add_argument("--max-components", type=int, default=12, help="Cap on selected pool size.")
    parser.add_argument("--core-stake", type=float, default=5.0)
    parser.add_argument("--core-cap", type=float, default=1000.0)
    parser.add_argument("--tail-stake", type=float, default=2.0)
    parser.add_argument("--tail-cap", type=float, default=250.0)
    parser.add_argument("--exclude-categories", nargs="*", default=[],
                        help="Drop these market categories before ranking (e.g. crypto).")
    parser.add_argument("--min-horizon-days", type=float, default=None,
                        help="Drop signals whose horizon-to-deadline is below this.")
    args = parser.parse_args()

    signals = pd.read_csv(args.signals, low_memory=False)
    signals["timestamp"] = pd.to_datetime(signals["timestamp"], utc=True, format="mixed")
    if args.exclude_categories:
        signals = signals[~signals["category"].isin(args.exclude_categories)]
    if args.min_horizon_days is not None:
        signals = signals[signals["horizon_days"].astype(float) >= args.min_horizon_days]
    tune_before = pd.Timestamp(args.tune_before, tz="UTC")
    tune = signals[signals["timestamp"] < tune_before].copy()

    rank = rank_components(tune)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rank.to_csv(args.output_dir / "component_ranking_tune.csv", index=False)

    pool = select_pool(rank, args.min_signals, args.max_components,
                       args.core_stake, args.core_cap, args.tail_stake, args.tail_cap)
    specs = [f"{c['strategy']}:{c['exit_rule']}:{c['sleeve']}:{c['stake']:g}:{c['monthly_cap']:g}:{c['priority']}" for c in pool]

    (args.output_dir / "selected_components.json").write_text(
        json.dumps({"tune_before": args.tune_before, "min_signals": args.min_signals, "components": specs}, indent=2) + "\n",
        encoding="utf-8",
    )

    pd.set_option("display.width", 200, "display.max_rows", 80)
    print("=== tune-period ranking (eligible >= %d signals shown) ===" % args.min_signals)
    show = rank[rank["tune_signals"] >= args.min_signals]
    print(show[["strategy", "exit_rule", "sleeve", "tune_signals", "mean_unit_return", "hit_rate", "lcb_mean", "without_top1_per_sig"]].round(3).to_string(index=False))
    print("\n=== selected pool (%d) ===" % len(specs))
    for s in specs:
        print(" ", s)


if __name__ == "__main__":
    main()
