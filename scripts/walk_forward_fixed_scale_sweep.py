#!/usr/bin/env python3
"""Fixed-strategy OOS scale sweep.

This evaluates one deployable strategy across multiple stake and bankroll scales
while reusing the expensive same-side exit-path attachment.
"""
from __future__ import annotations

import argparse
import json
import math
from copy import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from online_underdog_allocation import execution_model
from realistic_underdog_account import attach_exit_paths, write_csv
from walk_forward_oos import (
    attach_exit_liquidity_totals,
    fixed_strategy_key,
    load_arrays,
    parse_fixed_strategy,
    period_rows,
    replay_selected_oos,
    validation_candidates,
)


def parse_bankroll_scale(value: str) -> tuple[float, float]:
    parts = value.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("bankroll scales must use initial_cash:max_stake")
    initial_cash = float(parts[0])
    max_stake = float(parts[1])
    if initial_cash <= 0 or max_stake <= 0:
        raise argparse.ArgumentTypeError("initial_cash and max_stake must be positive")
    return initial_cash, max_stake


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    summaries = []
    for keys, group in df.groupby(
        ["scale_mode", "scale_label", "initial_cash", "max_stake", "test_months"],
        dropna=False,
    ):
        scale_mode, scale_label, initial_cash, max_stake, test_months = keys
        returns = group["account_return"].astype(float).to_numpy()
        profits = group["realized_profit"].astype(float).to_numpy()
        without_top1 = group["without_top_1"].astype(float).to_numpy()
        summaries.append({
            "scale_mode": scale_mode,
            "scale_label": scale_label,
            "initial_cash": float(initial_cash),
            "max_stake": float(max_stake),
            "test_months": int(test_months),
            "periods": int(len(group)),
            "mean_profit": float(np.mean(profits)),
            "median_profit": float(np.median(profits)),
            "mean_account_return": float(np.mean(returns)),
            "median_account_return": float(np.median(returns)),
            "positive_rate": float(np.mean(profits > 0)),
            "all_periods_profitable": bool(np.all(profits > 0)),
            "mean_without_top1": float(np.mean(without_top1)),
            "all_without_top1_positive": bool(np.all(without_top1 > 0)),
            "mean_capped_5x": float(group["profit_capped_at_5x_cost"].astype(float).mean()),
            "mean_capped_10x": float(group["profit_capped_at_10x_cost"].astype(float).mean()),
            "mean_capped_20x": float(group["profit_capped_at_20x_cost"].astype(float).mean()),
            "mean_entries": float(group["entries"].astype(float).mean()),
            "mean_deployed": float(group["deployed"].astype(float).mean()),
            "mean_max_drawdown": float(group["max_drawdown"].astype(float).mean()),
            "worst_max_drawdown": float(group["max_drawdown"].astype(float).max()),
            "worst_period_profit": float(np.min(profits)),
            "best_period_profit": float(np.max(profits)),
        })
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/underdog_optimization_kalshi"))
    parser.add_argument("--data-dir", type=Path, default=Path("archive/processed/underdog_events"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/oos_fixed_scale_sweep"))
    parser.add_argument(
        "--fixed-strategy",
        type=parse_fixed_strategy,
        default=parse_fixed_strategy("ungated:exclude_high_price:forecast_paced"),
    )
    parser.add_argument("--test-months", type=int, nargs="+", default=[6, 12])
    parser.add_argument("--min-train-months", type=int, default=12)
    parser.add_argument("--validation-months", type=int, default=6)
    parser.add_argument("--stride-months", type=int, default=None)
    parser.add_argument("--include-partial-final", action="store_true")
    parser.add_argument("--max-stakes", type=float, nargs="+", default=[1, 2, 5, 10, 25, 50, 75, 100, 150])
    parser.add_argument(
        "--bankroll-scales",
        type=parse_bankroll_scale,
        nargs="+",
        default=[
            parse_bankroll_scale("250:5"),
            parse_bankroll_scale("500:10"),
            parse_bankroll_scale("1000:15"),
            parse_bankroll_scale("2500:40"),
            parse_bankroll_scale("5000:75"),
        ],
    )
    parser.add_argument("--scenario", choices=["optimistic", "neutral", "conservative", "very_conservative"], default="conservative")
    parser.add_argument("--period-budget", type=float, default=5000.0)
    parser.add_argument("--budget-period", choices=["week", "month"], default="month")
    parser.add_argument("--base-stake", type=float, default=1.0)
    parser.add_argument("--kelly-fraction", type=float, default=0.10)
    parser.add_argument("--max-fraction", type=float, default=0.02)
    parser.add_argument("--reserve-fraction", type=float, default=0.30)
    parser.add_argument("--min-stake", type=float, default=1.0)
    parser.add_argument("--min-minutes-to-close", type=float, default=60.0)
    parser.add_argument("--max-category-locked-fraction", type=float, default=0.30)
    parser.add_argument("--max-regime-locked-fraction", type=float, default=0.30)
    parser.add_argument("--min-fit-trades", type=int, default=10)
    parser.add_argument("--shrink-k", type=float, default=100.0)
    parser.add_argument("--lcb-z", type=float, default=1.0)
    args = parser.parse_args()

    gate, profile, policy = args.fixed_strategy
    arrays, _ = load_arrays(args.report_dir, args.data_dir)
    execution = execution_model(args.report_dir)
    print("attaching same-side exit paths...", flush=True)
    attach_exit_paths(arrays, args.data_dir, execution)
    attach_exit_liquidity_totals(arrays)
    print("exit paths attached", flush=True)

    configs = []
    for max_stake in args.max_stakes:
        configs.append({
            "scale_mode": "max_stake",
            "scale_label": f"cash_5000_max_{max_stake:g}",
            "initial_cash": 5000.0,
            "period_budget": args.period_budget,
            "max_stake": float(max_stake),
        })
    for initial_cash, max_stake in args.bankroll_scales:
        configs.append({
            "scale_mode": "bankroll",
            "scale_label": f"cash_{initial_cash:g}_max_{max_stake:g}",
            "initial_cash": float(initial_cash),
            "period_budget": float(initial_cash),
            "max_stake": float(max_stake),
        })

    period_result_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for config in configs:
        run_args = copy(args)
        run_args.initial_cash = config["initial_cash"]
        run_args.period_budget = config["period_budget"]
        run_args.max_stake = config["max_stake"]
        run_args.selection_score = "robust"
        for months in sorted(set(args.test_months)):
            periods = period_rows(
                arrays,
                months,
                args.min_train_months,
                args.validation_months,
                args.stride_months,
                args.include_partial_final,
            )
            for period in periods:
                print(
                    f"evaluating {config['scale_label']} {period['period_id']} "
                    f"test={period['test_start'].date()}->{period['test_end'].date()}",
                    flush=True,
                )
                candidates, _ = validation_candidates(
                    period,
                    arrays,
                    execution,
                    [policy],
                    [profile],
                    run_args,
                )
                for row in candidates:
                    validation_rows.append({
                        **config,
                        **row,
                    })
                selected = next(
                    (row for row in candidates if fixed_strategy_key(row) == (gate, profile, policy)),
                    None,
                )
                if selected is None:
                    selected_rows.append({
                        **config,
                        "period_id": period["period_id"],
                        "test_months": period["test_months"],
                        "selection_status": "missing",
                        "fixed_strategy": ":".join(args.fixed_strategy),
                    })
                    continue
                selected_rows.append({
                    **config,
                    "period_id": period["period_id"],
                    "test_months": period["test_months"],
                    "test_start": period["test_start"].date(),
                    "test_end": period["test_end"].date(),
                    "selection_status": "fixed",
                    "fixed_strategy": ":".join(args.fixed_strategy),
                    "validation_realized_profit": selected["realized_profit"],
                    "validation_without_top_1": selected["without_top_1"],
                    "validation_entries": selected["entries"],
                })
                oos = replay_selected_oos(period, selected, arrays, execution, run_args)
                if oos is not None:
                    period_result_rows.append({
                        **config,
                        "fixed_strategy": ":".join(args.fixed_strategy),
                        **oos,
                    })

    summary_rows = summarize(period_result_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "scale_period_results.csv", period_result_rows)
    write_csv(args.output_dir / "scale_summary.csv", summary_rows)
    write_csv(args.output_dir / "scale_selected_strategies.csv", selected_rows)
    write_csv(args.output_dir / "scale_validation_leaderboard.csv", validation_rows)
    (args.output_dir / "summary.json").write_text(json.dumps({
        "fixed_strategy": ":".join(args.fixed_strategy),
        "test_months": args.test_months,
        "max_stakes": args.max_stakes,
        "bankroll_scales": [f"{cash:g}:{stake:g}" for cash, stake in args.bankroll_scales],
        "files": [
            "scale_period_results.csv",
            "scale_summary.csv",
            "scale_selected_strategies.csv",
            "scale_validation_leaderboard.csv",
        ],
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "period_rows": len(period_result_rows),
        "summary_rows": len(summary_rows),
    }, indent=2))


if __name__ == "__main__":
    main()
