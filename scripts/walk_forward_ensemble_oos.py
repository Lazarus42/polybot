#!/usr/bin/env python3
"""Walk-forward OOS tests for fixed-strategy bankroll ensembles."""
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


DEFAULT_ENSEMBLES: dict[str, list[tuple[str, float]]] = {
    "best_only": [
        ("ungated:exclude_high_price:forecast_paced", 1.0),
    ],
    "forecast_availability_mix": [
        ("ungated:exclude_high_price:forecast_paced", 0.50),
        ("ungated:exclude_high_price:availability", 0.50),
    ],
    "price_diversified": [
        ("ungated:exclude_high_price:forecast_paced", 0.50),
        ("ungated:mid_price_6_30c:availability", 0.30),
        ("ungated:price_16_30c:availability", 0.20),
    ],
    "conservative_mix": [
        ("ungated:exclude_high_price:availability", 0.50),
        ("ungated:mid_price_6_30c:availability", 0.30),
        ("selected_gate:low_mid_price_1_15c:flat_one", 0.20),
    ],
}


def parse_ensemble(value: str) -> tuple[str, list[tuple[str, float]]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("ensemble must use name=strategy:weight,...")
    name, body = value.split("=", 1)
    sleeves = []
    for item in body.split(","):
        strategy_text, weight_text = item.rsplit(":", 1)
        parse_fixed_strategy(strategy_text)
        weight = float(weight_text)
        if weight <= 0:
            raise argparse.ArgumentTypeError("ensemble weights must be positive")
        sleeves.append((strategy_text, weight))
    total = sum(weight for _, weight in sleeves)
    sleeves = [(strategy, weight / total) for strategy, weight in sleeves]
    return name, sleeves


def strategy_parts(strategy: str) -> tuple[str, str, str]:
    return parse_fixed_strategy(strategy)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    result = []
    for (ensemble, test_months), group in df.groupby(["ensemble", "test_months"], dropna=False):
        profits = group["realized_profit"].astype(float).to_numpy()
        returns = group["account_return"].astype(float).to_numpy()
        result.append({
            "ensemble": ensemble,
            "test_months": int(test_months),
            "periods": int(len(group)),
            "mean_profit": float(np.mean(profits)),
            "median_profit": float(np.median(profits)),
            "mean_account_return": float(np.mean(returns)),
            "median_account_return": float(np.median(returns)),
            "positive_rate": float(np.mean(profits > 0)),
            "mean_without_top1": float(group["without_top_1"].astype(float).mean()),
            "mean_capped_10x": float(group["profit_capped_at_10x_cost"].astype(float).mean()),
            "worst_period_profit": float(np.min(profits)),
            "best_period_profit": float(np.max(profits)),
            "mean_max_drawdown": float(group["max_drawdown"].astype(float).mean()),
            "worst_max_drawdown": float(group["max_drawdown"].astype(float).max()),
            "mean_entries": float(group["entries"].astype(float).mean()),
            "mean_deployed": float(group["deployed"].astype(float).mean()),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/underdog_optimization_kalshi"))
    parser.add_argument("--data-dir", type=Path, default=Path("archive/processed/underdog_events"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/oos_ensemble_tests"))
    parser.add_argument("--test-months", type=int, nargs="+", default=[1, 2, 6, 12])
    parser.add_argument("--min-train-months", type=int, default=12)
    parser.add_argument("--validation-months", type=int, default=6)
    parser.add_argument("--stride-months", type=int, default=None)
    parser.add_argument("--include-partial-final", action="store_true")
    parser.add_argument("--initial-cash", type=float, default=5000.0)
    parser.add_argument("--period-budget", type=float, default=5000.0)
    parser.add_argument("--max-stake", type=float, default=75.0)
    parser.add_argument("--reserve-fraction", type=float, default=0.30)
    parser.add_argument("--min-fit-trades", type=int, default=10)
    parser.add_argument("--scenario", choices=["optimistic", "neutral", "conservative", "very_conservative"], default="conservative")
    parser.add_argument("--budget-period", choices=["week", "month"], default="month")
    parser.add_argument("--base-stake", type=float, default=1.0)
    parser.add_argument("--kelly-fraction", type=float, default=0.10)
    parser.add_argument("--max-fraction", type=float, default=0.02)
    parser.add_argument("--min-stake", type=float, default=1.0)
    parser.add_argument("--min-minutes-to-close", type=float, default=60.0)
    parser.add_argument("--max-category-locked-fraction", type=float, default=0.30)
    parser.add_argument("--max-regime-locked-fraction", type=float, default=0.30)
    parser.add_argument("--drawdown-throttle-start", type=float, default=math.inf)
    parser.add_argument("--drawdown-throttle-stop", type=float, default=math.inf)
    parser.add_argument("--drawdown-throttle-min-scale", type=float, default=0.0)
    parser.add_argument("--shrink-k", type=float, default=100.0)
    parser.add_argument("--lcb-z", type=float, default=1.0)
    parser.add_argument("--ensembles", type=parse_ensemble, nargs="*", default=[])
    args = parser.parse_args()

    ensembles = dict(args.ensembles) if args.ensembles else DEFAULT_ENSEMBLES
    arrays, _ = load_arrays(args.report_dir, args.data_dir)
    execution = execution_model(args.report_dir)
    print("attaching same-side exit paths...", flush=True)
    attach_exit_paths(arrays, args.data_dir, execution)
    attach_exit_liquidity_totals(arrays)
    print("exit paths attached", flush=True)

    sleeve_rows: list[dict[str, Any]] = []
    ensemble_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []

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
            for ensemble_name, sleeves in ensembles.items():
                print(f"evaluating {ensemble_name} {period['period_id']}", flush=True)
                sleeve_results = []
                for strategy, weight in sleeves:
                    gate, profile, policy = strategy_parts(strategy)
                    run_args = copy(args)
                    run_args.initial_cash = args.initial_cash * weight
                    run_args.period_budget = args.period_budget * weight
                    run_args.max_stake = args.max_stake * weight
                    run_args.selection_score = "robust"
                    candidates, _ = validation_candidates(
                        period,
                        arrays,
                        execution,
                        [policy],
                        [profile],
                        run_args,
                    )
                    selected = next(
                        (row for row in candidates if fixed_strategy_key(row) == (gate, profile, policy)),
                        None,
                    )
                    selection_rows.append({
                        "ensemble": ensemble_name,
                        "period_id": period["period_id"],
                        "test_months": period["test_months"],
                        "strategy": strategy,
                        "weight": weight,
                        "selection_status": "found" if selected is not None else "missing",
                    })
                    if selected is None:
                        continue
                    oos = replay_selected_oos(period, selected, arrays, execution, run_args)
                    if oos is None:
                        continue
                    sleeve_row = {
                        "ensemble": ensemble_name,
                        "period_id": period["period_id"],
                        "test_months": period["test_months"],
                        "test_start": period["test_start"].date(),
                        "test_end": period["test_end"].date(),
                        "strategy": strategy,
                        "weight": weight,
                        "sleeve_initial_cash": run_args.initial_cash,
                        "sleeve_max_stake": run_args.max_stake,
                        **oos,
                    }
                    sleeve_rows.append(sleeve_row)
                    sleeve_results.append(sleeve_row)

                if not sleeve_results:
                    continue
                realized = sum(float(row["realized_profit"]) for row in sleeve_results)
                total_value = sum(float(row["total_account_value"]) for row in sleeve_results)
                locked = sum(float(row["locked_capital_end"]) for row in sleeve_results)
                deployed = sum(float(row["deployed"]) for row in sleeve_results)
                fees = sum(float(row["fees"]) for row in sleeve_results)
                entries = sum(int(row["entries"]) for row in sleeve_results)
                ensemble_rows.append({
                    "ensemble": ensemble_name,
                    "period_id": period["period_id"],
                    "test_months": period["test_months"],
                    "test_start": period["test_start"].date(),
                    "test_end": period["test_end"].date(),
                    "initial_cash": args.initial_cash,
                    "realized_profit": realized,
                    "total_account_value": total_value,
                    "account_return": total_value / args.initial_cash - 1.0,
                    "locked_capital_end": locked,
                    "deployed": deployed,
                    "fees": fees,
                    "entries": entries,
                    "max_drawdown": max(float(row["max_drawdown"]) * float(row["weight"]) for row in sleeve_results),
                    "without_top_1": sum(float(row["without_top_1"]) for row in sleeve_results),
                    "profit_capped_at_5x_cost": sum(float(row["profit_capped_at_5x_cost"]) for row in sleeve_results),
                    "profit_capped_at_10x_cost": sum(float(row["profit_capped_at_10x_cost"]) for row in sleeve_results),
                    "profit_capped_at_20x_cost": sum(float(row["profit_capped_at_20x_cost"]) for row in sleeve_results),
                    "sleeves": ",".join(f"{strategy}@{weight:.3f}" for strategy, weight in sleeves),
                })

    summary_rows = summarize(ensemble_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "ensemble_period_results.csv", ensemble_rows)
    write_csv(args.output_dir / "ensemble_summary.csv", summary_rows)
    write_csv(args.output_dir / "ensemble_sleeve_results.csv", sleeve_rows)
    write_csv(args.output_dir / "ensemble_selected_strategies.csv", selection_rows)
    (args.output_dir / "summary.json").write_text(json.dumps({
        "ensembles": ensembles,
        "test_months": args.test_months,
        "include_partial_final": args.include_partial_final,
        "files": [
            "ensemble_period_results.csv",
            "ensemble_summary.csv",
            "ensemble_sleeve_results.csv",
            "ensemble_selected_strategies.csv",
        ],
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "period_rows": len(ensemble_rows),
        "sleeve_rows": len(sleeve_rows),
    }, indent=2))


if __name__ == "__main__":
    main()
