#!/usr/bin/env python3
"""Account-level OOS replay for selected strategy-family diagnostic signals."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from long_holdout_weighting_experiments import add_months
from realistic_underdog_account import write_csv
from walk_forward_oos import month_floor


DEFAULT_STRATEGIES = [
    "base_rate_calibration_edge_02p:16-30c_single_1.25x:core",
    "base_rate_calibration_edge_03p:16-30c_single_1.25x:core",
    "underdog_attention_light:16-30c_single_1.25x:core",
    "momentum_24h_light:16-30c_single_1.25x:core",
    "favorite_fade_long_horizon_16_30:16-30c_single_1.25x:core",
    "pure_long_tail_16_30:16-30c_single_1.25x:core",
    "pure_long_tail_01_05:01-05c_single_2x:tail",
    "favorite_fade_near_deadline_01_15:01-05c_single_2x:tail",
]


def parse_strategy(value: str) -> dict[str, str]:
    parts = value.split(":")
    if len(parts) == 2:
        strategy, exit_rule = parts
        sleeve = "core"
    elif len(parts) == 3:
        strategy, exit_rule, sleeve = parts
    else:
        raise argparse.ArgumentTypeError("strategies must use strategy:exit_rule[:sleeve]")
    return {"strategy": strategy, "exit_rule": exit_rule, "sleeve": sleeve}


def period_rows(first: datetime, last: datetime, test_months: int, min_train_months: int, validation_months: int) -> list[dict[str, Any]]:
    test_start = add_months(first, min_train_months + validation_months)
    last_month_end = add_months(last, 1)
    rows = []
    period_number = 0
    while test_start < last_month_end:
        test_end = min(add_months(test_start, test_months), last_month_end)
        rows.append({
            "period_id": f"{test_months}m_{period_number:02d}_{test_start.date()}",
            "test_months": test_months,
            "validation_start": add_months(test_start, -validation_months),
            "test_start": test_start,
            "test_end": test_end,
        })
        period_number += 1
        test_start = add_months(test_start, test_months)
    return rows


def period_key(timestamp: pd.Timestamp, budget_period: str) -> Any:
    if budget_period == "month":
        return int(timestamp.year), int(timestamp.month)
    if budget_period == "week":
        iso = timestamp.isocalendar()
        return int(iso.year), int(iso.week)
    raise ValueError(budget_period)


def max_drawdown_from_profits(initial_cash: float, profits: list[float]) -> float:
    if not profits:
        return 0.0
    curve = initial_cash + np.cumsum(np.asarray(profits, dtype=float))
    peaks = np.maximum.accumulate(np.concatenate([[initial_cash], curve]))[1:]
    drawdowns = (peaks - curve) / np.maximum(peaks, 1e-12)
    return float(np.max(drawdowns))


def replay_account(
    signals: pd.DataFrame,
    end_time: pd.Timestamp,
    initial_cash: float,
    period_budget: float,
    budget_period: str,
    stake: float,
    tail_stake: float,
    sleeve: str,
    reserve_fraction: float,
    min_stake: float,
    max_trades_per_market: int,
) -> dict[str, Any]:
    cash = initial_cash
    realized_profit = 0.0
    deployed = 0.0
    entries = 0
    skipped = defaultdict(int)
    profits: list[float] = []
    trade_debits: list[float] = []
    period_spend: dict[Any, float] = defaultdict(float)
    market_counts: dict[int, int] = defaultdict(int)

    signals = signals.sort_values("timestamp")
    for row in signals.itertuples(index=False):
        timestamp = pd.Timestamp(row.timestamp)
        if timestamp >= end_time:
            break
        market_id = int(row.market_id)
        if market_counts[market_id] >= max_trades_per_market:
            skipped["market_trade_cap"] += 1
            continue
        pkey = period_key(timestamp, budget_period)
        reserve_floor = initial_cash * reserve_fraction
        target_stake = tail_stake if sleeve == "tail" else stake
        debit = min(
            target_stake,
            max(0.0, period_budget - period_spend[pkey]),
            max(0.0, cash - reserve_floor),
        )
        if debit < min_stake:
            skipped["below_min_stake_or_budget"] += 1
            continue
        unit_return = float(row.unit_return)
        if not math.isfinite(unit_return):
            skipped["nonfinite_return"] += 1
            continue
        profit = debit * unit_return
        cash += profit
        realized_profit += profit
        deployed += debit
        period_spend[pkey] += debit
        market_counts[market_id] += 1
        entries += 1
        profits.append(profit)
        trade_debits.append(debit)

    profit_array = np.asarray(profits, dtype=float)
    debit_array = np.asarray(trade_debits, dtype=float)
    sorted_profits = np.sort(profit_array)[::-1] if len(profit_array) else np.asarray([], dtype=float)
    return {
        "initial_cash": initial_cash,
        "available_cash_end": cash,
        "locked_capital_end": 0.0,
        "realized_profit": float(realized_profit),
        "total_account_value": cash,
        "account_return": cash / initial_cash - 1.0,
        "deployed": float(deployed),
        "entries": int(entries),
        "hit_rate": float((profit_array > 0).mean()) if len(profit_array) else 0.0,
        "gross_winnings": float(profit_array[profit_array > 0].sum()) if len(profit_array) else 0.0,
        "gross_losses": float(profit_array[profit_array < 0].sum()) if len(profit_array) else 0.0,
        "max_single_profit": float(sorted_profits[0]) if len(sorted_profits) else 0.0,
        "without_top_1": float(sorted_profits[1:].sum()) if len(sorted_profits) > 1 else 0.0,
        "without_top_3": float(sorted_profits[3:].sum()) if len(sorted_profits) > 3 else 0.0,
        "profit_capped_at_5x_cost": float(np.minimum(profit_array, debit_array * 5.0).sum()) if len(profit_array) else 0.0,
        "profit_capped_at_10x_cost": float(np.minimum(profit_array, debit_array * 10.0).sum()) if len(profit_array) else 0.0,
        "profit_capped_at_20x_cost": float(np.minimum(profit_array, debit_array * 20.0).sum()) if len(profit_array) else 0.0,
        "max_drawdown": max_drawdown_from_profits(initial_cash, profits),
        "skipped": dict(skipped),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    summary_rows = []
    for (selected_strategy, sleeve, months), group in df.groupby(["selected_strategy", "sleeve", "test_months"], dropna=False):
        profits = group["realized_profit"].astype(float).to_numpy()
        returns = group["account_return"].astype(float).to_numpy()
        summary_rows.append({
            "selected_strategy": selected_strategy,
            "sleeve": sleeve,
            "test_months": int(months),
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
            "mean_entries": float(group["entries"].astype(float).mean()),
            "mean_deployed": float(group["deployed"].astype(float).mean()),
            "worst_max_drawdown": float(group["max_drawdown"].astype(float).max()),
        })
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", type=Path, default=Path("reports/strategy_family_diagnostics/strategy_family_signals.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/strategy_family_oos"))
    parser.add_argument("--strategies", type=parse_strategy, nargs="+", default=[parse_strategy(value) for value in DEFAULT_STRATEGIES])
    parser.add_argument("--first-month", default="2022-11-01")
    parser.add_argument("--test-months", type=int, nargs="+", default=[1, 2, 6])
    parser.add_argument("--min-train-months", type=int, default=12)
    parser.add_argument("--validation-months", type=int, default=6)
    parser.add_argument("--initial-cash", type=float, default=5000.0)
    parser.add_argument("--period-budget", type=float, default=5000.0)
    parser.add_argument("--budget-period", choices=["week", "month"], default="month")
    parser.add_argument("--stake", type=float, default=5.0)
    parser.add_argument("--tail-stake", type=float, default=1.0)
    parser.add_argument("--reserve-fraction", type=float, default=0.30)
    parser.add_argument("--min-stake", type=float, default=1.0)
    parser.add_argument("--max-trades-per-market", type=int, default=1)
    args = parser.parse_args()

    signals = pd.read_csv(args.signals)
    signals["timestamp"] = pd.to_datetime(signals["timestamp"], utc=True)
    first_month = datetime.fromisoformat(args.first_month).replace(tzinfo=timezone.utc)
    last_month = month_floor(int(signals["timestamp"].max().timestamp()))

    rows = []
    for spec in args.strategies:
        strategy = spec["strategy"]
        exit_rule = spec["exit_rule"]
        sleeve = spec["sleeve"]
        selected_strategy = f"{strategy}:{exit_rule}"
        selected = signals[(signals["strategy"] == strategy) & (signals["exit_rule"] == exit_rule)].copy()
        if selected.empty:
            continue
        for months in sorted(set(args.test_months)):
            for period in period_rows(first_month, last_month, months, args.min_train_months, args.validation_months):
                test_start = pd.Timestamp(period["test_start"])
                test_end = pd.Timestamp(period["test_end"])
                if "period_id" in selected.columns and strategy.startswith("base_rate_calibration"):
                    period_signals = selected[
                        (selected["timestamp"] >= test_start)
                        & (selected["timestamp"] < test_end)
                        & (selected["test_months"].fillna(-1).astype(int) == months)
                    ]
                else:
                    period_signals = selected[(selected["timestamp"] >= test_start) & (selected["timestamp"] < test_end)]
                result = replay_account(
                    period_signals,
                    test_end,
                    args.initial_cash,
                    args.period_budget,
                    args.budget_period,
                    args.stake,
                    args.tail_stake,
                    sleeve,
                    args.reserve_fraction,
                    args.min_stake,
                    args.max_trades_per_market,
                )
                rows.append({
                    "experiment": "strategy_family_oos",
                    "selected_strategy": selected_strategy,
                    "strategy": strategy,
                    "exit_rule": exit_rule,
                    "sleeve": sleeve,
                    "period_id": period["period_id"],
                    "test_months": months,
                    "validation_months": args.validation_months,
                    "validation_start": period["validation_start"].date(),
                    "test_start": period["test_start"].date(),
                    "test_end": period["test_end"].date(),
                    **result,
                    "skipped": json.dumps(result["skipped"], sort_keys=True),
                })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "strategy_family_period_results.csv", rows)
    summary_rows = summarize(rows)
    write_csv(args.output_dir / "strategy_family_oos_summary.csv", summary_rows)
    (args.output_dir / "summary.json").write_text(json.dumps({
        "signals": str(args.signals),
        "strategies": [f"{item['strategy']}:{item['exit_rule']}:{item['sleeve']}" for item in args.strategies],
        "period_rows": len(rows),
        "summary_rows": len(summary_rows),
        "files": ["strategy_family_period_results.csv", "strategy_family_oos_summary.csv"],
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "period_rows": len(rows), "summary_rows": len(summary_rows)}, indent=2))


if __name__ == "__main__":
    main()
