#!/usr/bin/env python3
"""Causal capital-account experiments for the underdog bracket strategy.

This simulator selects brackets, category gates, and scheduled-resolution-horizon
gates using only pre-holdout data. During holdout it releases cash only when a
TP/SL fill or actual market close is reached in simulated time. Positions still
open at the evaluation boundary remain locked and are marked at cost because the
archive does not contain trustworthy point-in-time order-book snapshots.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
from typing import Any, Optional

import duckdb
import matplotlib.pyplot as plt
import numpy as np

from online_underdog_allocation import (
    CATEGORIES,
    WEEK_SECONDS,
    classify_market,
    execution_model,
    week_id,
    week_start,
)
from optimize_underdog_bracket import max_contracts_for_budget, posted_balance_change


DAY = 86400
SCENARIOS = {
    "optimistic": {"participation": 1.0, "entry_ticks": 0, "exit_ticks": 0},
    "neutral": {"participation": 0.25, "entry_ticks": 0, "exit_ticks": 0},
    "conservative": {"participation": 0.10, "entry_ticks": 1, "exit_ticks": 1},
    "very_conservative": {"participation": 0.05, "entry_ticks": 2, "exit_ticks": 2},
}
HORIZONS = (1, 3, 7, 14, 30, 60, 90, 180, math.inf)
CAPACITY_BUDGETS = (50, 100, 250, 500, 1000, 2500, 5000)
EVENT_COUNTS = (100, 250, 500, 1000, 2500, 5000)


def load_market_categories(data_dir: Path, market_ids: np.ndarray) -> np.ndarray:
    path = str((data_dir / "markets.parquet").resolve()).replace("'", "''")
    rows = duckdb.connect().execute(
        f"SELECT market_id, coalesce(slug, ''), coalesce(question, '') FROM read_parquet('{path}')"
    ).fetchall()
    mapping = {int(mid): classify_market(f"{slug} {question}") for mid, slug, question in rows}
    return np.asarray([mapping.get(int(mid), "other") for mid in market_ids])


def fit_brackets(
    levels: np.ndarray,
    cube: np.ndarray,
    fit_mask: np.ndarray,
    min_trades: int,
) -> dict[int, tuple[int, int]]:
    result = {}
    for level in range(1, 50):
        indexes = np.where(fit_mask & (levels == level))[0]
        if len(indexes) < min_trades:
            continue
        flat = int(np.argmax(cube[indexes].mean(axis=0)))
        result[level] = tuple(int(value) for value in np.unravel_index(flat, cube.shape[1:]))
    return result


def exact_entry(
    stake: float,
    price: float,
    model: dict[str, float],
) -> tuple[Decimal, float, float]:
    contracts, debit = max_contracts_for_budget(
        stake, price, model["fee_coefficient"], model["contract_step"]
    )
    if contracts <= 0 or debit <= 0:
        return Decimal("0"), 0.0, 0.0
    fee = float(debit) - float(contracts) * price
    return contracts, float(debit), fee


def exact_exit(
    contracts: Decimal,
    price: float,
    code: int,
    model: dict[str, float],
) -> tuple[float, float]:
    if code == 0 or price in (0.0, 1.0):
        proceeds = (contracts * Decimal(str(price)) * 100).to_integral_value(
            rounding=ROUND_FLOOR
        ) / 100
        return float(proceeds), 0.0
    proceeds = posted_balance_change(
        contracts,
        Decimal(str(price)),
        Decimal(str(model["fee_coefficient"])),
        "sell",
    )
    fee = float(contracts) * price - float(proceeds)
    return float(proceeds), fee


def adverse_price(price: float, ticks: int, tick: float, entry: bool) -> float:
    if price in (0.0, 1.0):
        return price
    value = price + ticks * tick if entry else price - ticks * tick
    return min(1.0 - tick, max(tick, value))


def horizon_bucket(days: float) -> str:
    for bound in (1, 3, 7, 14, 30, 60, 90, 180):
        if days <= bound:
            return f"<={bound}d"
    return ">180d"


def run_account(
    indexes: np.ndarray,
    end_time: int,
    selectors: dict[int, tuple[int, int]],
    arrays: dict[str, np.ndarray],
    allowed_categories: set[str],
    max_horizon: float,
    model: dict[str, float],
    scenario_name: str,
    initial_cash: float,
    weekly_budget: float,
    stake: float,
    availability_lambda: Optional[float] = None,
    max_entries: Optional[int] = None,
) -> dict[str, Any]:
    scenario = SCENARIOS[scenario_name]
    cash = initial_cash
    realized_profit = 0.0
    fees = 0.0
    deployed = 0.0
    dollar_days = 0.0
    peak = initial_cash
    max_drawdown = 0.0
    opened = 0
    exits = []
    open_positions: dict[int, dict[str, Any]] = {}
    weekly_spend: dict[int, float] = defaultdict(float)
    weekly_records: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    category_records: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    horizon_records: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    skipped = Counter()
    closed_records = []

    def account_value() -> float:
        return cash + sum(position["debit"] for position in open_positions.values())

    def release(until: int) -> None:
        nonlocal cash, realized_profit, fees, dollar_days, peak, max_drawdown
        while exits and exits[0][0] <= until:
            _, position_id = heapq.heappop(exits)
            position = open_positions.pop(position_id)
            proceeds, exit_fee = exact_exit(
                position["contracts"], position["exit_price"], position["exit_code"], model
            )
            profit = proceeds - position["debit"]
            cash += proceeds
            realized_profit += profit
            fees += exit_fee
            held_days = max(0.0, (position["exit_time"] - position["entry_time"]) / DAY)
            dollar_days += position["debit"] * held_days
            entry_week = int(week_id(float(position["entry_time"])))
            weekly_records[entry_week]["profit"] += profit
            weekly_records[entry_week]["exits"] += 1
            category_records[position["category"]]["profit"] += profit
            category_records[position["category"]]["exits"] += 1
            horizon_records[position["horizon_bucket"]]["profit"] += profit
            horizon_records[position["horizon_bucket"]]["exits"] += 1
            closed_records.append(profit)
            value = account_value()
            peak = max(peak, value)
            max_drawdown = max(max_drawdown, (peak - value) / peak if peak else 0.0)

    for index in indexes:
        entry_time = int(arrays["times"][index])
        if entry_time >= end_time:
            break
        release(entry_time)
        if max_entries is not None and opened >= max_entries:
            skipped["event_count_cap"] += 1
            continue
        level = int(arrays["levels"][index])
        selector = selectors.get(level)
        if selector is None:
            skipped["untrained_level"] += 1
            continue
        category = str(arrays["categories"][index])
        if category not in allowed_categories:
            skipped["category_gate"] += 1
            continue
        scheduled_end = int(arrays["scheduled_end"][index])
        scheduled_horizon = max(0.0, (scheduled_end - entry_time) / DAY)
        if scheduled_horizon > max_horizon:
            skipped["horizon_gate"] += 1
            continue
        week = int(week_id(float(entry_time)))
        remaining_week = max(0.0, weekly_budget - weekly_spend[week])
        if remaining_week <= 0:
            skipped["weekly_budget"] += 1
            continue
        target = stake if availability_lambda is None else weekly_budget / max(1.0, availability_lambda)
        target = min(target, remaining_week, cash)
        fill_cap = float(arrays["entry_fill"][index]) * scenario["participation"]
        if fill_cap < target:
            skipped["entry_liquidity_limited"] += 1
        target = min(target, fill_cap)
        entry_price = adverse_price(
            float(arrays["prices"][index]), scenario["entry_ticks"], model["price_tick"], True
        )
        contracts, debit, entry_fee = exact_entry(target, entry_price, model)
        if not contracts:
            skipped["minimum_order"] += 1
            continue
        tp_index, sl_index = selector
        exit_time = int(arrays["exit_times"][index, tp_index, sl_index])
        exit_price = float(arrays["exit_prices"][index, tp_index, sl_index])
        exit_fill = float(arrays["exit_fill"][index, tp_index, sl_index])
        exit_code = int(arrays["exit_codes"][index, tp_index, sl_index])
        if exit_code and (
            not math.isfinite(exit_fill)
            or exit_fill * scenario["participation"] < float(contracts) * exit_price
        ):
            # The first threshold-crossing fill cannot support our order. With no
            # order-book queue history, conservatively hold until actual close.
            exit_time = int(arrays["closed_times"][index])
            exit_price = 1.0 if bool(arrays["won"][index]) else 0.0
            exit_code = 0
            skipped["threshold_exit_insufficient_liquidity"] += 1
        else:
            exit_price = adverse_price(
                exit_price, scenario["exit_ticks"], model["price_tick"], False
            )
        cash -= debit
        fees += entry_fee
        deployed += debit
        weekly_spend[week] += debit
        opened += 1
        bucket = horizon_bucket(scheduled_horizon)
        position = {
            "position_id": int(index),
            "market_id": int(arrays["market_ids"][index]),
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "exit_code": exit_code,
            "contracts": contracts,
            "debit": debit,
            "category": category,
            "scheduled_horizon_days": scheduled_horizon,
            "horizon_bucket": bucket,
        }
        open_positions[int(index)] = position
        heapq.heappush(exits, (exit_time, int(index)))
        weekly_records[week]["entries"] += 1
        weekly_records[week]["deployed"] += debit
        category_records[category]["entries"] += 1
        category_records[category]["deployed"] += debit
        horizon_records[bucket]["entries"] += 1
        horizon_records[bucket]["deployed"] += debit

    release(end_time)
    for position in open_positions.values():
        held_days = max(0.0, (end_time - position["entry_time"]) / DAY)
        dollar_days += position["debit"] * held_days
        category_records[position["category"]]["locked_end"] += position["debit"]
        horizon_records[position["horizon_bucket"]]["locked_end"] += position["debit"]
    locked = sum(position["debit"] for position in open_positions.values())
    eventual_open_profit = 0.0
    unresolved_rows = []
    for position in open_positions.values():
        proceeds, _ = exact_exit(
            position["contracts"], position["exit_price"], position["exit_code"], model
        )
        eventual_open_profit += proceeds - position["debit"]
        unresolved_rows.append({
            "market_id": position["market_id"],
            "entry_time": datetime.fromtimestamp(position["entry_time"], timezone.utc).isoformat(),
            "planned_exit_time": datetime.fromtimestamp(position["exit_time"], timezone.utc).isoformat(),
            "category": position["category"],
            "scheduled_horizon_days": position["scheduled_horizon_days"],
            "locked_capital": position["debit"],
            "eventual_profit_audit_only": proceeds - position["debit"],
        })
    total_value = cash + locked
    profits = np.asarray(closed_records, dtype=float)
    gross_win = float(profits[profits > 0].sum()) if len(profits) else 0.0
    gross_loss = float(profits[profits < 0].sum()) if len(profits) else 0.0
    summary = {
        "initial_cash": initial_cash,
        "available_cash_end": cash,
        "locked_capital_end": locked,
        "open_positions_end": len(open_positions),
        "realized_profit": realized_profit,
        "unrealized_profit_marked_at_cost": 0.0,
        "total_account_value": total_value,
        "account_return": total_value / initial_cash - 1.0,
        "eventual_open_profit_audit_only": eventual_open_profit,
        "deployed": deployed,
        "fees": fees,
        "entries": opened,
        "resolved_exits": len(closed_records),
        "hit_rate": float((profits > 0).mean()) if len(profits) else 0.0,
        "gross_winnings": gross_win,
        "gross_losses": gross_loss,
        "profit_factor": gross_win / abs(gross_loss) if gross_loss else None,
        "dollar_days_locked": dollar_days,
        "annualized_return_on_locked_capital": realized_profit / dollar_days * 365 if dollar_days else 0.0,
        "max_drawdown": max_drawdown,
        "skipped": dict(skipped),
    }
    return {
        "summary": summary,
        "weekly": weekly_records,
        "categories": category_records,
        "horizons": horizon_records,
        "unresolved": unresolved_rows,
    }


def calibration_category_sets(
    calibration_indexes: np.ndarray,
    calibration_end: int,
    selectors: dict[int, tuple[int, int]],
    arrays: dict[str, np.ndarray],
) -> dict[str, set[str]]:
    returns: dict[str, list[float]] = defaultdict(list)
    for index in calibration_indexes:
        selector = selectors.get(int(arrays["levels"][index]))
        if selector is None:
            continue
        tp, sl = selector
        if int(arrays["exit_times"][index, tp, sl]) <= calibration_end:
            returns[str(arrays["categories"][index])].append(float(arrays["cube"][index, tp, sl]))
    all_values = [value for values in returns.values() for value in values]
    global_mean = float(np.mean(all_values)) if all_values else 0.0
    positive_lcb = set()
    for category, values in returns.items():
        data = np.asarray(values)
        mean = float(data.mean())
        shrunk = len(data) / (len(data) + 100) * mean + 100 / (len(data) + 100) * global_mean
        se = float(data.std(ddof=1) / math.sqrt(len(data))) if len(data) > 1 else math.inf
        if shrunk - se > 0:
            positive_lcb.add(category)
    return {
        "all": set(CATEGORIES),
        "no_crypto": set(CATEGORIES) - {"crypto"},
        "sports_other": {"sports", "other"},
        "positive_lcb": positive_lcb or set(CATEGORIES),
        **{f"only_{category}": {category} for category in CATEGORIES},
    }


def gate_score(result: dict[str, Any]) -> float:
    summary = result["summary"]
    return (
        summary["realized_profit"]
        - 0.0001 * summary["dollar_days_locked"]
        - 0.05 * summary["locked_capital_end"]
    )


def append_result(
    destination: list[dict[str, Any]],
    cut: float,
    experiment: str,
    parameter: str,
    categories: set[str],
    horizon: float,
    scenario: str,
    result: dict[str, Any],
) -> None:
    destination.append({
        "cut_fraction": cut,
        "experiment": experiment,
        "parameter": parameter,
        "categories": ",".join(sorted(categories)),
        "max_scheduled_horizon_days": "none" if math.isinf(horizon) else horizon,
        "fill_scenario": scenario,
        **result["summary"],
        "skipped": json.dumps(result["summary"]["skipped"], sort_keys=True),
    })


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/underdog_optimization_kalshi"))
    parser.add_argument("--data-dir", type=Path, default=Path("archive/processed/underdog_events"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/realistic_underdog_account"))
    parser.add_argument("--cut-fractions", type=float, nargs="+", default=[0.6, 0.7, 0.8])
    parser.add_argument("--inner-fit-fraction", type=float, default=0.7)
    parser.add_argument("--min-fit-trades", type=int, default=30)
    parser.add_argument("--initial-cash", type=float, default=5000.0)
    parser.add_argument("--weekly-budget", type=float, default=5000.0)
    args = parser.parse_args()

    data = np.load(args.report_dir / "strategy_cube.npz")
    required = {"scheduled_end_times", "closed_times", "exit_times", "exit_prices", "exit_fill_usd", "exit_codes"}
    missing = required - set(data.files)
    if missing:
        raise SystemExit(f"strategy cube missing {sorted(missing)}; rerun optimize_underdog_bracket.py")
    order = np.argsort(data["entry_times"], kind="stable")
    arrays = {
        "market_ids": data["market_ids"][order],
        "times": data["entry_times"][order],
        "prices": data["entry_prices"][order],
        "levels": data["entry_levels"][order],
        "scheduled_end": data["scheduled_end_times"][order],
        "closed_times": data["closed_times"][order],
        "entry_fill": data["entry_fill_usd"][order],
        "won": data["underdog_won"][order],
        "cube": data["returns"][order],
        "exit_times": data["exit_times"][order],
        "exit_prices": data["exit_prices"][order],
        "exit_fill": data["exit_fill_usd"][order],
        "exit_codes": data["exit_codes"][order],
    }
    arrays["categories"] = load_market_categories(args.data_dir, arrays["market_ids"])
    weeks = np.asarray(week_id(arrays["times"]))
    all_weeks = np.arange(int(weeks.min()), int(weeks.max()) + 1)
    model = execution_model(args.report_dir)
    summaries = []
    gate_rows = []
    weekly_rows = []
    category_rows = []
    horizon_rows = []
    unresolved_rows = []

    for cut in sorted(set(args.cut_fractions)):
        cut_position = min(len(all_weeks) - 1, max(2, int(len(all_weeks) * cut)))
        cut_week = int(all_weeks[cut_position])
        training_weeks = all_weeks[:cut_position]
        inner_position = min(len(training_weeks) - 1, max(1, int(len(training_weeks) * args.inner_fit_fraction)))
        calibration_week = int(training_weeks[inner_position])
        calibration_start = int(week_start(calibration_week).timestamp())
        holdout_start = int(week_start(cut_week).timestamp())
        holdout_end = int(week_start(int(all_weeks[-1]) + 1).timestamp())
        fit_mask = arrays["times"] < calibration_start
        selectors = fit_brackets(arrays["levels"], arrays["cube"], fit_mask, args.min_fit_trades)
        calibration_indexes = np.where(
            (arrays["times"] >= calibration_start) & (arrays["times"] < holdout_start)
        )[0]
        holdout_indexes = np.where(
            (arrays["times"] >= holdout_start) & (arrays["times"] < holdout_end)
        )[0]
        category_sets = calibration_category_sets(
            calibration_indexes, holdout_start, selectors, arrays
        )
        best = None
        for category_name, category_set in category_sets.items():
            for horizon in HORIZONS:
                result = run_account(
                    calibration_indexes, holdout_start, selectors, arrays, category_set,
                    horizon, model, "conservative", args.initial_cash,
                    args.weekly_budget, 1.0,
                )
                score = gate_score(result)
                gate_rows.append({
                    "cut_fraction": cut,
                    "calibration_start": datetime.fromtimestamp(calibration_start, timezone.utc).date(),
                    "holdout_start": datetime.fromtimestamp(holdout_start, timezone.utc).date(),
                    "category_gate": category_name,
                    "categories": ",".join(sorted(category_set)),
                    "max_scheduled_horizon_days": "none" if math.isinf(horizon) else horizon,
                    "score": score,
                    **result["summary"],
                    "skipped": json.dumps(result["summary"]["skipped"], sort_keys=True),
                })
                if best is None or score > best[0]:
                    best = (score, category_name, category_set, horizon)
        assert best is not None
        _, selected_name, selected_categories, selected_horizon = best

        experiment_results = []
        for scenario in SCENARIOS:
            result = run_account(
                holdout_indexes, holdout_end, selectors, arrays, selected_categories,
                selected_horizon, model, scenario, args.initial_cash,
                args.weekly_budget, 1.0,
            )
            append_result(summaries, cut, "fill_sensitivity", scenario, selected_categories, selected_horizon, scenario, result)
            experiment_results.append(("fill_sensitivity", scenario, result))
        ungated = run_account(
            holdout_indexes, holdout_end, selectors, arrays, set(CATEGORIES), math.inf,
            model, "conservative", args.initial_cash, args.weekly_budget, 1.0,
        )
        append_result(summaries, cut, "ungated_baseline", "$1", set(CATEGORIES), math.inf, "conservative", ungated)
        experiment_results.append(("ungated_baseline", "$1", ungated))

        calibration_eligible = sum(
            str(arrays["categories"][index]) in selected_categories
            and max(0.0, (int(arrays["scheduled_end"][index]) - int(arrays["times"][index])) / DAY) <= selected_horizon
            and int(arrays["levels"][index]) in selectors
            for index in calibration_indexes
        )
        calibration_weeks = max(1, cut_week - calibration_week)
        availability_lambda = calibration_eligible / calibration_weeks
        for budget in CAPACITY_BUDGETS:
            result = run_account(
                holdout_indexes, holdout_end, selectors, arrays, selected_categories,
                selected_horizon, model, "conservative", args.initial_cash,
                float(budget), 1.0, availability_lambda=availability_lambda,
            )
            append_result(summaries, cut, "capacity_frontier", str(budget), selected_categories, selected_horizon, "conservative", result)
            experiment_results.append(("capacity_frontier", str(budget), result))
        for count in EVENT_COUNTS:
            result = run_account(
                holdout_indexes, holdout_end, selectors, arrays, selected_categories,
                selected_horizon, model, "conservative", args.initial_cash,
                args.weekly_budget, 1.0, max_entries=count,
            )
            append_result(summaries, cut, "event_count_frontier", str(count), selected_categories, selected_horizon, "conservative", result)
            experiment_results.append(("event_count_frontier", str(count), result))

        for experiment, parameter, result in experiment_results:
            for week in range(cut_week, int(all_weeks[-1]) + 1):
                values = result["weekly"].get(week, {})
                weekly_rows.append({
                    "cut_fraction": cut, "experiment": experiment, "parameter": parameter,
                    "week": week_start(week).date(), "entries": values.get("entries", 0),
                    "exits": values.get("exits", 0), "deployed": values.get("deployed", 0),
                    "realized_profit_attributed_to_entry_week": values.get("profit", 0),
                })
            for category in CATEGORIES:
                values = result["categories"].get(category, {})
                category_rows.append({
                    "cut_fraction": cut, "experiment": experiment, "parameter": parameter,
                    "category": category, **dict(values),
                })
            for bucket, values in result["horizons"].items():
                horizon_rows.append({
                    "cut_fraction": cut, "experiment": experiment, "parameter": parameter,
                    "horizon_bucket": bucket, **dict(values),
                })
            for position in result["unresolved"]:
                unresolved_rows.append({
                    "cut_fraction": cut, "experiment": experiment, "parameter": parameter,
                    **position,
                })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "account_summary.csv", summaries)
    write_csv(args.output_dir / "gate_selection.csv", gate_rows)
    write_csv(args.output_dir / "weekly_accounts.csv", weekly_rows)
    write_csv(args.output_dir / "category_attribution.csv", category_rows)
    write_csv(args.output_dir / "horizon_attribution.csv", horizon_rows)
    write_csv(args.output_dir / "unresolved_positions.csv", unresolved_rows)
    if not unresolved_rows:
        (args.output_dir / "unresolved_positions.csv").write_text(
            "cut_fraction,experiment,parameter,market_id,entry_time,planned_exit_time,category,scheduled_horizon_days,locked_capital,eventual_profit_audit_only\n",
            encoding="utf-8",
        )

    capacity = [row for row in summaries if row["experiment"] == "capacity_frontier"]
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for cut in sorted(set(args.cut_fractions)):
        rows = sorted((row for row in capacity if row["cut_fraction"] == cut), key=lambda row: float(row["parameter"]))
        x = [float(row["parameter"]) for row in rows]
        axes[0].plot(x, [row["realized_profit"] for row in rows], marker="o", label=f"cut {cut:.0%}")
        axes[1].plot(x, [row["deployed"] for row in rows], marker="o")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Realized profit ($)")
    axes[0].legend()
    axes[1].set_ylabel("Total deployed ($)")
    axes[1].set_xlabel("Target weekly budget ($)")
    figure.tight_layout()
    figure.savefig(args.output_dir / "capacity_frontier.png", dpi=160)
    plt.close(figure)

    summary = {
        "causal_accounting": {
            "scheduled_end_date": "used as entry-time horizon proxy; archive has no metadata revision history",
            "actual_closed_time": "used only when simulated time reaches the close",
            "open_position_mark": "cost basis; future outcomes excluded from account value",
            "eventual_open_outcome": "audit-only field, excluded from realized profit and account value",
            "resolved_market_selection_bias": "not eliminated: compact source universe contains resolved binary markets",
            "event_cluster_limits": "not implemented: source data has no reliable event_id",
            "order_book_limit": "triggering trade liquidity is available; historical queue/depth snapshots are not",
        },
        "cuts": args.cut_fractions,
        "experiments": sorted({row["experiment"] for row in summaries}),
        "files": [
            "account_summary.csv", "gate_selection.csv", "weekly_accounts.csv",
            "category_attribution.csv", "horizon_attribution.csv",
            "unresolved_positions.csv", "capacity_frontier.png",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
