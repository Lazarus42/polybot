#!/usr/bin/env python3
"""Tune family-ensemble component caps on earlier OOS periods, then score holdout."""
from __future__ import annotations

import argparse
import itertools
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from realistic_underdog_account import write_csv
from replay_family_ensemble_oos import (
    DEFAULT_COMPONENTS,
    load_component_signals,
    parse_component,
    replay_ensemble,
)
from replay_strategy_family_oos import period_rows
from walk_forward_oos import month_floor


# Grids are keyed by sleeve rather than by named component, so the tuner can sweep an
# arbitrary candidate pool (e.g. a de-leaked ranking) instead of a hardcoded six.
# With a participation cap in force, per-market deployment is limited by liquidity, so
# monthly caps mainly govern how many thick markets a component compounds into per month.
SLEEVE_CAP_GRID = {
    "core": [500.0, 1500.0],
    "tail": [125.0, 375.0],
}

# Stakes span low/high to test heavier deployment; the participation cap clips these
# back on thin markets, so only the liquid tail of each component can actually use them.
SLEEVE_STAKE_GRID = {
    "core": [5.0, 15.0],
    "tail": [2.0, 5.0],
}


def component_map(components: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {item["component"]: dict(item) for item in components}


def candidate_components(base: list[dict[str, Any]], cap_values: dict[str, float], stake_values: dict[str, float]) -> list[dict[str, Any]]:
    mapped = component_map(base)
    result = []
    for component, item in mapped.items():
        updated = dict(item)
        if component in cap_values:
            updated["monthly_cap"] = cap_values[component]
        if component in stake_values:
            updated["stake"] = stake_values[component]
        result.append(updated)
    return result


def replay_periods(
    signals: pd.DataFrame,
    components: list[dict[str, Any]],
    periods: list[dict[str, Any]],
    args: argparse.Namespace,
    config_name: str,
) -> list[dict[str, Any]]:
    combined = load_component_signals(signals, components)
    rows = []
    for period in periods:
        test_start = pd.Timestamp(period["test_start"])
        test_end = pd.Timestamp(period["test_end"])
        period_signals = combined[(combined["timestamp"] >= test_start) & (combined["timestamp"] < test_end)]
        result = replay_ensemble(
            period_signals,
            test_end,
            args.initial_cash,
            args.period_budget,
            args.budget_period,
            args.reserve_fraction,
            args.min_stake,
            args.max_trades_per_market,
            args.max_components_per_market,
            args.participation_fraction,
            args.min_stake_fill_fraction,
        )
        rows.append({
            "config": config_name,
            "period_id": period["period_id"],
            "test_months": period["test_months"],
            "test_start": period["test_start"].date(),
            "test_end": period["test_end"].date(),
            **result,
            "component_counts": json.dumps(result["component_counts"], sort_keys=True),
            "component_profit": json.dumps(result["component_profit"], sort_keys=True),
            "component_deployed": json.dumps(result["component_deployed"], sort_keys=True),
            "skipped": json.dumps(result["skipped"], sort_keys=True),
        })
    return rows


def score(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return -1e9
    df = pd.DataFrame(rows)
    mean_profit = float(df["realized_profit"].mean())
    worst_profit = float(df["realized_profit"].min())
    worst_drawdown = float(df["max_drawdown"].max())
    positive_rate = float((df["realized_profit"] > 0).mean())
    # Favor steady profit, penalize tail loss and drawdown.
    return mean_profit + 0.25 * worst_profit - 250.0 * worst_drawdown + 10.0 * positive_rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", type=Path, default=Path("reports/strategy_family_diagnostics/strategy_family_signals.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/family_ensemble_tuning"))
    parser.add_argument("--base-components", type=parse_component, nargs="+", default=[parse_component(value) for value in DEFAULT_COMPONENTS])
    parser.add_argument("--first-month", default="2022-11-01")
    parser.add_argument("--tune-before", default="2025-05-01")
    parser.add_argument("--test-months", type=int, nargs="+", default=[1, 2, 6])
    parser.add_argument("--min-train-months", type=int, default=12)
    parser.add_argument("--validation-months", type=int, default=6)
    parser.add_argument("--initial-cash", type=float, default=5000.0)
    parser.add_argument("--period-budget", type=float, default=5000.0)
    parser.add_argument("--budget-period", choices=["week", "month"], default="month")
    parser.add_argument("--reserve-fraction", type=float, default=0.30)
    parser.add_argument("--min-stake", type=float, default=0.25,
                        help="Absolute minimum stake floor below which a trade is skipped.")
    parser.add_argument("--max-trades-per-market", type=int, default=1)
    parser.add_argument("--max-components-per-market", type=int, default=1)
    parser.add_argument("--participation-fraction", type=float, default=0.0,
                        help="Cap each trade at this fraction of entry_fill_usd (0 disables).")
    parser.add_argument("--min-stake-fill-fraction", type=float, default=0.0,
                        help="Market-dependent minimum: floor scales with this fraction of entry_fill_usd (0 disables).")
    parser.add_argument("--max-configs", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0, help="Seed for random grid sampling.")
    args = parser.parse_args()

    signals = pd.read_csv(args.signals)
    signals["timestamp"] = pd.to_datetime(signals["timestamp"], utc=True)
    first_month = datetime.fromisoformat(args.first_month).replace(tzinfo=timezone.utc)
    last_month = month_floor(int(signals["timestamp"].max().timestamp()))
    tune_before = datetime.fromisoformat(args.tune_before).replace(tzinfo=timezone.utc)

    periods = []
    for months in sorted(set(args.test_months)):
        periods.extend(period_rows(first_month, last_month, months, args.min_train_months, args.validation_months))
    tune_periods = [p for p in periods if p["test_end"] <= tune_before]
    holdout_periods = [p for p in periods if p["test_start"] >= tune_before]

    components = [item["component"] for item in args.base_components]
    cmap = component_map(args.base_components)
    cap_lists = [SLEEVE_CAP_GRID.get(cmap[component]["sleeve"], [cmap[component]["monthly_cap"]]) for component in components]
    stake_lists = [SLEEVE_STAKE_GRID.get(cmap[component]["sleeve"], [cmap[component]["stake"]]) for component in components]

    # Build the full cap x stake grid, then randomly sample when it exceeds the
    # config budget. Enumerating in order would only ever explore the low-cap
    # prefix of the grid, biasing the search; random sampling covers it evenly.
    all_combos = [
        (cap_tuple, stake_tuple)
        for cap_tuple in itertools.product(*cap_lists)
        for stake_tuple in itertools.product(*stake_lists)
    ]
    total_combos = len(all_combos)
    rng = random.Random(args.seed)
    if total_combos > args.max_configs:
        all_combos = rng.sample(all_combos, args.max_configs)

    rows = []
    best = None
    checked = 0
    for cap_tuple, stake_tuple in all_combos:
        cap_values = dict(zip(components, cap_tuple))
        stake_values = dict(zip(components, stake_tuple))
        config_name = "cfg_" + str(checked).zfill(5)
        candidate = candidate_components(args.base_components, cap_values, stake_values)
        tune_rows = replay_periods(signals, candidate, tune_periods, args, config_name)
        tune_score = score(tune_rows)
        rows.append({
            "config": config_name,
            "tune_score": tune_score,
            "tune_mean_profit": float(pd.DataFrame(tune_rows)["realized_profit"].mean()) if tune_rows else 0.0,
            "tune_worst_profit": float(pd.DataFrame(tune_rows)["realized_profit"].min()) if tune_rows else 0.0,
            "tune_worst_drawdown": float(pd.DataFrame(tune_rows)["max_drawdown"].max()) if tune_rows else 0.0,
            "components": json.dumps(candidate, sort_keys=True),
        })
        if best is None or tune_score > best[0]:
            best = (tune_score, config_name, candidate, tune_rows)
        checked += 1

    assert best is not None
    _, best_name, best_components, best_tune_rows = best
    holdout_rows = replay_periods(signals, best_components, holdout_periods, args, best_name)
    all_rows = []
    for row in best_tune_rows:
        row = dict(row)
        row["segment"] = "tune"
        all_rows.append(row)
    for row in holdout_rows:
        row = dict(row)
        row["segment"] = "holdout"
        all_rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "tuning_leaderboard.csv", rows)
    write_csv(args.output_dir / "best_config_period_results.csv", all_rows)
    (args.output_dir / "best_config.json").write_text(json.dumps({
        "best_config": best_name,
        "tune_score": best[0],
        "components": best_components,
        "tune_periods": len(tune_periods),
        "holdout_periods": len(holdout_periods),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "checked_configs": checked,
        "best_config": best_name,
        "tune_score": best[0],
        "tune_periods": len(tune_periods),
        "holdout_periods": len(holdout_periods),
    }, indent=2))


if __name__ == "__main__":
    main()
