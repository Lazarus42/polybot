#!/usr/bin/env python3
"""Independent $5,000 monthly cohorts for causal underdog strategies."""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

from online_underdog_allocation import CATEGORIES, execution_model, week_id, week_start
from realistic_underdog_account import (
    DAY,
    HORIZONS,
    attach_exit_paths,
    calibration_category_sets,
    fit_brackets,
    gate_score,
    load_market_categories,
    run_account,
)


def next_month(value: datetime) -> datetime:
    if value.month == 12:
        return datetime(value.year + 1, 1, 1, tzinfo=timezone.utc)
    return datetime(value.year, value.month + 1, 1, tzinfo=timezone.utc)


def month_count(start: int, end: int) -> float:
    return max(1.0, (end - start) / (365.25 / 12 * DAY))


def load_arrays(report_dir: Path, data_dir: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    data = np.load(report_dir / "strategy_cube.npz")
    required = {"underdog_sides"}
    missing = required - set(data.files)
    if missing:
        raise SystemExit(f"strategy cube missing {sorted(missing)}; rerun optimize_underdog_bracket.py")
    order = np.argsort(data["entry_times"], kind="stable")
    arrays = {
        "market_ids": data["market_ids"][order],
        "times": data["entry_times"][order],
        "sides": data["underdog_sides"][order],
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
    arrays["categories"] = load_market_categories(data_dir, arrays["market_ids"])
    return arrays, np.asarray(week_id(arrays["times"]))


def global_brackets(
    levels: np.ndarray,
    cube: np.ndarray,
    fit_mask: np.ndarray,
    min_level_trades: int,
) -> tuple[dict[int, tuple[int, int]], tuple[int, int], set[int]]:
    fit_indexes = np.where(fit_mask)[0]
    flat = int(np.argmax(cube[fit_indexes].mean(axis=0)))
    selector = tuple(int(value) for value in np.unravel_index(flat, cube.shape[1:]))
    trained_levels = {
        level for level in range(1, 50)
        if int(np.sum(fit_mask & (levels == level))) >= min_level_trades
    }
    return {level: selector for level in trained_levels}, selector, trained_levels


def price_ranges(trained_levels: set[int]) -> dict[str, set[int]]:
    result = {}
    starts = list(range(1, 50, 5))
    for start_index, start in enumerate(starts):
        for final_start in starts[start_index:]:
            end = min(49, final_start + 4)
            selected = {level for level in trained_levels if start <= level <= end}
            if selected:
                result[f"{start:02d}-{end:02d}c"] = selected
    result["all_trained_prices"] = set(trained_levels)
    return result


def select_market_gate(
    cut: float,
    calibration_indexes: np.ndarray,
    calibration_end: int,
    selectors: dict[int, tuple[int, int]],
    arrays: dict[str, np.ndarray],
    model: dict[str, float],
    initial_cash: float,
    monthly_budget: float,
) -> tuple[set[str], float, list[dict[str, Any]]]:
    candidates = calibration_category_sets(calibration_indexes, calibration_end, selectors, arrays)
    rows = []
    best = None
    for name, categories in candidates.items():
        for horizon in HORIZONS:
            result = run_account(
                calibration_indexes, calibration_end, selectors, arrays, categories,
                horizon, model, "conservative", initial_cash, monthly_budget,
                1.0, budget_period="month",
            )
            score = gate_score(result)
            rows.append({
                "cut_fraction": cut, "gate_type": "market",
                "candidate": name, "categories": ",".join(sorted(categories)),
                "max_horizon_days": "none" if math.isinf(horizon) else horizon,
                "score": score, **result["summary"],
            })
            if best is None or score > best[0]:
                best = (score, categories, horizon)
    assert best is not None
    return set(best[1]), float(best[2]), rows


def select_price_gate(
    cut: float,
    calibration_indexes: np.ndarray,
    calibration_end: int,
    selectors: dict[int, tuple[int, int]],
    ranges: dict[str, set[int]],
    categories: set[str],
    horizon: float,
    arrays: dict[str, np.ndarray],
    model: dict[str, float],
    initial_cash: float,
    monthly_budget: float,
) -> tuple[str, set[int], list[dict[str, Any]]]:
    rows = []
    best = None
    for name, levels in ranges.items():
        result = run_account(
            calibration_indexes, calibration_end, selectors, arrays, categories,
            horizon, model, "conservative", initial_cash, monthly_budget,
            1.0, allowed_levels=levels, budget_period="month",
        )
        score = gate_score(result)
        rows.append({
            "cut_fraction": cut, "gate_type": "global_price_range",
            "candidate": name, "levels": ",".join(str(value) for value in sorted(levels)),
            "score": score, **result["summary"],
        })
        if best is None or score > best[0]:
            best = (score, name, levels)
    assert best is not None
    return str(best[1]), set(best[2]), rows


def eligible_count(
    indexes: np.ndarray,
    selectors: dict[int, tuple[int, int]],
    arrays: dict[str, np.ndarray],
    categories: set[str],
    horizon: float,
    levels: Optional[set[int]],
) -> int:
    count = 0
    for index in indexes:
        level = int(arrays["levels"][index])
        if level not in selectors or (levels is not None and level not in levels):
            continue
        if str(arrays["categories"][index]) not in categories:
            continue
        scheduled_days = max(
            0.0,
            (int(arrays["scheduled_end"][index]) - int(arrays["times"][index])) / DAY,
        )
        if scheduled_days <= horizon:
            count += 1
    return count


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/underdog_optimization_kalshi"))
    parser.add_argument("--data-dir", type=Path, default=Path("archive/processed/underdog_events"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/monthly_underdog_experiments"))
    parser.add_argument("--cut-fractions", type=float, nargs="+", default=[0.6, 0.7, 0.8])
    parser.add_argument("--inner-fit-fraction", type=float, default=0.7)
    parser.add_argument("--min-fit-trades", type=int, default=30)
    parser.add_argument("--initial-cash", type=float, default=5000.0)
    parser.add_argument("--monthly-budget", type=float, default=5000.0)
    args = parser.parse_args()

    arrays, weeks = load_arrays(args.report_dir, args.data_dir)
    all_weeks = np.arange(int(weeks.min()), int(weeks.max()) + 1)
    execution = execution_model(args.report_dir)
    attach_exit_paths(arrays, args.data_dir, execution)
    monthly_rows = []
    gate_rows = []
    cut_rows = []

    for cut in sorted(set(args.cut_fractions)):
        cut_position = min(len(all_weeks) - 1, max(2, int(len(all_weeks) * cut)))
        cut_week = int(all_weeks[cut_position])
        training_weeks = all_weeks[:cut_position]
        inner_position = min(
            len(training_weeks) - 1,
            max(1, int(len(training_weeks) * args.inner_fit_fraction)),
        )
        calibration_week = int(training_weeks[inner_position])
        calibration_start = int(week_start(calibration_week).timestamp())
        holdout_start = int(week_start(cut_week).timestamp())
        holdout_end = int(week_start(int(all_weeks[-1]) + 1).timestamp())
        fit_mask = arrays["times"] < calibration_start
        per_level = fit_brackets(
            arrays["levels"], arrays["cube"], fit_mask, args.min_fit_trades
        )
        global_model, global_selector, trained_levels = global_brackets(
            arrays["levels"], arrays["cube"], fit_mask, args.min_fit_trades
        )
        calibration_indexes = np.where(
            (arrays["times"] >= calibration_start) & (arrays["times"] < holdout_start)
        )[0]
        categories, horizon, market_gate_rows = select_market_gate(
            cut, calibration_indexes, holdout_start, per_level, arrays, execution,
            args.initial_cash, args.monthly_budget,
        )
        gate_rows.extend(market_gate_rows)
        price_name, selected_prices, price_rows = select_price_gate(
            cut, calibration_indexes, holdout_start, global_model,
            price_ranges(trained_levels), categories, horizon, arrays, execution,
            args.initial_cash, args.monthly_budget,
        )
        gate_rows.extend(price_rows)
        calibration_months = month_count(calibration_start, holdout_start)
        strategies = {
            "ungated_per_level": (per_level, set(CATEGORIES), math.inf, None),
            "market_gated_per_level": (per_level, categories, horizon, None),
            "market_price_gated_global": (global_model, categories, horizon, selected_prices),
        }
        lambdas = {
            name: eligible_count(
                calibration_indexes, selectors, arrays, allowed_categories,
                max_horizon, allowed_levels,
            ) / calibration_months
            for name, (selectors, allowed_categories, max_horizon, allowed_levels)
            in strategies.items()
        }
        cut_rows.append({
            "cut_fraction": cut,
            "calibration_start": datetime.fromtimestamp(calibration_start, timezone.utc).date(),
            "holdout_start": datetime.fromtimestamp(holdout_start, timezone.utc).date(),
            "holdout_end": datetime.fromtimestamp(holdout_end, timezone.utc).date(),
            "selected_categories": ",".join(sorted(categories)),
            "selected_horizon_days": "none" if math.isinf(horizon) else horizon,
            "global_take_profit": float(np.load(args.report_dir / "strategy_cube.npz")["take_profits"][global_selector[0]]),
            "global_stop_loss": float(np.load(args.report_dir / "strategy_cube.npz")["stop_losses"][global_selector[1]]),
            "selected_price_range": price_name,
            "selected_price_levels": ",".join(str(value) for value in sorted(selected_prices)),
        })

        month = datetime.fromtimestamp(holdout_start, timezone.utc).replace(day=1, hour=0, minute=0, second=0)
        while int(month.timestamp()) < holdout_end:
            following = next_month(month)
            start = max(holdout_start, int(month.timestamp()))
            end = min(holdout_end, int(following.timestamp()))
            indexes = np.where((arrays["times"] >= start) & (arrays["times"] < end))[0]
            partial = start != int(month.timestamp()) or end != int(following.timestamp())
            for strategy, (selectors, allowed_categories, max_horizon, allowed_levels) in strategies.items():
                for sizing in ("one_dollar_until_exhausted", "availability_target_5000"):
                    result = run_account(
                        indexes, end, selectors, arrays, allowed_categories, max_horizon,
                        execution, "conservative", args.initial_cash,
                        args.monthly_budget, 1.0,
                        availability_lambda=lambdas[strategy] if sizing == "availability_target_5000" else None,
                        allowed_levels=allowed_levels,
                        budget_period="month",
                    )
                    monthly_rows.append({
                        "cut_fraction": cut,
                        "month": month.strftime("%Y-%m"),
                        "partial_month": partial,
                        "strategy": strategy,
                        "sizing": sizing,
                        "training_expected_opportunities_per_month": lambdas[strategy],
                        "monthly_budget": args.monthly_budget,
                        "budget_utilization": result["summary"]["deployed"] / args.monthly_budget,
                        **result["summary"],
                        "skipped": json.dumps(result["summary"]["skipped"], sort_keys=True),
                    })
            month = following

    aggregate_rows = []
    for cut in sorted(set(args.cut_fractions)):
        for strategy in sorted({row["strategy"] for row in monthly_rows}):
            for sizing in sorted({row["sizing"] for row in monthly_rows}):
                selected = [
                    row for row in monthly_rows
                    if row["cut_fraction"] == cut and row["strategy"] == strategy
                    and row["sizing"] == sizing and not row["partial_month"]
                ]
                profits = np.asarray([row["realized_profit"] for row in selected], dtype=float)
                aggregate_rows.append({
                    "cut_fraction": cut, "strategy": strategy, "sizing": sizing,
                    "full_months": len(selected),
                    "total_realized_profit": float(profits.sum()) if len(profits) else 0.0,
                    "mean_monthly_profit": float(profits.mean()) if len(profits) else 0.0,
                    "median_monthly_profit": float(np.median(profits)) if len(profits) else 0.0,
                    "positive_month_fraction": float((profits > 0).mean()) if len(profits) else 0.0,
                    "worst_month": float(profits.min()) if len(profits) else 0.0,
                    "best_month": float(profits.max()) if len(profits) else 0.0,
                    "total_deployed": sum(row["deployed"] for row in selected),
                    "mean_budget_utilization": float(np.mean([row["budget_utilization"] for row in selected])) if selected else 0.0,
                })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "monthly_results.csv", monthly_rows)
    write_csv(args.output_dir / "monthly_aggregate.csv", aggregate_rows)
    write_csv(args.output_dir / "gate_candidates.csv", gate_rows)
    write_csv(args.output_dir / "selected_models.csv", cut_rows)

    figure, axes = plt.subplots(len(args.cut_fractions), 1, figsize=(13, 10), sharex=False)
    if len(args.cut_fractions) == 1:
        axes = [axes]
    for axis, cut in zip(axes, sorted(set(args.cut_fractions))):
        for strategy in ("ungated_per_level", "market_gated_per_level", "market_price_gated_global"):
            selected = [
                row for row in monthly_rows if row["cut_fraction"] == cut
                and row["strategy"] == strategy
                and row["sizing"] == "one_dollar_until_exhausted"
            ]
            axis.plot(
                [row["month"] for row in selected],
                [row["realized_profit"] for row in selected],
                marker="o", label=strategy,
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(f"Cut {cut:.0%}: $1 per market, conservative fills")
        axis.set_ylabel("Realized profit ($)")
        axis.tick_params(axis="x", rotation=60)
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "monthly_one_dollar.png", dpi=160)
    plt.close(figure)
    print(json.dumps({"selected_models": cut_rows, "rows": len(monthly_rows)}, indent=2, default=str))


if __name__ == "__main__":
    main()
