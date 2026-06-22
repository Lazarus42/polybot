#!/usr/bin/env python3
"""Long-holdout sizing-policy experiments for the realistic underdog account.

Each cut uses an early fit segment for brackets and a later calibration segment for
market gates plus sizing weights. The holdout is then replayed as one continuous
account over longer windows so tail-event strategies are not judged only by short
weekly or monthly cohorts.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from online_underdog_allocation import CATEGORIES, execution_model, week_id, week_start
from realistic_underdog_account import (
    HORIZONS,
    SIZING_POLICIES,
    attach_exit_paths,
    calibration_category_sets,
    fit_brackets,
    fit_exit_candidates,
    fit_sizing_model,
    gate_score,
    load_market_categories,
    run_account,
    write_csv,
)


def add_months(value: datetime, months: int) -> datetime:
    total = value.month - 1 + months
    year = value.year + total // 12
    month = total % 12 + 1
    return value.replace(year=year, month=month)


def parse_holdout_months(values: list[str]) -> list[Optional[int]]:
    result = []
    for value in values:
        if value.lower() == "all":
            result.append(None)
        else:
            parsed = int(value)
            if parsed <= 0:
                raise argparse.ArgumentTypeError("holdout months must be positive or 'all'")
            result.append(parsed)
    return result


def load_arrays(report_dir: Path, data_dir: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    data = np.load(report_dir / "strategy_cube.npz")
    required = {
        "underdog_sides",
        "scheduled_end_times",
        "closed_times",
        "exit_times",
        "exit_prices",
        "exit_fill_usd",
        "exit_codes",
    }
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
    if "candidate_returns" in data.files:
        arrays.update({
            "candidate_returns": data["candidate_returns"][order],
            "candidate_exit_times": data["candidate_exit_times"][order],
            "candidate_exit_prices": data["candidate_exit_prices"][order],
            "candidate_exit_codes": data["candidate_exit_codes"][order],
            "candidate_names": data["candidate_names"],
            "candidate_families": data["candidate_families"],
            "candidate_regimes": data["candidate_regimes"],
            "candidate_policy_json": data["candidate_policy_json"],
        })
    arrays["categories"] = load_market_categories(data_dir, arrays["market_ids"])
    return arrays, np.asarray(week_id(arrays["times"]))


def select_gate(
    cut: float,
    calibration_indexes: np.ndarray,
    calibration_end: int,
    selectors: dict[int, tuple[int, int]],
    arrays: dict[str, np.ndarray],
    execution: dict[str, float],
    initial_cash: float,
    period_budget: float,
    budget_period: str,
) -> tuple[set[str], float, list[dict[str, Any]]]:
    candidates = calibration_category_sets(calibration_indexes, calibration_end, selectors, arrays)
    rows = []
    best = None
    for name, categories in candidates.items():
        for horizon in HORIZONS:
            result = run_account(
                calibration_indexes,
                calibration_end,
                selectors,
                arrays,
                categories,
                horizon,
                execution,
                "conservative",
                initial_cash,
                period_budget,
                1.0,
                budget_period=budget_period,
                sizing_policy="flat_one",
            )
            score = gate_score(result)
            row = {
                "cut_fraction": cut,
                "candidate": name,
                "categories": ",".join(sorted(categories)),
                "max_scheduled_horizon_days": "none" if math.isinf(horizon) else horizon,
                "score": score,
                **result["summary"],
            }
            row["skipped"] = json.dumps(result["summary"]["skipped"], sort_keys=True)
            rows.append(row)
            if best is None or score > best[0]:
                best = (score, categories, horizon)
    assert best is not None
    return set(best[1]), float(best[2]), rows


def sizing_audit_rows(
    cut: float,
    gate_name: str,
    model: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for granularity, stats in model["stats"].items():
        for key, value in stats.items():
            rows.append({
                "cut_fraction": cut,
                "gate": gate_name,
                "granularity": granularity,
                "bucket": key,
                **value,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/underdog_optimization_kalshi"))
    parser.add_argument("--data-dir", type=Path, default=Path("archive/processed/underdog_events"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/long_holdout_weighting"))
    parser.add_argument("--cut-fractions", type=float, nargs="+", default=[0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--holdout-months", nargs="+", default=["3", "6", "12", "all"])
    parser.add_argument(
        "--sizing-policies",
        nargs="+",
        choices=SIZING_POLICIES,
        default=["flat_one", "availability", "hybrid_floor_lcb", "fractional_kelly", "forecast_paced"],
    )
    parser.add_argument("--inner-fit-fraction", type=float, default=0.7)
    parser.add_argument("--min-fit-trades", type=int, default=30)
    parser.add_argument("--initial-cash", type=float, default=5000.0)
    parser.add_argument("--period-budget", type=float, default=5000.0)
    parser.add_argument("--budget-period", choices=["week", "month"], default="month")
    parser.add_argument("--base-stake", type=float, default=1.0)
    parser.add_argument("--max-stake", type=float, default=250.0)
    parser.add_argument("--kelly-fraction", type=float, default=0.10)
    parser.add_argument("--max-fraction", type=float, default=0.02)
    parser.add_argument("--reserve-fraction", type=float, default=0.25)
    parser.add_argument("--min-stake", type=float, default=1.0)
    parser.add_argument("--min-minutes-to-close", type=float, default=60.0)
    parser.add_argument("--max-category-locked-fraction", type=float, default=0.35)
    parser.add_argument("--max-regime-locked-fraction", type=float, default=0.35)
    parser.add_argument("--shrink-k", type=float, default=100.0)
    parser.add_argument("--lcb-z", type=float, default=1.0)
    args = parser.parse_args()

    holdout_months = parse_holdout_months(args.holdout_months)
    arrays, weeks = load_arrays(args.report_dir, args.data_dir)
    all_weeks = np.arange(int(weeks.min()), int(weeks.max()) + 1)
    execution = execution_model(args.report_dir)
    attach_exit_paths(arrays, args.data_dir, execution)

    account_rows = []
    gate_rows = []
    bucket_rows = []
    training_rows = []

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
        full_holdout_end = int(week_start(int(all_weeks[-1]) + 1).timestamp())

        fit_mask = arrays["times"] < calibration_start
        if "candidate_returns" in arrays:
            selectors = fit_exit_candidates(
                arrays["levels"], arrays["candidate_returns"], fit_mask, args.min_fit_trades
            )
        else:
            selectors = fit_brackets(arrays["levels"], arrays["cube"], fit_mask, args.min_fit_trades)
        calibration_indexes = np.where(
            (arrays["times"] >= calibration_start) & (arrays["times"] < holdout_start)
        )[0]
        selected_categories, selected_horizon, rows = select_gate(
            cut,
            calibration_indexes,
            holdout_start,
            selectors,
            arrays,
            execution,
            args.initial_cash,
            args.period_budget,
            args.budget_period,
        )
        gate_rows.extend(rows)
        gate_variants = {
            "selected_gate": (selected_categories, selected_horizon),
            "ungated": (set(CATEGORIES), math.inf),
        }
        training_rows.append({
            "cut_fraction": cut,
            "calibration_start": datetime.fromtimestamp(calibration_start, timezone.utc).date(),
            "holdout_start": datetime.fromtimestamp(holdout_start, timezone.utc).date(),
            "full_holdout_end": datetime.fromtimestamp(full_holdout_end, timezone.utc).date(),
            "fit_weeks": inner_position,
            "calibration_weeks": cut_week - calibration_week,
            "selected_categories": ",".join(sorted(selected_categories)),
            "selected_horizon_days": "none" if math.isinf(selected_horizon) else selected_horizon,
            "eligible_entry_levels": len(selectors),
        })

        for gate_name, (categories, horizon) in gate_variants.items():
            sizing_model = fit_sizing_model(
                calibration_indexes,
                holdout_start,
                selectors,
                arrays,
                categories,
                horizon,
                budget_period=args.budget_period,
                shrink_k=args.shrink_k,
                lcb_z=args.lcb_z,
            )
            bucket_rows.extend(sizing_audit_rows(cut, gate_name, sizing_model))

            for months in holdout_months:
                if months is None:
                    holdout_end = full_holdout_end
                    horizon_name = "all"
                else:
                    start_dt = datetime.fromtimestamp(holdout_start, timezone.utc)
                    holdout_end = min(int(add_months(start_dt, months).timestamp()), full_holdout_end)
                    horizon_name = f"{months}m"
                indexes = np.where(
                    (arrays["times"] >= holdout_start) & (arrays["times"] < holdout_end)
                )[0]
                for policy in args.sizing_policies:
                    result = run_account(
                        indexes,
                        holdout_end,
                        selectors,
                        arrays,
                        categories,
                        horizon,
                        execution,
                        "conservative",
                        args.initial_cash,
                        args.period_budget,
                        args.base_stake,
                        budget_period=args.budget_period,
                        sizing_policy=policy,
                        sizing_model=sizing_model,
                        max_stake=args.max_stake,
                        kelly_fraction=args.kelly_fraction,
                        max_fraction=args.max_fraction,
                        reserve_fraction=args.reserve_fraction,
                        min_stake=args.min_stake,
                        min_minutes_to_close=args.min_minutes_to_close,
                        max_category_locked_fraction=args.max_category_locked_fraction,
                        max_regime_locked_fraction=args.max_regime_locked_fraction,
                    )
                    account_rows.append({
                        "cut_fraction": cut,
                        "gate": gate_name,
                        "holdout_window": horizon_name,
                        "holdout_start": datetime.fromtimestamp(holdout_start, timezone.utc).date(),
                        "holdout_end": datetime.fromtimestamp(holdout_end, timezone.utc).date(),
                        "period_budget": args.period_budget,
                        "budget_period": args.budget_period,
                        "sizing_policy": policy,
                        **result["summary"],
                        "skipped": json.dumps(result["summary"]["skipped"], sort_keys=True),
                    })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "account_summary.csv", account_rows)
    write_csv(args.output_dir / "gate_selection.csv", gate_rows)
    write_csv(args.output_dir / "sizing_bucket_stats.csv", bucket_rows)
    write_csv(args.output_dir / "training_cuts.csv", training_rows)
    summary = {
        "policies": list(args.sizing_policies),
        "holdout_months": args.holdout_months,
        "cuts": args.cut_fractions,
        "budget_period": args.budget_period,
        "files": [
            "account_summary.csv",
            "gate_selection.csv",
            "sizing_bucket_stats.csv",
            "training_cuts.csv",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
