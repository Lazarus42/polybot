#!/usr/bin/env python3
"""Single-process drawdown-throttle OOS sweep.

This reuses one strategy-cube load and one same-side exit-path attachment across
all throttle variants.
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
from realistic_underdog_account import (
    attach_exit_paths,
    fit_brackets,
    fit_exit_candidates,
    fit_sizing_model,
    run_account,
    write_csv,
)
from walk_forward_oos import (
    attach_exit_liquidity_totals,
    apply_gate_profile,
    load_arrays,
    parse_fixed_strategy,
    period_rows,
    profile_description,
)


DEFAULT_THROTTLES = [
    ("no_throttle", math.inf, math.inf, 0.0),
    ("throttle_05_15", 0.05, 0.15, 0.0),
    ("throttle_10_20", 0.10, 0.20, 0.0),
    ("throttle_05_20_min25", 0.05, 0.20, 0.25),
]

PROFILE_CATEGORIES = {
    "ungated": {
        "exclude_high_price": "crypto,economics,entertainment,other,politics,sports,weather",
    }
}


def parse_throttle(value: str) -> tuple[str, float, float, float]:
    parts = value.split(":")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("throttle must use name:start:stop:min_scale")
    name, start, stop, min_scale = parts
    return name, float(start), float(stop), float(min_scale)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    summaries = []
    for (variant, months), group in df.groupby(["throttle_variant", "test_months"], dropna=False):
        returns = group["account_return"].astype(float).to_numpy()
        profits = group["realized_profit"].astype(float).to_numpy()
        summaries.append({
            "throttle_variant": variant,
            "test_months": int(months),
            "periods": len(group),
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
            "mean_max_drawdown": float(group["max_drawdown"].astype(float).mean()),
            "worst_max_drawdown": float(group["max_drawdown"].astype(float).max()),
        })
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/underdog_optimization_kalshi"))
    parser.add_argument("--data-dir", type=Path, default=Path("archive/processed/underdog_events"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/oos_best_strategy_throttle_sweep_fast"))
    parser.add_argument(
        "--fixed-strategy",
        type=parse_fixed_strategy,
        default=parse_fixed_strategy("ungated:exclude_high_price:forecast_paced"),
    )
    parser.add_argument("--throttles", type=parse_throttle, nargs="*", default=DEFAULT_THROTTLES)
    parser.add_argument("--test-months", type=int, nargs="+", default=[1, 2, 6, 12])
    parser.add_argument("--min-train-months", type=int, default=12)
    parser.add_argument("--validation-months", type=int, default=6)
    parser.add_argument("--stride-months", type=int, default=None)
    parser.add_argument("--include-partial-final", action="store_true")
    parser.add_argument("--min-fit-trades", type=int, default=10)
    parser.add_argument("--scenario", choices=["optimistic", "neutral", "conservative", "very_conservative"], default="conservative")
    parser.add_argument("--initial-cash", type=float, default=5000.0)
    parser.add_argument("--period-budget", type=float, default=5000.0)
    parser.add_argument("--budget-period", choices=["week", "month"], default="month")
    parser.add_argument("--base-stake", type=float, default=1.0)
    parser.add_argument("--max-stake", type=float, default=75.0)
    parser.add_argument("--kelly-fraction", type=float, default=0.10)
    parser.add_argument("--max-fraction", type=float, default=0.02)
    parser.add_argument("--reserve-fraction", type=float, default=0.30)
    parser.add_argument("--min-stake", type=float, default=1.0)
    parser.add_argument("--min-minutes-to-close", type=float, default=60.0)
    parser.add_argument("--max-category-locked-fraction", type=float, default=0.30)
    parser.add_argument("--max-regime-locked-fraction", type=float, default=0.30)
    parser.add_argument("--shrink-k", type=float, default=100.0)
    parser.add_argument("--lcb-z", type=float, default=1.0)
    args = parser.parse_args()

    gate, profile, policy = args.fixed_strategy
    if gate != "ungated" or profile != "exclude_high_price":
        raise SystemExit("fast throttle sweep currently supports ungated:exclude_high_price:* fixed strategies")
    arrays, _ = load_arrays(args.report_dir, args.data_dir)
    execution = execution_model(args.report_dir)
    print("attaching same-side exit paths...", flush=True)
    attach_exit_paths(arrays, args.data_dir, execution)
    attach_exit_liquidity_totals(arrays)
    print("exit paths attached", flush=True)

    period_rows_out: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    categories = {"sports", "crypto", "politics", "economics", "weather", "entertainment", "other"}
    horizon = math.inf
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
            test_start = int(period["test_start"].timestamp())
            test_end = int(period["test_end"].timestamp())
            fit_mask = arrays["times"] < test_start
            if "candidate_returns" in arrays:
                selectors = fit_exit_candidates(
                    arrays["levels"], arrays["candidate_returns"], fit_mask, args.min_fit_trades
                )
            else:
                selectors = fit_brackets(arrays["levels"], arrays["cube"], fit_mask, args.min_fit_trades)
            pretest_indexes = np.where(arrays["times"] < test_start)[0]
            fit_indexes = apply_gate_profile(
                pretest_indexes,
                arrays,
                profile,
                args.scenario,
                include_bucket_quality=False,
            )
            sizing_model = fit_sizing_model(
                fit_indexes,
                test_start,
                selectors,
                arrays,
                categories,
                horizon,
                budget_period=args.budget_period,
                shrink_k=args.shrink_k,
                lcb_z=args.lcb_z,
            )
            test_indexes = apply_gate_profile(
                np.where((arrays["times"] >= test_start) & (arrays["times"] < test_end))[0],
                arrays,
                profile,
                args.scenario,
                sizing_model=sizing_model,
            )
            for name, start, stop, min_scale in args.throttles:
                print(
                    f"evaluating {name} {period['period_id']} "
                    f"test={period['test_start'].date()}->{period['test_end'].date()}",
                    flush=True,
                )
                selected_rows.append({
                    "throttle_variant": name,
                    "throttle_start": start,
                    "throttle_stop": stop,
                    "throttle_min_scale": min_scale,
                    "period_id": period["period_id"],
                    "test_months": period["test_months"],
                    "test_start": period["test_start"].date(),
                    "test_end": period["test_end"].date(),
                    "selection_status": "fixed_fast",
                    "fixed_strategy": ":".join(args.fixed_strategy),
                    "eligible_test_opportunities": int(len(test_indexes)),
                    "eligible_entry_levels": int(len(selectors)),
                })
                result = run_account(
                    test_indexes,
                    test_end,
                    selectors,
                    arrays,
                    categories,
                    horizon,
                    execution,
                    args.scenario,
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
                    drawdown_throttle_start=start,
                    drawdown_throttle_stop=stop,
                    drawdown_throttle_min_scale=min_scale,
                )
                summary = result["summary"]
                period_rows_out.append({
                    "throttle_variant": name,
                    "throttle_start": start,
                    "throttle_stop": stop,
                    "throttle_min_scale": min_scale,
                    "fixed_strategy": ":".join(args.fixed_strategy),
                    "period_id": period["period_id"],
                    "test_months": period["test_months"],
                    "validation_months": period["validation_months"],
                    "validation_start": period["validation_start"].date(),
                    "test_start": period["test_start"].date(),
                    "test_end": period["test_end"].date(),
                    "selected_gate": gate,
                    "selected_gate_profile": profile,
                    "selected_gate_profile_description": profile_description(profile),
                    "selected_categories": ",".join(sorted(categories)),
                    "selected_max_scheduled_horizon_days": "none",
                    "selected_sizing_policy": policy,
                    "selected_strategy": ":".join(args.fixed_strategy),
                    **summary,
                    "skipped": json.dumps(summary["skipped"], sort_keys=True),
                })

    summary_rows = summarize(period_rows_out)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "throttle_period_results.csv", period_rows_out)
    write_csv(args.output_dir / "throttle_summary.csv", summary_rows)
    write_csv(args.output_dir / "throttle_selected_strategies.csv", selected_rows)
    write_csv(args.output_dir / "throttle_validation_leaderboard.csv", validation_rows)
    (args.output_dir / "summary.json").write_text(json.dumps({
        "fixed_strategy": ":".join(args.fixed_strategy),
        "throttles": [
            {"name": name, "start": start, "stop": stop, "min_scale": min_scale}
            for name, start, stop, min_scale in args.throttles
        ],
        "test_months": args.test_months,
        "files": [
            "throttle_period_results.csv",
            "throttle_summary.csv",
            "throttle_selected_strategies.csv",
            "throttle_validation_leaderboard.csv",
        ],
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "period_rows": len(period_rows_out),
        "summary_rows": len(summary_rows),
    }, indent=2))


if __name__ == "__main__":
    main()
