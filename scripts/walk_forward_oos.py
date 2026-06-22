#!/usr/bin/env python3
"""Clean walk-forward out-of-sample evaluation for underdog strategies.

For each non-overlapping forward period this script:
1. fits exit selectors using data before the validation window;
2. ranks deployable gate/sizing methods on the validation window only;
3. refits statistical components using all pre-test data; and
4. replays the single selected strategy on the unseen forward test window.

The resulting period rows are the primary surface for estimating expected
out-of-sample account returns for tail-dependent strategies.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from long_holdout_weighting_experiments import add_months, load_arrays, select_gate
from online_underdog_allocation import CATEGORIES, execution_model
from realistic_underdog_account import (
    SCENARIOS,
    SIZING_POLICIES,
    attach_exit_paths,
    bucket_stat_for_index,
    fit_brackets,
    fit_exit_candidates,
    fit_sizing_model,
    price_regime_for_level,
    run_account,
    write_csv,
)

GATE_PROFILES: dict[str, dict[str, Any]] = {
    "light": {},
    "liquid_entry_5": {"min_entry_fill_usd": 5.0},
    "liquid_entry_10": {"min_entry_fill_usd": 10.0},
    "liquid_entry_25": {"min_entry_fill_usd": 25.0},
    "liquid_entry_50": {"min_entry_fill_usd": 50.0},
    "liquid_entry_100": {"min_entry_fill_usd": 100.0},
    "exit_liquid_10": {"min_exit_path_usd": 10.0},
    "exit_liquid_25": {"min_exit_path_usd": 25.0},
    "exit_liquid_50": {"min_exit_path_usd": 50.0},
    "exit_liquid_100": {"min_exit_path_usd": 100.0},
    "exit_liquid_250": {"min_exit_path_usd": 250.0},
    "liquid_entry_exit_25": {"min_entry_fill_usd": 25.0, "min_exit_path_usd": 25.0},
    "liquid_entry_exit_50": {"min_entry_fill_usd": 50.0, "min_exit_path_usd": 50.0},
    "liquid_entry_exit_100": {"min_entry_fill_usd": 100.0, "min_exit_path_usd": 100.0},
    "low_price_1_5c": {"allowed_regimes": {"01-05c"}},
    "low_mid_price_1_15c": {"allowed_regimes": {"01-05c", "06-15c"}},
    "mid_price_6_30c": {"allowed_regimes": {"06-15c", "16-30c"}},
    "price_16_30c": {"allowed_regimes": {"16-30c"}},
    "high_price_31_49c": {"allowed_regimes": {"31-49c"}},
    "exclude_high_price": {"allowed_regimes": {"01-05c", "06-15c", "16-30c"}},
    "bucket_count_25": {"min_bucket_count": 25},
    "bucket_lcb_nonnegative": {"min_bucket_count": 10, "min_bucket_lcb": 0.0},
    "liquid_low_mid": {
        "allowed_regimes": {"01-05c", "06-15c"},
        "min_entry_fill_usd": 10.0,
        "min_exit_path_usd": 25.0,
    },
    "quality_liquid": {
        "min_entry_fill_usd": 10.0,
        "min_exit_path_usd": 25.0,
        "min_bucket_count": 25,
        "min_bucket_lcb": 0.0,
    },
}


def month_floor(timestamp: int) -> datetime:
    value = datetime.fromtimestamp(int(timestamp), timezone.utc)
    return value.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def fit_selectors(
    arrays: dict[str, np.ndarray],
    fit_end: int,
    min_fit_trades: int,
) -> dict[int, int | tuple[int, int]]:
    fit_mask = arrays["times"] < fit_end
    if "candidate_returns" in arrays:
        return fit_exit_candidates(
            arrays["levels"], arrays["candidate_returns"], fit_mask, min_fit_trades
        )
    return fit_brackets(arrays["levels"], arrays["cube"], fit_mask, min_fit_trades)


def indexes_between(arrays: dict[str, np.ndarray], start: int, end: int) -> np.ndarray:
    return np.where((arrays["times"] >= start) & (arrays["times"] < end))[0]


def indexes_before(arrays: dict[str, np.ndarray], end: int) -> np.ndarray:
    return np.where(arrays["times"] < end)[0]


def profile_names(values: list[str]) -> list[str]:
    unknown = sorted(set(values) - set(GATE_PROFILES))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown gate profile(s): {', '.join(unknown)}")
    return values


def parse_fixed_strategy(value: str) -> tuple[str, str, str]:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "fixed strategies must use gate:profile:policy, e.g. ungated:exclude_high_price:availability"
        )
    gate, profile, policy = parts
    if gate not in {"selected_gate", "ungated"}:
        raise argparse.ArgumentTypeError("fixed strategy gate must be selected_gate or ungated")
    if profile not in GATE_PROFILES:
        raise argparse.ArgumentTypeError(f"unknown gate profile: {profile}")
    if policy not in SIZING_POLICIES:
        raise argparse.ArgumentTypeError(f"unknown sizing policy: {policy}")
    return gate, profile, policy


def fixed_strategy_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["gate"]), str(row.get("gate_profile", "light")), str(row["sizing_policy"])


def attach_exit_liquidity_totals(arrays: dict[str, np.ndarray]) -> None:
    if "exit_path_offsets" not in arrays or "exit_path_usd" not in arrays:
        arrays["exit_path_total_usd"] = np.zeros(len(arrays["market_ids"]), dtype=np.float64)
        return
    offsets = arrays["exit_path_offsets"]
    usd = arrays["exit_path_usd"]
    totals = np.zeros(len(arrays["market_ids"]), dtype=np.float64)
    for index in range(len(totals)):
        totals[index] = float(usd[int(offsets[index]):int(offsets[index + 1])].sum())
    arrays["exit_path_total_usd"] = totals


def profile_description(name: str) -> str:
    profile = GATE_PROFILES[name]
    if not profile:
        return "light safety gates only"
    parts = []
    if "allowed_regimes" in profile:
        parts.append("regimes=" + "|".join(sorted(profile["allowed_regimes"])))
    for key in ("min_entry_fill_usd", "min_exit_path_usd", "min_bucket_count", "min_bucket_lcb"):
        if key in profile:
            parts.append(f"{key}={profile[key]}")
    return "; ".join(parts)


def apply_gate_profile(
    indexes: np.ndarray,
    arrays: dict[str, np.ndarray],
    profile_name: str,
    scenario_name: str,
    sizing_model: Optional[dict[str, Any]] = None,
    include_bucket_quality: bool = True,
) -> np.ndarray:
    profile = GATE_PROFILES[profile_name]
    if len(indexes) == 0 or not profile:
        return indexes

    keep = np.ones(len(indexes), dtype=bool)
    allowed_regimes = profile.get("allowed_regimes")
    if allowed_regimes is not None:
        keep &= np.asarray([
            price_regime_for_level(int(arrays["levels"][index])) in allowed_regimes
            for index in indexes
        ])

    if "min_entry_fill_usd" in profile:
        participation = float(SCENARIOS[scenario_name]["participation"])
        keep &= arrays["entry_fill"][indexes] * participation >= float(profile["min_entry_fill_usd"])

    if "min_exit_path_usd" in profile:
        keep &= arrays["exit_path_total_usd"][indexes] >= float(profile["min_exit_path_usd"])

    if include_bucket_quality and ("min_bucket_count" in profile or "min_bucket_lcb" in profile):
        if sizing_model is None:
            keep &= False
        else:
            bucket_keep = []
            for index in indexes:
                row = bucket_stat_for_index(int(index), arrays, sizing_model)
                enough_count = row.get("opportunities", 0) >= profile.get("min_bucket_count", 0)
                enough_lcb = row.get("lcb", 0.0) >= profile.get("min_bucket_lcb", -math.inf)
                bucket_keep.append(enough_count and enough_lcb)
            keep &= np.asarray(bucket_keep, dtype=bool)

    return indexes[keep]


def parse_skipped(summary: dict[str, Any]) -> str:
    return json.dumps(summary.get("skipped", {}), sort_keys=True)


def selection_score(summary: dict[str, Any], initial_cash: float, mode: str) -> float:
    """Rank validation runs by robustness, not just raw tail payoff."""
    entries = int(summary.get("entries", 0) or 0)
    if entries <= 0:
        return -1e12

    realized = float(summary.get("realized_profit", 0.0) or 0.0)
    without_top1 = float(summary.get("without_top_1", 0.0) or 0.0)
    max_single = float(summary.get("max_single_profit", 0.0) or 0.0)
    drawdown = float(summary.get("max_drawdown", 0.0) or 0.0)
    locked_end = float(summary.get("locked_capital_end", 0.0) or 0.0)
    top1_contribution = max(0.0, realized - without_top1)
    concentration_penalty = max(0.0, top1_contribution - max(0.0, without_top1))

    if mode == "raw_profit":
        return realized - initial_cash * drawdown - 0.01 * locked_end
    if mode == "without_top1":
        return without_top1 - initial_cash * drawdown - 0.03 * locked_end
    if mode == "drawdown_heavy":
        return 0.40 * realized + 0.60 * without_top1 - 2.0 * initial_cash * drawdown - 0.05 * locked_end
    if mode == "min_trades":
        trade_bonus = min(entries, 250) * 0.25
        return 0.25 * realized + 0.75 * without_top1 + trade_bonus - initial_cash * drawdown - 0.03 * locked_end
    if mode != "robust":
        raise ValueError(f"Unknown selection score mode: {mode}")
    return (
        0.35 * realized
        + 0.65 * without_top1
        - 0.50 * concentration_penalty
        - initial_cash * drawdown
        - 0.03 * locked_end
        - 0.02 * max_single
    )


def period_rows(
    arrays: dict[str, np.ndarray],
    test_months: int,
    min_train_months: int,
    validation_months: int,
    stride_months: Optional[int],
    include_partial_final: bool,
) -> list[dict[str, Any]]:
    first_month = month_floor(int(arrays["times"].min()))
    last_month_end = add_months(month_floor(int(arrays["times"].max())), 1)
    stride = stride_months or test_months
    test_start = add_months(first_month, min_train_months + validation_months)
    rows = []
    period_number = 0
    while test_start < last_month_end:
        test_end = add_months(test_start, test_months)
        if test_end > last_month_end and not include_partial_final:
            break
        test_end = min(test_end, last_month_end)
        validation_start = add_months(test_start, -validation_months)
        rows.append({
            "period_id": f"{test_months}m_{period_number:02d}_{test_start.date()}",
            "test_months": test_months,
            "validation_months": validation_months,
            "fit_start": first_month,
            "validation_start": validation_start,
            "test_start": test_start,
            "test_end": test_end,
        })
        period_number += 1
        test_start = add_months(test_start, stride)
    return rows


def horizon_value(value: float) -> str:
    return "none" if math.isinf(value) else str(value)


def validation_candidates(
    period: dict[str, Any],
    arrays: dict[str, np.ndarray],
    execution: dict[str, float],
    sizing_policies: list[str],
    gate_profiles: list[str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validation_start = int(period["validation_start"].timestamp())
    test_start = int(period["test_start"].timestamp())
    selectors = fit_selectors(arrays, validation_start, args.min_fit_trades)
    validation_indexes = indexes_between(arrays, validation_start, test_start)
    fit_indexes = indexes_before(arrays, validation_start)
    gate_rows: list[dict[str, Any]] = []
    if not selectors or len(validation_indexes) == 0:
        return [], gate_rows

    selected_categories, selected_horizon, gate_audit = select_gate(
        float(period["test_months"]),
        validation_indexes,
        test_start,
        selectors,
        arrays,
        execution,
        args.initial_cash,
        args.period_budget,
        args.budget_period,
    )
    for row in gate_audit:
        row.update({
            "period_id": period["period_id"],
            "validation_start": period["validation_start"].date(),
            "test_start": period["test_start"].date(),
            "test_months": period["test_months"],
        })
    gate_rows.extend(gate_audit)

    gate_variants = {
        "selected_gate": (selected_categories, selected_horizon),
        "ungated": (set(CATEGORIES), math.inf),
    }
    rows: list[dict[str, Any]] = []
    for gate_name, (categories, horizon) in gate_variants.items():
        for profile_name in gate_profiles:
            profile_fit_indexes = apply_gate_profile(
                fit_indexes,
                arrays,
                profile_name,
                args.scenario,
                include_bucket_quality=False,
            )
            sizing_model = fit_sizing_model(
                profile_fit_indexes,
                validation_start,
                selectors,
                arrays,
                categories,
                horizon,
                budget_period=args.budget_period,
                shrink_k=args.shrink_k,
                lcb_z=args.lcb_z,
            )
            profile_validation_indexes = apply_gate_profile(
                validation_indexes,
                arrays,
                profile_name,
                args.scenario,
                sizing_model=sizing_model,
            )
            for policy in sizing_policies:
                result = run_account(
                    profile_validation_indexes,
                    test_start,
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
                    drawdown_throttle_start=args.drawdown_throttle_start,
                    drawdown_throttle_stop=args.drawdown_throttle_stop,
                    drawdown_throttle_min_scale=args.drawdown_throttle_min_scale,
                )
                summary = result["summary"]
                rows.append({
                    "period_id": period["period_id"],
                    "test_months": period["test_months"],
                    "validation_start": period["validation_start"].date(),
                    "validation_end": period["test_start"].date(),
                    "gate": gate_name,
                    "gate_profile": profile_name,
                    "gate_profile_description": profile_description(profile_name),
                    "categories": ",".join(sorted(categories)),
                    "max_scheduled_horizon_days": horizon_value(horizon),
                    "sizing_policy": policy,
                    "eligible_validation_opportunities": int(len(profile_validation_indexes)),
                    "selection_score_mode": args.selection_score,
                    "selection_score": selection_score(summary, args.initial_cash, args.selection_score),
                    **summary,
                    "skipped": parse_skipped(summary),
                })
    return rows, gate_rows


def replay_selected_oos(
    period: dict[str, Any],
    selected: dict[str, Any],
    arrays: dict[str, np.ndarray],
    execution: dict[str, float],
    args: argparse.Namespace,
) -> Optional[dict[str, Any]]:
    test_start = int(period["test_start"].timestamp())
    test_end = int(period["test_end"].timestamp())
    selectors = fit_selectors(arrays, test_start, args.min_fit_trades)
    if not selectors:
        return None

    categories = set(str(selected["categories"]).split(",")) if selected["categories"] else set()
    horizon_text = str(selected["max_scheduled_horizon_days"])
    horizon = math.inf if horizon_text == "none" else float(horizon_text)
    pretest_indexes = indexes_before(arrays, test_start)
    profile_name = str(selected.get("gate_profile", "light"))
    profile_fit_indexes = apply_gate_profile(
        pretest_indexes,
        arrays,
        profile_name,
        args.scenario,
        include_bucket_quality=False,
    )
    sizing_model = fit_sizing_model(
        profile_fit_indexes,
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
        indexes_between(arrays, test_start, test_end),
        arrays,
        profile_name,
        args.scenario,
        sizing_model=sizing_model,
    )
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
        sizing_policy=str(selected["sizing_policy"]),
        sizing_model=sizing_model,
        max_stake=args.max_stake,
        kelly_fraction=args.kelly_fraction,
        max_fraction=args.max_fraction,
        reserve_fraction=args.reserve_fraction,
        min_stake=args.min_stake,
        min_minutes_to_close=args.min_minutes_to_close,
        max_category_locked_fraction=args.max_category_locked_fraction,
        max_regime_locked_fraction=args.max_regime_locked_fraction,
        drawdown_throttle_start=args.drawdown_throttle_start,
        drawdown_throttle_stop=args.drawdown_throttle_stop,
        drawdown_throttle_min_scale=args.drawdown_throttle_min_scale,
    )
    summary = result["summary"]
    return {
        "period_id": period["period_id"],
        "test_months": period["test_months"],
        "validation_months": period["validation_months"],
        "validation_start": period["validation_start"].date(),
        "test_start": period["test_start"].date(),
        "test_end": period["test_end"].date(),
        "selected_gate": selected["gate"],
        "selected_gate_profile": profile_name,
        "selected_gate_profile_description": selected.get("gate_profile_description", profile_description(profile_name)),
        "selected_categories": selected["categories"],
        "selected_max_scheduled_horizon_days": selected["max_scheduled_horizon_days"],
        "selected_sizing_policy": selected["sizing_policy"],
        "selected_strategy": f"{selected['gate']}:{profile_name}:{selected['sizing_policy']}",
        "validation_selection_score": selected["selection_score"],
        "validation_realized_profit": selected["realized_profit"],
        "validation_without_top_1": selected["without_top_1"],
        "validation_entries": selected["entries"],
        **summary,
        "skipped": parse_skipped(summary),
    }


def summarize_oos(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    summaries = []
    groupings: list[tuple[str, list[str]]] = [("all", ["test_months"])]
    if "selected_strategy" in df.columns and df["selected_strategy"].nunique() > 1:
        groupings.append(("selected_strategy", ["test_months", "selected_strategy"]))
    for scope, columns in groupings:
        grouped = df.groupby(columns, dropna=False)
        for key, group in grouped:
            if not isinstance(key, tuple):
                key = (key,)
            key_values = dict(zip(columns, key))
            test_months = int(key_values["test_months"])
            returns = group["account_return"].astype(float).to_numpy()
            profits = group["realized_profit"].astype(float).to_numpy()
            count = len(group)
            mean_return = float(np.mean(returns))
            se_return = float(np.std(returns, ddof=1) / math.sqrt(count)) if count > 1 else 0.0
            compound = float(np.prod(1.0 + returns) - 1.0)
            years = max(1e-9, count * float(test_months) / 12.0)
            row = {
                "summary_scope": scope,
                "test_months": test_months,
                "periods": count,
                "mean_profit": float(np.mean(profits)),
                "median_profit": float(np.median(profits)),
                "mean_account_return": mean_return,
                "median_account_return": float(np.median(returns)),
                "return_standard_error": se_return,
                "return_ci95_low": mean_return - 1.96 * se_return,
                "return_ci95_high": mean_return + 1.96 * se_return,
                "positive_rate": float(np.mean(profits > 0)),
                "mean_without_top1": float(group["without_top_1"].astype(float).mean()),
                "worst_period_profit": float(np.min(profits)),
                "best_period_profit": float(np.max(profits)),
                "mean_entries": float(group["entries"].astype(float).mean()),
                "mean_deployed": float(group["deployed"].astype(float).mean()),
                "mean_max_drawdown": float(group["max_drawdown"].astype(float).mean()),
                "worst_max_drawdown": float(group["max_drawdown"].astype(float).max()),
                "compound_return": compound,
                "annualized_compound_return": float((1.0 + compound) ** (1.0 / years) - 1.0),
            }
            if "selected_strategy" in key_values:
                row["selected_strategy"] = key_values["selected_strategy"]
            summaries.append(row)
    return summaries


def write_visuals(rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        return
    visuals = output_dir / "visuals"
    visuals.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["test_months", "test_start"])

    for test_months, group in df.groupby("test_months"):
        labels = group["test_start"].astype(str).tolist()
        x = np.arange(len(group))
        fig, axis = plt.subplots(figsize=(max(9, len(group) * 1.8), 5.5))
        axis.bar(x, group["account_return"] * 100.0)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=35, ha="right")
        axis.set_ylabel("account return (%)")
        axis.set_title(f"Walk-Forward OOS Returns ({int(test_months)}m windows)")
        fig.tight_layout()
        fig.savefig(visuals / f"oos_period_returns_{int(test_months)}m.png", dpi=160)
        plt.close(fig)

    horizons = sorted(df["test_months"].unique())
    data = [df.loc[df["test_months"] == months, "account_return"].to_numpy() * 100.0 for months in horizons]
    fig, axis = plt.subplots(figsize=(9, 6))
    axis.boxplot(data, labels=[f"{int(months)}m" for months in horizons], showmeans=True)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("account return (%)")
    axis.set_title("Clean OOS Return Distribution")
    fig.tight_layout()
    fig.savefig(visuals / "oos_return_distribution.png", dpi=160)
    plt.close(fig)

    df["top1_contribution"] = df["realized_profit"] - df["without_top_1"]
    labels = df["test_months"].astype(str) + "m " + df["test_start"].astype(str)
    x = np.arange(len(df))
    fig, axis = plt.subplots(figsize=(max(11, len(df) * 1.25), 6))
    width = 0.35
    axis.bar(x - width / 2, df["realized_profit"], width, label="realized")
    axis.bar(x + width / 2, df["without_top_1"], width, label="without top 1")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=50, ha="right", fontsize=8)
    axis.set_ylabel("profit ($)")
    axis.set_title("OOS Profit vs. Same Period With Largest Winner Removed")
    axis.legend()
    fig.tight_layout()
    fig.savefig(visuals / "oos_tail_dependence.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/underdog_optimization_kalshi"))
    parser.add_argument("--data-dir", type=Path, default=Path("archive/processed/underdog_events"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/walk_forward_oos"))
    parser.add_argument("--test-months", type=int, nargs="+", default=[6, 12])
    parser.add_argument("--min-train-months", type=int, default=12)
    parser.add_argument("--validation-months", type=int, default=6)
    parser.add_argument("--stride-months", type=int, default=None)
    parser.add_argument("--include-partial-final", action="store_true")
    parser.add_argument(
        "--selection-score",
        choices=["robust", "raw_profit", "without_top1", "drawdown_heavy", "min_trades"],
        default="robust",
    )
    parser.add_argument(
        "--fixed-strategies",
        nargs="*",
        type=parse_fixed_strategy,
        default=[],
        help="Evaluate fixed gate:profile:policy combos instead of selecting the best validation score.",
    )
    parser.add_argument(
        "--sizing-policies",
        nargs="+",
        choices=SIZING_POLICIES,
        default=["flat_one", "availability", "hybrid_floor_lcb", "forecast_paced"],
    )
    parser.add_argument(
        "--gate-profiles",
        nargs="+",
        choices=sorted(GATE_PROFILES),
        default=[
            "light",
            "liquid_entry_25",
            "liquid_entry_exit_25",
            "low_price_1_5c",
            "low_mid_price_1_15c",
            "exclude_high_price",
            "bucket_count_25",
            "bucket_lcb_nonnegative",
            "liquid_low_mid",
            "quality_liquid",
        ],
    )
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
    parser.add_argument("--drawdown-throttle-start", type=float, default=math.inf)
    parser.add_argument("--drawdown-throttle-stop", type=float, default=math.inf)
    parser.add_argument("--drawdown-throttle-min-scale", type=float, default=0.0)
    parser.add_argument("--shrink-k", type=float, default=100.0)
    parser.add_argument("--lcb-z", type=float, default=1.0)
    args = parser.parse_args()

    arrays, _ = load_arrays(args.report_dir, args.data_dir)
    execution = execution_model(args.report_dir)
    print("attaching same-side exit paths...", flush=True)
    attach_exit_paths(arrays, args.data_dir, execution)
    attach_exit_liquidity_totals(arrays)
    print("exit paths attached", flush=True)

    validation_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    oos_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    periods_by_horizon = []

    for months in sorted(set(args.test_months)):
        periods = period_rows(
            arrays,
            months,
            args.min_train_months,
            args.validation_months,
            args.stride_months,
            args.include_partial_final,
        )
        periods_by_horizon.extend(periods)
        for period in periods:
            print(
                "evaluating "
                f"{period['period_id']} "
                f"validation={period['validation_start'].date()}->{period['test_start'].date()} "
                f"test={period['test_start'].date()}->{period['test_end'].date()}",
                flush=True,
            )
            candidates, gates = validation_candidates(
                period, arrays, execution, list(args.sizing_policies), list(args.gate_profiles), args
            )
            gate_rows.extend(gates)
            validation_rows.extend(candidates)
            if not candidates:
                selected_rows.append({
                    "period_id": period["period_id"],
                    "test_months": period["test_months"],
                    "test_start": period["test_start"].date(),
                    "test_end": period["test_end"].date(),
                    "selection_status": "no_validation_candidates",
                })
                continue
            if args.fixed_strategies:
                by_key = {fixed_strategy_key(row): row for row in candidates}
                selected_candidates = []
                for fixed in args.fixed_strategies:
                    selected = by_key.get(fixed)
                    if selected is None:
                        selected_rows.append({
                            "period_id": period["period_id"],
                            "test_months": period["test_months"],
                            "test_start": period["test_start"].date(),
                            "test_end": period["test_end"].date(),
                            "selection_status": "fixed_strategy_missing",
                            "requested_fixed_strategy": ":".join(fixed),
                        })
                    else:
                        selected_candidates.append(selected)
            else:
                selected_candidates = [max(candidates, key=lambda row: float(row["selection_score"]))]

            for selected in selected_candidates:
                selected_row = {
                    "period_id": period["period_id"],
                    "test_months": period["test_months"],
                    "test_start": period["test_start"].date(),
                    "test_end": period["test_end"].date(),
                    "selection_status": "fixed" if args.fixed_strategies else "selected",
                    "selection_score_mode": args.selection_score,
                    "selected_gate": selected["gate"],
                    "selected_gate_profile": selected["gate_profile"],
                    "selected_gate_profile_description": selected["gate_profile_description"],
                    "selected_categories": selected["categories"],
                    "selected_max_scheduled_horizon_days": selected["max_scheduled_horizon_days"],
                    "selected_sizing_policy": selected["sizing_policy"],
                    "validation_selection_score": selected["selection_score"],
                    "validation_realized_profit": selected["realized_profit"],
                    "validation_without_top_1": selected["without_top_1"],
                    "validation_entries": selected["entries"],
                }
                selected_rows.append(selected_row)
                oos = replay_selected_oos(period, selected, arrays, execution, args)
                if oos is not None:
                    oos["selection_status"] = selected_row["selection_status"]
                    oos["selection_score_mode"] = args.selection_score
                    oos_rows.append(oos)

    summary_rows = summarize_oos(oos_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "oos_period_results.csv", oos_rows)
    write_csv(args.output_dir / "validation_leaderboard.csv", validation_rows)
    write_csv(args.output_dir / "selected_strategies.csv", selected_rows)
    write_csv(args.output_dir / "gate_selection.csv", gate_rows)
    write_csv(args.output_dir / "oos_summary.csv", summary_rows)
    write_visuals(oos_rows, args.output_dir)

    summary = {
        "periods": len(oos_rows),
        "test_months": sorted(set(args.test_months)),
        "min_train_months": args.min_train_months,
        "validation_months": args.validation_months,
        "sizing_policies": list(args.sizing_policies),
        "gate_profiles": list(args.gate_profiles),
        "selection_score": args.selection_score,
        "fixed_strategies": [":".join(strategy) for strategy in args.fixed_strategies],
        "files": [
            "oos_period_results.csv",
            "oos_summary.csv",
            "selected_strategies.csv",
            "validation_leaderboard.csv",
            "gate_selection.csv",
            "visuals/",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
