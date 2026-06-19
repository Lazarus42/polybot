#!/usr/bin/env python3
"""Causal, weekly walk-forward allocation tests for the underdog strategy.

The strategy cube contains the realized return for every entry and bracket. This
script splits each training period chronologically: the early fit segment selects
the bracket by entry-price level, and the later calibration segment estimates
weights, opportunity rates, thresholds, and sizing. Test weeks are then replayed
in timestamp order without exposing future arrivals to the allocator.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
from typing import Any, Optional, Union

import duckdb
import matplotlib.pyplot as plt
import numpy as np

from optimize_underdog_bracket import max_contracts_for_budget, posted_balance_change


WEEK_SECONDS = 7 * 24 * 60 * 60
MONDAY_EPOCH = datetime(1970, 1, 5, tzinfo=timezone.utc).timestamp()
CATEGORIES = ("sports", "crypto", "politics", "economics", "weather", "entertainment", "other")
WEIGHT_METHODS = ("equal", "raw_roi", "shrunk_roi", "lcb_roi", "profit_volume", "roi_availability")
POLICIES = (
    "equal_one",
    "observed_normalized",
    "normalized_cash",
    "two_stage_cash",
    "two_stage_redistribute",
    "availability_fixed",
    "rank_greedy",
    "fractional_edge",
)


def week_id(timestamp: Union[np.ndarray, float]) -> Union[np.ndarray, int]:
    values = np.floor((np.asarray(timestamp) - MONDAY_EPOCH) / WEEK_SECONDS).astype(np.int64)
    return int(values) if values.ndim == 0 else values


def week_start(value: int) -> datetime:
    return datetime.fromtimestamp(MONDAY_EPOCH + value * WEEK_SECONDS, timezone.utc)


def classify_market(text: str) -> str:
    value = text.lower()
    patterns = {
        "sports": r"\b(nfl|nba|mlb|nhl|ncaa|soccer|football|basketball|baseball|hockey|tennis|ufc|fight|game|match|score|win the|champion|super bowl|world cup)\b",
        "crypto": r"\b(bitcoin|btc|ethereum|eth|crypto|solana|dogecoin|token|blockchain)\b",
        "politics": r"\b(election|president|senate|congress|governor|democrat|republican|trump|biden|primary|vote|cabinet|parliament)\b",
        "economics": r"\b(fed|inflation|cpi|gdp|unemployment|interest rate|recession|stock|nasdaq|s&p|dow|price of|market cap)\b",
        "weather": r"\b(weather|temperature|rain|snow|hurricane|storm|tornado|degrees|precipitation)\b",
        "entertainment": r"\b(oscar|emmy|grammy|movie|film|album|song|box office|celebrity|award|tv show)\b",
    }
    for category, pattern in patterns.items():
        if re.search(pattern, value):
            return category
    return "other"


def load_categories(data_dir: Path, market_ids: np.ndarray) -> np.ndarray:
    path = str((data_dir / "markets.parquet").resolve()).replace("'", "''")
    rows = duckdb.connect().execute(
        f"SELECT market_id, coalesce(slug, ''), coalesce(question, '') FROM read_parquet('{path}')"
    ).fetchall()
    mapping = {int(market_id): classify_market(f"{slug} {question}") for market_id, slug, question in rows}
    return np.asarray([mapping.get(int(market_id), "other") for market_id in market_ids])


def execution_model(report_dir: Path) -> dict[str, float]:
    summary = json.loads((report_dir / "holdout_portfolio_summary.json").read_text())
    model = summary["execution_model"]
    return {key: float(model[key]) for key in ("fee_coefficient", "price_tick", "contract_step")}


def infer_exit_price(entry_price: float, strategy_return: float, fee: float, tick: float) -> float:
    entry_cost = entry_price + fee * entry_price * (1.0 - entry_price)
    net_exit = max(0.0, min(1.0, (1.0 + strategy_return) * entry_cost))
    if net_exit <= 1e-7:
        return 0.0
    if net_exit >= 1.0 - 1e-6:
        return 1.0
    if fee == 0:
        price = net_exit
    else:
        price = (-(1.0 - fee) + math.sqrt((1.0 - fee) ** 2 + 4.0 * fee * net_exit)) / (2.0 * fee)
    return max(tick, min(1.0 - tick, round(price / tick) * tick)) if tick else price


def execute(stake: float, entry_price: float, strategy_return: float, model: dict[str, float]) -> tuple[float, float, bool]:
    contracts, debit = max_contracts_for_budget(
        stake, entry_price, model["fee_coefficient"], model["contract_step"]
    )
    if contracts <= 0 or debit <= 0:
        return 0.0, 0.0, False
    exit_price = infer_exit_price(
        entry_price, strategy_return, model["fee_coefficient"], model["price_tick"]
    )
    if exit_price in (0.0, 1.0):
        proceeds = (contracts * Decimal(str(exit_price)) * 100).to_integral_value(
            rounding=ROUND_FLOOR
        ) / 100
    else:
        proceeds = posted_balance_change(
            contracts,
            Decimal(str(exit_price)),
            Decimal(str(model["fee_coefficient"])),
            "sell",
        )
    return float(debit), float(proceeds - debit), True


def stats(values: np.ndarray, global_mean: float, shrink_k: float, lcb_z: float) -> dict[str, float]:
    count = len(values)
    if not count:
        return {"n": 0, "mean": 0.0, "shrunk": global_mean, "se": float("inf"), "lcb": 0.0, "second": 0.0}
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(count)) if count > 1 else float("inf")
    alpha = count / (count + shrink_k)
    shrunk = alpha * mean + (1.0 - alpha) * global_mean
    return {
        "n": count,
        "mean": mean,
        "shrunk": shrunk,
        "se": se,
        "lcb": shrunk - lcb_z * se if math.isfinite(se) else 0.0,
        "second": float(np.mean(values * values)),
    }


def fit_model(
    levels: np.ndarray,
    categories: np.ndarray,
    weeks: np.ndarray,
    cube: np.ndarray,
    fit_mask: np.ndarray,
    calibration_mask: np.ndarray,
    min_fit_trades: int,
    shrink_k: float,
    lcb_z: float,
) -> dict[str, Any]:
    selectors: dict[int, tuple[int, int]] = {}
    for level in range(1, 50):
        indexes = np.where(fit_mask & (levels == level))[0]
        if len(indexes) >= min_fit_trades:
            flat = int(np.argmax(cube[indexes].mean(axis=0)))
            selectors[level] = tuple(int(value) for value in np.unravel_index(flat, cube.shape[1:]))

    selected = np.full(len(levels), np.nan, dtype=float)
    for level, (tp_index, sl_index) in selectors.items():
        indexes = np.where(calibration_mask & (levels == level))[0]
        selected[indexes] = cube[indexes, tp_index, sl_index]
    valid = calibration_mask & np.isfinite(selected)
    global_mean = float(selected[valid].mean()) if valid.any() else 0.0
    level_stats = {
        level: stats(selected[valid & (levels == level)], global_mean, shrink_k, lcb_z)
        for level in selectors
    }
    calibration_week_values = weeks[calibration_mask]
    calibration_weeks = max(
        1,
        int(calibration_week_values.max() - calibration_week_values.min() + 1)
        if len(calibration_week_values) else 1,
    )
    category_stats = {}
    for category in CATEGORIES:
        values = selected[valid & (categories == category)]
        category_stats[category] = stats(values, global_mean, shrink_k, lcb_z)
        category_stats[category]["lambda"] = float(np.sum(valid & (categories == category)) / calibration_weeks)

    proxy = np.zeros(len(levels), dtype=float)
    for level, values in level_stats.items():
        proxy[levels == level] = max(0.0, values["shrunk"])
    positive_proxy = proxy[valid & (proxy > 0)]
    tau = float(np.quantile(positive_proxy, 0.25)) if len(positive_proxy) else 0.0
    return {
        "selectors": selectors,
        "level_stats": level_stats,
        "category_stats": category_stats,
        "global_mean": global_mean,
        "tau": tau,
        "calibration_weeks": calibration_weeks,
    }


def category_weights(method: str, model: dict[str, Any]) -> dict[str, float]:
    raw = {}
    for category, value in model["category_stats"].items():
        if method == "equal":
            score = 1.0 if value["n"] else 0.0
        elif method == "raw_roi":
            score = max(0.0, value["mean"])
        elif method == "shrunk_roi":
            score = max(0.0, value["shrunk"])
        elif method == "lcb_roi":
            score = max(0.0, value["lcb"])
        elif method == "profit_volume":
            score = max(0.0, value["mean"] * value["n"])
        elif method == "roi_availability":
            score = max(0.0, value["shrunk"]) * value["lambda"]
        else:
            raise ValueError(method)
        raw[category] = score
    total = sum(raw.values())
    return {category: value / total if total else 0.0 for category, value in raw.items()}


def selected_return(index: int, level: int, cube: np.ndarray, model: dict[str, Any]) -> Optional[float]:
    selector = model["selectors"].get(level)
    if selector is None:
        return None
    return float(cube[index, selector[0], selector[1]])


def future_quality(
    current_category: str,
    elapsed: float,
    observed: Counter,
    weights: dict[str, float],
    model: dict[str, Any],
    category_only: bool = False,
) -> float:
    result = 0.0
    for category, category_stat in model["category_stats"].items():
        if category_only and category != current_category:
            continue
        expected_by_clock = category_stat["lambda"] * max(0.0, 1.0 - elapsed)
        expected_by_count = max(0.0, category_stat["lambda"] - observed[category])
        expected = min(expected_by_clock, expected_by_count)
        mean_proxy = 0.0
        level_values = [max(0.0, value["shrunk"]) for value in model["level_stats"].values()]
        if level_values:
            mean_proxy = float(np.mean(level_values))
        result += expected * weights.get(category, 0.0) * mean_proxy
    return result


def replay_week(
    indexes: np.ndarray,
    policy: str,
    weight_method: str,
    budget: float,
    one_dollar_stake: float,
    levels: np.ndarray,
    categories: np.ndarray,
    times: np.ndarray,
    prices: np.ndarray,
    entry_fill_usd: np.ndarray,
    cube: np.ndarray,
    model: dict[str, Any],
    execution: dict[str, float],
    max_stake: float,
    kelly_fraction: float,
    participation: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    weights = category_weights(weight_method, model)
    remaining = budget
    category_remaining = {category: budget * weights.get(category, 0.0) for category in CATEGORIES}
    observed: Counter = Counter()
    week_value = week_id(float(times[indexes[0]]))
    start_time = MONDAY_EPOCH + week_value * WEEK_SECONDS
    records = []
    cumulative = peak = max_drawdown = 0.0
    expected_ranked = sum(value["lambda"] for value in model["category_stats"].values()) * 0.25
    fixed_rank_stake = budget / max(1.0, expected_ranked)

    for index in indexes:
        if remaining <= 0.005:
            break
        level = int(levels[index])
        category = str(categories[index])
        outcome = selected_return(int(index), level, cube, model)
        level_stat = model["level_stats"].get(level)
        if outcome is None or level_stat is None:
            continue
        elapsed = min(1.0, max(0.0, (float(times[index]) - start_time) / WEEK_SECONDS))
        observed[category] += 1
        proxy = max(0.0, level_stat["shrunk"])
        edge = max(0.0, proxy - model["tau"])
        quality = weights.get(category, 0.0) * proxy
        future = future_quality(category, elapsed, observed, weights, model)
        stake = 0.0

        if policy == "equal_one":
            stake = one_dollar_stake
        elif policy == "observed_normalized":
            stake = remaining * quality / (quality + future) if quality + future > 0 else 0.0
        elif policy == "normalized_cash":
            cash_quality = max(model["tau"], model["global_mean"], 0.01)
            stake = remaining * weights.get(category, 0.0) * edge / (
                weights.get(category, 0.0) * edge + future + cash_quality
            ) if edge > 0 else 0.0
        elif policy in {"two_stage_cash", "two_stage_redistribute"}:
            category_future = future_quality(category, elapsed, observed, weights, model, True)
            bucket = category_remaining[category]
            stake = bucket * quality / (quality + category_future) if quality + category_future > 0 else 0.0
            if policy == "two_stage_redistribute" and observed[category] > model["category_stats"][category]["lambda"]:
                # Causal redistribution: only excess arrivals may draw from uncommitted cash.
                stake = max(stake, min(remaining, budget * weights.get(category, 0.0) / max(1.0, model["category_stats"][category]["lambda"])))
        elif policy == "availability_fixed":
            lam = model["category_stats"][category]["lambda"]
            base = budget * weights.get(category, 0.0) / max(1.0, lam)
            category_mean = max(0.01, model["category_stats"][category]["shrunk"])
            stake = base * min(2.0, proxy / category_mean)
        elif policy == "rank_greedy":
            if proxy >= model["tau"] and proxy > 0:
                stake = fixed_rank_stake
        elif policy == "fractional_edge":
            second = max(level_stat["second"], 1e-6)
            conservative_edge = max(0.0, level_stat["lcb"])
            fraction = min(0.05, kelly_fraction * conservative_edge / second)
            stake = budget * fraction
        else:
            raise ValueError(policy)

        stake = min(stake, remaining, max_stake, float(entry_fill_usd[index]) * participation)
        if policy.startswith("two_stage"):
            stake = min(stake, category_remaining[category])
        debit, profit, executed = execute(stake, float(prices[index]), outcome, execution)
        if not executed:
            continue
        remaining -= debit
        if policy.startswith("two_stage"):
            category_remaining[category] -= debit
        cumulative += profit
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
        records.append({
            "market_id": int(index),
            "entry_time": datetime.fromtimestamp(int(times[index]), timezone.utc).isoformat(),
            "entry_level": level,
            "category": category,
            "proxy": proxy,
            "stake_budget": stake,
            "debit": debit,
            "profit": profit,
            "profitable": profit > 0,
        })

    deployed = sum(row["debit"] for row in records)
    profit = sum(row["profit"] for row in records)
    return {
        "week": week_start(week_value).date().isoformat(),
        "opportunities": len(indexes),
        "bets": len(records),
        "deployed": deployed,
        "profit": profit,
        "roi_on_budget": profit / budget,
        "roi_on_deployed": profit / deployed if deployed else 0.0,
        "budget_utilization": deployed / budget,
        "hit_rate": sum(row["profitable"] for row in records) / len(records) if records else 0.0,
        "within_week_realized_sequence_drawdown": max_drawdown,
    }, records


def maximum_drawdown(profits: list[float], budget: float) -> float:
    equity = peak = budget
    result = 0.0
    for profit in profits:
        equity += profit
        peak = max(peak, equity)
        result = max(result, (peak - equity) / peak if peak else 0.0)
    return result


def summarize(rows: list[dict[str, Any]], budget: float) -> dict[str, Any]:
    profits = [float(row["profit"]) for row in rows]
    deployed = sum(float(row["deployed"]) for row in rows)
    total_budget = budget * len(rows)
    mean_profit = float(np.mean(profits)) if profits else 0.0
    std_profit = float(np.std(profits, ddof=1)) if len(profits) > 1 else 0.0
    return {
        "weeks": len(rows),
        "total_profit": sum(profits),
        "roi_on_total_weekly_budget": sum(profits) / total_budget if total_budget else 0.0,
        "roi_on_deployed": sum(profits) / deployed if deployed else 0.0,
        "budget_utilization": deployed / total_budget if total_budget else 0.0,
        "positive_week_fraction": sum(value > 0 for value in profits) / len(profits) if profits else 0.0,
        "mean_weekly_profit": mean_profit,
        "median_weekly_profit": float(np.median(profits)) if profits else 0.0,
        "weekly_sharpe_like": mean_profit / std_profit if std_profit else 0.0,
        "max_weekly_equity_drawdown": maximum_drawdown(profits, budget),
        "bets": sum(int(row["bets"]) for row in rows),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows: list[dict[str, Any]], output: Path) -> None:
    selected = [row for row in rows if row["weight_method"] in {"equal", "lcb_roi"}]
    labels = sorted({f"{row['policy']}\n{row['weight_method']}" for row in selected})
    cuts = sorted({row["cut_fraction"] for row in selected})
    lookup = {(row["cut_fraction"], f"{row['policy']}\n{row['weight_method']}"): row for row in selected}
    figure, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    x = np.arange(len(labels))
    width = 0.8 / len(cuts)
    for offset, cut in enumerate(cuts):
        values = [lookup.get((cut, label), {}).get("roi_on_total_weekly_budget", 0) * 100 for label in labels]
        utilization = [lookup.get((cut, label), {}).get("budget_utilization", 0) * 100 for label in labels]
        axes[0].bar(x + (offset - (len(cuts) - 1) / 2) * width, values, width, label=f"cut {cut:.0%}")
        axes[1].bar(x + (offset - (len(cuts) - 1) / 2) * width, utilization, width)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("ROI on weekly budget (%)")
    axes[0].legend()
    axes[1].set_ylabel("Budget utilization (%)")
    axes[1].set_xticks(x, labels, rotation=75, ha="right")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/underdog_optimization_kalshi"))
    parser.add_argument("--data-dir", type=Path, default=Path("archive/processed/underdog_events"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/online_underdog_allocation"))
    parser.add_argument("--cut-fractions", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8])
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--weight-methods", nargs="+", choices=WEIGHT_METHODS, default=list(WEIGHT_METHODS))
    parser.add_argument("--weekly-budget", type=float, default=5000.0)
    parser.add_argument("--one-dollar-stake", type=float, default=1.0)
    parser.add_argument("--inner-fit-fraction", type=float, default=0.7)
    parser.add_argument("--min-fit-trades", type=int, default=30)
    parser.add_argument("--shrink-k", type=float, default=100.0)
    parser.add_argument("--lcb-z", type=float, default=1.0)
    parser.add_argument("--max-stake", type=float, default=250.0)
    parser.add_argument("--kelly-fraction", type=float, default=0.1)
    parser.add_argument(
        "--participation", type=float, default=0.10,
        help="maximum fraction of the triggering archived fill used at entry",
    )
    parser.add_argument("--write-trades", action="store_true")
    args = parser.parse_args()
    if any(not 0 < value < 1 for value in args.cut_fractions):
        parser.error("cut fractions must be between zero and one")
    if not 0 < args.inner_fit_fraction < 1:
        parser.error("inner fit fraction must be between zero and one")

    data = np.load(args.report_dir / "strategy_cube.npz")
    order = np.argsort(data["entry_times"], kind="stable")
    market_ids = data["market_ids"][order]
    times = data["entry_times"][order]
    prices = data["entry_prices"][order]
    if "entry_fill_usd" not in data.files:
        raise SystemExit("strategy_cube.npz lacks entry_fill_usd; rerun optimize_underdog_bracket.py")
    entry_fill_usd = data["entry_fill_usd"][order]
    levels = data["entry_levels"][order]
    cube = data["returns"][order]
    categories = load_categories(args.data_dir, market_ids)
    weeks = week_id(times)
    active_weeks = np.unique(weeks)
    all_weeks = np.arange(int(active_weeks.min()), int(active_weeks.max()) + 1, dtype=np.int64)
    execution = execution_model(args.report_dir)
    weekly_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []

    for cut_fraction in sorted(set(args.cut_fractions)):
        cut_position = min(len(all_weeks) - 1, max(2, int(len(all_weeks) * cut_fraction)))
        cut_week = int(all_weeks[cut_position])
        training_weeks = all_weeks[:cut_position]
        inner_position = min(len(training_weeks) - 1, max(1, int(len(training_weeks) * args.inner_fit_fraction)))
        calibration_start = int(training_weeks[inner_position])
        fit_mask = weeks < calibration_start
        calibration_mask = (weeks >= calibration_start) & (weeks < cut_week)
        model = fit_model(
            levels, categories, weeks, cube, fit_mask, calibration_mask,
            args.min_fit_trades, args.shrink_k, args.lcb_z,
        )
        training_rows.append({
            "cut_fraction": cut_fraction,
            "fit_start": week_start(int(all_weeks[0])).date().isoformat(),
            "calibration_start": week_start(calibration_start).date().isoformat(),
            "holdout_start": week_start(cut_week).date().isoformat(),
            "holdout_end": week_start(int(all_weeks[-1]) + 1).date().isoformat(),
            "fit_weeks": inner_position,
            "calibration_weeks": model["calibration_weeks"],
            "holdout_weeks": int(np.sum(all_weeks >= cut_week)),
            "eligible_entry_levels": len(model["selectors"]),
            "calibration_global_roi": model["global_mean"],
            "proxy_threshold": model["tau"],
        })
        for policy in args.policies:
            methods = ["equal"] if policy == "equal_one" else args.weight_methods
            for method in methods:
                group = []
                category_profit: dict[str, float] = defaultdict(float)
                category_deployed: dict[str, float] = defaultdict(float)
                category_bets: Counter = Counter()
                for test_week in all_weeks[all_weeks >= cut_week]:
                    indexes = np.where(weeks == test_week)[0]
                    if len(indexes):
                        result, records = replay_week(
                            indexes, policy, method, args.weekly_budget, args.one_dollar_stake,
                            levels, categories, times, prices, entry_fill_usd, cube, model, execution,
                            args.max_stake, args.kelly_fraction, args.participation,
                        )
                    else:
                        result, records = ({
                            "week": week_start(int(test_week)).date().isoformat(),
                            "opportunities": 0, "bets": 0, "deployed": 0.0,
                            "profit": 0.0, "roi_on_budget": 0.0,
                            "roi_on_deployed": 0.0, "budget_utilization": 0.0,
                            "hit_rate": 0.0, "within_week_realized_sequence_drawdown": 0.0,
                        }, [])
                    row = {"cut_fraction": cut_fraction, "policy": policy, "weight_method": method, **result}
                    weekly_rows.append(row)
                    group.append(row)
                    for record in records:
                        category = record["category"]
                        category_profit[category] += record["profit"]
                        category_deployed[category] += record["debit"]
                        category_bets[category] += 1
                        if args.write_trades:
                            trade_rows.append({
                                "cut_fraction": cut_fraction, "policy": policy,
                                "weight_method": method, "week": result["week"],
                                "market_id": int(market_ids[int(record["market_id"])]),
                                **{key: value for key, value in record.items() if key != "market_id"},
                            })
                aggregate_rows.append({
                    "cut_fraction": cut_fraction,
                    "holdout_start": week_start(cut_week).date().isoformat(),
                    "policy": policy,
                    "weight_method": method,
                    **summarize(group, args.weekly_budget),
                })
                for category in CATEGORIES:
                    category_rows.append({
                        "cut_fraction": cut_fraction, "policy": policy,
                        "weight_method": method, "category": category,
                        "bets": category_bets[category],
                        "deployed": category_deployed[category],
                        "profit": category_profit[category],
                        "roi_on_deployed": category_profit[category] / category_deployed[category]
                        if category_deployed[category] else 0.0,
                    })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "training_cuts.csv", training_rows)
    write_csv(args.output_dir / "weekly_results.csv", weekly_rows)
    write_csv(args.output_dir / "aggregate_results.csv", aggregate_rows)
    write_csv(args.output_dir / "category_results.csv", category_rows)
    if args.write_trades:
        write_csv(args.output_dir / "trade_results.csv", trade_rows)
    plot_results(aggregate_rows, args.output_dir / "allocation_comparison.png")
    summary = {
        "causality": {
            "bracket_selection": "early chronological training segment only",
            "weight_calibration": "later chronological training segment only",
            "allocation": "timestamp-order replay using training arrival rates, clock time, and remaining weekly budget",
            "future_holdout_opportunities_visible": False,
            "outcomes_used_only_after_stake": True,
        },
        "execution_model": execution,
        "weekly_budget": args.weekly_budget,
        "one_dollar_stake": args.one_dollar_stake,
        "entry_fill_participation": args.participation,
        "market_type": "entry-time-known semantic category derived from slug and question",
        "cuts": training_rows,
        "best_by_total_profit": sorted(aggregate_rows, key=lambda row: row["total_profit"], reverse=True)[:20],
        "files": ["training_cuts.csv", "weekly_results.csv", "aggregate_results.csv", "category_results.csv", "allocation_comparison.png"],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
