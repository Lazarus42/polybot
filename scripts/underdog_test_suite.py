#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
import heapq
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from optimize_underdog_bracket import max_contracts_for_budget, posted_balance_change
from decimal import Decimal, ROUND_FLOOR


DEFAULT_REPORT = Path("reports/underdog_optimization_kalshi")


def parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes"}


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for key in (
                "entry_level_cents",
                "market_id",
            ):
                row[key] = int(row[key])
            for key in (
                "entry_price",
                "exit_price",
                "entry_fill_usd",
                "historical_volume",
                "take_profit",
                "stop_loss",
                "training_roi",
            ):
                value = row.get(key, "")
                row[key] = float(value) if value not in ("", None) else None
            row["entry_time"] = datetime.fromisoformat(row["entry_time"]).replace(tzinfo=timezone.utc)
            exit_time = row.get("exit_time")
            row["exit_time"] = (
                datetime.fromisoformat(exit_time).replace(tzinfo=timezone.utc)
                if exit_time
                else row["entry_time"]
            )
            row["underdog_won"] = parse_bool(row.get("underdog_won", "false"))
            rows.append(row)
    return rows


def load_execution(report_dir: Path) -> dict:
    summary = json.loads((report_dir / "holdout_portfolio_summary.json").read_text())
    model = summary["execution_model"]
    return {
        "fee_coefficient": float(model["fee_coefficient"]),
        "price_tick": float(model["price_tick"]),
        "contract_step": float(model["contract_step"]),
    }


def trade_result(
    row: dict,
    budget: float,
    execution: dict,
    entry_ticks: int = 0,
    exit_ticks: int = 0,
    participation: Optional[float] = None,
    hold_to_resolution: bool = False,
) -> dict:
    if participation is not None and row.get("entry_fill_usd") is not None:
        budget = min(budget, max(0.0, row["entry_fill_usd"] * participation))
    tick = execution["price_tick"]
    entry_price = min(1.0 - tick, row["entry_price"] + entry_ticks * tick)
    if hold_to_resolution:
        exit_price = 1.0 if row["underdog_won"] else 0.0
        exit_type = "resolution_win" if row["underdog_won"] else "resolution_loss"
    else:
        exit_type = row["exit_type"]
        exit_price = row["exit_price"]
        if exit_type in {"take_profit", "stop_loss"}:
            exit_price = max(tick, exit_price - exit_ticks * tick)
    contracts, debit = max_contracts_for_budget(
        budget,
        entry_price,
        execution["fee_coefficient"],
        execution["contract_step"],
    )
    if contracts <= 0 or debit <= 0:
        return {"executed": False, "budget": budget, "debit": 0.0, "pnl": 0.0, "return": 0.0}
    if exit_type.startswith("resolution"):
        proceeds = (contracts * Decimal(str(exit_price)) * 100).to_integral_value(
            rounding=ROUND_FLOOR
        ) / 100
    else:
        proceeds = posted_balance_change(
            contracts,
            Decimal(str(exit_price)),
            Decimal(str(execution["fee_coefficient"])),
            "sell",
        )
    pnl = float(proceeds - debit)
    return {
        "executed": True,
        "budget": budget,
        "debit": float(debit),
        "pnl": pnl,
        "return": pnl / float(debit),
        "contracts": float(contracts),
    }


def allocate(
    rows: list[dict],
    capital: float,
    execution: dict,
    entry_ticks: int = 0,
    exit_ticks: int = 0,
    participation: Optional[float] = None,
    hold_to_resolution: bool = False,
) -> dict:
    if not rows:
        return {"opportunities": 0, "executed": 0, "profit": 0.0, "roi": 0.0}
    level_roi = {}
    level_counts = Counter(row["entry_level_cents"] for row in rows)
    for row in rows:
        level_roi[row["entry_level_cents"]] = max(0.0, row["training_roi"])
    total_weight = sum(level_roi.values())
    if total_weight <= 0:
        return {"opportunities": len(rows), "executed": 0, "profit": 0.0, "roi": 0.0}
    results = []
    for row in rows:
        level = row["entry_level_cents"]
        budget = capital * level_roi[level] / total_weight / level_counts[level]
        results.append(
            trade_result(
                row,
                budget,
                execution,
                entry_ticks,
                exit_ticks,
                participation,
                hold_to_resolution,
            )
        )
    executed = [result for result in results if result["executed"]]
    profit = sum(result["pnl"] for result in results)
    returns = [result["return"] for result in executed]
    return {
        "opportunities": len(rows),
        "executed": len(executed),
        "capital_deployed": sum(result["debit"] for result in results),
        "profit": profit,
        "final_value": capital + profit,
        "roi": profit / capital,
        "profitable_rate": sum(value > 0 for value in returns) / len(returns) if returns else None,
        "full_loss_rate": sum(value <= -1 for value in returns) / len(returns) if returns else None,
    }


def quantiles(values: Iterable[float]) -> dict:
    data = np.asarray(list(values), dtype=float)
    if not len(data):
        return {}
    return {
        "min": float(data.min()),
        "p05": float(np.quantile(data, 0.05)),
        "p25": float(np.quantile(data, 0.25)),
        "median": float(np.median(data)),
        "mean": float(data.mean()),
        "p75": float(np.quantile(data, 0.75)),
        "p95": float(np.quantile(data, 0.95)),
        "max": float(data.max()),
        "probability_positive": float((data > 0).mean()),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(value, indent=2, default=str))
    print(f"Wrote {path}")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_weeks(args, rows, execution):
    rows = sorted(rows, key=lambda row: row["entry_time"])
    timestamps = [row["entry_time"].timestamp() for row in rows]
    start_min = rows[0]["entry_time"]
    start_max = rows[-1]["entry_time"] - timedelta(days=7)
    if start_max <= start_min:
        raise SystemExit("Holdout is shorter than seven days")
    candidate_starts = []
    start = datetime.combine(start_min.date(), datetime.min.time(), tzinfo=timezone.utc)
    if start < start_min:
        start += timedelta(days=1)
    while start <= start_max:
        candidate_starts.append(start)
        start += timedelta(days=1)
    candidate_results = []
    for window_id, start in enumerate(candidate_starts):
        end = start + timedelta(days=7)
        left = bisect.bisect_left(timestamps, start.timestamp())
        right = bisect.bisect_left(timestamps, end.timestamp())
        result = allocate(rows[left:right], args.capital, execution)
        candidate_results.append(
            {"window_id": window_id, "start": start, "end": end, **result}
        )
    rng = np.random.default_rng(args.seed)
    selected = rng.integers(0, len(candidate_results), size=args.samples)
    results = [
        {"sample": index, **candidate_results[int(window_id)]}
        for index, window_id in enumerate(selected)
    ]
    output = args.output_dir / "random_weeks.csv"
    write_csv(output, results)
    rois = [row["roi"] for row in results]
    summary = {
        "samples": args.samples,
        "holdout_start": start_min,
        "holdout_end": rows[-1]["entry_time"],
        "independent_non_overlapping_weeks_available": int(
            (rows[-1]["entry_time"] - start_min).days / 7
        ),
        "distinct_daily_start_windows": len(candidate_results),
        "warning": "Samples bootstrap the distinct daily-start windows; sample count does not equal independent weeks.",
        "weekly_roi_distribution": quantiles(rois),
        "weekly_profit_distribution": quantiles(row["profit"] for row in results),
        "weekly_trade_count_distribution": quantiles(row["executed"] for row in results),
        "rows": str(output),
    }
    write_json(args.output_dir / "random_weeks_summary.json", summary)


def test_periods(args, rows, execution):
    groups = defaultdict(list)
    for row in rows:
        date = row["entry_time"]
        monday = (date - timedelta(days=date.weekday())).date()
        groups[str(monday)].append(row)
    results = []
    for period, selected in sorted(groups.items()):
        results.append({"week": period, **allocate(selected, args.capital, execution)})
    write_csv(args.output_dir / "calendar_weeks.csv", results)
    write_json(
        args.output_dir / "calendar_weeks_summary.json",
        {"weeks": len(results), "roi_distribution": quantiles(row["roi"] for row in results)},
    )


def test_tail(args, rows, execution):
    baseline = allocate(rows, args.capital, execution)
    # Use the already fee/minimum-adjusted contribution to avoid reallocating after deleting winners.
    contributions = sorted(
        (float(row.get("dollar_pnl") or 0) for row in rows if float(row.get("dollar_pnl") or 0) > 0),
        reverse=True,
    )
    results = []
    for fraction in args.fractions:
        removed = int(np.ceil(len(contributions) * fraction))
        removed_profit = sum(contributions[:removed])
        profit = baseline["profit"] - removed_profit
        results.append(
            {
                "removed_top_winner_fraction": fraction,
                "removed_winners": removed,
                "profit": profit,
                "roi": profit / args.capital,
            }
        )
    write_json(args.output_dir / "tail_removal.json", {"baseline": baseline, "results": results})


def test_bootstrap(args, rows, execution):
    baseline = allocate(rows, args.capital, execution)
    key_format = "%Y-%m-%d" if args.cluster == "day" else "%G-W%V"
    clusters = defaultdict(float)
    for row in rows:
        clusters[row["entry_time"].strftime(key_format)] += float(row.get("dollar_pnl") or 0)
    values = np.array(list(clusters.values()), dtype=float)
    rng = np.random.default_rng(args.seed)
    samples = rng.choice(values, size=(args.samples, len(values)), replace=True).sum(axis=1)
    write_json(
        args.output_dir / f"bootstrap_{args.cluster}.json",
        {
            "cluster": args.cluster,
            "clusters": len(values),
            "baseline": baseline,
            "portfolio_roi_distribution": quantiles(samples / args.capital),
        },
    )


def test_stability(args):
    grid = list(csv.DictReader(args.report_dir.joinpath("grid_results.csv").open()))
    best = list(csv.DictReader(args.report_dir.joinpath("best_by_entry_level.csv").open()))
    tps = sorted({float(row["take_profit"]) for row in grid})
    sls = sorted({float(row["stop_loss"]) for row in grid})
    lookup = {
        (int(row["entry_level_cents"]), float(row["take_profit"]), float(row["stop_loss"]), row["split"]): float(row["roi"])
        for row in grid
    }
    results = []
    for selected in best:
        level = int(selected["entry_level_cents"])
        tp = float(selected["take_profit"])
        sl = float(selected["stop_loss"])
        ti, si = tps.index(tp), sls.index(sl)
        neighbors = []
        for t_index in range(max(0, ti - 1), min(len(tps), ti + 2)):
            for s_index in range(max(0, si - 1), min(len(sls), si + 2)):
                if t_index == ti and s_index == si:
                    continue
                value = lookup.get((level, tps[t_index], sls[s_index], "train"))
                if value is not None:
                    neighbors.append(value)
        results.append(
            {
                "entry_level_cents": level,
                "take_profit": tp,
                "stop_loss": sl,
                "selected_train_roi": float(selected["train_roi"]),
                "neighbor_count": len(neighbors),
                "positive_neighbor_fraction": sum(value > 0 for value in neighbors) / len(neighbors),
                "worst_neighbor_roi": min(neighbors),
                "mean_neighbor_roi": float(np.mean(neighbors)),
            }
        )
    write_csv(args.output_dir / "parameter_stability.csv", results)
    write_json(
        args.output_dir / "parameter_stability_summary.json",
        {
            "levels": len(results),
            "levels_with_all_positive_neighbors": sum(
                row["positive_neighbor_fraction"] == 1 for row in results
            ),
            "levels_with_negative_neighbor": sum(row["worst_neighbor_roi"] < 0 for row in results),
        },
    )


def test_stress(args, rows, execution):
    results = []
    for entry_ticks in args.entry_ticks:
        for exit_ticks in args.exit_ticks:
            result = allocate(
                rows,
                args.capital,
                execution,
                entry_ticks=entry_ticks,
                exit_ticks=exit_ticks,
            )
            results.append({"entry_adverse_ticks": entry_ticks, "exit_adverse_ticks": exit_ticks, **result})
    write_csv(args.output_dir / "execution_stress.csv", results)
    write_json(args.output_dir / "execution_stress_summary.json", {"results": results})


def test_liquidity(args, rows, execution):
    results = []
    for participation in args.participation:
        result = allocate(rows, args.capital, execution, participation=participation)
        results.append({"fill_participation": participation, **result})
    write_csv(args.output_dir / "liquidity_caps.csv", results)
    write_json(args.output_dir / "liquidity_caps_summary.json", {"results": results})


def test_baselines(args, rows, execution):
    optimized = allocate(rows, args.capital, execution)
    hold = allocate(rows, args.capital, execution, hold_to_resolution=True)
    equal_rows = [dict(row, training_roi=1.0) for row in rows]
    equal_weight = allocate(equal_rows, args.capital, execution)
    write_json(
        args.output_dir / "baselines.json",
        {"optimized": optimized, "hold_underdog": hold, "equal_level_weight": equal_weight},
    )


def test_filter(args, rows, execution):
    include = re.compile(args.include, re.I) if args.include else None
    exclude = re.compile(args.exclude, re.I) if args.exclude else None
    selected = []
    for row in rows:
        text = f"{row.get('slug', '')} {row.get('question', '')}"
        if row.get("historical_volume", 0) < args.min_volume:
            continue
        if include and not include.search(text):
            continue
        if exclude and exclude.search(text):
            continue
        selected.append(row)
    write_json(
        args.output_dir / "filtered.json",
        {
            "input_rows": len(rows),
            "selected_rows": len(selected),
            "min_volume": args.min_volume,
            "include": args.include,
            "exclude": args.exclude,
            "result": allocate(selected, args.capital, execution),
        },
    )


def test_bankroll(args, rows, execution):
    rows = sorted(rows, key=lambda row: row["entry_time"])
    level_roi = {}
    for row in rows:
        level_roi[row["entry_level_cents"]] = max(0.0, row["training_roi"])
    total_roi = sum(level_roi.values())
    cash = args.capital
    realized_equity = args.capital
    peak = args.capital
    max_drawdown = 0.0
    exits = []
    entered = skipped_cash = skipped_minimum = 0
    max_concurrent = 0
    for row in rows:
        while exits and exits[0][0] <= row["entry_time"]:
            _, _, proceeds, _debit = heapq.heappop(exits)
            cash += proceeds
            realized_equity = cash + sum(item[3] for item in exits)
            peak = max(peak, realized_equity)
            max_drawdown = max(max_drawdown, (peak - realized_equity) / peak)
        target = realized_equity * level_roi[row["entry_level_cents"]] / total_roi * args.exposure_scale
        budget = min(cash, target)
        if args.participation is not None:
            budget = min(budget, row["entry_fill_usd"] * args.participation)
        if budget <= 0:
            skipped_cash += 1
            continue
        result = trade_result(row, budget, execution)
        if not result["executed"]:
            skipped_minimum += 1
            continue
        cash -= result["debit"]
        proceeds = result["debit"] + result["pnl"]
        heapq.heappush(
            exits,
            (row["exit_time"], row["market_id"], proceeds, result["debit"]),
        )
        entered += 1
        max_concurrent = max(max_concurrent, len(exits))
    while exits:
        _, _, proceeds, _debit = heapq.heappop(exits)
        cash += proceeds
    write_json(
        args.output_dir / "rolling_bankroll.json",
        {
            "initial_capital": args.capital,
            "final_value": cash,
            "roi": cash / args.capital - 1,
            "entered": entered,
            "skipped_cash": skipped_cash,
            "skipped_minimum": skipped_minimum,
            "max_concurrent_positions": max_concurrent,
            "max_realized_drawdown": max_drawdown,
            "exposure_scale": args.exposure_scale,
            "fill_participation": args.participation,
        },
    )


def test_walk_forward(args):
    data = np.load(args.report_dir / "strategy_cube.npz")
    times = data["entry_times"]
    levels = data["entry_levels"]
    cube = data["returns"]
    tps = data["take_profits"]
    sls = data["stop_losses"]
    start = datetime.fromtimestamp(int(times.min()), timezone.utc) + timedelta(weeks=args.train_weeks)
    end = datetime.fromtimestamp(int(times.max()), timezone.utc)
    results = []
    fold = 0
    while start + timedelta(weeks=args.test_weeks) <= end + timedelta(days=1):
        train_start = start - timedelta(weeks=args.train_weeks)
        test_end = start + timedelta(weeks=args.test_weeks)
        train_mask = (times >= train_start.timestamp()) & (times < start.timestamp())
        test_mask = (times >= start.timestamp()) & (times < test_end.timestamp())
        selected_returns = []
        weights = []
        for level in range(1, 50):
            train_indexes = np.where(train_mask & (levels == level))[0]
            test_indexes = np.where(test_mask & (levels == level))[0]
            if len(train_indexes) < args.min_train_trades or not len(test_indexes):
                continue
            means = cube[train_indexes].mean(axis=0)
            best_flat = int(np.argmax(means))
            tp_index, sl_index = np.unravel_index(best_flat, means.shape)
            train_roi = float(means[tp_index, sl_index])
            test_values = cube[test_indexes, tp_index, sl_index]
            selected_returns.extend(test_values.tolist())
            weights.extend([max(0.0, train_roi) / len(test_values)] * len(test_values))
        if selected_returns:
            values = np.asarray(selected_returns)
            weight_values = np.asarray(weights)
            weighted_roi = (
                float(np.average(values, weights=weight_values)) if weight_values.sum() else 0.0
            )
            results.append(
                {
                    "fold": fold,
                    "train_start": train_start,
                    "test_start": start,
                    "test_end": test_end,
                    "test_trades": len(values),
                    "unweighted_roi": float(values.mean()),
                    "training_roi_weighted_roi": weighted_roi,
                }
            )
        fold += 1
        start += timedelta(weeks=args.step_weeks)
    write_csv(args.output_dir / "walk_forward.csv", results)
    write_json(
        args.output_dir / "walk_forward_summary.json",
        {
            "folds": len(results),
            "weighted_roi_distribution": quantiles(
                row["training_roi_weighted_roi"] for row in results
            ),
            "profitable_fold_fraction": sum(
                row["training_roi_weighted_roi"] > 0 for row in results
            )
            / len(results),
            "take_profit_grid": tps.tolist(),
            "stop_loss_grid": sls.tolist(),
        },
    )


def add_common(parser):
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/underdog_tests"))
    parser.add_argument("--capital", type=float, default=5_000)


def main() -> None:
    parser = argparse.ArgumentParser(description="Falsification and robustness tests for the underdog strategy.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("weeks", "periods", "tail", "bootstrap", "stability", "stress", "liquidity", "baselines", "filter", "bankroll", "walk-forward"):
        child = subparsers.add_parser(name)
        add_common(child)
    subparsers.choices["weeks"].add_argument("--samples", type=int, default=10_000)
    subparsers.choices["weeks"].add_argument("--seed", type=int, default=42)
    subparsers.choices["tail"].add_argument("--fractions", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.10])
    subparsers.choices["bootstrap"].add_argument("--cluster", choices=["day", "week"], default="day")
    subparsers.choices["bootstrap"].add_argument("--samples", type=int, default=10_000)
    subparsers.choices["bootstrap"].add_argument("--seed", type=int, default=42)
    subparsers.choices["stress"].add_argument("--entry-ticks", type=int, nargs="+", default=[0, 1, 2])
    subparsers.choices["stress"].add_argument("--exit-ticks", type=int, nargs="+", default=[0, 1, 2])
    subparsers.choices["liquidity"].add_argument("--participation", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.25, 1.0])
    subparsers.choices["filter"].add_argument("--min-volume", type=float, default=0)
    subparsers.choices["filter"].add_argument("--include")
    subparsers.choices["filter"].add_argument("--exclude")
    subparsers.choices["bankroll"].add_argument("--exposure-scale", type=float, default=1.0)
    subparsers.choices["bankroll"].add_argument("--participation", type=float)
    subparsers.choices["walk-forward"].add_argument("--train-weeks", type=int, default=12)
    subparsers.choices["walk-forward"].add_argument("--test-weeks", type=int, default=1)
    subparsers.choices["walk-forward"].add_argument("--step-weeks", type=int, default=1)
    subparsers.choices["walk-forward"].add_argument("--min-train-trades", type=int, default=30)
    args = parser.parse_args()

    if args.command == "stability":
        test_stability(args)
        return
    if args.command == "walk-forward":
        test_walk_forward(args)
        return
    trades_path = args.report_dir / "holdout_policy_trades.csv"
    rows = load_rows(trades_path)
    execution = load_execution(args.report_dir)
    functions = {
        "weeks": test_weeks,
        "periods": test_periods,
        "tail": test_tail,
        "bootstrap": test_bootstrap,
        "stress": test_stress,
        "liquidity": test_liquidity,
        "baselines": test_baselines,
        "filter": test_filter,
        "bankroll": test_bankroll,
    }
    functions[args.command](args, rows, execution)


if __name__ == "__main__":
    main()
