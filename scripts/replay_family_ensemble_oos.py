#!/usr/bin/env python3
"""Combined ensemble replay for promoted strategy-family signals."""
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

from realistic_underdog_account import write_csv
from replay_strategy_family_oos import max_drawdown_from_profits, period_key, period_rows
from walk_forward_oos import month_floor


DEFAULT_COMPONENTS = [
    "underdog_attention_light:16-30c_single_1.25x:core:5:500:5",
    "favorite_fade_long_horizon_16_30:16-30c_single_1.25x:core:5:750:4",
    "pure_long_tail_16_30:16-30c_single_1.25x:core:2:1000:3",
    "momentum_24h_light:16-30c_single_1.25x:core:5:250:2",
    "pure_long_tail_01_05:01-05c_single_2x:tail:1:125:1",
    "favorite_fade_near_deadline_01_15:01-05c_single_2x:tail:1:125:0",
]


def parse_component(value: str) -> dict[str, Any]:
    parts = value.split(":")
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            "components must use strategy:exit_rule:sleeve:stake:monthly_cap:priority"
        )
    strategy, exit_rule, sleeve, stake, monthly_cap, priority = parts
    return {
        "strategy": strategy,
        "exit_rule": exit_rule,
        "sleeve": sleeve,
        "stake": float(stake),
        "monthly_cap": float(monthly_cap),
        "priority": int(priority),
        "component": f"{strategy}:{exit_rule}",
    }


def max_drawdown_from_profit_times(initial_cash: float, profits: list[tuple[pd.Timestamp, float]]) -> float:
    if not profits:
        return 0.0
    ordered = [profit for _, profit in sorted(profits, key=lambda item: item[0])]
    return max_drawdown_from_profits(initial_cash, ordered)


def load_component_signals(signals: pd.DataFrame, components: list[dict[str, Any]]) -> pd.DataFrame:
    frames = []
    for component in components:
        selected = signals[
            (signals["strategy"] == component["strategy"])
            & (signals["exit_rule"] == component["exit_rule"])
        ].copy()
        if selected.empty:
            continue
        selected["component"] = component["component"]
        selected["sleeve"] = component["sleeve"]
        selected["stake"] = component["stake"]
        selected["monthly_cap"] = component["monthly_cap"]
        selected["priority"] = component["priority"]
        frames.append(selected)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.sort_values(["timestamp", "priority", "component"], kind="stable")
    return combined


def replay_ensemble(
    signals: pd.DataFrame,
    end_time: pd.Timestamp,
    initial_cash: float,
    period_budget: float,
    budget_period: str,
    reserve_fraction: float,
    min_stake: float,
    max_trades_per_market: int,
    max_components_per_market: int,
    participation_fraction: float = 0.0,
    min_stake_fill_fraction: float = 0.0,
    slippage_model: str = "none",
    slippage_coef: float = 0.0,
    stake_fill_fraction: float = 0.0,
    max_stake: float = float("inf"),
) -> dict[str, Any]:
    cash = initial_cash
    realized_profit = 0.0
    deployed = 0.0
    slippage_cost = 0.0
    entries = 0
    skipped = defaultdict(int)
    profits: list[float] = []
    profit_times: list[tuple[pd.Timestamp, float]] = []
    trade_debits: list[float] = []
    period_spend: dict[Any, float] = defaultdict(float)
    component_spend: dict[tuple[Any, str], float] = defaultdict(float)
    market_counts: dict[int, int] = defaultdict(int)
    market_components: dict[int, set[str]] = defaultdict(set)
    component_counts = defaultdict(int)
    component_profit = defaultdict(float)
    component_deployed = defaultdict(float)

    for row in signals.itertuples(index=False):
        timestamp = pd.Timestamp(row.timestamp)
        if timestamp >= end_time:
            break
        market_id = int(row.market_id)
        component = str(row.component)
        if market_counts[market_id] >= max_trades_per_market:
            skipped["market_trade_cap"] += 1
            continue
        if component not in market_components[market_id] and len(market_components[market_id]) >= max_components_per_market:
            skipped["market_component_cap"] += 1
            continue

        pkey = period_key(timestamp, budget_period)
        component_key = (pkey, component)
        reserve_floor = initial_cash * reserve_fraction
        # Liquidity-aware participation cap: never stake more than a fraction of the
        # actual archived triggering fill. Markets with unknown fill size are skipped
        # rather than assumed liquid.
        fill = float(getattr(row, "entry_fill_usd", float("nan")))
        participation_cap = float("inf")
        if participation_fraction and participation_fraction > 0.0:
            participation_cap = participation_fraction * fill if math.isfinite(fill) else 0.0
        # Liquidity-scaled target stake: in deep markets, scale the intended position up
        # with the triggering fill (floored at the flat base stake, capped at max_stake)
        # instead of always staking the flat base. Participation remains the hard ceiling,
        # so set stake_fill_fraction <= participation_fraction for it to govern. Thin
        # markets keep the base stake (then get sized down by participation as before).
        target_stake = float(row.stake)
        if stake_fill_fraction and stake_fill_fraction > 0.0 and math.isfinite(fill):
            target_stake = min(max(float(row.stake), stake_fill_fraction * fill), max_stake)
        debit = min(
            target_stake,
            max(0.0, period_budget - period_spend[pkey]),
            max(0.0, float(row.monthly_cap) - component_spend[component_key]),
            max(0.0, cash - reserve_floor),
            participation_cap,
        )
        # Market-dependent minimum stake: scales with the market's own liquidity so
        # thin tail markets get a proportionally smaller floor (less money committed).
        # It is never allowed above the participation cap, so it can't cause perverse
        # skips on markets the cap already sized down.
        effective_min = min_stake
        if min_stake_fill_fraction and min_stake_fill_fraction > 0.0 and math.isfinite(fill):
            effective_min = max(min_stake, min_stake_fill_fraction * fill)
        # Never let the floor exceed what we'd actually stake (target stake or the
        # participation cap), so a small stake into a deep market is not perversely
        # skipped; the floor only filters dust / budget-starved trades.
        effective_min = min(effective_min, target_stake)
        if math.isfinite(participation_cap):
            effective_min = min(effective_min, participation_cap)
        if debit < effective_min:
            skipped["below_min_stake_or_budget"] += 1
            continue
        unit_return = float(row.unit_return)
        if not math.isfinite(unit_return):
            skipped["nonfinite_return"] += 1
            continue

        # Market-impact / slippage: taking a larger share of the triggering fill
        # walks the book, worsening the effective entry price by factor (1 + s).
        # Fewer shares at a worse basis scale the gross payoff by 1/(1 + s), so
        #   r_eff = (1 + unit_return) / (1 + s) - 1.
        # This leaves a total loss (~ -1) essentially unchanged while shaving
        # winners, which is why sizing up should erode net edge. Applied only when
        # the triggering fill is known; otherwise no impact is charged.
        effective_return = unit_return
        if slippage_coef > 0.0 and slippage_model != "none" and math.isfinite(fill) and fill > 0.0:
            ratio = debit / fill
            if slippage_model == "linear":
                s = slippage_coef * ratio
            elif slippage_model == "sqrt":
                s = slippage_coef * math.sqrt(ratio)
            else:
                raise ValueError(f"Unknown slippage_model: {slippage_model}")
            effective_return = (1.0 + unit_return) / (1.0 + s) - 1.0

        gross_profit = debit * unit_return
        profit = debit * effective_return
        slippage_cost += gross_profit - profit
        cash += profit
        realized_profit += profit
        deployed += debit
        period_spend[pkey] += debit
        component_spend[component_key] += debit
        market_counts[market_id] += 1
        market_components[market_id].add(component)
        component_counts[component] += 1
        component_profit[component] += profit
        component_deployed[component] += debit
        entries += 1
        profits.append(profit)
        profit_times.append((timestamp, profit))
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
        "slippage_cost": float(slippage_cost),
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
        "max_drawdown": max_drawdown_from_profit_times(initial_cash, profit_times),
        "component_counts": dict(component_counts),
        "component_profit": dict(component_profit),
        "component_deployed": dict(component_deployed),
        "skipped": dict(skipped),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    summaries = []
    for months, group in df.groupby("test_months", dropna=False):
        profits = group["realized_profit"].astype(float).to_numpy()
        summaries.append({
            "selected_strategy": "family_signal_ensemble",
            "test_months": int(months),
            "periods": int(len(group)),
            "mean_profit": float(np.mean(profits)),
            "median_profit": float(np.median(profits)),
            "mean_account_return": float(group["account_return"].astype(float).mean()),
            "median_account_return": float(group["account_return"].astype(float).median()),
            "positive_rate": float(np.mean(profits > 0)),
            "mean_without_top1": float(group["without_top_1"].astype(float).mean()),
            "mean_capped_10x": float(group["profit_capped_at_10x_cost"].astype(float).mean()),
            "worst_period_profit": float(np.min(profits)),
            "best_period_profit": float(np.max(profits)),
            "mean_entries": float(group["entries"].astype(float).mean()),
            "mean_deployed": float(group["deployed"].astype(float).mean()),
            "worst_max_drawdown": float(group["max_drawdown"].astype(float).max()),
        })
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", type=Path, default=Path("reports/strategy_family_diagnostics/strategy_family_signals.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/family_ensemble_oos"))
    parser.add_argument("--components", type=parse_component, nargs="+", default=[parse_component(value) for value in DEFAULT_COMPONENTS])
    parser.add_argument("--first-month", default="2022-11-01")
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
    args = parser.parse_args()

    signals = pd.read_csv(args.signals)
    signals["timestamp"] = pd.to_datetime(signals["timestamp"], utc=True)
    combined = load_component_signals(signals, args.components)
    if combined.empty:
        raise SystemExit("No component signals matched.")

    first_month = datetime.fromisoformat(args.first_month).replace(tzinfo=timezone.utc)
    last_month = month_floor(int(signals["timestamp"].max().timestamp()))
    rows = []
    for months in sorted(set(args.test_months)):
        for period in period_rows(first_month, last_month, months, args.min_train_months, args.validation_months):
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
                "experiment": "family_signal_ensemble",
                "selected_strategy": "family_signal_ensemble",
                "period_id": period["period_id"],
                "test_months": months,
                "validation_months": args.validation_months,
                "validation_start": period["validation_start"].date(),
                "test_start": period["test_start"].date(),
                "test_end": period["test_end"].date(),
                **result,
                "component_counts": json.dumps(result["component_counts"], sort_keys=True),
                "component_profit": json.dumps(result["component_profit"], sort_keys=True),
                "component_deployed": json.dumps(result["component_deployed"], sort_keys=True),
                "skipped": json.dumps(result["skipped"], sort_keys=True),
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "family_ensemble_period_results.csv", rows)
    summary_rows = summarize(rows)
    write_csv(args.output_dir / "family_ensemble_summary.csv", summary_rows)
    (args.output_dir / "summary.json").write_text(json.dumps({
        "signals": str(args.signals),
        "components": [
            f"{item['component']}:{item['sleeve']}:stake={item['stake']}:cap={item['monthly_cap']}:priority={item['priority']}"
            for item in args.components
        ],
        "period_rows": len(rows),
        "summary_rows": len(summary_rows),
        "files": ["family_ensemble_period_results.csv", "family_ensemble_summary.csv"],
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "period_rows": len(rows), "summary_rows": len(summary_rows)}, indent=2))


if __name__ == "__main__":
    main()
