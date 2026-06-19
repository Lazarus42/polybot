#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path
from typing import Optional

import duckdb
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Entry:
    market_id: int
    entry_time: object
    entry_price: float
    underdog_side: str
    winner_side: str
    end_date: object
    closed_time: object
    slug: str
    question: str
    historical_volume: float
    entry_fill_usd: float
    split: str = ""
    take_profit_hits: Optional[list] = None
    stop_loss_hits: Optional[list] = None


def parse_values(value: str) -> list[float]:
    return sorted({float(item) for item in value.split(",")})


def entry_level(price: float) -> int:
    return max(1, min(49, int(price * 100 + 1e-9)))


def quantize_price(price: float, tick: float) -> float:
    if tick <= 0:
        return price
    ticks = math.floor(price / tick + 0.5)
    return min(1.0 - tick, max(tick, ticks * tick))


def continuous_return(entry_price: float, exit_price: float, fee_coefficient: float) -> float:
    entry_fee = fee_coefficient * entry_price * (1.0 - entry_price)
    entry_cost = entry_price + entry_fee
    exit_fee = fee_coefficient * exit_price * (1.0 - exit_price) if 0 < exit_price < 1 else 0.0
    return (exit_price - exit_fee) / entry_cost - 1.0


def posted_balance_change(
    contracts: Decimal,
    price: Decimal,
    fee_coefficient: Decimal,
    action: str,
) -> Decimal:
    raw_fee = fee_coefficient * contracts * price * (Decimal("1") - price)
    trade_fee = raw_fee.quantize(Decimal("0.0001"), rounding=ROUND_CEILING)
    revenue = contracts * price * (Decimal("-1") if action == "buy" else Decimal("1"))
    balance_change = revenue - trade_fee
    return (balance_change * 100).to_integral_value(rounding=ROUND_FLOOR) / 100


def max_contracts_for_budget(
    budget: float,
    price: float,
    fee_coefficient: float,
    contract_step: float,
) -> tuple[Decimal, Decimal]:
    budget_value = Decimal(str(budget))
    price_value = Decimal(str(price))
    fee_value = Decimal(str(fee_coefficient))
    step = Decimal(str(contract_step))
    if budget_value <= 0 or price_value <= 0 or step <= 0:
        return Decimal("0"), Decimal("0")
    high = max(0, int((budget_value / (price_value * step)).to_integral_value(rounding=ROUND_FLOOR)))
    low = 0
    while low < high:
        middle = (low + high + 1) // 2
        contracts = step * middle
        debit = -posted_balance_change(contracts, price_value, fee_value, "buy")
        if debit <= budget_value:
            low = middle
        else:
            high = middle - 1
    contracts = step * low
    debit = -posted_balance_change(contracts, price_value, fee_value, "buy") if low else Decimal("0")
    return contracts, debit


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize underdog brackets by 1-cent entry level.")
    parser.add_argument("--data-dir", type=Path, default=Path("archive/processed/underdog_events"))
    parser.add_argument("--confirmation-minutes", type=float, default=5)
    parser.add_argument("--entry-delay-minutes", type=float, default=0)
    parser.add_argument("--entry-min", type=float, default=0.01)
    parser.add_argument("--entry-max", type=float, default=0.49)
    parser.add_argument("--take-profits", default="1.1,1.25,1.5,1.75,2,2.5,3,4,5,7.5,10,15,20")
    parser.add_argument("--stop-losses", default="0,0.01,0.025,0.05,0.075,0.1,0.15,0.25,0.4,0.5,0.6,0.75,0.9")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--min-train-trades", type=int, default=30)
    parser.add_argument("--fee-coefficient", type=float, default=0.0)
    parser.add_argument("--price-tick", type=float, default=0.0)
    parser.add_argument("--contract-step", type=float, default=0.0)
    parser.add_argument("--initial-capital", type=float, default=5_000.0)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/underdog_optimization"))
    args = parser.parse_args()
    if not 0 < args.entry_min <= args.entry_max < 1:
        parser.error("entry bounds must satisfy 0 < min <= max < 1")
    if not 0 < args.train_fraction < 1:
        parser.error("train fraction must be between 0 and 1")
    take_profits = parse_values(args.take_profits)
    stop_losses = parse_values(args.stop_losses)
    if any(value <= 1 for value in take_profits):
        parser.error("all take-profit multipliers must exceed 1")
    if any(not 0 <= value < 1 for value in stop_losses):
        parser.error("all stop-loss multipliers must be between 0 (disabled) and 1")

    sorted_fills = args.data_dir / "fills_sorted.parquet"
    if not sorted_fills.exists():
        raise SystemExit(f"Missing clustered dataset: {sorted_fills}")
    fills_path = str(sorted_fills.resolve()).replace("'", "''")
    markets_path = str((args.data_dir / "markets.parquet").resolve()).replace("'", "''")
    connection = duckdb.connect()
    connection.execute("SET threads = 1")
    market_metadata = {
        row[0]: row[1:]
        for row in connection.execute(
            f"SELECT market_id, first_trade_time, winner_side, end_date, closed_time, slug, question, historical_volume FROM read_parquet('{markets_path}') WHERE first_trade_time IS NOT NULL"
        ).fetchall()
    }
    cursor = connection.execute(
        f"SELECT market_id, timestamp, side, price, usd_amount FROM read_parquet('{fills_path}')"
    )
    entries = {}
    current_market = None
    first_trade_time = None
    winner_side = None
    end_date = None
    closed_time = None
    slug = None
    question = None
    historical_volume = None
    confirmation_time = None
    underdog_side = None
    eligible_time = None
    current_entry = None
    rows_scanned = 0
    for batch in iter(lambda: cursor.fetchmany(100_000), []):
        for market_id, timestamp, side, price, fill_usd in batch:
            rows_scanned += 1
            price = quantize_price(price, args.price_tick)
            if current_market is not None and market_id < current_market:
                raise RuntimeError("fills_sorted.parquet is not ordered by market_id")
            if market_id != current_market:
                current_market = market_id
                metadata = market_metadata.get(market_id)
                (
                    first_trade_time,
                    winner_side,
                    end_date,
                    closed_time,
                    slug,
                    question,
                    historical_volume,
                ) = metadata if metadata else (None, None, None, None, None, None, None)
                confirmation_time = None
                underdog_side = None
                eligible_time = None
                current_entry = None
            if first_trade_time is None:
                continue
            if underdog_side is None:
                confirmation_cutoff = first_trade_time + timedelta(minutes=args.confirmation_minutes)
                if timestamp < confirmation_cutoff or not args.entry_min <= price <= args.entry_max:
                    continue
                confirmation_time = timestamp
                underdog_side = side
                eligible_time = timestamp + timedelta(minutes=args.entry_delay_minutes)
            if current_entry is None:
                if side != underdog_side or timestamp < eligible_time or not args.entry_min <= price <= args.entry_max:
                    continue
                current_entry = Entry(
                    market_id=market_id,
                    entry_time=timestamp,
                    entry_price=price,
                    underdog_side=underdog_side,
                    winner_side=winner_side,
                    end_date=end_date,
                    closed_time=closed_time,
                    slug=slug or "",
                    question=question or "",
                    historical_volume=historical_volume or 0.0,
                    entry_fill_usd=fill_usd,
                    take_profit_hits=[None] * len(take_profits),
                    stop_loss_hits=[None] * len(stop_losses),
                )
                entries[market_id] = current_entry
                continue
            if side != current_entry.underdog_side:
                continue
            ratio = price / current_entry.entry_price
            for index, threshold in enumerate(take_profits):
                if current_entry.take_profit_hits[index] is None and ratio >= threshold:
                    current_entry.take_profit_hits[index] = (
                        rows_scanned,
                        timestamp,
                        price,
                        fill_usd,
                    )
            for index, threshold in enumerate(stop_losses):
                if current_entry.stop_loss_hits[index] is None and ratio <= threshold:
                    current_entry.stop_loss_hits[index] = (
                        rows_scanned,
                        timestamp,
                        price,
                        fill_usd,
                    )

    if not entries:
        raise SystemExit("No eligible entries found")
    ordered_times = sorted(entry.entry_time for entry in entries.values())
    split_time = ordered_times[int((len(ordered_times) - 1) * args.train_fraction)]
    for entry in entries.values():
        entry.split = "train" if entry.entry_time <= split_time else "test"

    # (level, tp, sl, split) -> [count, pnl sum, profitable count]
    stats = defaultdict(lambda: [0, 0.0, 0])
    entry_list = list(entries.values())
    strategy_cube = np.empty(
        (len(entry_list), len(take_profits), len(stop_losses)), dtype=np.float32
    )
    hold_returns = np.empty(len(entry_list), dtype=np.float32)
    exit_times = np.empty_like(strategy_cube, dtype=np.int64)
    exit_prices = np.empty_like(strategy_cube, dtype=np.float32)
    exit_fill_cube = np.empty_like(strategy_cube, dtype=np.float32)
    exit_codes = np.empty_like(strategy_cube, dtype=np.uint8)
    for entry_index, entry in enumerate(entry_list):
        level = entry_level(entry.entry_price)
        resolution_price = 1.0 if entry.underdog_side == entry.winner_side else 0.0
        hold_returns[entry_index] = continuous_return(
            entry.entry_price, resolution_price, args.fee_coefficient
        )
        for tp_index, take_profit in enumerate(take_profits):
            tp_hit = entry.take_profit_hits[tp_index]
            for sl_index, stop_loss in enumerate(stop_losses):
                sl_hit = entry.stop_loss_hits[sl_index]
                if tp_hit is not None and (sl_hit is None or tp_hit[0] < sl_hit[0]):
                    exit_price = tp_hit[2]
                    exit_time = tp_hit[1]
                    exit_fill = tp_hit[3]
                    exit_code = 1
                elif sl_hit is not None:
                    exit_price = sl_hit[2]
                    exit_time = sl_hit[1]
                    exit_fill = sl_hit[3]
                    exit_code = 2
                else:
                    exit_price = resolution_price
                    exit_time = entry.closed_time
                    exit_fill = np.nan
                    exit_code = 0
                pnl = continuous_return(entry.entry_price, exit_price, args.fee_coefficient)
                strategy_cube[entry_index, tp_index, sl_index] = pnl
                exit_times[entry_index, tp_index, sl_index] = (
                    int(exit_time.timestamp()) if exit_time is not None else np.iinfo(np.int64).max
                )
                exit_prices[entry_index, tp_index, sl_index] = exit_price
                exit_fill_cube[entry_index, tp_index, sl_index] = exit_fill
                exit_codes[entry_index, tp_index, sl_index] = exit_code
                values = stats[(level, take_profit, stop_loss, entry.split)]
                values[0] += 1
                values[1] += pnl
                values[2] += pnl > 0

    grid_rows = []
    for (level, take_profit, stop_loss, split), (count, pnl, profitable) in sorted(stats.items()):
        grid_rows.append(
            {
                "entry_level_cents": level,
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "split": split,
                "trades": count,
                "roi": pnl / count,
                "profitable_trade_rate": profitable / count,
            }
        )

    by_key = {
        (row["entry_level_cents"], row["take_profit"], row["stop_loss"], row["split"]): row
        for row in grid_rows
    }
    best_rows = []
    for level in range(1, 50):
        candidates = [
            row for row in grid_rows
            if row["entry_level_cents"] == level
            and row["split"] == "train"
            and row["trades"] >= args.min_train_trades
        ]
        if not candidates:
            continue
        best_train = max(candidates, key=lambda row: row["roi"])
        test = by_key.get((level, best_train["take_profit"], best_train["stop_loss"], "test"))
        if test is None:
            continue
        best_rows.append(
            {
                "entry_level_cents": level,
                "take_profit": best_train["take_profit"],
                "stop_loss": best_train["stop_loss"],
                "train_trades": best_train["trades"],
                "train_roi": best_train["roi"],
                "test_trades": test["trades"],
                "test_roi": test["roi"],
                "test_profitable_trade_rate": test["profitable_trade_rate"],
            }
        )

    best_by_level = {row["entry_level_cents"]: row for row in best_rows}
    policy_rows = []
    for entry in entries.values():
        if entry.split != "test":
            continue
        level = entry_level(entry.entry_price)
        policy = best_by_level.get(level)
        if policy is None:
            continue
        take_profit = policy["take_profit"]
        stop_loss = policy["stop_loss"]
        tp_hit = entry.take_profit_hits[take_profits.index(take_profit)]
        sl_hit = entry.stop_loss_hits[stop_losses.index(stop_loss)]
        if tp_hit is not None and (sl_hit is None or tp_hit[0] < sl_hit[0]):
            exit_time = tp_hit[1]
            exit_price = tp_hit[2]
            exit_fill_usd = tp_hit[3]
            exit_type = "take_profit"
        elif sl_hit is not None:
            exit_time = sl_hit[1]
            exit_price = sl_hit[2]
            exit_fill_usd = sl_hit[3]
            exit_type = "stop_loss"
        else:
            won = entry.underdog_side == entry.winner_side
            exit_time = entry.closed_time
            exit_price = 1.0 if won else 0.0
            exit_fill_usd = None
            exit_type = "resolution_win" if won else "resolution_loss"
        policy_rows.append(
            {
                "market_id": entry.market_id,
                "slug": entry.slug,
                "question": entry.question,
                "historical_volume": entry.historical_volume,
                "entry_time": entry.entry_time,
                "exit_time": exit_time,
                "entry_level_cents": level,
                "entry_price": entry.entry_price,
                "entry_fill_usd": entry.entry_fill_usd,
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "exit_type": exit_type,
                "exit_price": exit_price,
                "exit_fill_usd": exit_fill_usd,
                "underdog_won": entry.underdog_side == entry.winner_side,
                "return": continuous_return(entry.entry_price, exit_price, args.fee_coefficient),
                "training_roi": policy["train_roi"],
            }
        )

    initial_capital = args.initial_capital
    positive_training_roi = {
        level: max(0.0, row["train_roi"]) for level, row in best_by_level.items()
    }
    total_training_roi = sum(positive_training_roi.values())
    level_counts = defaultdict(int)
    for row in policy_rows:
        level_counts[row["entry_level_cents"]] += 1
    for row in policy_rows:
        level = row["entry_level_cents"]
        level_weight = positive_training_roi[level] / total_training_roi
        row["level_portfolio_weight"] = level_weight
        budget = initial_capital * level_weight / level_counts[level]
        row["allocated_budget"] = budget
        row["model_return_before_order_rounding"] = row.pop("return")
        if args.contract_step > 0:
            contracts, debit = max_contracts_for_budget(
                budget,
                row["entry_price"],
                args.fee_coefficient,
                args.contract_step,
            )
            row["contracts"] = float(contracts)
            row["capital_deployed"] = float(debit)
            if contracts > 0 and debit > 0:
                if row["exit_type"].startswith("resolution"):
                    payout = (contracts * Decimal(str(row["exit_price"])) * 100).to_integral_value(
                        rounding=ROUND_FLOOR
                    ) / 100
                else:
                    payout = posted_balance_change(
                        contracts,
                        Decimal(str(row["exit_price"])),
                        Decimal(str(args.fee_coefficient)),
                        "sell",
                    )
                row["dollar_pnl"] = float(payout - debit)
                row["return"] = row["dollar_pnl"] / float(debit)
                row["executed"] = True
            else:
                row["dollar_pnl"] = 0.0
                row["return"] = None
                row["executed"] = False
        else:
            row["contracts"] = None
            row["capital_deployed"] = budget
            row["dollar_pnl"] = budget * row["model_return_before_order_rounding"]
            row["return"] = row["model_return_before_order_rounding"]
            row["executed"] = True
        row["idle_budget"] = budget - row["capital_deployed"]
        row["allocated_budget_return"] = row["dollar_pnl"] / budget if budget > 0 else 0.0

    executed_rows = [row for row in policy_rows if row["executed"]]
    returns = np.array([row["return"] for row in executed_rows], dtype=float)
    dollar_pnls = np.array([row["dollar_pnl"] for row in policy_rows], dtype=float)
    portfolio_profit = float(dollar_pnls.sum())
    portfolio_roi = portfolio_profit / initial_capital
    exit_counts = defaultdict(int)
    for row in executed_rows:
        exit_counts[row["exit_type"]] += 1

    rng = np.random.default_rng(42)
    bootstrap_returns = np.zeros(10_000, dtype=float)
    for level, weight in positive_training_roi.items():
        level_values = np.array(
            [
                row["allocated_budget_return"]
                for row in policy_rows
                if row["entry_level_cents"] == level
            ],
            dtype=float,
        )
        if not len(level_values) or weight <= 0:
            continue
        samples = rng.choice(level_values, size=(10_000, len(level_values)), replace=True)
        bootstrap_returns += (weight / total_training_roi) * samples.mean(axis=1)

    quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    portfolio_summary = {
        "allocation": "Allocate capital across entry-cent strategies proportional to fee-adjusted training ROI, then equally across each strategy's holdout opportunities.",
        "execution_model": {
            "fee_coefficient": args.fee_coefficient,
            "price_tick": args.price_tick,
            "contract_step": args.contract_step,
            "liquidity_role": "taker",
            "fills_per_order": 1,
            "rounding": "trade fee up to $0.0001, posted balance down to $0.01",
        },
        "initial_capital": initial_capital,
        "holdout_opportunities": len(policy_rows),
        "executed_trades": len(executed_rows),
        "zero_weight_opportunities": sum(row["allocated_budget"] == 0 for row in policy_rows),
        "skipped_below_minimum": sum(
            row["allocated_budget"] > 0 and not row["executed"] for row in policy_rows
        ),
        "capital_deployed": sum(row["capital_deployed"] for row in policy_rows),
        "idle_capital_from_rounding_and_minimums": sum(row["idle_budget"] for row in policy_rows),
        "profitable_trades": int((returns > 0).sum()),
        "profitable_trade_rate": float((returns > 0).mean()),
        "losing_trade_rate": float((returns < 0).mean()),
        "full_loss_rate": float((returns <= -1).mean()),
        "exit_counts": dict(exit_counts),
        "unweighted_mean_trade_return": float(returns.mean()),
        "median_trade_return": float(np.median(returns)),
        "trade_return_standard_deviation": float(returns.std()),
        "trade_return_quantiles": {
            str(quantile): float(np.quantile(returns, quantile)) for quantile in quantiles
        },
        "portfolio_profit": portfolio_profit,
        "final_value": initial_capital + portfolio_profit,
        "portfolio_roi": portfolio_roi,
        "bootstrap_portfolio_roi": {
            "iterations": 10_000,
            "mean": float(bootstrap_returns.mean()),
            "median": float(np.median(bootstrap_returns)),
            "p05": float(np.quantile(bootstrap_returns, 0.05)),
            "p95": float(np.quantile(bootstrap_returns, 0.95)),
            "probability_positive": float((bootstrap_returns > 0).mean()),
        },
        "level_allocations": [
            {
                "entry_level_cents": level,
                "training_roi": best_by_level[level]["train_roi"],
                "portfolio_weight": positive_training_roi[level] / total_training_roi,
                "allocated_dollars": initial_capital * positive_training_roi[level] / total_training_roi,
                "holdout_trades": level_counts[level],
                "executed_trades": sum(
                    row["executed"]
                    for row in policy_rows
                    if row["entry_level_cents"] == level
                ),
            }
            for level in sorted(best_by_level)
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = args.output_dir / "grid_results.csv"
    best_path = args.output_dir / "best_by_entry_level.csv"
    policy_path = args.output_dir / "holdout_policy_trades.csv"
    portfolio_path = args.output_dir / "holdout_portfolio_summary.json"
    cube_path = args.output_dir / "strategy_cube.npz"
    with grid_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grid_rows[0]))
        writer.writeheader()
        writer.writerows(grid_rows)
    with best_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(best_rows[0]))
        writer.writeheader()
        writer.writerows(best_rows)
    with policy_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(policy_rows[0]))
        writer.writeheader()
        writer.writerows(policy_rows)
    portfolio_path.write_text(json.dumps(portfolio_summary, indent=2, default=str) + "\n", encoding="utf-8")
    np.savez_compressed(
        cube_path,
        market_ids=np.array([entry.market_id for entry in entry_list], dtype=np.int64),
        entry_times=np.array(
            [int(entry.entry_time.timestamp()) for entry in entry_list], dtype=np.int64
        ),
        scheduled_end_times=np.array(
            [int(entry.end_date.timestamp()) if entry.end_date is not None else np.iinfo(np.int64).max for entry in entry_list],
            dtype=np.int64,
        ),
        closed_times=np.array(
            [int(entry.closed_time.timestamp()) if entry.closed_time is not None else np.iinfo(np.int64).max for entry in entry_list],
            dtype=np.int64,
        ),
        entry_prices=np.array([entry.entry_price for entry in entry_list], dtype=np.float32),
        entry_fill_usd=np.array(
            [entry.entry_fill_usd for entry in entry_list], dtype=np.float64
        ),
        entry_levels=np.array(
            [entry_level(entry.entry_price) for entry in entry_list], dtype=np.int8
        ),
        historical_volumes=np.array(
            [entry.historical_volume for entry in entry_list], dtype=np.float64
        ),
        underdog_won=np.array(
            [entry.underdog_side == entry.winner_side for entry in entry_list], dtype=bool
        ),
        hold_returns=hold_returns,
        returns=strategy_cube,
        exit_times=exit_times,
        exit_prices=exit_prices,
        exit_fill_usd=exit_fill_cube,
        exit_codes=exit_codes,
        take_profits=np.array(take_profits, dtype=np.float32),
        stop_losses=np.array(stop_losses, dtype=np.float32),
    )

    levels = [row["entry_level_cents"] for row in best_rows]
    test_rois = [100 * row["test_roi"] for row in best_rows]
    fig, (performance, parameters) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    colors = ["#18864b" if value >= 0 else "#c23b3b" for value in test_rois]
    performance.bar(levels, test_rois, color=colors, width=0.85)
    performance.axhline(0, color="#222222", linewidth=1)
    performance.set_ylabel("Holdout ROI (%)")
    performance.set_yscale("symlog", linthresh=10)
    performance.set_title("Optimized Underdog Brackets by Entry Price (70/30 chronological split)")
    performance.grid(axis="y", alpha=0.25)
    take_profit_line = parameters.plot(
        levels,
        [row["take_profit"] for row in best_rows],
        marker="o",
        color="#1f77b4",
        label="Take-profit multiple",
    )
    stop_axis = parameters.twinx()
    stop_line = stop_axis.plot(
        levels,
        [row["stop_loss"] for row in best_rows],
        marker="o",
        color="#ff7f0e",
        label="Stop-loss multiple (0 = disabled)",
    )
    parameters.set_xlabel("Underdog entry price (cents)")
    parameters.set_ylabel("Selected multiplier")
    stop_axis.set_ylabel("Selected stop-loss multiplier")
    stop_axis.set_ylim(-0.005, 0.105)
    parameters.set_xticks(range(1, 50, 2))
    parameters.grid(alpha=0.25)
    parameters.legend(take_profit_line + stop_line, [line.get_label() for line in take_profit_line + stop_line])
    fig.tight_layout()
    graph_path = args.output_dir / "best_by_entry_level.png"
    fig.savefig(graph_path, dpi=180)
    plt.close(fig)

    summary = {
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "take_profit_grid": take_profits,
        "stop_loss_grid": stop_losses,
        "entries": len(entries),
        "chronological_split_time": str(split_time),
        "levels_with_train_and_test_results": len(best_rows),
        "grid_results": str(grid_path),
        "best_by_entry_level": str(best_path),
        "graph": str(graph_path),
        "holdout_policy_trades": str(policy_path),
        "holdout_portfolio_summary": str(portfolio_path),
        "strategy_cube": str(cube_path),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
