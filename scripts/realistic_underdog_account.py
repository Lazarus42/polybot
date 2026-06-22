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
import bisect
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
SIZING_POLICIES = (
    "flat_one",
    "availability",
    "ev_weighted",
    "lcb_weighted",
    "hybrid_floor_ev",
    "hybrid_floor_lcb",
    "equal_positive_bucket",
    "fractional_kelly",
    "forecast_paced",
)
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


def fit_exit_candidates(
    levels: np.ndarray,
    candidate_returns: np.ndarray,
    fit_mask: np.ndarray,
    min_trades: int,
) -> dict[int, int]:
    result = {}
    for level in range(1, 50):
        indexes = np.where(fit_mask & (levels == level))[0]
        if len(indexes) < min_trades:
            continue
        means = np.nanmean(candidate_returns[indexes], axis=0)
        if np.all(np.isnan(means)):
            continue
        result[level] = int(np.nanargmax(means))
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


def quantize_price(price: float, tick: float) -> float:
    if tick <= 0:
        return price
    ticks = math.floor(price / tick + 0.5)
    return min(1.0 - tick, max(tick, ticks * tick))


def fill_contract_capacity(
    fill_usd: float,
    price: float,
    participation: float,
    contract_step: float,
) -> Decimal:
    if fill_usd <= 0 or price <= 0 or participation <= 0:
        return Decimal("0")
    raw = Decimal(str(fill_usd * participation)) / Decimal(str(price))
    if contract_step <= 0:
        return raw
    step = Decimal(str(contract_step))
    return (raw / step).to_integral_value(rounding=ROUND_FLOOR) * step


def attach_exit_paths(
    arrays: dict[str, np.ndarray],
    data_dir: Path,
    model: dict[str, float],
) -> None:
    """Attach post-entry same-side fill paths used for partial exit replay."""
    if "sides" not in arrays:
        raise SystemExit("strategy cube missing underdog_sides; rerun optimize_underdog_bracket.py")
    fills_path = data_dir / "fills_sorted.parquet"
    if not fills_path.exists():
        raise SystemExit(f"Missing clustered fill dataset: {fills_path}")

    indexes_by_market: dict[int, list[int]] = defaultdict(list)
    for index, market_id in enumerate(arrays["market_ids"]):
        indexes_by_market[int(market_id)].append(index)
    market_ids = sorted(indexes_by_market)
    if not market_ids:
        arrays["exit_path_offsets"] = np.zeros(1, dtype=np.int64)
        arrays["exit_path_times"] = np.array([], dtype=np.int64)
        arrays["exit_path_prices"] = np.array([], dtype=np.float32)
        arrays["exit_path_usd"] = np.array([], dtype=np.float64)
        return

    connection = duckdb.connect()
    connection.execute("SET threads = 1")
    connection.execute("CREATE TEMP TABLE selected_markets(market_id BIGINT)")
    connection.executemany(
        "INSERT INTO selected_markets VALUES (?)",
        [(market_id,) for market_id in market_ids],
    )
    path = str(fills_path.resolve()).replace("'", "''")
    cursor = connection.execute(
        f"""
        SELECT f.market_id, f.timestamp, f.side, f.price, f.usd_amount
        FROM read_parquet('{path}') AS f
        JOIN selected_markets AS s USING (market_id)
        ORDER BY f.market_id, f.timestamp
        """
    )

    fills_by_market_side: dict[tuple[int, str], list[tuple[int, float, float]]] = defaultdict(list)
    for batch in iter(lambda: cursor.fetchmany(100_000), []):
        for market_id, timestamp, side, price, usd_amount in batch:
            fills_by_market_side[(int(market_id), str(side))].append((
                int(timestamp.timestamp()),
                quantize_price(float(price), model["price_tick"]),
                float(usd_amount),
            ))

    offsets = [0]
    times: list[int] = []
    prices: list[float] = []
    usd_values: list[float] = []
    sorted_fill_times: dict[tuple[int, str], list[int]] = {}
    for key, rows in fills_by_market_side.items():
        rows.sort(key=lambda row: row[0])
        sorted_fill_times[key] = [row[0] for row in rows]

    for index, market_id in enumerate(arrays["market_ids"]):
        key = (int(market_id), str(arrays["sides"][index]))
        rows = fills_by_market_side.get(key, [])
        row_times = sorted_fill_times.get(key, [])
        start = bisect.bisect_right(row_times, int(arrays["times"][index]))
        end = bisect.bisect_right(row_times, int(arrays["closed_times"][index]))
        for timestamp_value, price, usd_amount in rows[start:end]:
            times.append(timestamp_value)
            prices.append(price)
            usd_values.append(usd_amount)
        offsets.append(len(times))
    arrays["exit_path_offsets"] = np.asarray(offsets, dtype=np.int64)
    arrays["exit_path_times"] = np.asarray(times, dtype=np.int64)
    arrays["exit_path_prices"] = np.asarray(prices, dtype=np.float32)
    arrays["exit_path_usd"] = np.asarray(usd_values, dtype=np.float64)


def build_exit_legs(
    index: int,
    selector: int | tuple[int, int],
    contracts: Decimal,
    arrays: dict[str, np.ndarray],
    model: dict[str, float],
    scenario: dict[str, float],
) -> tuple[list[dict[str, Any]], Counter]:
    if not isinstance(selector, tuple):
        return build_candidate_exit_legs(index, selector, contracts, arrays, model, scenario)

    tp_index, sl_index = selector
    exit_time = int(arrays["exit_times"][index, tp_index, sl_index])
    exit_price = float(arrays["exit_prices"][index, tp_index, sl_index])
    exit_code = int(arrays["exit_codes"][index, tp_index, sl_index])
    closed_time = int(arrays["closed_times"][index])
    resolution_price = 1.0 if bool(arrays["won"][index]) else 0.0
    skipped = Counter()

    if not exit_code:
        return ([{
            "time": closed_time,
            "contracts": contracts,
            "price": resolution_price,
            "code": 0,
        }], skipped)

    if "exit_path_offsets" not in arrays:
        return ([{
            "time": exit_time,
            "contracts": contracts,
            "price": adverse_price(exit_price, scenario["exit_ticks"], model["price_tick"], False),
            "code": exit_code,
        }], skipped)

    remaining = contracts
    legs: list[dict[str, Any]] = []
    start = int(arrays["exit_path_offsets"][index])
    end = int(arrays["exit_path_offsets"][index + 1])
    for path_index in range(start, end):
        fill_time = int(arrays["exit_path_times"][path_index])
        if fill_time < exit_time:
            continue
        if fill_time > closed_time:
            break
        observed_price = float(arrays["exit_path_prices"][path_index])
        fill_price = adverse_price(
            observed_price,
            scenario["exit_ticks"],
            model["price_tick"],
            False,
        )
        capacity = fill_contract_capacity(
            float(arrays["exit_path_usd"][path_index]),
            observed_price,
            scenario["participation"],
            model["contract_step"],
        )
        if capacity <= 0:
            skipped["threshold_exit_zero_liquidity"] += 1
            continue
        sold = min(remaining, capacity)
        if sold <= 0:
            continue
        legs.append({
            "time": fill_time,
            "contracts": sold,
            "price": fill_price,
            "code": exit_code,
        })
        remaining -= sold
        if remaining <= 0:
            break

    if remaining > 0:
        if legs:
            skipped["threshold_exit_partial_then_resolution"] += 1
        else:
            skipped["threshold_exit_no_liquidity_then_resolution"] += 1
        legs.append({
            "time": closed_time,
            "contracts": remaining,
            "price": resolution_price,
            "code": 0,
        })
    elif len(legs) > 1:
        skipped["threshold_exit_multi_fill"] += 1
    return legs, skipped


def build_candidate_exit_legs(
    index: int,
    candidate_id: int,
    contracts: Decimal,
    arrays: dict[str, np.ndarray],
    model: dict[str, float],
    scenario: dict[str, float],
) -> tuple[list[dict[str, Any]], Counter]:
    policy = json.loads(str(arrays["candidate_policy_json"][candidate_id]))
    entry_price = float(arrays["prices"][index])
    entry_time = int(arrays["times"][index])
    closed_time = int(arrays["closed_times"][index])
    resolution_price = 1.0 if bool(arrays["won"][index]) else 0.0
    skipped = Counter()
    start = int(arrays.get("exit_path_offsets", np.array([0]))[index]) if "exit_path_offsets" in arrays else 0
    end = int(arrays["exit_path_offsets"][index + 1]) if "exit_path_offsets" in arrays else 0
    tranches = list(policy["tranches"])
    tranche_sold = [Decimal("0") for _ in tranches]
    remaining = contracts
    legs: list[dict[str, Any]] = []
    current_tranche = 0
    stop_loss = float(policy.get("stop_loss", 0.0))
    stop_price = entry_price * stop_loss if stop_loss > 0 else None

    for path_index in range(start, end):
        fill_time = int(arrays["exit_path_times"][path_index])
        if fill_time <= entry_time:
            continue
        if fill_time > closed_time or remaining <= 0:
            break
        observed_price = float(arrays["exit_path_prices"][path_index])
        if stop_price is not None and not legs and observed_price <= stop_price:
            fill_price = adverse_price(
                observed_price, scenario["exit_ticks"], model["price_tick"], False
            )
            capacity = fill_contract_capacity(
                float(arrays["exit_path_usd"][path_index]),
                observed_price,
                scenario["participation"],
                model["contract_step"],
            )
            sold = min(remaining, capacity)
            if sold > 0:
                legs.append({"time": fill_time, "contracts": sold, "price": fill_price, "code": 2})
                remaining -= sold
            if remaining > 0:
                skipped["stop_exit_partial_then_resolution"] += 1
                legs.append({
                    "time": closed_time,
                    "contracts": remaining,
                    "price": resolution_price,
                    "code": 0,
                })
            return legs, skipped

        capacity = fill_contract_capacity(
            float(arrays["exit_path_usd"][path_index]),
            observed_price,
            scenario["participation"],
            model["contract_step"],
        )
        if capacity <= 0:
            continue
        while current_tranche < len(tranches) and capacity > 0 and remaining > 0:
            tranche = tranches[current_tranche]
            target_price = entry_price * float(tranche["multiplier"])
            if target_price >= 0.995 or observed_price < target_price:
                break
            desired = contracts * Decimal(str(float(tranche["fraction"])))
            needed = max(Decimal("0"), desired - tranche_sold[current_tranche])
            sold = min(remaining, capacity, needed)
            if sold <= 0:
                current_tranche += 1
                continue
            fill_price = adverse_price(
                observed_price, scenario["exit_ticks"], model["price_tick"], False
            )
            legs.append({"time": fill_time, "contracts": sold, "price": fill_price, "code": 3})
            tranche_sold[current_tranche] += sold
            remaining -= sold
            capacity -= sold
            if tranche_sold[current_tranche] >= desired:
                current_tranche += 1

    if remaining > 0:
        if legs:
            skipped["candidate_exit_partial_then_resolution"] += 1
        else:
            skipped["candidate_exit_no_liquidity_then_resolution"] += 1
        legs.append({
            "time": closed_time,
            "contracts": remaining,
            "price": resolution_price,
            "code": 0,
        })
    elif len(legs) > 1:
        skipped["candidate_exit_multi_fill"] += 1
    return legs, skipped


def horizon_bucket(days: float) -> str:
    for bound in (1, 3, 7, 14, 30, 60, 90, 180):
        if days <= bound:
            return f"<={bound}d"
    return ">180d"


def price_regime_for_level(level: int) -> str:
    if level <= 5:
        return "01-05c"
    if level <= 15:
        return "06-15c"
    if level <= 30:
        return "16-30c"
    return "31-49c"


def liquidity_bucket(fill_usd: float) -> str:
    for bound in (10, 25, 50, 100, 250, 500, 1000, 2500):
        if fill_usd <= bound:
            return f"<={bound}"
    return ">2500"


def period_key_for_time(timestamp: int, budget_period: str) -> Any:
    if budget_period == "month":
        entry_date = datetime.fromtimestamp(timestamp, timezone.utc)
        return entry_date.year, entry_date.month
    if budget_period == "week":
        return int(week_id(float(timestamp)))
    raise ValueError(f"Unknown budget period: {budget_period}")


def period_bounds(timestamp: int, budget_period: str) -> tuple[int, int]:
    value = datetime.fromtimestamp(timestamp, timezone.utc)
    if budget_period == "week":
        start = int(week_start(int(week_id(float(timestamp)))).timestamp())
        return start, start + WEEK_SECONDS
    if budget_period == "month":
        start_dt = value.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if start_dt.month == 12:
            end_dt = start_dt.replace(year=start_dt.year + 1, month=1)
        else:
            end_dt = start_dt.replace(month=start_dt.month + 1)
        return int(start_dt.timestamp()), int(end_dt.timestamp())
    raise ValueError(f"Unknown budget period: {budget_period}")


def selector_return(
    index: int,
    selector: int | tuple[int, int],
    arrays: dict[str, np.ndarray],
) -> float:
    if isinstance(selector, tuple):
        tp, sl = selector
        return float(arrays["cube"][index, tp, sl])
    return float(arrays["candidate_returns"][index, selector])


def selector_exit_time(
    index: int,
    selector: int | tuple[int, int],
    arrays: dict[str, np.ndarray],
) -> int:
    if isinstance(selector, tuple):
        tp, sl = selector
        return int(arrays["exit_times"][index, tp, sl])
    return int(arrays["candidate_exit_times"][index, selector])


def opportunity_keys(index: int, arrays: dict[str, np.ndarray]) -> dict[str, str]:
    entry_time = int(arrays["times"][index])
    scheduled_days = max(0.0, (int(arrays["scheduled_end"][index]) - entry_time) / DAY)
    level = int(arrays["levels"][index])
    category = str(arrays["categories"][index])
    regime = price_regime_for_level(level)
    horizon = horizon_bucket(scheduled_days)
    liquidity = liquidity_bucket(float(arrays["entry_fill"][index]))
    return {
        "global": "global",
        "regime": f"regime:{regime}",
        "level": f"level:{level}",
        "category": f"category:{category}",
        "level_category": f"level:{level}|category:{category}",
        "rich": f"regime:{regime}|level:{level}|category:{category}|horizon:{horizon}|liquidity:{liquidity}",
    }


def is_eligible_opportunity(
    index: int,
    selectors: dict[int, int | tuple[int, int]],
    arrays: dict[str, np.ndarray],
    allowed_categories: set[str],
    max_horizon: float,
    allowed_levels: Optional[set[int]] = None,
) -> bool:
    level = int(arrays["levels"][index])
    if level not in selectors or (allowed_levels is not None and level not in allowed_levels):
        return False
    if str(arrays["categories"][index]) not in allowed_categories:
        return False
    scheduled_days = max(
        0.0,
        (int(arrays["scheduled_end"][index]) - int(arrays["times"][index])) / DAY,
    )
    return scheduled_days <= max_horizon


def selected_return_before(
    index: int,
    selectors: dict[int, int | tuple[int, int]],
    arrays: dict[str, np.ndarray],
    calibration_end: int,
) -> Optional[float]:
    selector = selectors.get(int(arrays["levels"][index]))
    if selector is None:
        return None
    if selector_exit_time(index, selector, arrays) > calibration_end:
        return None
    return selector_return(index, selector, arrays)


def summarize_returns(
    values: list[float],
    global_mean: float,
    shrink_k: float,
    lcb_z: float,
) -> dict[str, float]:
    count = len(values)
    if not count:
        return {
            "n": 0,
            "mean": 0.0,
            "shrunk": global_mean,
            "se": math.inf,
            "lcb": 0.0,
            "second": 0.0,
        }
    data = np.asarray(values, dtype=float)
    mean = float(data.mean())
    alpha = count / (count + shrink_k)
    shrunk = alpha * mean + (1.0 - alpha) * global_mean
    se = float(data.std(ddof=1) / math.sqrt(count)) if count > 1 else math.inf
    return {
        "n": count,
        "mean": mean,
        "shrunk": shrunk,
        "se": se,
        "lcb": shrunk - lcb_z * se if math.isfinite(se) else 0.0,
        "second": float(np.mean(data * data)),
    }


def fit_sizing_model(
    indexes: np.ndarray,
    calibration_end: int,
    selectors: dict[int, int | tuple[int, int]],
    arrays: dict[str, np.ndarray],
    allowed_categories: set[str],
    max_horizon: float,
    allowed_levels: Optional[set[int]] = None,
    budget_period: str = "week",
    shrink_k: float = 100.0,
    lcb_z: float = 1.0,
) -> dict[str, Any]:
    bucket_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    bucket_counts: dict[str, Counter] = defaultdict(Counter)
    periods = set()
    realized_values = []

    for index in indexes:
        if not is_eligible_opportunity(
            index, selectors, arrays, allowed_categories, max_horizon, allowed_levels
        ):
            continue
        periods.add(period_key_for_time(int(arrays["times"][index]), budget_period))
        keys = opportunity_keys(index, arrays)
        for granularity, key in keys.items():
            bucket_counts[granularity][key] += 1
        value = selected_return_before(index, selectors, arrays, calibration_end)
        if value is None:
            continue
        realized_values.append(value)
        for granularity, key in keys.items():
            bucket_values[granularity][key].append(value)

    period_count = max(1, len(periods))
    global_mean = float(np.mean(realized_values)) if realized_values else 0.0
    stats_by_granularity: dict[str, dict[str, dict[str, float]]] = {}
    for granularity, counts in bucket_counts.items():
        stats_by_granularity[granularity] = {}
        for key, count in counts.items():
            row = summarize_returns(
                bucket_values[granularity].get(key, []),
                global_mean,
                shrink_k,
                lcb_z,
            )
            row["opportunities"] = int(count)
            row["lambda"] = count / period_count
            stats_by_granularity[granularity][key] = row

    rich_stats = stats_by_granularity.get("rich", {})
    weight_sums = {
        metric: sum(max(0.0, row.get(metric, 0.0)) for row in rich_stats.values())
        for metric in ("shrunk", "lcb")
    }
    return {
        "period_count": period_count,
        "global_mean": global_mean,
        "global_lambda": sum(bucket_counts.get("global", Counter()).values()) / period_count,
        "stats": stats_by_granularity,
        "weight_sums": weight_sums,
        "positive_rich_buckets": {
            metric: sum(1 for row in rich_stats.values() if row.get(metric, 0.0) > 0)
            for metric in ("shrunk", "lcb")
        },
    }


def bucket_stat_for_index(
    index: int,
    arrays: dict[str, np.ndarray],
    sizing_model: dict[str, Any],
) -> dict[str, float]:
    keys = opportunity_keys(index, arrays)
    for granularity in ("rich", "level_category", "level", "regime", "category", "global"):
        row = sizing_model.get("stats", {}).get(granularity, {}).get(keys[granularity])
        if row is not None and row.get("opportunities", 0) > 0:
            return row
    return {
        "n": 0,
        "mean": 0.0,
        "shrunk": sizing_model.get("global_mean", 0.0),
        "se": math.inf,
        "lcb": 0.0,
        "second": 0.0,
        "opportunities": 0,
        "lambda": max(1.0, sizing_model.get("global_lambda", 1.0)),
    }


def planned_stake(
    index: int,
    arrays: dict[str, np.ndarray],
    policy: str,
    sizing_model: Optional[dict[str, Any]],
    period_budget: float,
    base_stake: float,
    availability_lambda: Optional[float],
    kelly_fraction: float,
    max_fraction: float,
    cash: float = math.inf,
    reserve_floor: float = 0.0,
    remaining_period_fraction: float = 1.0,
) -> float:
    if policy not in SIZING_POLICIES:
        raise ValueError(f"Unknown sizing policy: {policy}")
    if policy == "flat_one":
        return base_stake

    if sizing_model is None:
        lam = max(1.0, availability_lambda or 1.0)
        return period_budget / lam

    if policy == "availability":
        lam = max(1.0, sizing_model.get("global_lambda", availability_lambda or 1.0))
        return period_budget / lam

    row = bucket_stat_for_index(index, arrays, sizing_model)
    lam = max(1.0, row.get("lambda", 1.0))

    if policy == "fractional_kelly":
        second = max(row.get("second", 0.0), 1e-6)
        edge = max(0.0, row.get("lcb", 0.0))
        fraction = min(max_fraction, kelly_fraction * edge / second)
        return period_budget * fraction

    if policy == "forecast_paced":
        deployable = max(0.0, cash - reserve_floor)
        expected_remaining = max(1.0, row.get("lambda", 1.0) * max(0.05, remaining_period_fraction))
        edge = max(row.get("lcb", 0.0), 0.25 * max(0.0, row.get("shrunk", 0.0)))
        global_edge = max(abs(sizing_model.get("global_mean", 0.0)), 0.05)
        quality = max(0.25, min(3.0, 1.0 + edge / global_edge))
        return base_stake + deployable / expected_remaining * quality

    metric = "lcb" if "lcb" in policy or policy == "equal_positive_bucket" else "shrunk"
    weight = max(0.0, row.get(metric, 0.0))
    floor = base_stake if policy.startswith("hybrid_floor") else 0.0
    if weight <= 0:
        return floor

    if policy == "equal_positive_bucket":
        buckets = max(1, sizing_model["positive_rich_buckets"].get(metric, 0))
        return period_budget / buckets / lam

    overlay_fraction = 1.0
    if policy.startswith("hybrid_floor"):
        overlay_fraction = 0.70
    denom = sizing_model["weight_sums"].get(metric, 0.0)
    if denom <= 0:
        return floor
    overlay = period_budget * overlay_fraction * weight / denom / lam
    return floor + overlay


def run_account(
    indexes: np.ndarray,
    end_time: int,
    selectors: dict[int, int | tuple[int, int]],
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
    allowed_levels: Optional[set[int]] = None,
    budget_period: str = "week",
    sizing_policy: Optional[str] = None,
    sizing_model: Optional[dict[str, Any]] = None,
    max_stake: float = math.inf,
    kelly_fraction: float = 0.10,
    max_fraction: float = 0.02,
    reserve_fraction: float = 0.25,
    min_stake: float = 1.0,
    min_minutes_to_close: float = 60.0,
    max_category_locked_fraction: float = 0.35,
    max_regime_locked_fraction: float = 0.35,
    drawdown_throttle_start: float = math.inf,
    drawdown_throttle_stop: float = math.inf,
    drawdown_throttle_min_scale: float = 0.0,
    safety_gates: bool = True,
) -> dict[str, Any]:
    scenario = SCENARIOS[scenario_name]
    if sizing_policy is None:
        sizing_policy = "availability" if availability_lambda is not None else "flat_one"
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
    period_spend: dict[Any, float] = defaultdict(float)
    weekly_records: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    category_records: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    horizon_records: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    skipped = Counter()
    closed_records = []
    closed_cost_records = []
    closed_record_times = []
    exit_family_counts: Counter = Counter()
    candidate_name_counts: Counter = Counter()
    exit_sequence = 0

    def account_value() -> float:
        return cash + sum(position["remaining_debit"] for position in open_positions.values())

    def locked_for(field: str, value: str) -> float:
        return sum(
            position["remaining_debit"]
            for position in open_positions.values()
            if str(position.get(field)) == value
        )

    def drawdown_scale() -> float:
        if not math.isfinite(drawdown_throttle_start):
            return 1.0
        value = account_value()
        current_drawdown = (peak - value) / peak if peak else 0.0
        if current_drawdown <= drawdown_throttle_start:
            return 1.0
        if (
            not math.isfinite(drawdown_throttle_stop)
            or drawdown_throttle_stop <= drawdown_throttle_start
            or current_drawdown >= drawdown_throttle_stop
        ):
            return max(0.0, drawdown_throttle_min_scale)
        span = drawdown_throttle_stop - drawdown_throttle_start
        progress = (current_drawdown - drawdown_throttle_start) / span
        return max(
            0.0,
            1.0 - progress * (1.0 - max(0.0, drawdown_throttle_min_scale)),
        )

    def release(until: int) -> None:
        nonlocal cash, realized_profit, fees, dollar_days, peak, max_drawdown
        while exits and exits[0][0] <= until:
            _, _, position_id, leg = heapq.heappop(exits)
            position = open_positions.get(position_id)
            if position is None:
                continue
            proceeds, exit_fee = exact_exit(
                leg["contracts"], leg["price"], leg["code"], model
            )
            if position["contracts"] <= 0:
                cost_basis = Decimal("0")
            else:
                cost_basis = (
                    Decimal(str(position["debit"]))
                    * leg["contracts"]
                    / position["contracts"]
                )
            profit = proceeds - float(cost_basis)
            cash += proceeds
            realized_profit += profit
            fees += exit_fee
            held_days = max(0.0, (leg["time"] - position["entry_time"]) / DAY)
            dollar_days += float(cost_basis) * held_days
            position["remaining_contracts"] -= leg["contracts"]
            position["remaining_debit"] -= float(cost_basis)
            position["closed_cost_basis"] += float(cost_basis)
            position["realized_profit"] += profit
            entry_week = int(week_id(float(position["entry_time"])))
            weekly_records[entry_week]["profit"] += profit
            category_records[position["category"]]["profit"] += profit
            horizon_records[position["horizon_bucket"]]["profit"] += profit
            if position["remaining_contracts"] <= 0:
                open_positions.pop(position_id, None)
                weekly_records[entry_week]["exits"] += 1
                category_records[position["category"]]["exits"] += 1
                horizon_records[position["horizon_bucket"]]["exits"] += 1
                closed_records.append(position["realized_profit"])
                closed_cost_records.append(position["closed_cost_basis"])
                closed_record_times.append(leg["time"])
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
        if isinstance(selector, tuple):
            exit_family = "legacy_tp_sl"
            candidate_name = "legacy_tp_sl"
        else:
            exit_family = str(arrays.get("candidate_families", np.array(["candidate"]))[selector])
            candidate_name = str(arrays.get("candidate_names", np.array([f"candidate_{selector}"]))[selector])
        if allowed_levels is not None and level not in allowed_levels:
            skipped["price_gate"] += 1
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
        if safety_gates and int(arrays["closed_times"][index]) - entry_time < min_minutes_to_close * 60:
            skipped["safety_too_close_to_close"] += 1
            continue
        week = int(week_id(float(entry_time)))
        period_key = period_key_for_time(entry_time, budget_period)
        remaining_period = max(0.0, weekly_budget - period_spend[period_key])
        if remaining_period <= 0:
            skipped[f"{budget_period}_budget"] += 1
            continue
        regime = price_regime_for_level(level)
        reserve_floor = initial_cash * reserve_fraction
        if safety_gates and cash <= reserve_floor:
            skipped["reserve_floor"] += 1
            continue
        category_headroom = initial_cash * max_category_locked_fraction - locked_for("category", category)
        if safety_gates and category_headroom <= 0:
            skipped["category_locked_cap"] += 1
            continue
        regime_headroom = initial_cash * max_regime_locked_fraction - locked_for("price_regime", regime)
        if safety_gates and regime_headroom <= 0:
            skipped["price_regime_locked_cap"] += 1
            continue
        period_start, period_end = period_bounds(entry_time, budget_period)
        remaining_fraction = max(0.0, (period_end - entry_time) / max(1, period_end - period_start))
        target = planned_stake(
            index,
            arrays,
            sizing_policy,
            sizing_model,
            weekly_budget,
            stake,
            availability_lambda,
            kelly_fraction,
            max_fraction,
            cash=cash,
            reserve_floor=reserve_floor,
            remaining_period_fraction=remaining_fraction,
        )
        throttle_scale = drawdown_scale()
        if throttle_scale <= 0:
            skipped["drawdown_throttle"] += 1
            continue
        if throttle_scale < 1.0:
            skipped["drawdown_throttle_scaled"] += 1
            target *= throttle_scale
        target = min(
            target,
            remaining_period,
            max(0.0, cash - reserve_floor if safety_gates else cash),
            max_stake,
            category_headroom if safety_gates else math.inf,
            regime_headroom if safety_gates else math.inf,
        )
        fill_cap = float(arrays["entry_fill"][index]) * scenario["participation"]
        if safety_gates and fill_cap < min_stake:
            skipped["safety_entry_liquidity_below_min_stake"] += 1
            continue
        if fill_cap < target:
            skipped["entry_liquidity_limited"] += 1
        target = min(target, fill_cap)
        if safety_gates and target < min_stake:
            skipped["safety_target_below_min_stake"] += 1
            continue
        entry_price = adverse_price(
            float(arrays["prices"][index]), scenario["entry_ticks"], model["price_tick"], True
        )
        contracts, debit, entry_fee = exact_entry(target, entry_price, model)
        if not contracts:
            skipped["minimum_order"] += 1
            continue
        exit_legs, exit_skips = build_exit_legs(
            index, selector, contracts, arrays, model, scenario
        )
        skipped.update(exit_skips)
        cash -= debit
        fees += entry_fee
        deployed += debit
        period_spend[period_key] += debit
        opened += 1
        exit_family_counts[exit_family] += 1
        candidate_name_counts[candidate_name] += 1
        bucket = horizon_bucket(scheduled_horizon)
        final_leg = exit_legs[-1]
        position = {
            "position_id": int(index),
            "market_id": int(arrays["market_ids"][index]),
            "entry_time": entry_time,
            "exit_time": final_leg["time"],
            "entry_price": entry_price,
            "contracts": contracts,
            "remaining_contracts": contracts,
            "debit": debit,
            "remaining_debit": debit,
            "closed_cost_basis": 0.0,
            "realized_profit": 0.0,
            "category": category,
            "price_regime": regime,
            "scheduled_horizon_days": scheduled_horizon,
            "horizon_bucket": bucket,
        }
        open_positions[int(index)] = position
        for leg in exit_legs:
            exit_sequence += 1
            heapq.heappush(exits, (leg["time"], exit_sequence, int(index), leg))
        weekly_records[week]["entries"] += 1
        weekly_records[week]["deployed"] += debit
        category_records[category]["entries"] += 1
        category_records[category]["deployed"] += debit
        horizon_records[bucket]["entries"] += 1
        horizon_records[bucket]["deployed"] += debit

    release(end_time)
    for position in open_positions.values():
        held_days = max(0.0, (end_time - position["entry_time"]) / DAY)
        dollar_days += position["remaining_debit"] * held_days
        category_records[position["category"]]["locked_end"] += position["remaining_debit"]
        horizon_records[position["horizon_bucket"]]["locked_end"] += position["remaining_debit"]
    locked = sum(position["remaining_debit"] for position in open_positions.values())
    eventual_open_profit = 0.0
    unresolved_rows = []
    future_proceeds: dict[int, float] = defaultdict(float)
    for _, _, position_id, leg in exits:
        if position_id in open_positions:
            proceeds, _ = exact_exit(leg["contracts"], leg["price"], leg["code"], model)
            future_proceeds[position_id] += proceeds
    for position in open_positions.values():
        eventual_profit = future_proceeds[position["position_id"]] - position["remaining_debit"]
        eventual_open_profit += eventual_profit
        unresolved_rows.append({
            "market_id": position["market_id"],
            "entry_time": datetime.fromtimestamp(position["entry_time"], timezone.utc).isoformat(),
            "planned_exit_time": datetime.fromtimestamp(position["exit_time"], timezone.utc).isoformat(),
            "category": position["category"],
            "scheduled_horizon_days": position["scheduled_horizon_days"],
            "locked_capital": position["remaining_debit"],
            "eventual_profit_audit_only": eventual_profit,
        })
    total_value = cash + locked
    profits = np.asarray(closed_records, dtype=float)
    gross_win = float(profits[profits > 0].sum()) if len(profits) else 0.0
    gross_loss = float(profits[profits < 0].sum()) if len(profits) else 0.0
    sorted_profits = np.sort(profits)[::-1] if len(profits) else np.array([], dtype=float)
    profit_without_top = {
        f"without_top_{count}": float(sorted_profits[count:].sum()) if len(sorted_profits) > count else 0.0
        for count in (1, 3, 5, 10)
    }
    closed_costs = np.asarray(closed_cost_records, dtype=float)
    capped_profits = {}
    for cap_multiple in (5, 10, 20):
        if len(profits) and len(closed_costs) == len(profits):
            capped = np.minimum(profits, closed_costs * cap_multiple)
            capped_profits[f"profit_capped_at_{cap_multiple}x_cost"] = float(capped.sum())
        else:
            capped_profits[f"profit_capped_at_{cap_multiple}x_cost"] = 0.0
    first_large_winner_days = None
    first_entry_time = int(arrays["times"][indexes[0]]) if len(indexes) else end_time
    for profit, exit_time in zip(closed_records, closed_record_times):
        if profit >= 5.0:
            first_large_winner_days = max(0.0, (exit_time - first_entry_time) / DAY)
            break
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
        "sizing_policy": sizing_policy,
        "reserve_fraction": reserve_fraction,
        "min_stake": min_stake,
        "max_category_locked_fraction": max_category_locked_fraction,
        "max_regime_locked_fraction": max_regime_locked_fraction,
        "drawdown_throttle_start": drawdown_throttle_start,
        "drawdown_throttle_stop": drawdown_throttle_stop,
        "drawdown_throttle_min_scale": drawdown_throttle_min_scale,
        "resolved_exits": len(closed_records),
        "hit_rate": float((profits > 0).mean()) if len(profits) else 0.0,
        "gross_winnings": gross_win,
        "gross_losses": gross_loss,
        "max_single_profit": float(sorted_profits[0]) if len(sorted_profits) else 0.0,
        **profit_without_top,
        **capped_profits,
        "time_to_first_large_winner_days": first_large_winner_days,
        "exit_family_counts": dict(exit_family_counts),
        "candidate_name_counts_top10": dict(candidate_name_counts.most_common(10)),
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
    selectors: dict[int, int | tuple[int, int]],
    arrays: dict[str, np.ndarray],
) -> dict[str, set[str]]:
    returns: dict[str, list[float]] = defaultdict(list)
    for index in calibration_indexes:
        selector = selectors.get(int(arrays["levels"][index]))
        if selector is None:
            continue
        if selector_exit_time(index, selector, arrays) <= calibration_end:
            returns[str(arrays["categories"][index])].append(
                selector_return(index, selector, arrays)
            )
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
    required = {
        "scheduled_end_times",
        "closed_times",
        "exit_times",
        "exit_prices",
        "exit_fill_usd",
        "exit_codes",
        "underdog_sides",
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
    arrays["categories"] = load_market_categories(args.data_dir, arrays["market_ids"])
    weeks = np.asarray(week_id(arrays["times"]))
    all_weeks = np.arange(int(weeks.min()), int(weeks.max()) + 1)
    model = execution_model(args.report_dir)
    attach_exit_paths(arrays, args.data_dir, model)
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
        if "candidate_returns" in arrays:
            selectors = fit_exit_candidates(
                arrays["levels"], arrays["candidate_returns"], fit_mask, args.min_fit_trades
            )
        else:
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
