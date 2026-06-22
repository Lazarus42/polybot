#!/usr/bin/env python3
"""Walk-forward OOS tests for exact-price market-making underdog brackets.

This strategy buys underdogs below 50c and learns exact entry-level exit
brackets such as "buy 5c, sell 4c stop or 6c target". The regular model uses
coarse event/liquidity buckets. The advanced model additionally uses recent
pre-entry market activity from fills_sorted.parquet.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

import duckdb
import numpy as np
import pandas as pd

from long_holdout_weighting_experiments import load_arrays
from online_underdog_allocation import CATEGORIES, execution_model
from optimize_underdog_bracket import max_contracts_for_budget
from realistic_underdog_account import (
    DAY,
    SCENARIOS,
    adverse_price,
    attach_exit_paths,
    exact_exit,
    fill_contract_capacity,
    horizon_bucket,
    price_regime_for_level,
    write_csv,
)
from walk_forward_oos import add_months, month_floor, period_rows

SELECTION_SCORES = (
    "mean",
    "sharpe",
    "lcb",
    "capped_mean",
    "capped_sharpe",
)


def liquidity_bucket(value: float) -> str:
    for bound in (1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000):
        if value <= bound:
            return f"<={bound}"
    return ">5000"


def volume_bucket(value: float) -> str:
    for bound in (100, 1000, 5000, 25000, 100000, 500000, 1_000_000):
        if value <= bound:
            return f"<={bound}"
    return ">1000000"


def signed_bucket(value: float, cuts: tuple[float, ...]) -> str:
    for bound in cuts:
        if value <= bound:
            return f"<={bound:g}"
    return f">{cuts[-1]:g}"


def candidate_brackets(level: int, max_tp_level: int = 99) -> list[tuple[Optional[int], int]]:
    stop_steps = [1, 2, 3, 5, 8, 13, 21]
    stops: list[Optional[int]] = [None]
    stops.extend(sorted({level - step for step in stop_steps if level - step >= 1}, reverse=True))
    tp_steps = [1, 2, 3, 5, 8, 13, 21, 34]
    targets = {level + step for step in tp_steps if level + step <= max_tp_level}
    if level <= 10:
        targets.update({level * mult for mult in (2, 3, 5, 10) if level * mult <= max_tp_level})
    return [(stop, target) for stop in stops for target in sorted(targets) if target > level]


def bracket_name(bracket: tuple[Optional[int], int]) -> str:
    stop, target = bracket
    return f"sl_{'none' if stop is None else stop}c_tp_{target}c"


def parse_bracket(name: str) -> tuple[Optional[int], int]:
    stop_text, target_text = name.removeprefix("sl_").split("c_tp_")
    stop = None if stop_text == "none" else int(stop_text)
    return stop, int(target_text.removesuffix("c"))


def exact_entry(stake: float, price: float, model: dict[str, float]) -> tuple[Decimal, float, float]:
    contracts, debit = max_contracts_for_budget(
        stake, price, model["fee_coefficient"], model["contract_step"]
    )
    if contracts <= 0 or debit <= 0:
        return Decimal("0"), 0.0, 0.0
    fee = float(debit) - float(contracts) * price
    return contracts, float(debit), fee


def path_bounds(index: int, arrays: dict[str, np.ndarray]) -> tuple[int, int]:
    return int(arrays["exit_path_offsets"][index]), int(arrays["exit_path_offsets"][index + 1])


def bracket_unit_return(
    index: int,
    bracket: tuple[Optional[int], int],
    arrays: dict[str, np.ndarray],
    model: dict[str, float],
    scenario_name: str,
    stake: float = 1.0,
) -> float:
    scenario = SCENARIOS[scenario_name]
    entry_price = adverse_price(
        float(arrays["prices"][index]), scenario["entry_ticks"], model["price_tick"], True
    )
    contracts, debit, _ = exact_entry(stake, entry_price, model)
    if not contracts or debit <= 0:
        return np.nan

    stop_level, target_level = bracket
    stop_price = None if stop_level is None else stop_level / 100.0
    target_price = target_level / 100.0
    closed_time = int(arrays["closed_times"][index])
    resolution_price = 1.0 if bool(arrays["won"][index]) else 0.0
    start, end = path_bounds(index, arrays)

    triggered = False
    remaining = contracts
    proceeds_total = 0.0
    for path_index in range(start, end):
        observed_price = float(arrays["exit_path_prices"][path_index])
        if not triggered:
            if observed_price >= target_price:
                triggered = True
            elif stop_price is not None and observed_price <= stop_price:
                triggered = True
            else:
                continue
        capacity = fill_contract_capacity(
            float(arrays["exit_path_usd"][path_index]),
            observed_price,
            scenario["participation"],
            model["contract_step"],
        )
        if capacity <= 0:
            continue
        sold = min(remaining, capacity)
        exit_price = adverse_price(observed_price, scenario["exit_ticks"], model["price_tick"], False)
        proceeds, _ = exact_exit(sold, exit_price, 3, model)
        proceeds_total += proceeds
        remaining -= sold
        if remaining <= 0:
            break

    if remaining > 0:
        proceeds, _ = exact_exit(remaining, resolution_price, 0, model)
        proceeds_total += proceeds
    return proceeds_total / debit - 1.0


def bracket_exit_legs(
    index: int,
    bracket: tuple[Optional[int], int],
    contracts: Decimal,
    arrays: dict[str, np.ndarray],
    model: dict[str, float],
    scenario_name: str,
) -> tuple[list[dict[str, Any]], Counter]:
    scenario = SCENARIOS[scenario_name]
    stop_level, target_level = bracket
    stop_price = None if stop_level is None else stop_level / 100.0
    target_price = target_level / 100.0
    closed_time = int(arrays["closed_times"][index])
    resolution_price = 1.0 if bool(arrays["won"][index]) else 0.0
    start, end = path_bounds(index, arrays)
    skipped = Counter()
    triggered = False
    trigger_kind = "resolution"
    remaining = contracts
    legs: list[dict[str, Any]] = []

    for path_index in range(start, end):
        fill_time = int(arrays["exit_path_times"][path_index])
        observed_price = float(arrays["exit_path_prices"][path_index])
        if not triggered:
            if observed_price >= target_price:
                triggered = True
                trigger_kind = "take_profit"
            elif stop_price is not None and observed_price <= stop_price:
                triggered = True
                trigger_kind = "stop_loss"
            else:
                continue
        capacity = fill_contract_capacity(
            float(arrays["exit_path_usd"][path_index]),
            observed_price,
            scenario["participation"],
            model["contract_step"],
        )
        if capacity <= 0:
            skipped[f"{trigger_kind}_zero_liquidity"] += 1
            continue
        sold = min(remaining, capacity)
        if sold <= 0:
            continue
        exit_price = adverse_price(observed_price, scenario["exit_ticks"], model["price_tick"], False)
        legs.append({"time": fill_time, "contracts": sold, "price": exit_price, "code": 3})
        remaining -= sold
        if remaining <= 0:
            break

    if remaining > 0:
        skipped[f"{trigger_kind}_then_resolution"] += 1
        legs.append({"time": closed_time, "contracts": remaining, "price": resolution_price, "code": 0})
    elif len(legs) > 1:
        skipped[f"{trigger_kind}_multi_fill"] += 1
    return legs, skipped


def regular_features(index: int, arrays: dict[str, np.ndarray]) -> dict[str, str]:
    level = int(arrays["levels"][index])
    category = str(arrays["categories"][index])
    horizon_days = max(0.0, (int(arrays["scheduled_end"][index]) - int(arrays["times"][index])) / DAY)
    entry_liq = liquidity_bucket(float(arrays["entry_fill"][index]))
    hist_volume = volume_bucket(float(arrays.get("historical_volumes", np.zeros(len(arrays["levels"])))[index]))
    regime = price_regime_for_level(level)
    horizon = horizon_bucket(horizon_days)
    return {
        "level": f"level:{level}",
        "level_liq": f"level:{level}|liq:{entry_liq}",
        "level_category": f"level:{level}|cat:{category}",
        "level_horizon": f"level:{level}|horizon:{horizon}",
        "rich": f"level:{level}|cat:{category}|horizon:{horizon}|liq:{entry_liq}|histvol:{hist_volume}",
        "regime": f"regime:{regime}",
    }


def advanced_features(index: int, arrays: dict[str, np.ndarray]) -> dict[str, str]:
    features = regular_features(index, arrays)
    volume_24h = volume_bucket(float(arrays["recent_volume_24h"][index]))
    count_24h = liquidity_bucket(float(arrays["recent_count_24h"][index]))
    velocity = signed_bucket(float(arrays["recent_price_move_24h"][index]), (-0.10, -0.03, 0.0, 0.03, 0.10))
    features["advanced"] = f"{features['rich']}|vol24:{volume_24h}|cnt24:{count_24h}|vel24:{velocity}"
    return features


def feature_keys(index: int, arrays: dict[str, np.ndarray], feature_mode: str) -> list[str]:
    features = advanced_features(index, arrays) if feature_mode == "advanced" else regular_features(index, arrays)
    order = ["advanced", "rich", "level_liq", "level_category", "level_horizon", "level"]
    return [features[name] for name in order if name in features]


HOUR = 60 * 60
DAY_SECONDS = 24 * HOUR

# Feature columns produced by compute_window_features / attach_recent_features.
RECENT_FEATURE_KEYS = [
    "recent_volume_24h",
    "recent_count_24h",
    "recent_price_move_24h",
    "recent_price_move_1h",
    "recent_price_move_6h",
    "recent_price_move_7d",
    "recent_volatility_24h",
    "recent_accel_24h",
    "recent_flow_imbalance_24h",
]


def compute_window_features(
    row_times: list[int],
    prices: np.ndarray,
    usd_prefix: np.ndarray,
    signed_usd_prefix: np.ndarray,
    entry_time: int,
) -> dict[str, float]:
    """Pure window-feature math for one (market, side) fill history before an entry.

    ``row_times`` is ascending; ``prices`` aligns with it; ``usd_prefix`` and
    ``signed_usd_prefix`` are cumulative sums of USD and tick-signed USD respectively.
    Only fills strictly before ``entry_time`` are used (no look-ahead).
    """
    zero = {k: 0.0 for k in RECENT_FEATURE_KEYS}
    right = bisect.bisect_left(row_times, entry_time)
    if right <= 0:
        return zero

    def left_at(seconds_ago: int) -> int:
        return bisect.bisect_left(row_times, entry_time - seconds_ago)

    def move_over(seconds_ago: int) -> float:
        left = left_at(seconds_ago)
        if right <= left:
            return 0.0
        return float(prices[right - 1] - prices[left])

    l24 = left_at(DAY_SECONDS)
    l48 = left_at(2 * DAY_SECONDS)
    feats = dict(zero)
    feats["recent_price_move_1h"] = move_over(HOUR)
    feats["recent_price_move_6h"] = move_over(6 * HOUR)
    feats["recent_price_move_24h"] = move_over(DAY_SECONDS)
    feats["recent_price_move_7d"] = move_over(7 * DAY_SECONDS)

    # 24h volume / count (matches prior behaviour).
    usd_before = float(usd_prefix[l24 - 1]) if l24 > 0 else 0.0
    feats["recent_volume_24h"] = float(usd_prefix[right - 1]) - usd_before
    feats["recent_count_24h"] = float(right - l24)

    # 24h realized volatility of fill prices.
    if right - l24 >= 2:
        feats["recent_volatility_24h"] = float(np.std(prices[l24:right]))

    # Acceleration: move over the last 24h minus move over the prior 24h.
    if l24 > l48:
        prior = float(prices[l24 - 1] - prices[l48])
    else:
        prior = 0.0
    feats["recent_accel_24h"] = feats["recent_price_move_24h"] - prior

    # Order-flow imbalance: tick-signed USD / total USD over 24h, in [-1, 1].
    signed_before = float(signed_usd_prefix[l24 - 1]) if l24 > 0 else 0.0
    signed = float(signed_usd_prefix[right - 1]) - signed_before
    if feats["recent_volume_24h"] > 0:
        feats["recent_flow_imbalance_24h"] = signed / feats["recent_volume_24h"]
    return feats


def attach_recent_features(arrays: dict[str, np.ndarray], data_dir: Path) -> None:
    fills_path = data_dir / "fills_sorted.parquet"
    if not fills_path.exists():
        raise SystemExit(f"Missing fills dataset: {fills_path}")

    indexes_by_market: dict[int, list[int]] = defaultdict(list)
    for index, market_id in enumerate(arrays["market_ids"]):
        indexes_by_market[int(market_id)].append(index)
    market_ids = sorted(indexes_by_market)
    connection = duckdb.connect()
    connection.execute("SET threads = 1")
    connection.execute("CREATE TEMP TABLE selected_markets(market_id BIGINT)")
    connection.executemany("INSERT INTO selected_markets VALUES (?)", [(market_id,) for market_id in market_ids])
    path = str(fills_path.resolve()).replace("'", "''")
    cursor = connection.execute(
        f"""
        SELECT f.market_id, f.timestamp, f.side, f.price, f.usd_amount
        FROM read_parquet('{path}') AS f
        JOIN selected_markets AS s USING (market_id)
        ORDER BY f.market_id, f.side, f.timestamp
        """
    )

    fills: dict[tuple[int, str], list[tuple[int, float, float]]] = defaultdict(list)
    for batch in iter(lambda: cursor.fetchmany(200_000), []):
        for market_id, timestamp, side, price, usd_amount in batch:
            fills[(int(market_id), str(side))].append((int(timestamp.timestamp()), float(price), float(usd_amount)))

    times_by_key: dict[tuple[int, str], list[int]] = {}
    prices_by_key: dict[tuple[int, str], np.ndarray] = {}
    usd_prefix_by_key: dict[tuple[int, str], np.ndarray] = {}
    signed_prefix_by_key: dict[tuple[int, str], np.ndarray] = {}
    for key, rows in fills.items():
        prices = np.asarray([row[1] for row in rows], dtype=float)
        usd = np.asarray([row[2] for row in rows], dtype=float)
        # Tick rule: classify each fill as buy(+)/sell(-) by price change vs the prior fill.
        tick_sign = np.sign(np.diff(prices, prepend=prices[0] if len(prices) else 0.0))
        times_by_key[key] = [row[0] for row in rows]
        prices_by_key[key] = prices
        usd_prefix_by_key[key] = np.cumsum(usd, dtype=float)
        signed_prefix_by_key[key] = np.cumsum(tick_sign * usd, dtype=float)

    out = {k: np.zeros(len(arrays["market_ids"]), dtype=np.float64) for k in RECENT_FEATURE_KEYS}
    for index, market_id in enumerate(arrays["market_ids"]):
        key = (int(market_id), str(arrays["sides"][index]))
        row_times = times_by_key.get(key)
        if not row_times:
            continue
        feats = compute_window_features(
            row_times,
            prices_by_key[key],
            usd_prefix_by_key[key],
            signed_prefix_by_key[key],
            int(arrays["times"][index]),
        )
        for k, value in feats.items():
            out[k][index] = value
    for k, values in out.items():
        arrays[k] = values


def fit_bracket_model(
    indexes: np.ndarray,
    arrays: dict[str, np.ndarray],
    execution: dict[str, float],
    feature_mode: str,
    min_fit_trades: int,
    scenario_name: str,
    selection_score: str,
    return_cap: float,
    sigma_floor: float,
    lcb_z: float,
) -> dict[str, Any]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for index in indexes:
        level = int(arrays["levels"][index])
        if level < 1 or level >= 50:
            continue
        keys = feature_keys(int(index), arrays, feature_mode)
        for bracket in candidate_brackets(level):
            name = bracket_name(bracket)
            value = bracket_unit_return(int(index), bracket, arrays, execution, scenario_name)
            if not math.isfinite(value):
                continue
            for key in keys:
                values[key][name].append(value)

    selectors: dict[str, dict[str, Any]] = {}
    for key, bracket_values in values.items():
        best_name = None
        best_score = -math.inf
        best_metrics: dict[str, Any] = {}
        for name, vals in bracket_values.items():
            if len(vals) < min_fit_trades:
                continue
            metrics = bracket_metrics(vals, return_cap, sigma_floor, lcb_z)
            score = float(metrics[selection_score])
            if score > best_score:
                best_score = score
                best_name = name
                best_metrics = metrics
        if best_name is not None:
            selectors[key] = {
                "bracket": best_name,
                "selection_score": selection_score,
                "score": best_score,
                **best_metrics,
            }
    return {
        "feature_mode": feature_mode,
        "selection_score": selection_score,
        "return_cap": return_cap,
        "sigma_floor": sigma_floor,
        "lcb_z": lcb_z,
        "selectors": selectors,
    }


def bracket_metrics(
    values: list[float],
    return_cap: float,
    sigma_floor: float,
    lcb_z: float,
) -> dict[str, float]:
    data = np.asarray(values, dtype=float)
    capped = np.minimum(data, return_cap)
    mean = float(data.mean())
    capped_mean = float(capped.mean())
    std = float(data.std(ddof=1)) if len(data) > 1 else 0.0
    capped_std = float(capped.std(ddof=1)) if len(capped) > 1 else 0.0
    se = std / math.sqrt(len(data)) if len(data) > 1 else math.inf
    capped_se = capped_std / math.sqrt(len(capped)) if len(capped) > 1 else math.inf
    gross_win = float(data[data > 0].sum())
    gross_loss = float(data[data < 0].sum())
    sorted_values = np.sort(data)[::-1]
    top1 = float(sorted_values[0]) if len(sorted_values) else 0.0
    without_top1 = float(sorted_values[1:].sum()) if len(sorted_values) > 1 else 0.0
    return {
        "n": float(len(data)),
        "mean": mean,
        "capped_mean": capped_mean,
        "std": std,
        "capped_std": capped_std,
        "sharpe": mean / max(std, sigma_floor),
        "capped_sharpe": capped_mean / max(capped_std, sigma_floor),
        "lcb": mean - lcb_z * se if math.isfinite(se) else mean,
        "capped_lcb": capped_mean - lcb_z * capped_se if math.isfinite(capped_se) else capped_mean,
        "hit_rate": float((data > 0).mean()),
        "profit_factor": gross_win / abs(gross_loss) if gross_loss < 0 else math.inf,
        "max_return": top1,
        "without_top1_sum": without_top1,
        "top1_contribution": top1 / max(1e-9, abs(float(data.sum()))),
    }


def select_bracket(index: int, arrays: dict[str, np.ndarray], model: dict[str, Any]) -> Optional[tuple[str, tuple[Optional[int], int], str]]:
    for key in feature_keys(index, arrays, str(model["feature_mode"])):
        selected = model["selectors"].get(key)
        if selected is not None:
            name = str(selected["bracket"])
            return key, parse_bracket(name), name
    return None


def period_key(timestamp: int, budget_period: str) -> Any:
    value = datetime.fromtimestamp(timestamp, timezone.utc)
    if budget_period == "month":
        return value.year, value.month
    if budget_period == "week":
        return value.isocalendar().year, value.isocalendar().week
    raise ValueError(budget_period)


def run_market_making_account(
    indexes: np.ndarray,
    end_time: int,
    arrays: dict[str, np.ndarray],
    bracket_model: dict[str, Any],
    execution: dict[str, float],
    scenario_name: str,
    initial_cash: float,
    period_budget: float,
    budget_period: str,
    stake: float,
    max_stake: float,
    reserve_fraction: float,
    min_stake: float,
    max_horizon_days: float,
) -> dict[str, Any]:
    scenario = SCENARIOS[scenario_name]
    cash = initial_cash
    realized_profit = 0.0
    deployed = 0.0
    fees = 0.0
    entries = 0
    skipped = Counter()
    profits: list[float] = []
    period_spend: dict[Any, float] = defaultdict(float)
    bracket_counts = Counter()
    feature_counts = Counter()

    for index in indexes:
        entry_time = int(arrays["times"][index])
        if entry_time >= end_time:
            break
        level = int(arrays["levels"][index])
        if level >= 50:
            skipped["not_low_price"] += 1
            continue
        horizon_days = max(0.0, (int(arrays["scheduled_end"][index]) - entry_time) / DAY)
        if horizon_days > max_horizon_days:
            skipped["horizon_gate"] += 1
            continue
        selected = select_bracket(int(index), arrays, bracket_model)
        if selected is None:
            skipped["untrained_context"] += 1
            continue
        feature_key, bracket, bracket_label = selected
        pkey = period_key(entry_time, budget_period)
        remaining_period = max(0.0, period_budget - period_spend[pkey])
        reserve_floor = initial_cash * reserve_fraction
        target = min(stake, max_stake, remaining_period, max(0.0, cash - reserve_floor))
        fill_cap = float(arrays["entry_fill"][index]) * scenario["participation"]
        target = min(target, fill_cap)
        if target < min_stake:
            skipped["below_min_stake_or_liquidity"] += 1
            continue
        entry_price = adverse_price(float(arrays["prices"][index]), scenario["entry_ticks"], execution["price_tick"], True)
        contracts, debit, entry_fee = exact_entry(target, entry_price, execution)
        if not contracts:
            skipped["minimum_order"] += 1
            continue
        legs, exit_skips = bracket_exit_legs(int(index), bracket, contracts, arrays, execution, scenario_name)
        skipped.update(exit_skips)
        proceeds_total = 0.0
        exit_fees = 0.0
        for leg in legs:
            proceeds, exit_fee = exact_exit(leg["contracts"], leg["price"], leg["code"], execution)
            proceeds_total += proceeds
            exit_fees += exit_fee
        profit = proceeds_total - debit
        cash += profit
        realized_profit += profit
        deployed += debit
        fees += entry_fee + exit_fees
        period_spend[pkey] += debit
        entries += 1
        profits.append(profit)
        bracket_counts[bracket_label] += 1
        feature_counts[feature_key] += 1

    profit_array = np.asarray(profits, dtype=float)
    sorted_profits = np.sort(profit_array)[::-1] if len(profit_array) else np.array([], dtype=float)
    return {
        "initial_cash": initial_cash,
        "available_cash_end": cash,
        "locked_capital_end": 0.0,
        "realized_profit": realized_profit,
        "total_account_value": cash,
        "account_return": cash / initial_cash - 1.0,
        "deployed": deployed,
        "fees": fees,
        "entries": entries,
        "hit_rate": float((profit_array > 0).mean()) if len(profit_array) else 0.0,
        "gross_winnings": float(profit_array[profit_array > 0].sum()) if len(profit_array) else 0.0,
        "gross_losses": float(profit_array[profit_array < 0].sum()) if len(profit_array) else 0.0,
        "max_single_profit": float(sorted_profits[0]) if len(sorted_profits) else 0.0,
        "without_top_1": float(sorted_profits[1:].sum()) if len(sorted_profits) > 1 else 0.0,
        "without_top_3": float(sorted_profits[3:].sum()) if len(sorted_profits) > 3 else 0.0,
        "profit_factor": (
            float(profit_array[profit_array > 0].sum()) / abs(float(profit_array[profit_array < 0].sum()))
            if len(profit_array) and float(profit_array[profit_array < 0].sum()) < 0
            else None
        ),
        "top_brackets": dict(bracket_counts.most_common(10)),
        "top_feature_keys": dict(feature_counts.most_common(10)),
        "skipped": dict(skipped),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    summaries = []
    for (feature_mode, selection_score, months), group in df.groupby(
        ["feature_mode", "selection_score", "test_months"],
        dropna=False,
    ):
        profits = group["realized_profit"].astype(float).to_numpy()
        returns = group["account_return"].astype(float).to_numpy()
        summaries.append({
            "feature_mode": feature_mode,
            "selection_score": selection_score,
            "test_months": int(months),
            "periods": len(group),
            "mean_profit": float(np.mean(profits)),
            "median_profit": float(np.median(profits)),
            "mean_account_return": float(np.mean(returns)),
            "median_account_return": float(np.median(returns)),
            "positive_rate": float(np.mean(profits > 0)),
            "mean_without_top1": float(group["without_top_1"].astype(float).mean()),
            "worst_period_profit": float(np.min(profits)),
            "best_period_profit": float(np.max(profits)),
            "mean_entries": float(group["entries"].astype(float).mean()),
            "mean_deployed": float(group["deployed"].astype(float).mean()),
            "mean_hit_rate": float(group["hit_rate"].astype(float).mean()),
        })
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/underdog_optimization_kalshi"))
    parser.add_argument("--data-dir", type=Path, default=Path("archive/processed/underdog_events"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/market_making_oos"))
    parser.add_argument("--feature-modes", choices=["regular", "advanced"], nargs="+", default=["regular", "advanced"])
    parser.add_argument("--selection-scores", choices=SELECTION_SCORES, nargs="+", default=["capped_sharpe"])
    parser.add_argument("--return-cap", type=float, default=2.0)
    parser.add_argument("--sigma-floor", type=float, default=0.05)
    parser.add_argument("--lcb-z", type=float, default=1.0)
    parser.add_argument("--test-months", type=int, nargs="+", default=[1, 2, 6])
    parser.add_argument("--min-train-months", type=int, default=12)
    parser.add_argument("--validation-months", type=int, default=6)
    parser.add_argument("--stride-months", type=int, default=None)
    parser.add_argument("--include-partial-final", action="store_true")
    parser.add_argument("--min-fit-trades", type=int, default=25)
    parser.add_argument("--scenario", choices=["optimistic", "neutral", "conservative", "very_conservative"], default="conservative")
    parser.add_argument("--initial-cash", type=float, default=5000.0)
    parser.add_argument("--period-budget", type=float, default=5000.0)
    parser.add_argument("--budget-period", choices=["week", "month"], default="month")
    parser.add_argument("--stake", type=float, default=5.0)
    parser.add_argument("--max-stake", type=float, default=25.0)
    parser.add_argument("--reserve-fraction", type=float, default=0.30)
    parser.add_argument("--min-stake", type=float, default=1.0)
    parser.add_argument("--max-horizon-days", type=float, default=math.inf)
    parser.add_argument("--exclude-categories", nargs="*", default=[],
                        help="Drop these market categories from both fit and replay (e.g. crypto).")
    parser.add_argument("--min-horizon-days", type=float, default=0.0,
                        help="Drop entries with horizon-to-deadline below this (e.g. 1 to skip sub-daily markets).")
    args = parser.parse_args()

    arrays, _ = load_arrays(args.report_dir, args.data_dir)
    execution = execution_model(args.report_dir)
    print("attaching same-side exit paths...", flush=True)
    attach_exit_paths(arrays, args.data_dir, execution)
    print("exit paths attached", flush=True)
    if "advanced" in args.feature_modes:
        print("attaching recent pre-entry features...", flush=True)
        attach_recent_features(arrays, args.data_dir)
        print("recent features attached", flush=True)

    first_month = month_floor(int(arrays["times"].min()))

    # Category / horizon exclusion mask (applied to both fit and test index sets so
    # excluded markets never train the selectors or get traded). Used to fence off the
    # ultra-short-dated crypto markets that flooded the universe in 2025.
    allowed = np.ones(len(arrays["times"]), dtype=bool)
    if args.exclude_categories:
        cats = np.asarray([str(c) for c in arrays["categories"]])
        allowed &= ~np.isin(cats, list(args.exclude_categories))
    if args.min_horizon_days > 0:
        horizon = (arrays["scheduled_end"].astype(float) - arrays["times"].astype(float)) / DAY
        allowed &= horizon >= args.min_horizon_days
    print(f"category/horizon filter: excluded {int((~allowed).sum())} of {len(allowed)} entries", flush=True)

    rows: list[dict[str, Any]] = []
    selector_rows: list[dict[str, Any]] = []
    for feature_mode in args.feature_modes:
        for selection_score in args.selection_scores:
            for months in sorted(set(args.test_months)):
                for period in period_rows(
                    arrays,
                    months,
                    args.min_train_months,
                    args.validation_months,
                    args.stride_months,
                    args.include_partial_final,
                ):
                    fit_end = int(period["validation_start"].timestamp())
                    test_start = int(period["test_start"].timestamp())
                    test_end = int(period["test_end"].timestamp())
                    fit_indexes = np.where(arrays["times"] < fit_end)[0]
                    test_indexes = np.where((arrays["times"] >= test_start) & (arrays["times"] < test_end))[0]
                    fit_indexes = fit_indexes[allowed[fit_indexes]]
                    test_indexes = test_indexes[allowed[test_indexes]]
                    print(
                        f"fitting {feature_mode}/{selection_score} {period['period_id']} "
                        f"fit_end={period['validation_start'].date()} test={period['test_start'].date()}->{period['test_end'].date()}",
                        flush=True,
                    )
                    model = fit_bracket_model(
                        fit_indexes,
                        arrays,
                        execution,
                        feature_mode,
                        args.min_fit_trades,
                        args.scenario,
                        selection_score,
                        args.return_cap,
                        args.sigma_floor,
                        args.lcb_z,
                    )
                    selector_rows.append({
                        "feature_mode": feature_mode,
                        "selection_score": selection_score,
                        "period_id": period["period_id"],
                        "test_months": months,
                        "fit_end": period["validation_start"].date(),
                        "selectors": len(model["selectors"]),
                        "top_selectors": json.dumps(dict(list(model["selectors"].items())[:25]), sort_keys=True),
                    })
                    summary = run_market_making_account(
                        test_indexes,
                        test_end,
                        arrays,
                        model,
                        execution,
                        args.scenario,
                        args.initial_cash,
                        args.period_budget,
                        args.budget_period,
                        args.stake,
                        args.max_stake,
                        args.reserve_fraction,
                        args.min_stake,
                        args.max_horizon_days,
                    )
                    rows.append({
                        "feature_mode": feature_mode,
                        "selection_score": selection_score,
                        "period_id": period["period_id"],
                        "test_months": months,
                        "validation_start": period["validation_start"].date(),
                        "test_start": period["test_start"].date(),
                        "test_end": period["test_end"].date(),
                        **summary,
                        "top_brackets": json.dumps(summary["top_brackets"], sort_keys=True),
                        "top_feature_keys": json.dumps(summary["top_feature_keys"], sort_keys=True),
                        "skipped": json.dumps(summary["skipped"], sort_keys=True),
                    })

    summary_rows = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "market_making_period_results.csv", rows)
    write_csv(args.output_dir / "market_making_summary.csv", summary_rows)
    write_csv(args.output_dir / "market_making_selectors.csv", selector_rows)
    (args.output_dir / "summary.json").write_text(json.dumps({
        "feature_modes": args.feature_modes,
        "selection_scores": args.selection_scores,
        "return_cap": args.return_cap,
        "sigma_floor": args.sigma_floor,
        "lcb_z": args.lcb_z,
        "test_months": args.test_months,
        "stake": args.stake,
        "max_stake": args.max_stake,
        "min_fit_trades": args.min_fit_trades,
        "first_month": str(first_month.date()),
        "files": [
            "market_making_period_results.csv",
            "market_making_summary.csv",
            "market_making_selectors.csv",
        ],
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "period_rows": len(rows),
        "summary_rows": len(summary_rows),
    }, indent=2))


if __name__ == "__main__":
    main()
