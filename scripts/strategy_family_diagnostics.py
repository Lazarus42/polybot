#!/usr/bin/env python3
"""Exploratory diagnostics for non-portfolio strategy families.

This emits trade-level signal rows and simple summaries for candidate strategy
families before wiring them into the full account allocator.
"""
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

from long_holdout_weighting_experiments import load_arrays
from realistic_underdog_account import DAY, load_market_categories, price_regime_for_level, write_csv
from walk_forward_market_making_oos import attach_recent_features
from walk_forward_oos import period_rows


PRICE_BUCKETS = [
    (1, 3, "01-03c"),
    (4, 5, "04-05c"),
    (6, 10, "06-10c"),
    (11, 15, "11-15c"),
    (16, 20, "16-20c"),
    (21, 30, "21-30c"),
    (31, 40, "31-40c"),
    (41, 49, "41-49c"),
]

HORIZON_BUCKETS = [
    (0, 1, "0-1d"),
    (1, 7, "1-7d"),
    (7, 30, "7-30d"),
    (30, 90, "30-90d"),
    (90, math.inf, "90d+"),
]

LIQUIDITY_BUCKETS = [
    (0, 5, "<=5"),
    (5, 25, "5-25"),
    (25, 100, "25-100"),
    (100, 500, "100-500"),
    (500, math.inf, "500+"),
]

DEFAULT_EXIT_BY_REGIME = {
    "01-05c": "01-05c_single_2x",
    "06-15c": "06-15c_single_1.5x",
    "16-30c": "16-30c_single_1.25x",
    "31-49c": "31-49c_single_1.1x",
}

RULE_VARIANTS = [
    {
        "name": "pure_long_tail_01_05",
        "min_level": 1,
        "max_level": 5,
        "min_horizon": 1,
        "min_volume": 0,
        "min_count": 0,
        "min_move": None,
        "max_move": None,
    },
    {
        "name": "pure_long_tail_06_15",
        "min_level": 6,
        "max_level": 15,
        "min_horizon": 1,
        "min_volume": 0,
        "min_count": 0,
        "min_move": None,
        "max_move": None,
    },
    {
        "name": "pure_long_tail_16_30",
        "min_level": 16,
        "max_level": 30,
        "min_horizon": 1,
        "min_volume": 0,
        "min_count": 0,
        "min_move": None,
        "max_move": None,
    },
    {
        "name": "underdog_attention_light",
        "min_level": 1,
        "max_level": 30,
        "min_horizon": 1,
        "min_volume": 25,
        "min_count": 2,
        "min_move": 0.002,
        "max_move": None,
    },
    {
        "name": "underdog_attention",
        "min_level": 1,
        "max_level": 30,
        "min_horizon": 1,
        "min_volume": 50,
        "min_count": 3,
        "min_move": 0.005,
        "max_move": None,
    },
    {
        "name": "underdog_attention_strong",
        "min_level": 1,
        "max_level": 30,
        "min_horizon": 1,
        "min_volume": 250,
        "min_count": 8,
        "min_move": 0.02,
        "max_move": None,
    },
    {
        "name": "underdog_resurrection_01_03",
        "min_level": 1,
        "max_level": 3,
        "min_horizon": 1,
        "min_volume": 25,
        "min_count": 2,
        "min_move": 0.002,
        "max_move": None,
    },
    {
        "name": "underdog_resurrection_01_05",
        "min_level": 1,
        "max_level": 5,
        "min_horizon": 1,
        "min_volume": 25,
        "min_count": 2,
        "min_move": 0.002,
        "max_move": None,
    },
    {
        "name": "favorite_fade_long_horizon_01_05",
        "min_level": 1,
        "max_level": 5,
        "min_horizon": 30,
        "min_volume": 0,
        "min_count": 0,
        "min_move": None,
        "max_move": None,
    },
    {
        "name": "favorite_fade_long_horizon_06_15",
        "min_level": 6,
        "max_level": 15,
        "min_horizon": 30,
        "min_volume": 0,
        "min_count": 0,
        "min_move": None,
        "max_move": None,
    },
    {
        "name": "favorite_fade_long_horizon_16_30",
        "min_level": 16,
        "max_level": 30,
        "min_horizon": 30,
        "min_volume": 0,
        "min_count": 0,
        "min_move": None,
        "max_move": None,
    },
    {
        "name": "favorite_fade_near_deadline_01_15",
        "min_level": 1,
        "max_level": 15,
        "min_horizon": 0,
        "max_horizon": 7,
        "min_volume": 0,
        "min_count": 0,
        "min_move": None,
        "max_move": None,
    },
    {
        "name": "favorite_fade_near_deadline_16_30",
        "min_level": 16,
        "max_level": 30,
        "min_horizon": 0,
        "max_horizon": 7,
        "min_volume": 0,
        "min_count": 0,
        "min_move": None,
        "max_move": None,
    },
    {
        "name": "post_spike_favorite_fade_light",
        "min_level": 5,
        "max_level": 30,
        "min_horizon": 7,
        "min_volume": 25,
        "min_count": 2,
        "min_move": None,
        "max_move": -0.01,
    },
    {
        "name": "post_spike_favorite_fade",
        "min_level": 5,
        "max_level": 30,
        "min_horizon": 7,
        "min_volume": 50,
        "min_count": 3,
        "min_move": None,
        "max_move": -0.03,
    },
    {
        "name": "post_spike_favorite_fade_strong",
        "min_level": 5,
        "max_level": 30,
        "min_horizon": 7,
        "min_volume": 250,
        "min_count": 8,
        "min_move": None,
        "max_move": -0.08,
    },
    {
        "name": "momentum_24h_light",
        "min_level": 5,
        "max_level": 49,
        "min_horizon": 1,
        "min_volume": 50,
        "min_count": 3,
        "min_move": 0.01,
        "max_move": None,
    },
    {
        "name": "momentum_24h",
        "min_level": 5,
        "max_level": 49,
        "min_horizon": 1,
        "min_volume": 100,
        "min_count": 5,
        "min_move": 0.03,
        "max_move": None,
    },
    {
        "name": "momentum_24h_strong",
        "min_level": 5,
        "max_level": 49,
        "min_horizon": 1,
        "min_volume": 500,
        "min_count": 10,
        "min_move": 0.08,
        "max_move": None,
    },
    {
        "name": "capitulation_bounce_light",
        "min_level": 5,
        "max_level": 49,
        "min_horizon": 1,
        "min_volume": 25,
        "min_count": 2,
        "min_move": None,
        "max_move": -0.03,
    },
    {
        "name": "capitulation_bounce_strong",
        "min_level": 5,
        "max_level": 49,
        "min_horizon": 1,
        "min_volume": 100,
        "min_count": 5,
        "min_move": None,
        "max_move": -0.08,
    },
    {
        "name": "pre_deadline_attention_ramp",
        "min_level": 1,
        "max_level": 49,
        "min_horizon": 1,
        "max_horizon": 14,
        "min_volume": 50,
        "min_count": 3,
        "min_move": 0.005,
        "max_move": None,
    },
    {
        "name": "pre_deadline_momentum",
        "min_level": 5,
        "max_level": 49,
        "min_horizon": 1,
        "max_horizon": 14,
        "min_volume": 100,
        "min_count": 5,
        "min_move": 0.03,
        "max_move": None,
    },
    {
        "name": "liquid_low_mid_underdog",
        "min_level": 1,
        "max_level": 30,
        "min_horizon": 1,
        "min_entry_fill": 100,
        "min_volume": 0,
        "min_count": 0,
        "min_move": None,
        "max_move": None,
    },
]


# ---------------------------------------------------------------------------
# Expanded candidate families (2026-06): a broad sweep of momentum, reversion,
# late-game, attention, and structural-carry hypotheses across price bands and
# horizons. All use only existing entry features (price level, horizon, recent
# 24h move/volume/count, entry fill). Generate signals, then rank for alpha with
# scripts/rank_components_tune.py before promoting any into the ensemble.
# ---------------------------------------------------------------------------
EXPANDED_RULE_VARIANTS = [
    # --- Momentum: buy after a recent up-move; hypothesis = trend continuation ---
    {"name": "momentum_up_06_15", "min_level": 6, "max_level": 15, "min_horizon": 1, "min_volume": 50, "min_count": 3, "min_move": 0.01, "max_move": None},
    {"name": "momentum_up_16_30", "min_level": 16, "max_level": 30, "min_horizon": 1, "min_volume": 50, "min_count": 3, "min_move": 0.01, "max_move": None},
    {"name": "momentum_up_31_49", "min_level": 31, "max_level": 49, "min_horizon": 1, "min_volume": 50, "min_count": 3, "min_move": 0.01, "max_move": None},
    {"name": "momentum_up_strong_06_30", "min_level": 6, "max_level": 30, "min_horizon": 1, "min_volume": 100, "min_count": 5, "min_move": 0.05, "max_move": None},
    {"name": "momentum_up_long_horizon_06_30", "min_level": 6, "max_level": 30, "min_horizon": 30, "min_volume": 50, "min_count": 3, "min_move": 0.02, "max_move": None},
    {"name": "momentum_up_near_deadline_06_30", "min_level": 6, "max_level": 30, "min_horizon": 0, "max_horizon": 7, "min_volume": 50, "min_count": 3, "min_move": 0.02, "max_move": None},
    {"name": "momentum_up_high_conviction_16_49", "min_level": 16, "max_level": 49, "min_horizon": 1, "min_volume": 250, "min_count": 8, "min_move": 0.03, "max_move": None},

    # --- Reversion / capitulation: buy after a recent down-move; hypothesis = overreaction bounce ---
    {"name": "reversion_down_06_15", "min_level": 6, "max_level": 15, "min_horizon": 1, "min_volume": 50, "min_count": 3, "min_move": None, "max_move": -0.01},
    {"name": "reversion_down_16_30", "min_level": 16, "max_level": 30, "min_horizon": 1, "min_volume": 50, "min_count": 3, "min_move": None, "max_move": -0.02},
    {"name": "reversion_down_31_49", "min_level": 31, "max_level": 49, "min_horizon": 1, "min_volume": 50, "min_count": 3, "min_move": None, "max_move": -0.03},
    {"name": "reversion_deep_06_30", "min_level": 6, "max_level": 30, "min_horizon": 1, "min_volume": 100, "min_count": 5, "min_move": None, "max_move": -0.06},
    {"name": "reversion_down_long_horizon_06_30", "min_level": 6, "max_level": 30, "min_horizon": 30, "min_volume": 50, "min_count": 3, "min_move": None, "max_move": -0.03},
    {"name": "reversion_down_near_deadline_06_30", "min_level": 6, "max_level": 30, "min_horizon": 0, "max_horizon": 7, "min_volume": 50, "min_count": 3, "min_move": None, "max_move": -0.03},
    {"name": "reversion_crashed_longshot_01_05", "min_level": 1, "max_level": 5, "min_horizon": 1, "min_volume": 25, "min_count": 2, "min_move": None, "max_move": -0.02},

    # --- Late-game: entries close to the scheduled deadline, by band ---
    {"name": "late_game_24h_01_05", "min_level": 1, "max_level": 5, "min_horizon": 0, "max_horizon": 1, "min_volume": 0, "min_count": 0, "min_move": None, "max_move": None},
    {"name": "late_game_24h_06_15", "min_level": 6, "max_level": 15, "min_horizon": 0, "max_horizon": 1, "min_volume": 0, "min_count": 0, "min_move": None, "max_move": None},
    {"name": "late_game_24h_16_30", "min_level": 16, "max_level": 30, "min_horizon": 0, "max_horizon": 1, "min_volume": 0, "min_count": 0, "min_move": None, "max_move": None},
    {"name": "late_game_24h_31_49", "min_level": 31, "max_level": 49, "min_horizon": 0, "max_horizon": 1, "min_volume": 0, "min_count": 0, "min_move": None, "max_move": None},
    {"name": "late_game_48h_06_30", "min_level": 6, "max_level": 30, "min_horizon": 0, "max_horizon": 2, "min_volume": 0, "min_count": 0, "min_move": None, "max_move": None},
    {"name": "late_game_favorite_drift_31_49", "min_level": 31, "max_level": 49, "min_horizon": 0, "max_horizon": 3, "min_volume": 0, "min_count": 0, "min_move": None, "max_move": None},
    {"name": "late_game_momentum_16_30", "min_level": 16, "max_level": 30, "min_horizon": 0, "max_horizon": 2, "min_volume": 50, "min_count": 3, "min_move": 0.02, "max_move": None},
    {"name": "late_game_capitulation_16_30", "min_level": 16, "max_level": 30, "min_horizon": 0, "max_horizon": 2, "min_volume": 50, "min_count": 3, "min_move": None, "max_move": -0.03},

    # --- Attention / flow: recent activity surge as a standalone signal ---
    {"name": "attention_surge_06_15", "min_level": 6, "max_level": 15, "min_horizon": 1, "min_volume": 100, "min_count": 5, "min_move": None, "max_move": None},
    {"name": "attention_surge_16_30", "min_level": 16, "max_level": 30, "min_horizon": 1, "min_volume": 100, "min_count": 5, "min_move": None, "max_move": None},
    {"name": "attention_surge_31_49", "min_level": 31, "max_level": 49, "min_horizon": 1, "min_volume": 100, "min_count": 5, "min_move": None, "max_move": None},
    {"name": "attention_liquid_16_30", "min_level": 16, "max_level": 30, "min_horizon": 1, "min_entry_fill": 100, "min_volume": 0, "min_count": 0, "min_move": None, "max_move": None},

    # --- Structural carry: pure price-band exposure in bands not already covered ---
    {"name": "pure_mid_favorite_31_49", "min_level": 31, "max_level": 49, "min_horizon": 1, "min_volume": 0, "min_count": 0, "min_move": None, "max_move": None},
    {"name": "pure_deep_longshot_01_03", "min_level": 1, "max_level": 3, "min_horizon": 1, "min_volume": 0, "min_count": 0, "min_move": None, "max_move": None},

    # --- Horizon-conditioned carry: same band, different time-to-deadline windows ---
    {"name": "mid_horizon_fade_06_30", "min_level": 6, "max_level": 30, "min_horizon": 7, "max_horizon": 30, "min_volume": 0, "min_count": 0, "min_move": None, "max_move": None},
    {"name": "long_horizon_carry_31_49", "min_level": 31, "max_level": 49, "min_horizon": 30, "min_volume": 0, "min_count": 0, "min_move": None, "max_move": None},
    {"name": "long_horizon_underdog_01_15", "min_level": 1, "max_level": 15, "min_horizon": 60, "min_volume": 0, "min_count": 0, "min_move": None, "max_move": None},
]

# ---------------------------------------------------------------------------
# Feature-based families (2026-06): use the richer pre-entry features added to
# attach_recent_features (multi-timescale moves, acceleration, 24h volatility,
# tick-rule order-flow imbalance). These require regenerating signals with recent
# features attached (i.e. NOT --skip-recent-features).
# ---------------------------------------------------------------------------
# Loosened gates (2026-06): the move/flow/vol feature condition is itself the signal,
# so we drop the volume/count activity floors (which previously starved these of
# samples), soften thresholds, and widen bands. Downstream ranking still enforces a
# min-signal count and a lower-confidence-bound filter, so noise is handled there.
FEATURE_RULE_VARIANTS = [
    # Multi-timescale momentum (trend continuation at different lookbacks)
    {"name": "momentum_7d_up_06_30", "min_level": 6, "max_level": 30, "min_horizon": 1, "min_volume": 0, "min_count": 0, "min_move_7d": 0.03},
    {"name": "momentum_7d_up_16_49", "min_level": 16, "max_level": 49, "min_horizon": 1, "min_volume": 0, "min_count": 0, "min_move_7d": 0.03},
    {"name": "momentum_6h_up_06_30", "min_level": 6, "max_level": 30, "min_horizon": 1, "min_volume": 0, "min_count": 0, "min_move_6h": 0.015},
    {"name": "momentum_1h_burst_06_30", "min_level": 6, "max_level": 30, "min_horizon": 1, "min_volume": 0, "min_count": 0, "min_move_1h": 0.015},
    # Multi-timescale reversion (bounce after a drop) -- positive tune edge in pilot
    {"name": "reversion_7d_down_06_30", "min_level": 6, "max_level": 30, "min_horizon": 1, "min_volume": 0, "min_count": 0, "max_move_7d": -0.03},
    {"name": "reversion_7d_down_16_49", "min_level": 16, "max_level": 49, "min_horizon": 1, "min_volume": 0, "min_count": 0, "max_move_7d": -0.03},
    {"name": "reversion_7d_down_01_15", "min_level": 1, "max_level": 15, "min_horizon": 1, "min_volume": 0, "min_count": 0, "max_move_7d": -0.03},
    {"name": "reversion_6h_down_06_30", "min_level": 6, "max_level": 30, "min_horizon": 1, "min_volume": 0, "min_count": 0, "max_move_6h": -0.015},
    {"name": "reversion_1h_spike_06_30", "min_level": 6, "max_level": 30, "min_horizon": 1, "min_volume": 0, "min_count": 0, "max_move_1h": -0.015},
    # Acceleration: trend strengthening vs. fading
    {"name": "accel_up_06_30", "min_level": 6, "max_level": 30, "min_horizon": 1, "min_volume": 0, "min_count": 0, "min_accel": 0.015},
    {"name": "decel_reversion_06_30", "min_level": 6, "max_level": 30, "min_horizon": 1, "min_volume": 0, "min_count": 0, "max_accel": -0.015},
    {"name": "decel_reversion_16_49", "min_level": 16, "max_level": 49, "min_horizon": 1, "min_volume": 0, "min_count": 0, "max_accel": -0.015},
    # Order-flow imbalance (tick-rule net buy/sell pressure)
    {"name": "flow_buy_pressure_06_30", "min_level": 6, "max_level": 30, "min_horizon": 1, "min_volume": 0, "min_count": 0, "min_flow": 0.2},
    {"name": "flow_sell_pressure_reversion_06_30", "min_level": 6, "max_level": 30, "min_horizon": 1, "min_volume": 0, "min_count": 0, "max_flow": -0.2},
    {"name": "flow_sell_pressure_reversion_16_49", "min_level": 16, "max_level": 49, "min_horizon": 1, "min_volume": 0, "min_count": 0, "max_flow": -0.2},
    # Volatility regimes
    {"name": "high_vol_breakout_06_30", "min_level": 6, "max_level": 30, "min_horizon": 1, "min_volume": 0, "min_count": 0, "min_vol": 0.02, "min_move": 0.015},
    {"name": "low_vol_carry_06_30", "min_level": 6, "max_level": 30, "min_horizon": 1, "min_volume": 0, "min_count": 3, "max_vol": 0.015},
]

RULE_VARIANTS = RULE_VARIANTS + EXPANDED_RULE_VARIANTS + FEATURE_RULE_VARIANTS


def bucket_value(value: float, buckets: list[tuple[float, float, str]]) -> str:
    for lower, upper, label in buckets:
        if lower <= value <= upper:
            return label
    return "other"


def horizon_days(arrays: dict[str, np.ndarray], index: int) -> float:
    return max(0.0, (int(arrays["scheduled_end"][index]) - int(arrays["times"][index])) / DAY)


def exit_index_by_name(arrays: dict[str, np.ndarray]) -> dict[str, int]:
    return {str(name): i for i, name in enumerate(arrays["candidate_names"])}


def default_exit_name(arrays: dict[str, np.ndarray], index: int) -> str | None:
    regime = price_regime_for_level(int(arrays["levels"][index]))
    name = DEFAULT_EXIT_BY_REGIME.get(regime)
    return name if name in exit_index_by_name(arrays) else None


def signal_row(
    arrays: dict[str, np.ndarray],
    index: int,
    family: str,
    exit_name: str,
    unit_return: float,
    expected_edge: float | None,
    confidence: float | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    timestamp = int(arrays["times"][index])
    level = int(arrays["levels"][index])
    n_rows = len(arrays["levels"])
    def feat(name: str) -> float:
        return float(arrays.get(name, np.zeros(n_rows))[index])
    recent_move = feat("recent_price_move_24h")
    recent_volume = feat("recent_volume_24h")
    recent_count = feat("recent_count_24h")
    recent_move_1h = feat("recent_price_move_1h")
    recent_move_6h = feat("recent_price_move_6h")
    recent_move_7d = feat("recent_price_move_7d")
    recent_volatility_24h = feat("recent_volatility_24h")
    recent_accel_24h = feat("recent_accel_24h")
    recent_flow_imbalance_24h = feat("recent_flow_imbalance_24h")
    row = {
        "timestamp": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(),
        "market_id": int(arrays["market_ids"][index]),
        "event_cluster_id": int(arrays["market_ids"][index]),
        "strategy": family,
        "side": str(arrays["sides"][index]),
        "action": "buy",
        "limit_price": float(arrays["prices"][index]),
        "target_size": 1.0,
        "expected_edge": expected_edge,
        "confidence": confidence,
        "max_holding_period": "exit_rule_defined",
        "exit_rule": exit_name,
        "unit_return": unit_return,
        "hold_return": float(arrays["hold_returns"][index]),
        "won": bool(arrays["won"][index]),
        "category": str(arrays["categories"][index]),
        "price_level": level,
        "price_regime": price_regime_for_level(level),
        "price_bucket": bucket_value(level, PRICE_BUCKETS),
        "horizon_days": horizon_days(arrays, index),
        "horizon_bucket": bucket_value(horizon_days(arrays, index), HORIZON_BUCKETS),
        "entry_fill_usd": float(arrays["entry_fill"][index]),
        "historical_volume": float(arrays.get("historical_volumes", np.zeros(len(arrays["levels"])))[index]),
        "recent_volume_24h": recent_volume,
        "recent_count_24h": recent_count,
        "recent_price_move_24h": recent_move,
        "recent_price_move_1h": recent_move_1h,
        "recent_price_move_6h": recent_move_6h,
        "recent_price_move_7d": recent_move_7d,
        "recent_volatility_24h": recent_volatility_24h,
        "recent_accel_24h": recent_accel_24h,
        "recent_flow_imbalance_24h": recent_flow_imbalance_24h,
    }
    if extra:
        row.update(extra)
    row["features"] = json.dumps({
        "category": row["category"],
        "price_bucket": row["price_bucket"],
        "horizon_bucket": row["horizon_bucket"],
        "entry_liquidity_bucket": bucket_value(row["entry_fill_usd"], LIQUIDITY_BUCKETS),
        "recent_volume_24h": recent_volume,
        "recent_count_24h": recent_count,
        "recent_price_move_24h": recent_move,
    }, sort_keys=True)
    return row


def fixed_exit_return(arrays: dict[str, np.ndarray], index_by_exit: dict[str, int], index: int) -> tuple[str, float] | None:
    name = default_exit_name(arrays, index)
    if name is None:
        return None
    return name, float(arrays["candidate_returns"][index, index_by_exit[name]])


# Optional variant filter keys for the richer pre-entry features, mapped to the
# feature column they constrain. min_* is a floor, max_* is a ceiling.
EXTRA_FEATURE_CONDITIONS = {
    "min_move_1h": "recent_price_move_1h", "max_move_1h": "recent_price_move_1h",
    "min_move_6h": "recent_price_move_6h", "max_move_6h": "recent_price_move_6h",
    "min_move_7d": "recent_price_move_7d", "max_move_7d": "recent_price_move_7d",
    "min_accel": "recent_accel_24h", "max_accel": "recent_accel_24h",
    "min_flow": "recent_flow_imbalance_24h", "max_flow": "recent_flow_imbalance_24h",
    "min_vol": "recent_volatility_24h", "max_vol": "recent_volatility_24h",
}


def variant_matches(
    variant: dict[str, Any],
    level: int,
    horizon: float,
    entry_fill: float,
    volume: float,
    count: float,
    move: float,
    feats: dict[str, float] | None = None,
) -> bool:
    if level < int(variant["min_level"]) or level > int(variant["max_level"]):
        return False
    if horizon < float(variant.get("min_horizon", 0.0)):
        return False
    if "max_horizon" in variant and horizon > float(variant["max_horizon"]):
        return False
    if entry_fill < float(variant.get("min_entry_fill", 0.0)):
        return False
    if volume < float(variant.get("min_volume", 0.0)):
        return False
    if count < float(variant.get("min_count", 0.0)):
        return False
    min_move = variant.get("min_move")
    if min_move is not None and move < float(min_move):
        return False
    max_move = variant.get("max_move")
    if max_move is not None and move > float(max_move):
        return False
    for cond_key, feat_key in EXTRA_FEATURE_CONDITIONS.items():
        if cond_key not in variant:
            continue
        value = (feats or {}).get(feat_key, 0.0)
        if cond_key.startswith("min_") and value < float(variant[cond_key]):
            return False
        if cond_key.startswith("max_") and value > float(variant[cond_key]):
            return False
    return True


def variant_edge_and_confidence(variant: dict[str, Any], volume: float, count: float, move: float) -> tuple[float | None, float | None]:
    if variant.get("min_move") is not None:
        edge = move
    elif variant.get("max_move") is not None:
        edge = -move
    else:
        edge = None
    volume_scale = max(1.0, float(variant.get("min_volume", 100.0)) * 4.0)
    count_scale = max(1.0, float(variant.get("min_count", 5.0)) * 4.0)
    confidence = min(1.0, 0.5 * min(1.0, volume / volume_scale) + 0.5 * min(1.0, count / count_scale))
    if variant.get("min_volume", 0.0) == 0 and variant.get("min_count", 0.0) == 0:
        confidence = None
    return edge, confidence


def add_rule_based_signals(
    arrays: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    min_entry_fill: float,
) -> None:
    index_by_exit = exit_index_by_name(arrays)
    zeros = np.zeros(len(arrays["levels"]))
    recent_move = arrays.get("recent_price_move_24h", zeros)
    recent_volume = arrays.get("recent_volume_24h", zeros)
    recent_count = arrays.get("recent_count_24h", zeros)
    extra_feature_arrays = {
        name: arrays.get(name, zeros)
        for name in set(EXTRA_FEATURE_CONDITIONS.values())
    }

    for index in range(len(arrays["levels"])):
        level = int(arrays["levels"][index])
        if float(arrays["entry_fill"][index]) < min_entry_fill:
            continue
        fixed = fixed_exit_return(arrays, index_by_exit, index)
        if fixed is None:
            continue
        exit_name, unit_return = fixed
        horizon = horizon_days(arrays, index)
        entry_fill = float(arrays["entry_fill"][index])
        move = float(recent_move[index])
        volume = float(recent_volume[index])
        count = float(recent_count[index])
        feats = {name: float(values[index]) for name, values in extra_feature_arrays.items()}

        for variant in RULE_VARIANTS:
            if not variant_matches(variant, level, horizon, entry_fill, volume, count, move, feats):
                continue
            edge, confidence = variant_edge_and_confidence(variant, volume, count, move)
            rows.append(signal_row(
                arrays,
                index,
                str(variant["name"]),
                exit_name,
                unit_return,
                edge,
                confidence,
                {"rule_variant": str(variant["name"])},
            ))


def calibration_key(arrays: dict[str, np.ndarray], index: int) -> tuple[str, str, str, str]:
    level = int(arrays["levels"][index])
    return (
        str(arrays["categories"][index]),
        bucket_value(level, PRICE_BUCKETS),
        bucket_value(horizon_days(arrays, index), HORIZON_BUCKETS),
        bucket_value(float(arrays["entry_fill"][index]), LIQUIDITY_BUCKETS),
    )


def fit_calibration(
    arrays: dict[str, np.ndarray],
    fit_indexes: np.ndarray,
    min_bucket_trades: int,
    shrink_k: float,
) -> dict[tuple[str, str, str, str], dict[str, float]]:
    grouped: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    for index in fit_indexes:
        grouped[calibration_key(arrays, int(index))].append(int(index))
    model = {}
    for key, indexes in grouped.items():
        if len(indexes) < min_bucket_trades:
            continue
        idx = np.asarray(indexes, dtype=int)
        p_empirical = float(arrays["won"][idx].mean())
        p_market = float(arrays["prices"][idx].mean())
        w = len(idx) / (len(idx) + shrink_k)
        p_shrunk = w * p_empirical + (1.0 - w) * p_market
        model[key] = {
            "n": float(len(idx)),
            "p_empirical": p_empirical,
            "p_market": p_market,
            "p_shrunk": p_shrunk,
            "bucket_edge": p_shrunk - p_market,
        }
    return model


def add_base_rate_signals(
    arrays: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    min_entry_fill: float,
    min_bucket_trades: int,
    shrink_k: float,
    edge_thresholds: list[float],
    test_months: list[int],
    min_train_months: int,
    validation_months: int,
) -> None:
    index_by_exit = exit_index_by_name(arrays)
    seen: set[tuple[int, str]] = set()
    for months in sorted(set(test_months)):
        periods = period_rows(
            arrays,
            months,
            min_train_months,
            validation_months,
            None,
            True,
        )
        for period in periods:
            fit_end = int(period["validation_start"].timestamp())
            test_start = int(period["test_start"].timestamp())
            test_end = int(period["test_end"].timestamp())
            fit_indexes = np.where(arrays["times"] < fit_end)[0]
            test_indexes = np.where((arrays["times"] >= test_start) & (arrays["times"] < test_end))[0]
            model = fit_calibration(arrays, fit_indexes, min_bucket_trades, shrink_k)
            for index in test_indexes:
                index = int(index)
                if float(arrays["entry_fill"][index]) < min_entry_fill:
                    continue
                key = calibration_key(arrays, index)
                stats = model.get(key)
                if not stats:
                    continue
                edge = float(stats["p_shrunk"] - float(arrays["prices"][index]))
                fixed = fixed_exit_return(arrays, index_by_exit, index)
                if fixed is None:
                    continue
                exit_name, unit_return = fixed
                for threshold in edge_thresholds:
                    if edge < threshold:
                        continue
                    family = f"base_rate_calibration_edge_{int(round(threshold * 100)):02d}p"
                    dedupe = (index, str(period["period_id"]), family)
                    if dedupe in seen:
                        continue
                    seen.add(dedupe)
                    rows.append(signal_row(
                        arrays,
                        index,
                        family,
                        exit_name,
                        unit_return,
                        edge,
                        min(1.0, float(stats["n"]) / (float(stats["n"]) + shrink_k)),
                        {
                            "period_id": str(period["period_id"]),
                            "test_months": int(period["test_months"]),
                            "bucket_n": stats["n"],
                            "bucket_p_empirical": stats["p_empirical"],
                            "bucket_p_market": stats["p_market"],
                            "bucket_p_shrunk": stats["p_shrunk"],
                            "bucket_key": "|".join(key),
                            "edge_threshold": threshold,
                        },
                    ))


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    summaries = []
    for (strategy, exit_rule), group in df.groupby(["strategy", "exit_rule"], dropna=False):
        returns = group["unit_return"].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        hold_returns = group["hold_return"].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if not len(returns):
            continue
        sorted_returns = np.sort(returns)[::-1]
        summaries.append({
            "strategy": strategy,
            "exit_rule": exit_rule,
            "signals": int(len(group)),
            "mean_unit_return": float(np.mean(returns)),
            "median_unit_return": float(np.median(returns)),
            "hit_rate": float(np.mean(returns > 0)),
            "mean_hold_return": float(np.mean(hold_returns)) if len(hold_returns) else np.nan,
            "hold_hit_rate": float(np.mean(hold_returns > 0)) if len(hold_returns) else np.nan,
            "without_top_1_sum": float(sorted_returns[1:].sum()) if len(sorted_returns) > 1 else 0.0,
            "worst_unit_return": float(np.min(returns)),
            "best_unit_return": float(np.max(returns)),
            "p05_unit_return": float(np.quantile(returns, 0.05)),
            "p95_unit_return": float(np.quantile(returns, 0.95)),
            "mean_price_level": float(group["price_level"].astype(float).mean()),
            "mean_entry_fill_usd": float(group["entry_fill_usd"].astype(float).mean()),
            "mean_recent_volume_24h": float(group["recent_volume_24h"].astype(float).mean()),
            "mean_recent_price_move_24h": float(group["recent_price_move_24h"].astype(float).mean()),
        })
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/underdog_optimization_kalshi"))
    parser.add_argument("--data-dir", type=Path, default=Path("archive/processed/underdog_events"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/strategy_family_diagnostics"))
    parser.add_argument("--test-months", type=int, nargs="+", default=[1, 2, 6])
    parser.add_argument("--min-train-months", type=int, default=12)
    parser.add_argument("--validation-months", type=int, default=6)
    parser.add_argument("--min-entry-fill", type=float, default=2.0,
                        help="Minimum triggering-fill USD for a market to enter the signal set. "
                             "Lower values admit thinner tail markets (sized down by the "
                             "participation cap and market-dependent minimum at replay time).")
    parser.add_argument("--min-bucket-trades", type=int, default=50)
    parser.add_argument("--shrink-k", type=float, default=200.0)
    parser.add_argument("--edge-thresholds", type=float, nargs="+", default=[0.02, 0.03, 0.05, 0.08])
    parser.add_argument("--skip-recent-features", action="store_true")
    args = parser.parse_args()

    arrays, _ = load_arrays(args.report_dir, args.data_dir)
    data = np.load(args.report_dir / "strategy_cube.npz")
    order = np.argsort(data["entry_times"], kind="stable")
    if "historical_volumes" not in arrays:
        arrays["historical_volumes"] = data["historical_volumes"][order]
    if "hold_returns" not in arrays:
        arrays["hold_returns"] = data["hold_returns"][order]
    if "candidate_returns" not in arrays:
        raise SystemExit("strategy cube needs candidate_returns; rerun optimize_underdog_bracket.py")
    if args.skip_recent_features:
        zeros = np.zeros(len(arrays["levels"]), dtype=np.float64)
        arrays["recent_volume_24h"] = zeros
        arrays["recent_count_24h"] = zeros
        arrays["recent_price_move_24h"] = zeros
    else:
        print("attaching recent pre-entry features...", flush=True)
        attach_recent_features(arrays, args.data_dir)
        print("recent features attached", flush=True)
    arrays["categories"] = load_market_categories(args.data_dir, arrays["market_ids"])

    rows: list[dict[str, Any]] = []
    add_rule_based_signals(arrays, rows, args.min_entry_fill)
    add_base_rate_signals(
        arrays,
        rows,
        args.min_entry_fill,
        args.min_bucket_trades,
        args.shrink_k,
        args.edge_thresholds,
        args.test_months,
        args.min_train_months,
        args.validation_months,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "strategy_family_signals.csv", rows)
    summary_rows = summarize(rows)
    write_csv(args.output_dir / "strategy_family_summary.csv", summary_rows)
    (args.output_dir / "summary.json").write_text(json.dumps({
        "signals": len(rows),
        "summary_rows": len(summary_rows),
        "files": ["strategy_family_signals.csv", "strategy_family_summary.csv"],
        "notes": "Exploratory unit-return diagnostics; not a full account allocator.",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "signals": len(rows), "summary_rows": len(summary_rows)}, indent=2))


if __name__ == "__main__":
    main()
