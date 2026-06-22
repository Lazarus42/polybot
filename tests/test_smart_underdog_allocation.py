from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from optimize_underdog_bracket import exit_policy_templates, feasible_multiplier
from realistic_underdog_account import run_account


def base_arrays(count: int = 10) -> dict[str, np.ndarray]:
    times = np.arange(count, dtype=np.int64) * 60
    return {
        "market_ids": np.arange(1, count + 1, dtype=np.int64),
        "times": times,
        "sides": np.array(["YES"] * count),
        "prices": np.array([0.20] * count, dtype=np.float32),
        "levels": np.array([20] * count, dtype=np.int8),
        "scheduled_end": np.array([86_400] * count, dtype=np.int64),
        "closed_times": np.array([86_400] * count, dtype=np.int64),
        "entry_fill": np.array([10_000.0] * count, dtype=np.float64),
        "won": np.array([False] * count),
        "cube": np.array([[[0.0]]] * count, dtype=np.float32),
        "exit_times": np.array([[[86_400]]] * count, dtype=np.int64),
        "exit_prices": np.array([[[0.0]]] * count, dtype=np.float32),
        "exit_fill": np.array([[[np.nan]]] * count, dtype=np.float32),
        "exit_codes": np.array([[[0]]] * count, dtype=np.uint8),
        "categories": np.array(["sports"] * count),
        "exit_path_offsets": np.arange(count + 1, dtype=np.int64),
        "exit_path_times": np.array([1_000] * count, dtype=np.int64),
        "exit_path_prices": np.array([0.40] * count, dtype=np.float32),
        "exit_path_usd": np.array([10_000.0] * count, dtype=np.float64),
    }


def sizing_model(lambda_value: float = 100.0) -> dict:
    row = {
        "n": 100,
        "mean": 0.2,
        "shrunk": 0.2,
        "se": 0.01,
        "lcb": 0.19,
        "second": 0.25,
        "opportunities": 100,
        "lambda": lambda_value,
    }
    key = "regime:16-30c|level:20|category:sports|horizon:<=1d|liquidity:>2500"
    return {
        "period_count": 1,
        "global_mean": 0.1,
        "global_lambda": lambda_value,
        "stats": {
            "rich": {key: row},
            "level_category": {"level:20|category:sports": row},
            "level": {"level:20": row},
            "regime": {"regime:16-30c": row},
            "category": {"category:sports": row},
            "global": {"global": row},
        },
        "weight_sums": {"shrunk": 0.2, "lcb": 0.19},
        "positive_rich_buckets": {"shrunk": 1, "lcb": 1},
    }


class SmartUnderdogAllocationTests(unittest.TestCase):
    def test_forecast_pacing_does_not_exhaust_bankroll_early(self) -> None:
        arrays = base_arrays(10)
        result = run_account(
            np.arange(10),
            86_400,
            {20: (0, 0)},
            arrays,
            {"sports"},
            float("inf"),
            {"fee_coefficient": 0.0, "price_tick": 0.01, "contract_step": 0.01},
            "optimistic",
            5_000.0,
            5_000.0,
            1.0,
            budget_period="week",
            sizing_policy="forecast_paced",
            sizing_model=sizing_model(100.0),
            max_stake=250.0,
            reserve_fraction=0.25,
        )
        self.assertLess(result["summary"]["deployed"], 5_000.0)
        self.assertGreater(result["summary"]["available_cash_end"], 1_250.0)

    def test_reserve_floor_blocks_entries(self) -> None:
        arrays = base_arrays(1)
        result = run_account(
            np.array([0]),
            86_400,
            {20: (0, 0)},
            arrays,
            {"sports"},
            float("inf"),
            {"fee_coefficient": 0.0, "price_tick": 0.01, "contract_step": 0.01},
            "optimistic",
            5_000.0,
            5_000.0,
            1.0,
            sizing_policy="forecast_paced",
            sizing_model=sizing_model(1.0),
            reserve_fraction=1.0,
        )
        self.assertEqual(result["summary"]["entries"], 0)
        self.assertEqual(result["summary"]["skipped"]["reserve_floor"], 1)

    def test_category_cap_blocks_additional_locked_capital(self) -> None:
        arrays = base_arrays(3)
        result = run_account(
            np.arange(3),
            10_000,
            {20: (0, 0)},
            arrays,
            {"sports"},
            float("inf"),
            {"fee_coefficient": 0.0, "price_tick": 0.01, "contract_step": 0.01},
            "optimistic",
            5_000.0,
            5_000.0,
            1_000.0,
            sizing_policy="flat_one",
            max_category_locked_fraction=0.05,
            max_regime_locked_fraction=1.0,
            reserve_fraction=0.0,
        )
        self.assertGreater(result["summary"]["skipped"]["category_locked_cap"], 0)

    def test_price_regime_candidate_catalog(self) -> None:
        policies = exit_policy_templates()
        self.assertFalse(feasible_multiplier(0.49, 3.0))
        self.assertTrue(any(policy["regime"] == "01-05c" and "50x" in policy["name"] for policy in policies))
        self.assertTrue(any(policy["regime"] == "31-49c" and policy["family"] == "high_price_harvest" for policy in policies))

    def test_partial_ladder_recovers_basis_and_keeps_runner(self) -> None:
        arrays = base_arrays(1)
        policy = {
            "name": "test_basis_2x_runner_5x",
            "family": "basis_recovery",
            "regime": "16-30c",
            "stop_loss": 0.5,
            "tranches": [
                {"multiplier": 2.0, "fraction": 0.5},
                {"multiplier": 5.0, "fraction": 0.5},
            ],
            "runner_to_resolution": True,
        }
        arrays.update({
            "candidate_returns": np.array([[0.0]], dtype=np.float32),
            "candidate_exit_times": np.array([[86_400]], dtype=np.int64),
            "candidate_exit_prices": np.array([[0.0]], dtype=np.float32),
            "candidate_exit_codes": np.array([[3]], dtype=np.uint8),
            "candidate_names": np.array([policy["name"]]),
            "candidate_families": np.array([policy["family"]]),
            "candidate_regimes": np.array([policy["regime"]]),
            "candidate_policy_json": np.array([json.dumps(policy)]),
        })
        result = run_account(
            np.array([0]),
            86_400,
            {20: 0},
            arrays,
            {"sports"},
            float("inf"),
            {"fee_coefficient": 0.0, "price_tick": 0.0, "contract_step": 0.01},
            "optimistic",
            100.0,
            100.0,
            10.0,
            sizing_policy="flat_one",
            reserve_fraction=0.0,
        )
        self.assertGreater(result["summary"]["realized_profit"], -0.02)
        self.assertEqual(result["summary"]["exit_family_counts"]["basis_recovery"], 1)


if __name__ == "__main__":
    unittest.main()
