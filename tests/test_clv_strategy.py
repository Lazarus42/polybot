from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from clv_strategy_backtest import clv_long_short_backtest


def _frame(signal=True, n_months=24, per_month=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_months):
        pred = rng.normal(size=per_month)
        # if signal: realized move correlates with pred; else pure noise
        move = (0.05 * pred if signal else 0.0) + rng.normal(scale=0.05, size=per_month)
        for i in range(per_month):
            rows.append({"period": f"2024-{m:02d}", "pred": pred[i], "fwd_move": move[i], "fill": 100.0})
    return pd.DataFrame(rows)


class TestCLVBacktest(unittest.TestCase):
    def test_signal_profits_at_zero_cost(self):
        r = clv_long_short_backtest(_frame(signal=True, seed=1), roundtrip_cost=0.0)
        self.assertGreater(r["mean_monthly_return"], 0.0)
        self.assertGreater(r["net_dollars_per_month"], 0.0)
        self.assertGreater(r["positive_month_rate"], 0.6)

    def test_cost_kills_edge(self):
        cheap = clv_long_short_backtest(_frame(signal=True, seed=2), roundtrip_cost=0.0)
        dear = clv_long_short_backtest(_frame(signal=True, seed=2), roundtrip_cost=0.10)
        self.assertGreater(cheap["mean_monthly_return"], dear["mean_monthly_return"])
        self.assertLess(dear["mean_monthly_return"], 0.0)   # big cost -> net negative

    def test_no_signal_is_flat(self):
        r = clv_long_short_backtest(_frame(signal=False, seed=3), roundtrip_cost=0.0)
        self.assertLess(abs(r["mean_monthly_return"]), 0.01)

    def test_capacity_scales_with_fill(self):
        small = _frame(signal=True, seed=4); small["fill"] = 50.0
        big = _frame(signal=True, seed=4); big["fill"] = 500.0
        rs = clv_long_short_backtest(small, 0.0, participation=0.1)
        rb = clv_long_short_backtest(big, 0.0, participation=0.1)
        self.assertGreater(rb["mean_deployed_per_month"], rs["mean_deployed_per_month"])


if __name__ == "__main__":
    unittest.main()
