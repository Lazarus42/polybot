from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from forward_return_predictability import (
    information_coefficient, long_short_return, walk_forward_ic,
    feature_ic_breakdown, bucketed_ic, net_decile_long_short,
)


class TestIC(unittest.TestCase):
    def test_perfect_and_zero(self):
        x = np.arange(100.0)
        self.assertAlmostEqual(information_coefficient(x, x), 1.0, places=6)
        self.assertAlmostEqual(information_coefficient(x, -x), -1.0, places=6)
        rng = np.random.default_rng(0)
        ic = information_coefficient(rng.normal(size=2000), rng.normal(size=2000))
        self.assertLess(abs(ic), 0.1)

    def test_long_short_positive_when_aligned(self):
        rng = np.random.default_rng(1)
        pred = rng.normal(size=1000)
        realized = pred + rng.normal(scale=0.5, size=1000)
        self.assertGreater(long_short_return(pred, realized), 0.0)


class TestWalkForwardIC(unittest.TestCase):
    def _frame(self, signal_strength, n=3000, seed=0):
        rng = np.random.default_rng(seed)
        f1 = rng.normal(size=n); f2 = rng.normal(size=n); f3 = rng.normal(size=n)
        target = signal_strength * f1 + rng.normal(scale=1.0, size=n)
        return pd.DataFrame({"entry_ts": np.arange(n), "f1": f1, "f2": f2, "f3": f3, "y": target})

    def test_recovers_planted_signal(self):
        r = walk_forward_ic(self._frame(0.8, seed=2), ["f1", "f2", "f3"], "y", n_folds=5)
        self.assertGreater(r["oos_ic"], 0.2)            # real predictability -> positive OOS IC
        self.assertGreater(r["oos_long_short"], 0.0)

    def test_no_signal_gives_zero_ic(self):
        r = walk_forward_ic(self._frame(0.0, seed=3), ["f1", "f2", "f3"], "y", n_folds=5)
        self.assertLess(abs(r["oos_ic"]), 0.08)         # pure noise -> ~0 OOS IC

    def test_no_leakage_too_few_rows(self):
        r = walk_forward_ic(self._frame(0.8, n=50), ["f1", "f2", "f3"], "y", n_folds=5)
        self.assertTrue(np.isnan(r["oos_ic"]))

    def test_feature_breakdown_isolates_driver(self):
        # only f1 carries signal -> its univariate IC should dominate
        d = self._frame(0.8, seed=4)
        ics = feature_ic_breakdown(d, ["f1", "f2", "f3"], "y")
        self.assertGreater(ics["f1"], 0.2)
        self.assertGreater(ics["f1"], abs(ics["f2"]) + 0.1)

    def test_net_cost_reduces_edge(self):
        d = self._frame(0.8, seed=5)
        gross = net_decile_long_short(d, ["f1", "f2", "f3"], "y", cost=0.0)
        net = net_decile_long_short(d, ["f1", "f2", "f3"], "y", cost=0.05)
        self.assertGreater(gross, net)


if __name__ == "__main__":
    unittest.main()
