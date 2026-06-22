from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from estimate_effective_spread import roll_spread, roll_spread_from_cov


class TestRoll(unittest.TestCase):
    def test_from_cov(self):
        self.assertAlmostEqual(roll_spread_from_cov(-0.0001), 0.02, places=6)
        self.assertEqual(roll_spread_from_cov(0.0), 0.0)     # no bounce -> 0
        self.assertEqual(roll_spread_from_cov(0.0005), 0.0)  # positive cov -> 0

    def test_recovers_planted_spread(self):
        # Roll model: trades at mid +/- s/2 with random side; recovers full spread s
        rng = np.random.default_rng(0)
        s = 0.04
        side = rng.choice([-1.0, 1.0], size=20000)
        prices = 0.5 + side * (s / 2)
        est = roll_spread(prices)
        self.assertAlmostEqual(est, s, delta=0.005)

    def test_trend_gives_zero(self):
        prices = np.linspace(0.2, 0.8, 5000)   # monotone, no bounce
        self.assertLess(roll_spread(prices), 1e-3)   # ~0 (modulo float noise)


if __name__ == "__main__":
    unittest.main()
