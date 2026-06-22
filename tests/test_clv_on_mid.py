from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from clv_on_mid import asof_mid, forward_moves_from_mid


class TestAsofMid(unittest.TestCase):
    def test_lookup(self):
        t = np.array([0.0, 10.0, 20.0, 30.0])
        m = np.array([0.5, 0.6, 0.55, 0.7])
        self.assertEqual(asof_mid(t, m, 15.0), 0.6)   # last at/before 15 -> t=10
        self.assertEqual(asof_mid(t, m, 30.0), 0.7)
        self.assertTrue(np.isnan(asof_mid(t, m, -5.0)))  # before first point
        self.assertTrue(np.isnan(asof_mid(np.array([]), np.array([]), 5.0)))


class TestForwardMovesFromMid(unittest.TestCase):
    def test_forward_move_on_mid(self):
        # token A mid rises 0.50 -> 0.60 over an hour
        hist = {"A": (np.array([0.0, 3600.0, 7200.0]), np.array([0.50, 0.60, 0.55]))}
        entries = pd.DataFrame({"token": ["A"], "entry_ts": [0.0]})
        out = forward_moves_from_mid(entries, hist, [1.0, 2.0])
        self.assertAlmostEqual(out["fwd_move_1.0h"].iloc[0], 0.10)   # 0.60 - 0.50
        self.assertAlmostEqual(out["fwd_move_2.0h"].iloc[0], 0.05)   # 0.55 - 0.50

    def test_missing_token_is_nan(self):
        entries = pd.DataFrame({"token": ["Z"], "entry_ts": [0.0]})
        out = forward_moves_from_mid(entries, {}, [1.0])
        self.assertTrue(np.isnan(out["fwd_move_1.0h"].iloc[0]))


if __name__ == "__main__":
    unittest.main()
