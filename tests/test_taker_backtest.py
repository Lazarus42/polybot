"""Tests for the taker pickoff P&L core (scripts/taker_backtest.py)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from taker_backtest import taker_pnl, _is_ephemeral


class TestTakerPnl(unittest.TestCase):
    def test_buy_profits_when_mid_rises(self):
        # bought at ask 0.51, mid moves to 0.55 -> +0.04/share * 100
        self.assertAlmostEqual(taker_pnl("BUY", 0.51, 0.55, 100), 4.0)

    def test_buy_loses_when_flat(self):
        # bought at ask 0.51, mid stays 0.50 -> paid the half-spread, lose 0.01*100
        self.assertAlmostEqual(taker_pnl("BUY", 0.51, 0.50, 100), -1.0)

    def test_sell_profits_when_mid_falls(self):
        # sold at bid 0.49, mid drops to 0.45 -> +0.04/share
        self.assertAlmostEqual(taker_pnl("SELL", 0.49, 0.45, 100), 4.0)

    def test_sell_loses_when_flat(self):
        self.assertAlmostEqual(taker_pnl("SELL", 0.49, 0.50, 100), -1.0)

    def test_must_overcome_spread(self):
        # bought at ask 0.51; mid rises only to 0.505 -> still negative (didn't clear the spread)
        self.assertLess(taker_pnl("BUY", 0.51, 0.505, 100), 0.0)


class TestClassify(unittest.TestCase):
    def test_crypto_detected(self):
        self.assertTrue(_is_ephemeral("Ethereum Up or Down - 3:50PM"))
        self.assertFalse(_is_ephemeral("Will Spain win the World Cup?"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
