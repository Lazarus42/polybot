from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from related_market_residual import basket_arb_trade, backtest_basket_arb


class TestBasketArb(unittest.TestCase):
    def test_underround_buys_and_profits(self):
        # YES prices sum to 0.90 -> buy complete set for 0.90, winner in set -> pnl +0.10
        r = basket_arb_trade([0.4, 0.3, 0.2], winner_in_set=True, fee=0.0, edge_buffer=0.02)
        self.assertEqual(r["traded"], 1)
        self.assertAlmostEqual(r["cost"], 0.9)
        self.assertAlmostEqual(r["pnl"], 0.1)

    def test_fairly_priced_is_skipped(self):
        r = basket_arb_trade([0.34, 0.33, 0.33], winner_in_set=True, edge_buffer=0.02)
        self.assertEqual(r["traded"], 0)            # sum ~1, no edge -> no trade

    def test_overround_skipped(self):
        r = basket_arb_trade([0.5, 0.4, 0.3], winner_in_set=True, edge_buffer=0.02)
        self.assertEqual(r["traded"], 0)

    def test_incomplete_partition_missing_winner_loses(self):
        # underround because an outcome is MISSING; winner is the missing one -> pay, collect 0
        r = basket_arb_trade([0.3, 0.3], winner_in_set=False, edge_buffer=0.02)
        self.assertEqual(r["traded"], 1)
        self.assertAlmostEqual(r["pnl"], -0.6)      # this is the partition-incompleteness risk

    def test_fee_erodes_edge(self):
        cheap = basket_arb_trade([0.45, 0.45], winner_in_set=True, fee=0.0, edge_buffer=0.02)
        feed = basket_arb_trade([0.45, 0.45], winner_in_set=True, fee=0.1, edge_buffer=0.02)
        self.assertLess(feed["pnl"], cheap["pnl"])

    def test_backtest_aggregates(self):
        events = [
            {"yes_prices": [0.4, 0.3, 0.2], "winner_in_set": True},   # +0.10
            {"yes_prices": [0.34, 0.33, 0.33], "winner_in_set": True},  # skip
            {"yes_prices": [0.3, 0.3], "winner_in_set": False},        # -0.60
        ]
        s = backtest_basket_arb(events, fee=0.0, edge_buffer=0.02)
        self.assertEqual(s["events_traded"], 2)
        self.assertAlmostEqual(s["total_pnl"], 0.1 - 0.6)


if __name__ == "__main__":
    unittest.main()
