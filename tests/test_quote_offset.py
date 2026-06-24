"""Tests for the quote_offset (s*) knob on Quoter and the depth_optimal wiring.

quote_offset makes the Quoter rest each side a fixed PRICE distance from the mid (the s* from
optimal_spread) instead of pegging to the touch. We check: (1) placement is mid-relative and
symmetric; (2) reward is scored at that mid-distance; (3) resting deeper than the touch takes
strictly fewer fills than the legacy touch-pegged quoter on the same tape; (4) inventory still
SKEWS the pair (no side is pulled); and (5) paper_sim computes a sane per-token offset.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from quoter import Quoter  # noqa: E402


class TestPlacement(unittest.TestCase):
    def test_quotes_rest_at_mid_plus_minus_offset(self):
        q = Quoter(our_size=100, inventory_cap=1000, quote_offset=0.02)   # 2 cents
        q.on_quote(0.0, bid=0.49, ask=0.51, bid_size=500, ask_size=500)   # mid 0.50
        self.assertAlmostEqual(q.our_bid, 0.48)
        self.assertAlmostEqual(q.our_ask, 0.52)

    def test_offset_zero_is_legacy_touch(self):
        q = Quoter(our_size=100, inventory_cap=1000, quote_offset=0.0)
        q.on_quote(0.0, bid=0.49, ask=0.51, bid_size=500, ask_size=500)
        self.assertAlmostEqual(q.our_bid, 0.49)     # pegged to the touch
        self.assertAlmostEqual(q.our_ask, 0.51)

    def test_reward_scored_at_offset_distance(self):
        # reward score uses (mid - our_bid) in cents == the offset; deeper offset => less reward
        v = 3.0
        tight = Quoter(our_size=100, inventory_cap=1000, quote_offset=0.01,
                       reward_pool=1440.0, reward_v_cents=v, reward_min_size=0)
        wide = Quoter(our_size=100, inventory_cap=1000, quote_offset=0.02,
                      reward_pool=1440.0, reward_v_cents=v, reward_min_size=0)
        for q in (tight, wide):
            q.on_quote(0.0, bid=0.49, ask=0.51, bid_size=500, ask_size=500)
            # need competing depth, else capture share is 100% regardless of our score/depth
            q.credit_sample(0.50, q_bid_book=50.0, q_ask_book=50.0)
        self.assertGreater(tight.reward, wide.reward)


class TestFillRate(unittest.TestCase):
    def _run(self, offset):
        q = Quoter(our_size=100, inventory_cap=10000, quote_offset=offset)
        # mid steady at 0.50, touch at 0.49/0.51. Trades print walking through prices on both sides.
        t = 0.0
        for _ in range(50):
            q.on_quote(t, bid=0.49, ask=0.51, bid_size=500, ask_size=500)
            # aggressive sells stepping down to 0.485, aggressive buys stepping up to 0.515
            q.on_trade(t + 0.1, price=0.485, side="SELL", size=50)
            q.on_trade(t + 0.2, price=0.515, side="BUY", size=50)
            t += 1.0
        return len(q.fills)

    def test_deeper_offset_takes_fewer_fills(self):
        touch = self._run(0.0)        # pegged at 0.49/0.51 -> trades at .485/.515 reach us
        deep = self._run(0.03)        # rest at 0.47/0.53 -> the .485/.515 trades do NOT reach us
        self.assertGreater(touch, 0)
        self.assertLess(deep, touch)


class TestInventorySkewNotPull(unittest.TestCase):
    def test_inventory_skews_both_sides_no_side_pulled(self):
        q = Quoter(our_size=100, inventory_cap=1000, quote_offset=0.02, inv_skew=0.02)
        q.inv = 500   # long half the cap -> reservation price should shift DOWN (encourage selling)
        q.on_quote(0.0, bid=0.49, ask=0.51, bid_size=500, ask_size=500)
        # both sides still quoted (neither pulled to None), and the pair is shifted, not gated off
        self.assertIsNotNone(q.our_bid)
        self.assertIsNotNone(q.our_ask)
        off = 0.02 * (500 / 1000)
        self.assertAlmostEqual(q.our_bid, 0.50 - 0.02 - off)
        self.assertAlmostEqual(q.our_ask, 0.50 + 0.02 - off)


class TestExtremeLiquidation(unittest.TestCase):
    def test_dumps_inventory_when_price_leaves_band(self):
        q = Quoter(our_size=100, inventory_cap=1000, min_mid=0.10, max_mid=0.90,
                   liq_outside_band=True)
        q.on_quote(0.0, 0.49, 0.51, 500, 500)        # in band
        q.inv = 300                                   # pretend we built a long
        q.on_quote(1.0, 0.93, 0.95, 500, 500)        # mid 0.94 -> outside band
        self.assertEqual(q.inv, 0.0)                  # liquidated
        self.assertEqual(q.n_flats, 1)

    def test_no_dump_when_flag_off(self):
        q = Quoter(our_size=100, inventory_cap=1000, min_mid=0.10, max_mid=0.90,
                   liq_outside_band=False)
        q.on_quote(0.0, 0.49, 0.51, 500, 500)
        q.inv = 300
        q.on_quote(1.0, 0.93, 0.95, 500, 500)
        self.assertEqual(q.inv, 300)                  # only stops quoting, keeps position


class TestStopLoss(unittest.TestCase):
    def test_cuts_position_when_underwater(self):
        q = Quoter(our_size=100, inventory_cap=1000, stop_loss_cents=3.0)
        q.on_quote(0.0, 0.59, 0.61, 0, 0)            # no queue ahead -> our bid at 0.59
        q.on_trade(0.1, price=0.59, side="SELL", size=100)   # we BUY at 0.59 (long), entry 0.59
        self.assertGreater(q.inv, 0)
        q.on_quote(1.0, 0.53, 0.55, 0, 0)            # mid 0.54 -> 5c underwater vs 0.59 (> 3c stop)
        self.assertEqual(q.inv, 0.0)                  # stopped out
        self.assertEqual(q.n_flats, 1)

    def test_holds_when_within_stop(self):
        q = Quoter(our_size=100, inventory_cap=1000, stop_loss_cents=3.0)
        q.on_quote(0.0, 0.59, 0.61, 0, 0)
        q.on_trade(0.1, price=0.59, side="SELL", size=100)
        q.on_quote(1.0, 0.585, 0.605, 0, 0)          # mid 0.595 -> only 0.5c under, hold
        self.assertGreater(q.inv, 0)


class TestPaperSimWiring(unittest.TestCase):
    def test_optimal_offset_is_interior_and_sane(self):
        import paper_sim as ps
        # test the function's logic with explicit fast-fill-decay params (independent of the live
        # DEPTH_PARAMS, which are the real measured values where fills barely decay -> s* = 0).
        old = ps.DEPTH_PARAMS
        ps.DEPTH_PARAMS = dict(a=20.0, k=1.0, eta0=0.6, size=1.0)
        try:
            s = ps.optimal_offset_cents(v_cents=3.0, per_min_reward=1.0, size=100)
            self.assertTrue(0.0 < s < 3.0)        # interior to the band
            self.assertEqual(ps.optimal_offset_cents(0.0, 1.0, 100), 0.0)
            lo = ps.optimal_offset_cents(3.0, 0.3, 100)
            hi = ps.optimal_offset_cents(3.0, 30.0, 100)
            self.assertGreaterEqual(lo, hi)       # richer reward pulls s* tighter
        finally:
            ps.DEPTH_PARAMS = old

    def test_depth_optimal_config_registered(self):
        import paper_sim as ps
        self.assertIn("depth_optimal", ps.CONFIGS)
        self.assertTrue(ps.CONFIGS["depth_optimal"].get("_optimal"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
