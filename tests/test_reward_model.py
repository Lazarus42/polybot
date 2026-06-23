from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from reward_model import adjusted_mid, capture_share, order_score, q_min, side_score


class TestOrderScore(unittest.TestCase):
    def test_quadratic_and_bounds(self):
        self.assertAlmostEqual(order_score(3, 0), 1.0)            # at the mid: full score
        self.assertAlmostEqual(order_score(3, 3), 0.0)           # at the edge: zero
        self.assertAlmostEqual(order_score(3, 1), (2 / 3) ** 2)
        self.assertEqual(order_score(3, 4), 0.0)                 # beyond max spread: zero
        self.assertEqual(order_score(0, 0), 0.0)                 # degenerate v

    def test_polymarket_worked_example(self):
        # reproduce the docs' worked example exactly (v = 3 cents)
        # Q_ne: 100@1c, 200@2c, 100@1c
        q_ne = (order_score(3, 1) * 100 + order_score(3, 2) * 200 + order_score(3, 1) * 100)
        self.assertAlmostEqual(q_ne, 111.1111, places=3)
        # Q_no: 100@1.5c, 100@2c, 200@0.5c
        q_no = (order_score(3, 1.5) * 100 + order_score(3, 2) * 100 + order_score(3, 0.5) * 200)
        self.assertAlmostEqual(q_no, 175.0, places=3)
        # mid 0.50 is in [0.10,0.90]; two-sided present -> min dominates
        self.assertAlmostEqual(q_min(q_ne, q_no, 0.50), 111.1111, places=3)


class TestSideAndMin(unittest.TestCase):
    def test_side_score_filters_dust_and_far_orders(self):
        # mid 0.50, v=3c, min_size=50
        levels = [(0.49, 100),   # 1c, counts
                  (0.48, 200),   # 2c, counts
                  (0.45, 100),   # 5c, beyond max spread -> 0
                  (0.495, 10)]   # 0.5c but size < min_size -> excluded
        got = side_score(levels, 0.50, 3, 50)
        want = order_score(3, 1) * 100 + order_score(3, 2) * 200
        self.assertAlmostEqual(got, want)

    def test_qmin_single_sided_reduced_in_mid_range(self):
        # only one side present, mid in [0.10,0.90] -> scores at Q/c
        self.assertAlmostEqual(q_min(90.0, 0.0, 0.50, c=3.0), 30.0)

    def test_qmin_requires_two_sided_at_extremes(self):
        # mid 0.95 (extreme) with only one side -> zero
        self.assertEqual(q_min(90.0, 0.0, 0.95), 0.0)


class TestCaptureShare(unittest.TestCase):
    def test_below_min_size_earns_nothing(self):
        # our 20-lot quote with min_size=200 -> filtered out -> zero share
        book_b = [(0.49, 5000)]; book_a = [(0.51, 5000)]
        share = capture_share((0.49, 20), (0.51, 20), book_b, book_a,
                              mid=0.50, v_cents=3, min_size=200)
        self.assertEqual(share, 0.0)

    def test_share_grows_with_size_and_tightness(self):
        book_b = [(0.49, 1000)]; book_a = [(0.51, 1000)]
        small = capture_share((0.49, 500), (0.51, 500), book_b, book_a, 0.50, 3, 200)
        big = capture_share((0.49, 2000), (0.51, 2000), book_b, book_a, 0.50, 3, 200)
        self.assertGreater(big, small)
        self.assertGreater(small, 0.0)
        # quoting tighter (closer to mid) than the book beats quoting at the same distance
        tight = capture_share((0.499, 1000), (0.501, 1000), book_b, book_a, 0.50, 3, 200)
        same = capture_share((0.49, 1000), (0.51, 1000), book_b, book_a, 0.50, 3, 200)
        self.assertGreater(tight, same)

    def test_one_sided_quote_when_stepped_off(self):
        # stepping off the ask (None) should reduce share vs quoting both sides
        book_b = [(0.49, 1000)]; book_a = [(0.51, 1000)]
        both = capture_share((0.49, 1000), (0.51, 1000), book_b, book_a, 0.50, 3, 200)
        bid_only = capture_share((0.49, 1000), None, book_b, book_a, 0.50, 3, 200)
        self.assertGreater(both, bid_only)

    def test_adjusted_mid_filters_dust(self):
        b = [(0.49, 1000), (0.499, 5)]; a = [(0.51, 1000), (0.501, 5)]
        self.assertAlmostEqual(adjusted_mid(b, a, 200), 0.50)   # dust 0.499/0.501 ignored


if __name__ == "__main__":
    unittest.main()
