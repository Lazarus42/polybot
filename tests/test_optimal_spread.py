"""Tests for the reward / adverse-selection balancing equation (scripts/optimal_spread.py).

Validates: (1) the numeric solver agrees with a brute-force grid argmax; (2) the first-order
condition holds at the optimum, both by finite difference and against the hand-derived analytic
FOC; (3) the comparative statics are economically correct (more toxicity -> wider; more reward
-> tighter; faster fill-decay -> tighter); (4) edge cases (no adverse selection -> quote the
touch; overwhelming toxicity -> retreat to the band edge); and (5) the central claim -- the
reward subsidy can make quoting profitable when raw trading is not, and the optimum is then an
INTERIOR spread, never the touch.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from optimal_spread import (  # noqa: E402
    const_eta,
    exp_lambda,
    foc_closed_form_residual,
    linear_eta,
    net_deriv,
    net_rate,
    reward_subsidy,
    solve_optimal_spread,
)


def brute_argmax(v, r0, lam, eta, size=1.0, n=20000):
    best_s, best = 0.0, net_rate(0.0, v=v, r0=r0, lam=lam, eta=eta, size=size)
    for i in range(n + 1):
        s = v * i / n
        f = net_rate(s, v=v, r0=r0, lam=lam, eta=eta, size=size)
        if f > best:
            best, best_s = f, s
    return best_s, best


class TestComponents(unittest.TestCase):
    def test_subsidy_peaks_at_touch_and_decays(self):
        self.assertAlmostEqual(reward_subsidy(0.0, 3.0, 5.0), 5.0)      # score 1 at the touch
        self.assertAlmostEqual(reward_subsidy(3.0, 3.0, 5.0), 0.0)      # zero at the band edge
        self.assertEqual(reward_subsidy(4.0, 3.0, 5.0), 0.0)           # beyond band
        self.assertTrue(reward_subsidy(1.0, 3.0, 5.0) > reward_subsidy(2.0, 3.0, 5.0))

    def test_lambda_decays_with_depth(self):
        lam = exp_lambda(a=40.0, k=1.0)
        self.assertAlmostEqual(lam(0.0), 40.0)
        self.assertTrue(lam(0.5) > lam(1.0) > lam(2.0))


class TestSolverMatchesBrute(unittest.TestCase):
    def test_agreement_interior(self):
        lam = exp_lambda(a=40.0, k=1.2)
        eta = const_eta(0.6)
        r = solve_optimal_spread(v=3.0, r0=1.0, lam=lam, eta=eta)
        bs, _ = brute_argmax(3.0, 1.0, lam, eta)
        self.assertAlmostEqual(r["s_star"], bs, places=2)
        self.assertTrue(r["interior"])

    def test_agreement_linear_eta(self):
        lam = exp_lambda(a=25.0, k=0.8)
        eta = linear_eta(0.8, -0.1)          # deeper quoting sheds toxicity
        r = solve_optimal_spread(v=4.0, r0=0.5, lam=lam, eta=eta)
        bs, _ = brute_argmax(4.0, 0.5, lam, eta)
        self.assertAlmostEqual(r["s_star"], bs, places=2)


class TestFirstOrderCondition(unittest.TestCase):
    def test_finite_diff_zero_at_optimum(self):
        lam = exp_lambda(a=40.0, k=1.2)
        eta = const_eta(0.6)
        r = solve_optimal_spread(v=3.0, r0=1.0, lam=lam, eta=eta)
        self.assertAlmostEqual(
            net_deriv(r["s_star"], v=3.0, r0=1.0, lam=lam, eta=eta), 0.0, places=3)

    def test_matches_hand_derived_analytic_foc(self):
        # numeric optimum should be a root of the analytic Net'(s) for exp-lambda/const-eta
        a, k, eta0, v, r0 = 40.0, 1.2, 0.6, 3.0, 1.0
        lam, eta = exp_lambda(a, k), const_eta(eta0)
        r = solve_optimal_spread(v=v, r0=r0, lam=lam, eta=eta)
        resid = foc_closed_form_residual(r["s_star"], v=v, r0=r0, a=a, k=k, eta0=eta0)
        self.assertAlmostEqual(resid, 0.0, places=2)


class TestComparativeStatics(unittest.TestCase):
    def setUp(self):
        self.v = 3.0
        self.lam = exp_lambda(a=40.0, k=1.2)

    def _s(self, r0=1.0, eta0=0.6, lam=None):
        lam = lam or self.lam
        return solve_optimal_spread(v=self.v, r0=r0, lam=lam, eta=const_eta(eta0))["s_star"]

    def test_more_toxicity_widens(self):
        self.assertTrue(self._s(eta0=0.3) < self._s(eta0=0.6) < self._s(eta0=1.0))

    def test_more_reward_tightens(self):
        # bigger subsidy pulls quotes toward the touch
        self.assertTrue(self._s(r0=0.2) > self._s(r0=1.0) > self._s(r0=5.0))

    def test_faster_fill_decay_tightens(self):
        # if stepping out kills fills fast (big k), there is less to gain by backing off
        s_slow = self._s(lam=exp_lambda(a=40.0, k=0.5))
        s_fast = self._s(lam=exp_lambda(a=40.0, k=3.0))
        self.assertTrue(s_fast < s_slow)


class TestEdgeCases(unittest.TestCase):
    def test_zero_adverse_selection_spread_capture_optimum(self):
        # No toxicity, modest subsidy: you still don't quote the touch (s=0 captures zero spread
        # per fill). The trading term A*e^{-ks}*s peaks at s=1/k, so the optimum sits just inside
        # that, pulled in slightly by the subsidy.
        k = 1.2
        lam = exp_lambda(a=40.0, k=k)
        r = solve_optimal_spread(v=3.0, r0=1.0, lam=lam, eta=const_eta(0.0))
        self.assertTrue(0.0 < r["s_star"] < 1.0 / k)        # interior, inside the 1/k peak

    def test_dominant_subsidy_drives_to_touch(self):
        # When the reward subsidy overwhelms trading economics, the optimum collapses to the touch.
        lam = exp_lambda(a=40.0, k=1.2)
        r = solve_optimal_spread(v=3.0, r0=1e6, lam=lam, eta=const_eta(0.0))
        self.assertAlmostEqual(r["s_star"], 0.0, places=2)

    def test_overwhelming_toxicity_retreats_to_band_edge(self):
        # eta so large that any reachable fill loses money; with tiny subsidy, retreat to v
        lam = exp_lambda(a=40.0, k=0.2)
        r = solve_optimal_spread(v=3.0, r0=1e-6, lam=lam, eta=const_eta(10.0))
        self.assertGreater(r["s_star"], 2.5)


class TestRewardHarvestingRegime(unittest.TestCase):
    """The crux: subsidy rescues quoting that would lose money on trading alone -- but the
    optimal harvest is an interior spread, not the touch the live strategies use."""

    def test_subsidy_makes_unprofitable_quoting_profitable(self):
        lam = exp_lambda(a=40.0, k=1.2)
        eta = const_eta(1.0)                 # toxic: eta > s for small s
        no_sub = solve_optimal_spread(v=3.0, r0=0.0, lam=lam, eta=eta)
        with_sub = solve_optimal_spread(v=3.0, r0=3.0, lam=lam, eta=eta)
        # without subsidy the best you can do near the touch is non-positive trading PnL
        self.assertLessEqual(net_rate(0.0, v=3.0, r0=0.0, lam=lam, eta=eta), 0.0)
        # subsidy lifts the achievable net above the no-subsidy optimum
        self.assertGreater(with_sub["net_star"], no_sub["net_star"])

    def test_optimal_harvest_beats_touch(self):
        # the interior optimum must earn strictly more than pegging to the touch
        lam = exp_lambda(a=40.0, k=1.2)
        eta = const_eta(0.8)
        r = solve_optimal_spread(v=3.0, r0=2.0, lam=lam, eta=eta)
        self.assertGreater(r["net_star"], r["net_touch"])
        self.assertTrue(r["interior"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
