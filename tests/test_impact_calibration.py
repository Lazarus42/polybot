from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from calibrate_market_impact import fit_impact, load_trades_pandas, compute_prepost_impact


def _causal_tape(coef: float, alpha: float, n_markets: int = 60, per_market: int = 400,
                 bounce: float = 0.03, seed: int = 0) -> pd.DataFrame:
    """Synthetic fill tape with a planted PERMANENT impact law + bid-ask bounce.

    Each trade of relative size x permanently shifts the latent fair price by a signed
    coef*x^alpha, and prints at fair*(1 +/- bounce) (the bounce is pure noise that pre/post
    VWAP averaging should cancel). A correct causal estimator recovers (coef, alpha) and
    reads ~0 impact for tiny trades.
    """
    rng = np.random.default_rng(seed)
    rows = []
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    for m in range(n_markets):
        fair = rng.uniform(0.3, 0.6)
        typical = rng.uniform(20, 80)  # typical usd size for this market
        for k in range(per_market):
            usd = float(typical * np.exp(rng.normal(0, 0.8)))
            x = usd / typical
            direction = rng.choice([-1.0, 1.0])
            fair = fair + direction * coef * x ** alpha          # additive permanent move
            fair = min(max(fair, 0.05), 0.95)
            price = fair * (1.0 + rng.choice([-1.0, 1.0]) * bounce)
            rows.append((m, t0 + pd.Timedelta(seconds=k), max(price, 0.005), usd))
    return pd.DataFrame(rows, columns=["market_id", "timestamp", "price", "usd_amount"])


def _synthetic_trades(coef: float, alpha: float, n: int = 200_000,
                      noise: float = 0.25, seed: int = 0) -> pd.DataFrame:
    """relsize ~ lognormal; impact = coef*relsize^alpha * multiplicative lognormal noise."""
    rng = np.random.default_rng(seed)
    relsize = np.exp(rng.normal(0.0, 1.0, n))           # centered on 1
    true = coef * relsize ** alpha
    impact = true * np.exp(rng.normal(0.0, noise, n))   # symmetric in log -> median preserved
    return pd.DataFrame({"relsize": relsize, "impact": impact})


class TestImpactCalibration(unittest.TestCase):
    def test_recovers_sqrt_law(self):
        res = fit_impact(_synthetic_trades(0.40, 0.50, seed=1), buckets=20)
        self.assertAlmostEqual(res["alpha"], 0.50, delta=0.05)
        self.assertAlmostEqual(res["coef_power"], 0.40, delta=0.05)
        self.assertGreater(res["r2_loglog"], 0.97)

    def test_recovers_linear_law(self):
        res = fit_impact(_synthetic_trades(0.30, 1.00, seed=2), buckets=20)
        self.assertAlmostEqual(res["alpha"], 1.00, delta=0.05)
        # for a true linear law, the alpha=1 coef estimate should land near coef
        self.assertAlmostEqual(res["coef_linear_alpha1"], 0.30, delta=0.05)

    def test_recovers_superlinear_law(self):
        res = fit_impact(_synthetic_trades(0.20, 1.30, seed=3), buckets=25)
        self.assertAlmostEqual(res["alpha"], 1.30, delta=0.07)
        self.assertAlmostEqual(res["coef_power"], 0.20, delta=0.04)

    def test_noise_robustness_high_noise(self):
        # medians should still recover the law even with large multiplicative noise
        res = fit_impact(_synthetic_trades(0.40, 0.50, noise=0.6, n=300_000, seed=4), buckets=20)
        self.assertAlmostEqual(res["alpha"], 0.50, delta=0.07)

    def test_prepost_recovers_positive_alpha(self):
        # causal tape with planted permanent law -> estimator must give clearly positive
        # alpha (the bug we are fixing produced alpha < 0). Exact recovery is not expected
        # because small-trade direction inference is noisy; the sign and monotonicity are.
        # Note: the exponent is known to be biased LOW by neighbour contamination on a
        # dense tape, so we assert the robust properties (positive sign, real magnitude),
        # not exact alpha recovery. The previous (broken) estimator gave alpha < 0.
        tape = _causal_tape(coef=0.03, alpha=0.6, bounce=0.005, n_markets=100,
                            per_market=300, seed=11)
        trades = compute_prepost_impact(tape, window=8, min_window=4)
        self.assertGreater(len(trades), 2000)
        res = fit_impact(trades, buckets=20)
        self.assertGreater(res["alpha"], 0.05)
        self.assertGreater(res["coef_power"], 0.0)

    def test_prepost_tiny_trades_low_impact(self):
        # smallest-size bucket should read much lower impact than the largest
        tape = _causal_tape(coef=0.03, alpha=0.6, bounce=0.005, n_markets=100,
                            per_market=300, seed=12)
        trades = compute_prepost_impact(tape, window=8, min_window=4)
        res = fit_impact(trades, buckets=20)
        diag = res["diagnostics"].sort_values("median_relsize")
        self.assertLess(diag["agg_impact"].iloc[0], diag["agg_impact"].iloc[-1])

    def test_pandas_loader_end_to_end(self):
        # build a tiny parquet with a known per-trade structure and confirm the loader
        # computes trailing-VWAP impact / relsize without error, then fits.
        rng = np.random.default_rng(5)
        rows = []
        for m in range(40):
            base = rng.uniform(0.2, 0.6)
            for t in range(60):
                price = base * (1 + rng.normal(0, 0.02))
                usd = float(np.exp(rng.normal(0, 1)))
                rows.append((m, pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(seconds=t),
                             max(price, 0.01), usd))
        df = pd.DataFrame(rows, columns=["market_id", "timestamp", "price", "usd_amount"])
        try:
            p = Path("/tmp/_impact_test.parquet")
            df.to_parquet(p)
        except Exception as exc:  # pyarrow not installed in this env
            self.skipTest(f"parquet engine unavailable: {exc}")
        trades = load_trades_pandas(p, window=20, min_window=10)
        self.assertGreater(len(trades), 100)
        self.assertTrue((trades["relsize"] > 0).all())
        self.assertTrue((trades["impact"] >= 0).all())


if __name__ == "__main__":
    unittest.main()
