from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from walk_forward_passive_mm import simulate_passive_mm


def _meanrev_tape(value=0.5, theta=0.6, sigma=0.012, n=4000, seed=0):
    """Mean-reverting (Ornstein-Uhlenbeck) flow around a constant fair value: prints
    oscillate across the quotes and revert. This is the canonical *uninformed* flow a
    passive MM profits from — fills get marked back toward the stable mid."""
    rng = np.random.default_rng(seed)
    x = 0.0
    out = []
    for _ in range(n):
        x += -theta * x + sigma * rng.standard_normal()
        out.append(value + x)
    prices = np.clip(np.array(out), 0.02, 0.98)
    return prices, np.full(n, 100.0)


def _noise_tape(value=0.5, bounce=0.03, n=4000, seed=0):
    """Pure i.i.d. volatility: every print overshoots the quotes (no mean structure).
    Here every fill is a 'run-through' = adverse selection, so realized pnl is below the
    gross spread captured — the simulator must price that cost."""
    rng = np.random.default_rng(seed)
    prices = value + rng.choice([-bounce, bounce], size=n) * rng.uniform(0.8, 1.2, n)
    prices = np.clip(prices, 0.02, 0.98)
    return prices, np.full(n, 100.0)


def _trend_tape(start=0.3, end=0.7, n=4000, noise=0.005, seed=0):
    """Monotone drift: price ramps one direction. A passive MM gets adversely selected
    (keeps selling into a rising market / buying a falling one) and should lose on inventory."""
    rng = np.random.default_rng(seed)
    base = np.linspace(start, end, n)
    prices = np.clip(base + rng.normal(0, noise, n), 0.02, 0.98)
    contracts = np.full(n, 100.0)
    return prices, contracts


class TestPassiveMM(unittest.TestCase):
    def test_mean_reverting_flow_profitable(self):
        # uninformed, mean-reverting flow: MM should net positive after marking inventory
        prices, contracts = _meanrev_tape(seed=1)
        r = simulate_passive_mm(prices, contracts, half_spread=0.01, quote_size=10.0,
                                inventory_cap=100.0, ref_window=20, min_ref=5)
        self.assertGreater(r["pnl"], 0.0)
        self.assertGreater(r["n_buy"] + r["n_sell"], 50)

    def test_pure_volatility_has_adverse_selection_cost(self):
        # i.i.d. overshoot flow: gross spread is captured but realized pnl is strictly less
        # (adverse selection is a positive cost the simulator must reflect)
        prices, contracts = _noise_tape(bounce=0.03, seed=1)
        r = simulate_passive_mm(prices, contracts, half_spread=0.01, quote_size=10.0,
                                inventory_cap=100.0, ref_window=20, min_ref=5)
        self.assertGreater(r["spread_captured"], 0.0)
        self.assertLess(r["pnl"], r["spread_captured"])

    def test_trend_loses_to_adverse_selection(self):
        prices, contracts = _trend_tape(0.3, 0.7, seed=2)
        r = simulate_passive_mm(prices, contracts, half_spread=0.01, quote_size=10.0,
                                inventory_cap=100.0, ref_window=20, min_ref=5)
        # spread is nominally captured, but realized pnl is negative once inventory is marked
        self.assertLess(r["pnl"], 0.0)

    def test_inventory_cap_respected(self):
        # one-sided flow (only sells crossing our bid) must not breach the long cap
        prices = np.concatenate([np.full(40, 0.5), np.full(200, 0.40)])  # drops, lifts our bid repeatedly
        contracts = np.full(len(prices), 1000.0)
        cap = 50.0
        r = simulate_passive_mm(prices, contracts, half_spread=0.01, quote_size=20.0,
                                inventory_cap=cap, ref_window=20, min_ref=5)
        # net contracts bought minus sold cannot exceed the cap (we flatten at end, so check
        # via deployed/price staying bounded): deployed <= cap * price ceiling
        self.assertLessEqual(r["deployed"], cap * 0.5 + 1e-6)

    def test_wide_spread_no_fills(self):
        prices, contracts = _noise_tape(bounce=0.03, seed=3)
        r = simulate_passive_mm(prices, contracts, half_spread=0.20, quote_size=10.0,
                                inventory_cap=100.0, ref_window=20, min_ref=5)
        self.assertEqual(r["n_buy"] + r["n_sell"], 0)
        self.assertEqual(r["pnl"], 0.0)

    def test_fees_reduce_pnl(self):
        prices, contracts = _noise_tape(bounce=0.03, seed=4)
        base = simulate_passive_mm(prices, contracts, half_spread=0.01, quote_size=10.0,
                                   inventory_cap=100.0, ref_window=20, min_ref=5)
        feed = simulate_passive_mm(prices, contracts, half_spread=0.01, quote_size=10.0,
                                   inventory_cap=100.0, fee_rate=0.02, ref_window=20, min_ref=5)
        self.assertLess(feed["pnl"], base["pnl"])
        self.assertGreater(feed["fees"], 0.0)

    def test_vol_gate_avoids_sharp_move(self):
        # calm, then a sharp resolution-like ramp. The vol gate should stop quoting during
        # the ramp, cutting fills and improving pnl vs ungated.
        calm = np.full(120, 0.5)
        ramp = np.linspace(0.5, 0.97, 60)
        prices = np.concatenate([calm, ramp])
        contracts = np.full(len(prices), 100.0)
        ungated = simulate_passive_mm(prices, contracts, half_spread=0.01, quote_size=10.0,
                                      inventory_cap=100.0, ref_window=20, min_ref=5)
        gated = simulate_passive_mm(prices, contracts, half_spread=0.01, quote_size=10.0,
                                    inventory_cap=100.0, ref_window=20, min_ref=5, vol_gate=0.05)
        self.assertLess(gated["n_buy"] + gated["n_sell"], ungated["n_buy"] + ungated["n_sell"])
        self.assertGreater(gated["pnl"], ungated["pnl"])

    def test_resolution_gate_trims_tail(self):
        import pandas as pd
        from walk_forward_passive_mm import run_over_markets
        prices, contracts = _trend_tape(0.3, 0.9, n=1000, seed=7)
        df = pd.DataFrame({"market_id": 1, "price": prices, "contracts": contracts,
                           "month": np.zeros(len(prices), int)})
        full = run_over_markets(df, half_spread=0.01, quote_size=10.0, inventory_cap=100.0,
                                ref_window=20, min_ref=5)
        trimmed = run_over_markets(df, resolution_gate_frac=0.3, half_spread=0.01, quote_size=10.0,
                                   inventory_cap=100.0, ref_window=20, min_ref=5)
        # dropping the drifting tail should reduce the loss
        self.assertGreater(trimmed["total_pnl"], full["total_pnl"])

    def test_monthly_pnl_sums_to_total(self):
        prices, contracts = _noise_tape(seed=5)
        months = np.arange(len(prices)) // 1000  # 4 pseudo-months
        r = simulate_passive_mm(prices, contracts, half_spread=0.01, quote_size=10.0,
                                inventory_cap=100.0, ref_window=20, min_ref=5, months=months)
        self.assertAlmostEqual(sum(r["monthly_pnl"].values()), r["gross_cash"], places=4)


if __name__ == "__main__":
    unittest.main()
