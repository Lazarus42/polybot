from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from replay_family_ensemble_oos import replay_ensemble


def _one_trade(unit_return: float, fill: float, stake: float = 1000.0) -> pd.DataFrame:
    """A single signal, sized so the participation cap is the binding constraint."""
    return pd.DataFrame([{
        "timestamp": pd.Timestamp("2024-01-01", tz="UTC"),
        "market_id": 1,
        "component": "c",
        "stake": stake,
        "monthly_cap": 1e9,
        "priority": 1,
        "unit_return": unit_return,
        "entry_fill_usd": fill,
    }])


def _run(df: pd.DataFrame, **kw):
    return replay_ensemble(
        df, pd.Timestamp("2024-02-01", tz="UTC"),
        initial_cash=1e6, period_budget=1e9, budget_period="month",
        reserve_fraction=0.0, min_stake=0.0, max_trades_per_market=1,
        max_components_per_market=1, participation_fraction=0.10, **kw,
    )


class TestSlippageModel(unittest.TestCase):
    def test_off_by_default(self):
        # debit = 0.10 * fill = 100; no slippage -> profit = 100 * 1.0
        r = _run(_one_trade(1.0, fill=1000.0))
        self.assertAlmostEqual(r["realized_profit"], 100.0, places=6)
        self.assertEqual(r["slippage_cost"], 0.0)

    def test_linear_impact_shaves_winner(self):
        # debit = 100, ratio = debit/fill = 100/1000 = 0.10, s = 0.5*0.10 = 0.05
        # r_eff = (1+1.0)/(1+0.05) - 1 = 0.904761...; profit = 100 * r_eff
        r = _run(_one_trade(1.0, fill=1000.0), slippage_model="linear", slippage_coef=0.5)
        s = 0.5 * 0.10
        expected = 100.0 * ((1.0 + 1.0) / (1.0 + s) - 1.0)
        self.assertAlmostEqual(r["realized_profit"], expected, places=6)
        self.assertAlmostEqual(r["slippage_cost"], 100.0 * 1.0 - expected, places=6)

    def test_total_loss_unchanged(self):
        # unit_return = -1 (lose the whole stake): impact must not change the loss
        base = _run(_one_trade(-1.0, fill=1000.0))
        slipped = _run(_one_trade(-1.0, fill=1000.0), slippage_model="linear", slippage_coef=0.9)
        self.assertAlmostEqual(base["realized_profit"], slipped["realized_profit"], places=6)
        self.assertAlmostEqual(slipped["slippage_cost"], 0.0, places=6)

    def test_sqrt_vs_linear_at_small_participation(self):
        # at ratio<1, sqrt(ratio) > ratio, so sqrt impact is harsher on small trades
        lin = _run(_one_trade(1.0, fill=1000.0), slippage_model="linear", slippage_coef=0.5)
        sq = _run(_one_trade(1.0, fill=1000.0), slippage_model="sqrt", slippage_coef=0.5)
        self.assertLess(sq["realized_profit"], lin["realized_profit"])

    def test_no_impact_when_fill_unknown(self):
        # unknown fill + active participation cap -> cap is 0, so nothing is deployed
        # and crucially no slippage is charged (impact needs a known fill).
        df = _one_trade(1.0, fill=float("nan"))
        r = _run(df, slippage_model="linear", slippage_coef=0.5)
        self.assertEqual(r["deployed"], 0.0)
        self.assertEqual(r["slippage_cost"], 0.0)

    def test_bigger_trade_pays_more_impact(self):
        # same return, larger participation fraction -> larger ratio -> more cost per $
        small = replay_ensemble(
            _one_trade(1.0, fill=1000.0), pd.Timestamp("2024-02-01", tz="UTC"),
            1e6, 1e9, "month", 0.0, 0.0, 1, 1,
            participation_fraction=0.10, slippage_model="linear", slippage_coef=0.5,
        )
        big = replay_ensemble(
            _one_trade(1.0, fill=1000.0), pd.Timestamp("2024-02-01", tz="UTC"),
            1e6, 1e9, "month", 0.0, 0.0, 1, 1,
            participation_fraction=0.50, slippage_model="linear", slippage_coef=0.5,
        )
        # cost per deployed dollar should rise with participation
        self.assertLess(small["slippage_cost"] / small["deployed"],
                        big["slippage_cost"] / big["deployed"])


class TestLiquidityScaledStake(unittest.TestCase):
    def _run(self, df, **kw):
        return replay_ensemble(
            df, pd.Timestamp("2024-02-01", tz="UTC"),
            initial_cash=1e6, period_budget=1e9, budget_period="month",
            reserve_fraction=0.0, min_stake=0.0, max_trades_per_market=1,
            max_components_per_market=1, **kw,
        )

    def test_off_keeps_flat_stake(self):
        # base stake $5, fill $1000, 20% participation -> flat behavior = min(5, 200) = 5
        r = self._run(_one_trade(1.0, fill=1000.0, stake=5.0), participation_fraction=0.20)
        self.assertAlmostEqual(r["deployed"], 5.0, places=6)

    def test_scales_up_in_deep_market(self):
        # stake_fill_fraction 0.10 on a $1000 fill -> target $100 (>> base $5), capped by
        # participation 0.20 (=$200), so debit = $100
        r = self._run(_one_trade(1.0, fill=1000.0, stake=5.0),
                      participation_fraction=0.20, stake_fill_fraction=0.10)
        self.assertAlmostEqual(r["deployed"], 100.0, places=6)

    def test_max_stake_caps_exposure(self):
        r = self._run(_one_trade(1.0, fill=1000.0, stake=5.0),
                      participation_fraction=0.20, stake_fill_fraction=0.10, max_stake=40.0)
        self.assertAlmostEqual(r["deployed"], 40.0, places=6)

    def test_participation_still_hard_ceiling(self):
        # stake_fill_fraction 0.50 but participation 0.10 -> participation governs = $100
        r = self._run(_one_trade(1.0, fill=1000.0, stake=5.0),
                      participation_fraction=0.10, stake_fill_fraction=0.50)
        self.assertAlmostEqual(r["deployed"], 100.0, places=6)

    def test_thin_market_keeps_base_floor(self):
        # tiny fill $20: target = max(base 5, 0.10*20=2) = 5, participation 0.20*20=4 caps -> $4
        r = self._run(_one_trade(1.0, fill=20.0, stake=5.0),
                      participation_fraction=0.20, stake_fill_fraction=0.10)
        self.assertAlmostEqual(r["deployed"], 4.0, places=6)


if __name__ == "__main__":
    unittest.main()
