from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import paper_sim as ps
from quoter import Holder


class TestHolderPriceExits(unittest.TestCase):
    def _buy(self, holder, t=0.0, bid=0.04, ask=0.05):
        holder.on_quote(t, bid, ask, ask_levels=[(ask, 1000.0)])
        assert holder.bought and holder.inv > 0

    def test_take_profit_sells_at_bid(self):
        h = Holder(100.0, buy_lo=0.01, buy_hi=0.05, take_profit_price=0.15)
        self._buy(h)
        cash_after_buy = h.cash
        h.on_quote(60.0, 0.10, 0.12)               # mid 0.11 < 0.15: still holding
        self.assertGreater(h.inv, 0)
        h.on_quote(120.0, 0.14, 0.18)              # mid 0.16 >= 0.15: sell all at bid 0.14
        self.assertEqual(h.inv, 0.0)
        self.assertAlmostEqual(h.cash, cash_after_buy + 100.0 * 0.14)
        self.assertEqual(h.n_flats, 1)
        self.assertEqual(h.fills[-1][1], -1)

    def test_no_reentry_after_exit(self):
        h = Holder(100.0, buy_lo=0.01, buy_hi=0.05, take_profit_price=0.15)
        self._buy(h)
        h.on_quote(60.0, 0.15, 0.17)               # exit
        h.on_quote(120.0, 0.03, 0.05)              # back in band: must NOT re-buy
        self.assertEqual(h.inv, 0.0)
        self.assertTrue(h.bought)

    def test_stop_loss_sells_at_bid(self):
        h = Holder(100.0, buy_lo=0.90, buy_hi=0.99, stop_loss_price=0.60)
        h.on_quote(0.0, 0.91, 0.93, ask_levels=[(0.93, 1000.0)])
        self.assertGreater(h.inv, 0)
        h.on_quote(60.0, 0.70, 0.74)               # mid 0.72 > 0.60: hold
        self.assertGreater(h.inv, 0)
        h.on_quote(120.0, 0.55, 0.61)              # mid 0.58 <= 0.60: cut at bid 0.55
        self.assertEqual(h.inv, 0.0)

    def test_disabled_exits_hold_to_resolution(self):
        h = Holder(100.0, buy_lo=0.01, buy_hi=0.05)
        self._buy(h)
        h.on_quote(60.0, 0.90, 0.95)               # huge pop, but no take-profit configured
        self.assertGreater(h.inv, 0)


class TestTokenMetaTags(unittest.TestCase):
    def test_load_token_meta_carries_gate_fields(self):
        man = {"token_meta": {"T1": {"reward_daily_est": 10, "rewards_min_size": 50,
                                     "rewards_max_spread": 0.03, "question": "Q?",
                                     "category": "sports", "horizon_days": 7.5, "neg_risk": True},
                              "T2": {"reward_daily_est": 5, "rewards_min_size": 50,
                                     "rewards_max_spread": 0.03, "question": "Q2?"}}}
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "manifest.json"
            p.write_text(json.dumps(man))
            meta = ps.load_token_meta(p)
        self.assertEqual(meta["T1"]["category"], "sports")
        self.assertEqual(meta["T1"]["horizon_days"], 7.5)
        self.assertTrue(meta["T1"]["neg_risk"])
        self.assertIsNone(meta["T2"]["category"])
        self.assertIsNone(meta["T2"]["horizon_days"])
        self.assertFalse(meta["T2"]["neg_risk"])


class TestPerConfigMarketGate(unittest.TestCase):
    def _sim(self, out, extra_configs):
        ps.CONFIGS.update(extra_configs)
        meta = {"T": {"pool": 1440.0, "min_size": 100.0, "v_cents": 3.0, "question": "Will T win?",
                      "category": "politics", "horizon_days": 30.0, "neg_risk": False}}
        return ps.PaperSim(meta, size=200.0, inv_cap_mult=5.0, configs=list(extra_configs),
                           fill_model="prorata", capture_mult=1.0, out_dir=out, rotate_minutes=15.0)

    def test_horizon_gate_blocks_far_market_and_unknown_passes(self):
        cfgs = {"h_gate": dict(holder="longshot", buy_lo=0.01, buy_hi=0.05, max_horizon_days=14.0)}
        with tempfile.TemporaryDirectory() as td:
            sim = self._sim(Path(td), cfgs)
            self.assertFalse(sim._ensure("T"))                 # 30d > 14d gate
            sim.meta["T"]["horizon_days"] = None               # unknown horizon must PASS
            self.assertTrue(sim._ensure("T"))
            self.assertIn(("h_gate", "T"), sim.q)

    def test_min_horizon_gate_blocks_near_market_and_unknown_passes(self):
        cfgs = {"nh_gate": dict(holder="tail", buy_lo=0.55, buy_hi=0.90, min_horizon_days=7.0)}
        with tempfile.TemporaryDirectory() as td:
            sim = self._sim(Path(td), cfgs)
            sim.meta["T"]["horizon_days"] = 2.0
            self.assertFalse(sim._ensure("T"))                 # 2d < 7d min gate
            sim.meta["T"]["horizon_days"] = None               # unknown horizon must PASS
            self.assertTrue(sim._ensure("T"))

    def test_category_gate(self):
        cfgs = {"c_gate": dict(holder="longshot", buy_lo=0.01, buy_hi=0.05,
                               categories={"sports", "esports"})}
        with tempfile.TemporaryDirectory() as td:
            sim = self._sim(Path(td), cfgs)
            self.assertFalse(sim._ensure("T"))                 # politics not in {sports, esports}
            sim.meta["T"]["category"] = "sports"
            self.assertTrue(sim._ensure("T"))

    def test_maker_category_gate_and_kwargs_popped(self):
        cfgs = {"m_gate": dict(categories={"politics"}, roc_ceil=0.0, no_cull=True)}
        with tempfile.TemporaryDirectory() as td:
            sim = self._sim(Path(td), cfgs)
            self.assertTrue(sim._ensure("T"))                  # politics matches -> Quoter built
            self.assertIn(("m_gate", "T"), sim.q)              # (categories popped, not a kwarg)

    def test_exclude_categories_gate(self):
        cfgs = {"x_gate": dict(holder="longshot", buy_lo=0.01, buy_hi=0.05,
                               exclude_categories={"politics", "crypto"})}
        with tempfile.TemporaryDirectory() as td:
            sim = self._sim(Path(td), cfgs)
            self.assertFalse(sim._ensure("T"))                 # politics is excluded
            sim.meta["T"]["category"] = "sports"
            self.assertTrue(sim._ensure("T"))

    def test_max_reward_pool_gate(self):
        cfgs = {"p_gate": dict(holder="longshot", buy_lo=0.01, buy_hi=0.05, max_reward_pool=1.0)}
        with tempfile.TemporaryDirectory() as td:
            sim = self._sim(Path(td), cfgs)
            self.assertFalse(sim._ensure("T"))                 # pool=1440 > 1.0
            sim.meta["T"]["pool"] = 0.5                        # tiny pool passes
            self.assertTrue(sim._ensure("T"))

    def test_holder_size_mult(self):
        cfgs = {"big_clip": dict(holder="longshot", buy_lo=0.01, buy_hi=0.05, size_mult=5.0),
                "one_clip": dict(holder="longshot", buy_lo=0.01, buy_hi=0.05)}
        with tempfile.TemporaryDirectory() as td:
            sim = self._sim(Path(td), cfgs)
            self.assertTrue(sim._ensure("T"))
            self.assertEqual(sim.q[("big_clip", "T")].our_size,
                             5.0 * sim.q[("one_clip", "T")].our_size)

    def test_all_service_configs_construct(self):
        # every config named in the systemd unit must exist in CONFIGS and build a working
        # Quoter/Holder for a plain durable market (catches name typos before deploy)
        svc = (Path(__file__).resolve().parents[1] / "deploy" / "polybot-paper-sim.service").read_text()
        args = svc.split("--configs")[1].split("--size")[0].replace("\\", " ").split()
        self.assertGreaterEqual(len(args), 26)
        missing = [a for a in args if a not in ps.CONFIGS]
        self.assertEqual(missing, [])
        with tempfile.TemporaryDirectory() as td:
            meta = {"T": {"pool": 1440.0, "min_size": 100.0, "v_cents": 3.0,
                          "question": "Will T win?", "category": "tech",
                          "horizon_days": 10.0, "neg_risk": False}}
            sim = ps.PaperSim(meta, size=200.0, inv_cap_mult=5.0, configs=args,
                              fill_model="prorata", capture_mult=1.0, out_dir=Path(td),
                              rotate_minutes=15.0)
            self.assertTrue(sim._ensure("T"))   # no constructor blowups across the full set

    def test_old_meta_without_tag_fields_passes_gates(self):
        cfgs = {"h_gate2": dict(holder="longshot", buy_lo=0.01, buy_hi=0.05, max_horizon_days=14.0)}
        with tempfile.TemporaryDirectory() as td:
            sim = self._sim(Path(td), cfgs)
            sim.meta["T"] = {"pool": 1440.0, "min_size": 100.0, "v_cents": 3.0,
                             "question": "Will T win?"}       # legacy meta, no tag keys
            self.assertTrue(sim._ensure("T"))


if __name__ == "__main__":
    unittest.main()
