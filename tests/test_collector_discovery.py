from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from collect_clob_book import select_target_markets, select_target_events, _is_crypto


def ev(slug, markets, title="", tags=None, neg_risk=False, end_date=None):
    return {"slug": slug, "title": title, "markets": markets,
            "tags": [{"slug": t} for t in (tags or [])], "negRisk": neg_risk, "endDate": end_date}


def mk(slug, liq, move=0.0, vol24=0.0, q="", toks=("a", "b")):
    return {"slug": slug, "question": q, "liquidityNum": liq, "oneDayPriceChange": move,
            "volume24hr": vol24, "clobTokenIds": json.dumps(list(toks))}


class TestDiscovery(unittest.TestCase):
    def test_excludes_crypto(self):
        self.assertTrue(_is_crypto(mk("bitcoin-up-or-down-15m-123", 1e5)))
        self.assertTrue(_is_crypto({"slug": "x", "question": "Will ETH be above $4000?"}))
        self.assertFalse(_is_crypto(mk("will-trump-pardon-someone", 1e5)))

    def test_min_liquidity_filter(self):
        markets = [mk("low", 1000, toks=("LOW1", "LOW2")), mk("ok", 50000, toks=("OK1", "OK2"))]
        toks = select_target_markets(markets, n_markets=10, min_liquidity=5000)
        self.assertNotIn("LOW1", toks)   # below floor excluded
        self.assertIn("OK1", toks)       # above floor kept

    def test_picks_both_liquid_and_volatile(self):
        markets = [
            mk("liquid-calm", liq=1e6, move=0.0, toks=("L1", "L2")),     # very liquid, no move
            mk("mid-volatile", liq=2e4, move=0.30, toks=("V1", "V2")),   # moderate liq, big move
            mk("crypto-eth", liq=1e6, move=0.5, q="ETH up?", toks=("C1", "C2")),  # excluded
            mk("filler", liq=1e4, move=0.01, toks=("F1", "F2")),
        ]
        toks = select_target_markets(markets, n_markets=2, min_liquidity=5000)
        # both the liquid-calm and the volatile market should be represented; crypto excluded
        self.assertIn("L1", toks)
        self.assertIn("V1", toks)
        self.assertNotIn("C1", toks)

    def test_returns_both_tokens_per_market(self):
        toks = select_target_markets([mk("m", 1e5, toks=("t1", "t2"))], n_markets=1, min_liquidity=5000)
        self.assertEqual(set(toks), {"t1", "t2"})


class TestTaxonomy(unittest.TestCase):
    def test_meta_and_rewards_tagged(self):
        from collect_clob_book import select_target_event_records, _market_has_rewards
        m = mk("s", 1e5, toks=("S1", "S2")); m["rewardsMaxSpread"] = 3.0
        self.assertTrue(_market_has_rewards(m))
        event = ev("senate-race", [m], title="Senate", tags=["politics", "elections"],
                   neg_risk=True, end_date="2027-01-01T00:00:00Z")
        recs = select_target_event_records([event], n_events=4, min_liquidity=5000)
        r = recs[0]
        self.assertEqual(r["category"], "politics")
        self.assertTrue(r["neg_risk"])
        self.assertTrue(r["rewards"])
        self.assertIn("horizon_days", r)


class TestEventSelection(unittest.TestCase):
    def test_captures_one_token_per_outcome_of_multioutcome_event(self):
        multi = ev("election", [mk("c-a", 1e5, toks=("A1", "A2")),
                                mk("c-b", 1e5, toks=("B1", "B2")),
                                mk("c-c", 1e5, toks=("C1", "C2"))], title="Who wins?")
        toks = select_target_events([multi], n_events=3, min_liquidity=5000)
        for t in ("A1", "B1", "C1"):
            self.assertIn(t, toks)            # token1 of each outcome (for basket arb)
        for t in ("A2", "B2", "C2"):
            self.assertNotIn(t, toks)         # NO tokens dropped (redundant)

    def test_caps_markets_per_event(self):
        big = ev("giant", [mk(f"c{i}", 1e5 - i, toks=(f"T{i}a", f"T{i}b")) for i in range(40)])
        toks = select_target_events([big], n_events=1, min_liquidity=5000)
        self.assertLessEqual(len(toks), 15)   # capped at max_markets_per_event (15)

    def test_excludes_crypto_event_and_respects_token_cap(self):
        crypto = ev("eth-stuff", [mk("m", 1e6, q="Will ETH rise?", toks=("X1", "X2"))], title="ETH")
        normal = ev("senate", [mk("s", 1e5, toks=("S1", "S2"))], title="Senate race")
        toks = select_target_events([crypto, normal], n_events=5, min_liquidity=5000, max_tokens=2)
        self.assertNotIn("X1", toks)
        self.assertLessEqual(len(toks), 2)


if __name__ == "__main__":
    unittest.main()
