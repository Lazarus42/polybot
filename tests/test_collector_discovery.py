from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from collect_clob_book import (select_target_markets, select_target_events, _is_crypto,
                               tokens_to_add, _due, resolved_asset_ids)


def ev(slug, markets, title="", tags=None, neg_risk=False, end_date=None):
    return {"slug": slug, "title": title, "markets": markets,
            "tags": [{"slug": t} for t in (tags or [])], "negRisk": neg_risk, "endDate": end_date}


def mk(slug, liq, move=0.0, vol24=0.0, q="", toks=("a", "b"), price=0.5, mid=None):
    return {"slug": slug, "question": q, "liquidityNum": liq, "oneDayPriceChange": move,
            "volume24hr": vol24, "clobTokenIds": json.dumps(list(toks)), "bestAsk": price,
            "id": mid or slug}


class TestDiscovery(unittest.TestCase):
    def test_recurring_always_excluded_substantive_crypto_optional(self):
        from collect_clob_book import _event_excluded
        recurring = ev("btc-up-or-down-15m-123", [mk("m", 1e5)], title="BTC up or down")
        subst = ev("btc-100k-2026", [mk("m", 1e5, q="Will BTC be above $100k by 2026?")], title="BTC 100k")
        normal = ev("trump-pardon", [mk("m", 1e5)], title="Will Trump pardon someone?")
        # recurring excluded even with exclude_crypto=False
        self.assertTrue(_event_excluded(recurring, exclude_crypto=False))
        # substantive crypto kept when not excluding, dropped when excluding
        self.assertFalse(_event_excluded(subst, exclude_crypto=False))
        self.assertTrue(_event_excluded(subst, exclude_crypto=True))
        self.assertFalse(_event_excluded(normal, exclude_crypto=True))

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


class TestLongshotBucket(unittest.TestCase):
    def test_cheap_legs_captured_even_when_not_top_liquidity(self):
        # big event: 15 liquid favorites + 2 penny longshots beyond the per-event cap
        favs = [mk(f"fav{i}", 1e5 - i, toks=(f"F{i}", f"f{i}"), price=0.5, mid=f"fav{i}")
                for i in range(15)]
        longs = [mk("long-a", 100, toks=("LA", "la"), price=0.03, mid="long-a"),
                 mk("long-b", 100, toks=("LB", "lb"), price=0.05, mid="long-b")]
        e = ev("race", favs + longs, title="Who wins?")
        toks = select_target_events([e], n_events=5, min_liquidity=5000)
        self.assertIn("LA", toks)   # penny legs captured despite low liquidity
        self.assertIn("LB", toks)


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


class TestCollectAll(unittest.TestCase):
    def test_all_tokens_includes_low_liq_penny_and_excludes_crypto(self):
        from collect_clob_book import all_tokens_from_events
        events = [
            ev("race", [mk("fav", 1e5, toks=("F", "f"), price=0.6, mid="fav"),
                        mk("dust", 200, toks=("D", "d"), price=0.03, mid="dust")], title="Who wins?"),
            ev("eth", [mk("m", 1e6, q="Will ETH rise?", toks=("E", "e"), mid="m")], title="ETH"),
            ev("dead", [mk("x", 50, toks=("X", "x"), price=0.5, mid="x")], title="dead market"),
        ]
        toks, meta = all_tokens_from_events(events, min_liquidity=1000, exclude_crypto=True)
        self.assertIn("F", toks)              # liquid favorite
        self.assertIn("D", toks)              # penny longshot kept despite low liquidity
        self.assertNotIn("E", toks)           # crypto excluded
        self.assertNotIn("X", toks)           # below floor and not a penny longshot
        self.assertIn("category", meta["F"])  # per-token meta carried

    def test_shard(self):
        from collect_all import shard
        s = shard(list(range(1000)), 450)
        self.assertEqual([len(x) for x in s], [450, 450, 100])


class TestIncentiveMode(unittest.TestCase):
    def _rmkt(self, slug, toks, pool, liq=1e5):
        m = mk(slug, liq, toks=toks, mid=slug)
        m["rewardsMaxSpread"] = 3.0
        m["clobRewards"] = [{"rewardsDailyRate": pool}]
        return m

    def test_incentive_keeps_only_sports_rewards_ranked_by_pool(self):
        from collect_clob_book import select_target_event_records
        nba = ev("nba-lakers", [self._rmkt("nba", ("NBA1", "NBA2"), 7700)],
                 title="Lakers win?", tags=["sports", "nba"])
        soccer = ev("epl-game", [self._rmkt("epl", ("EPL1", "EPL2"), 10000)],
                    title="EPL", tags=["sports", "soccer"])
        politics = ev("senate", [self._rmkt("sen", ("SEN1", "SEN2"), 99999)],
                      title="Senate", tags=["politics"])           # rich but wrong category
        sports_norew = ev("nfl-noreward", [mk("nfl", 1e5, toks=("NFL1", "NFL2"), mid="nfl")],
                          title="NFL", tags=["sports", "nfl"])     # sports but no rewards
        recs = select_target_event_records([nba, soccer, politics, sports_norew],
                                           n_events=10, min_liquidity=5000,
                                           incentive_categories={"sports"})
        slugs = [r["slug"] for r in recs]
        self.assertEqual(slugs, ["epl-game", "nba-lakers"])   # only sports+rewards, by pool desc
        self.assertEqual(recs[0]["bucket"], "incentive")
        self.assertAlmostEqual(recs[0]["reward_est"], 10000.0)
        self.assertNotIn("senate", slugs)        # wrong category dropped
        self.assertNotIn("nfl-noreward", slugs)  # no rewards dropped


class TestRediscovery(unittest.TestCase):
    def test_tokens_to_add_only_new(self):
        current = {"a", "b"}
        self.assertEqual(tokens_to_add(current, ["a", "c", "b", "d", "c"]), ["c", "d"])
        self.assertEqual(tokens_to_add(current, ["a", "b"]), [])

    def test_due(self):
        self.assertTrue(_due(0.0, 200.0, 180.0))     # 200s elapsed >= 180s
        self.assertFalse(_due(100.0, 200.0, 180.0))  # only 100s elapsed
        self.assertFalse(_due(0.0, 200.0, 0.0))      # interval 0 = disabled

    def test_resolved_asset_ids(self):
        payload = [{"event_type": "market_resolved", "winning_asset_id": "W1", "winning_outcome": "Yes"},
                   {"event_type": "price_change", "price_changes": []}]
        self.assertEqual(resolved_asset_ids(payload), ["W1"])
        self.assertEqual(resolved_asset_ids({"event_type": "book", "asset_id": "X"}), [])


if __name__ == "__main__":
    unittest.main()
