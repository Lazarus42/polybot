from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from arb_monitor import (basket_opportunities, evaluate, extract_targets, make_scan_record,
                         ordered_token_set, rank_opportunities, top_of_book)


def leg(token, bid=None, ask=None):
    return {"token": token, "bid": bid, "ask": ask}


class TestTopOfBook(unittest.TestCase):
    def test_best_is_max_bid_min_ask_regardless_of_order(self):
        book = {"bids": [{"price": "0.40", "size": "10"}, {"price": "0.45", "size": "5"}],
                "asks": [{"price": "0.55", "size": "7"}, {"price": "0.50", "size": "3"}]}
        tob = top_of_book(book)
        self.assertEqual(tob["bid"], (0.45, 5.0))
        self.assertEqual(tob["ask"], (0.50, 3.0))

    def test_zero_size_and_junk_levels_ignored(self):
        book = {"bids": [{"price": "0.99", "size": "0"}, {"price": "bad", "size": "5"}],
                "asks": []}
        tob = top_of_book(book)
        self.assertIsNone(tob["bid"])
        self.assertIsNone(tob["ask"])

    def test_empty_book(self):
        tob = top_of_book({})
        self.assertIsNone(tob["bid"])
        self.assertIsNone(tob["ask"])


class TestBasketMath(unittest.TestCase):
    def test_buy_basket_arb(self):
        legs = [leg("a", ask=(0.30, 100)), leg("b", ask=(0.30, 100)), leg("c", ask=(0.35, 100))]
        opps = basket_opportunities(legs, threshold=0.005)
        self.assertEqual(len(opps), 1)
        o = opps[0]
        self.assertEqual(o["side"], "buy")
        self.assertEqual(o["n_legs"], 3)
        self.assertAlmostEqual(o["sum"], 0.95)
        self.assertAlmostEqual(o["deviation"], 0.05)
        self.assertEqual([l["token"] for l in o["legs"]], ["a", "b", "c"])

    def test_missing_ask_invalidates_buy_but_not_sell(self):
        # leg b has no ask -> the basket cannot be bought; sell side still evaluable
        legs = [leg("a", bid=(0.40, 10), ask=(0.30, 10)),
                leg("b", bid=(0.40, 10), ask=None),
                leg("c", bid=(0.25, 10), ask=(0.30, 10))]
        opps = basket_opportunities(legs, threshold=0.005)
        self.assertEqual([o["side"] for o in opps], ["sell"])
        self.assertAlmostEqual(opps[0]["sum"], 1.05)
        self.assertAlmostEqual(opps[0]["deviation"], 0.05)

    def test_missing_bid_invalidates_sell(self):
        legs = [leg("a", bid=(0.60, 10)), leg("b", bid=None), leg("c", bid=(0.60, 10))]
        self.assertEqual(basket_opportunities(legs, threshold=0.005), [])

    def test_no_arb_when_sum_fair(self):
        legs = [leg("a", bid=(0.49, 10), ask=(0.51, 10)),
                leg("b", bid=(0.49, 10), ask=(0.51, 10))]
        self.assertEqual(basket_opportunities(legs, threshold=0.005), [])

    def test_empty_legs_produce_nothing(self):
        self.assertEqual(basket_opportunities([], threshold=0.005), [])

    def test_depth_bound_is_min_usd_across_legs(self):
        # touch depth in USD = size * price per leg; bound is the MIN, here leg b: 10 * 0.40 = $4
        legs = [leg("a", ask=(0.50, 100)), leg("b", ask=(0.40, 10))]
        o = basket_opportunities(legs, threshold=0.005)[0]
        self.assertEqual(o["min_touch_depth_usd"], 4.0)

    def test_threshold_edge_inclusive(self):
        # deviation exactly at threshold (1 - 0.995 = 0.005) must be logged
        legs = [leg("a", ask=(0.495, 10)), leg("b", ask=(0.5, 10))]
        opps = basket_opportunities(legs, threshold=0.005)
        self.assertEqual(len(opps), 1)
        self.assertAlmostEqual(opps[0]["deviation"], 0.005)

    def test_threshold_edge_just_below_excluded(self):
        legs = [leg("a", ask=(0.4951, 10)), leg("b", ask=(0.5, 10))]
        self.assertEqual(basket_opportunities(legs, threshold=0.005), [])


class TestParityMath(unittest.TestCase):
    """YES/NO parity is the 2-leg basket case."""

    def test_parity_buy(self):
        legs = [leg("yes", ask=(0.40, 50)), leg("no", ask=(0.55, 20))]
        o = basket_opportunities(legs, threshold=0.005)[0]
        self.assertEqual(o["side"], "buy")
        self.assertAlmostEqual(o["sum"], 0.95)
        self.assertAlmostEqual(o["deviation"], 0.05)
        self.assertEqual(o["min_touch_depth_usd"], 11.0)   # min(0.40*50=20, 0.55*20=11)

    def test_parity_sell(self):
        legs = [leg("yes", bid=(0.60, 10)), leg("no", bid=(0.42, 10))]
        o = basket_opportunities(legs, threshold=0.005)[0]
        self.assertEqual(o["side"], "sell")
        self.assertAlmostEqual(o["deviation"], 0.02)

    def test_parity_one_sided_book_no_buy(self):
        legs = [leg("yes", ask=(0.40, 50)), leg("no", ask=None)]
        self.assertEqual(basket_opportunities(legs, threshold=0.005), [])


class TestRanking(unittest.TestCase):
    def test_ranked_by_deviation_times_depth_and_capped(self):
        opps = [{"deviation": 0.01 * (i % 7 + 1), "min_touch_depth_usd": float(i + 1)}
                for i in range(60)]
        top = rank_opportunities(opps, max_n=50)
        self.assertEqual(len(top), 50)
        scores = [o["deviation"] * o["min_touch_depth_usd"] for o in top]
        self.assertEqual(scores, sorted(scores, reverse=True))
        # the kept set is exactly the 50 best by score
        all_scores = sorted((o["deviation"] * o["min_touch_depth_usd"] for o in opps),
                            reverse=True)
        self.assertAlmostEqual(sum(scores), sum(all_scores[:50]))


class TestScanRecord(unittest.TestCase):
    def test_shape_and_zero_opportunity_run(self):
        rec = make_scan_record("2026-07-09T00:00:00Z", 12, 340, [], 45.678)
        self.assertEqual(set(rec), {"ts", "n_events_checked", "n_binary_checked",
                                    "n_opportunities", "max_deviation", "runtime_s"})
        self.assertEqual(rec["n_opportunities"], 0)
        self.assertEqual(rec["max_deviation"], 0.0)
        self.assertEqual(rec["runtime_s"], 45.68)
        self.assertEqual(rec["n_events_checked"], 12)
        self.assertEqual(rec["n_binary_checked"], 340)

    def test_max_deviation_over_opportunities(self):
        opps = [{"deviation": 0.006}, {"deviation": 0.031}, {"deviation": 0.012}]
        rec = make_scan_record("t", 1, 2, opps, 1.0)
        self.assertEqual(rec["n_opportunities"], 3)
        self.assertAlmostEqual(rec["max_deviation"], 0.031)


def market(slug, tokens, outcomes=("Yes", "No"), **kw):
    m = {"slug": slug, "clobTokenIds": json.dumps(list(tokens)),
         "outcomes": json.dumps(list(outcomes))}
    m.update(kw)
    return m


class TestExtractTargets(unittest.TestCase):
    def test_negrisk_event_with_closed_leg_dropped_from_basket(self):
        ev = {"slug": "election", "negRisk": True, "markets": [
            market("a", ["101", "102"]),
            market("b", ["201", "202"]),
            market("c", ["301", "302"]),
            market("d", ["401", "402"], closed=True),      # resolved leg: excluded, basket ok
        ]}
        negrisk, binary = extract_targets([ev])
        self.assertEqual(len(negrisk), 1)
        self.assertEqual(negrisk[0]["slug"], "election")
        self.assertEqual(negrisk[0]["yes_tokens"], ["101", "201", "301"])
        self.assertEqual(len(binary), 3)                    # closed market not a parity target

    def test_tokens_stay_strings(self):
        # token ids are ~77-digit integers; they must never be parsed to int/float
        big = "21742633143463906290569050155826241533067272736897614950488156847949938836455"
        ev = {"slug": "e", "negRisk": True, "markets": [
            market("m1", [big, "2"]), market("m2", ["3", "4"]), market("m3", ["5", "6"])]}
        negrisk, binary = extract_targets([ev])
        self.assertEqual(negrisk[0]["yes_tokens"][0], big)
        self.assertIsInstance(binary[0]["yes"], str)

    def test_negrisk_needs_three_open_markets(self):
        ev = {"slug": "e", "negRisk": True, "markets": [
            market("m1", ["1", "2"]), market("m2", ["3", "4"])]}
        negrisk, binary = extract_targets([ev])
        self.assertEqual(negrisk, [])
        self.assertEqual(len(binary), 2)                    # still parity candidates

    def test_non_negrisk_event_is_binary_only(self):
        ev = {"slug": "e", "markets": [market("m1", ["1", "2"]),
                                       market("m2", ["3", "4"]),
                                       market("m3", ["5", "6"])]}
        negrisk, binary = extract_targets([ev])
        self.assertEqual(negrisk, [])
        self.assertEqual(len(binary), 3)

    def test_yes_index_respects_outcome_order(self):
        ev = {"slug": "e", "markets": [market("m1", ["10", "20"], outcomes=("No", "Yes"))]}
        _, binary = extract_targets([ev])
        self.assertEqual(binary[0], {"slug": "m1", "yes": "20", "no": "10"})

    def test_malformed_leg_invalidates_whole_basket(self):
        # an open outcome market without usable tokens -> we cannot price the full basket
        ev = {"slug": "e", "negRisk": True, "markets": [
            market("m1", ["1", "2"]), market("m2", ["3", "4"]), market("m3", ["5", "6"]),
            {"slug": "broken", "clobTokenIds": "not-json"}]}
        negrisk, _ = extract_targets([ev])
        self.assertEqual(negrisk, [])

    def test_ordered_token_set_dedupes(self):
        negrisk = [{"slug": "e", "yes_tokens": ["1", "3", "9"]}]
        binary = [{"slug": "m1", "yes": "1", "no": "2"}, {"slug": "m2", "yes": "3", "no": "4"}]
        self.assertEqual(ordered_token_set(negrisk, binary), ["1", "2", "3", "4", "9"])


class TestEvaluate(unittest.TestCase):
    def _books(self, quotes):
        # quotes: token -> (bid_price, ask_price), size 10 at both touches
        return {t: {"asset_id": t,
                    "bids": [{"price": str(b), "size": "10"}],
                    "asks": [{"price": str(a), "size": "10"}]}
                for t, (b, a) in quotes.items()}

    def test_counts_and_kinds(self):
        negrisk = [{"slug": "ev", "yes_tokens": ["1", "3", "5"]}]
        binary = [{"slug": "m1", "yes": "1", "no": "2"},
                  {"slug": "mx", "yes": "7", "no": "8"}]     # 8 missing -> not checked
        books = self._books({"1": (0.28, 0.30), "2": (0.63, 0.65),
                             "3": (0.28, 0.30), "5": (0.28, 0.30), "7": (0.5, 0.5)})
        opps, n_ev, n_bin = evaluate(negrisk, binary, books, ts="T", threshold=0.005)
        self.assertEqual(n_ev, 1)
        self.assertEqual(n_bin, 1)                           # mx skipped: missing book
        kinds = sorted(o["kind"] for o in opps)
        self.assertEqual(kinds, ["negrisk_buy", "parity_buy"])   # 0.90 basket, 0.95 parity
        for o in opps:
            self.assertEqual(set(o), {"ts", "kind", "slug", "n_legs", "sum", "deviation",
                                      "min_touch_depth_usd", "legs"})
            self.assertEqual(o["ts"], "T")

    def test_partial_basket_books_skip_event(self):
        negrisk = [{"slug": "ev", "yes_tokens": ["1", "3", "5"]}]
        books = self._books({"1": (0.1, 0.2), "3": (0.1, 0.2)})   # leg 5 unfetched
        opps, n_ev, _ = evaluate(negrisk, [], books, ts="T")
        self.assertEqual(n_ev, 0)
        self.assertEqual(opps, [])


if __name__ == "__main__":
    unittest.main()
