from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np

from build_event_groups import (
    parse_threshold, classify_group, build_event_groups, cohort_by_overlap,
    sum_to_one_overround, ladder_monotonicity_breaks, measure_opportunity_surface,
)


class TestCohorting(unittest.TestCase):
    def test_simultaneous_collapses_to_one(self):
        # three markets all live over the same window -> one cohort
        starts = np.array([0, 1, 2]); ends = np.array([10, 10, 10])
        self.assertEqual(len(set(cohort_by_overlap(starts, ends))), 1)

    def test_recurring_disjoint_splits(self):
        # three non-overlapping windows (recurring series) -> three cohorts
        starts = np.array([0, 100, 200]); ends = np.array([10, 110, 210])
        self.assertEqual(len(set(cohort_by_overlap(starts, ends))), 3)

    def test_chain_merges(self):
        # overlapping chain A-B-C (A∩B, B∩C, but not A∩C) is still one connected cohort
        starts = np.array([0, 5, 12]); ends = np.array([6, 13, 20])
        self.assertEqual(len(set(cohort_by_overlap(starts, ends))), 1)

    def test_temporal_split_in_build(self):
        # one ticker, two disjoint time windows -> build_event_groups makes two events
        m = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "question": ["Will A win?", "Will B win?", "Will A win?", "Will B win?"],
            "market_slug": ["a-1", "b-1", "a-2", "b-2"],
            "ticker": ["race", "race", "race", "race"],
            "createdAt": ["2024-01-01", "2024-01-01", "2024-06-01", "2024-06-01"],
            "closedTime": ["2024-02-01", "2024-02-01", "2024-07-01", "2024-07-01"],
        })
        g = build_event_groups(m, use_time=True)
        self.assertEqual(g["event_id"].nunique(), 2)


class TestThresholdParsing(unittest.TestCase):
    def test_units(self):
        self.assertEqual(parse_threshold("Will BTC reach $100k?"), 100_000)
        self.assertEqual(parse_threshold("Above $1.5 million?"), 1_500_000)
        self.assertEqual(parse_threshold("Over 250,000 votes?"), 250_000)
        self.assertIsNone(parse_threshold("Will Team A win the cup?"))


class TestClassification(unittest.TestCase):
    def test_ladder(self):
        qs = ["Will BTC be above $90k?", "Will BTC be above $100k?", "Will BTC be above $110k?"]
        self.assertEqual(classify_group(qs), "ladder")

    def test_categorical(self):
        qs = ["Will Alice win?", "Will Bob win?", "Will Carol win?"]
        self.assertEqual(classify_group(qs), "categorical")

    def test_singleton(self):
        self.assertEqual(classify_group(["Will it rain?"]), "singleton")

    def test_who_wins_is_categorical(self):
        qs = ["Will Alice win the 2024 mayoral election?",
              "Will Bob win the 2024 mayoral election?",
              "Will Carol win the 2024 mayoral election?"]
        self.assertEqual(classify_group(qs), "categorical")

    def test_price_buckets_are_categorical(self):
        qs = ["Will ETH be between $4,275 and $4,300 on Sep 10?",
              "Will ETH be between $4,300 and $4,325 on Sep 10?",
              "Will ETH be between $4,325 and $4,350 on Sep 10?"]
        self.assertEqual(classify_group(qs), "categorical")

    def test_thematic_bundle_excluded(self):
        # a UFC card / sports week: independent matchups sharing a ticker, NOT exclusive
        qs = ["Pereira vs. Rountree Jr.", "Pennington vs. Pena", "Bautista vs. Aldo"]
        self.assertEqual(classify_group(qs), "thematic")


class TestConstraintMath(unittest.TestCase):
    def test_overround(self):
        self.assertAlmostEqual(sum_to_one_overround([0.5, 0.4, 0.2]), 0.1)   # sum 1.1
        self.assertAlmostEqual(sum_to_one_overround([0.3, 0.3, 0.3]), -0.1)  # underround

    def test_ladder_no_break(self):
        r = ladder_monotonicity_breaks([90, 100, 110], [0.8, 0.6, 0.4])      # monotone down
        self.assertEqual(r["n_breaks"], 0)

    def test_ladder_break(self):
        # higher strike priced HIGHER than a lower strike = arbitrage
        r = ladder_monotonicity_breaks([90, 100, 110], [0.5, 0.6, 0.4])
        self.assertEqual(r["n_breaks"], 1)
        self.assertAlmostEqual(r["max_gap"], 0.1)

    def test_ladder_unsorted_input(self):
        # thresholds out of order on input must still be evaluated sorted
        r = ladder_monotonicity_breaks([110, 90, 100], [0.4, 0.8, 0.6])
        self.assertEqual(r["n_breaks"], 0)


class TestBuildGroups(unittest.TestCase):
    def _markets(self):
        return pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "question": ["Will BTC be above $90k?", "Will BTC be above $100k?",
                         "Will BTC be above $110k?", "Will Alice win?", "Will Bob win?"],
            "market_slug": ["btc-above-90k", "btc-above-100k", "btc-above-110k",
                            "race-alice", "race-bob"],
            "ticker": ["btc-ladder", "btc-ladder", "btc-ladder", "the-race", "the-race"],
        })

    def test_grouping_and_types(self):
        g = build_event_groups(self._markets())
        self.assertEqual(g[g["id"] == "1"]["event_type"].iloc[0], "ladder")
        self.assertEqual(g[g["id"] == "4"]["event_type"].iloc[0], "categorical")
        self.assertEqual(g["event_id"].nunique(), 2)

    def test_coverage_gate_excludes_incomplete_partitions(self):
        m = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "question": ["Will A win the cup?", "Will B win the cup?",
                         "Will C win the cup?", "Will D win the cup?"],
            "market_slug": ["a", "b", "c", "d"], "ticker": ["cup"] * 4,
        })
        g = build_event_groups(m, use_time=False)
        full, _, _ = measure_opportunity_surface(g, {"1": .3, "2": .3, "3": .2, "4": .2}, min_coverage=0.9)
        partial, _, _ = measure_opportunity_surface(g, {"1": .3, "2": .3}, min_coverage=0.9)
        self.assertEqual(full["categorical_events_priced"], 1)     # complete -> measured
        self.assertEqual(partial["categorical_events_priced"], 0)  # half priced -> excluded

    def test_surface_measurement(self):
        g = build_event_groups(self._markets())
        # planted prices: ladder is monotone-clean; categorical overround +0.2
        yes = {"1": 0.8, "2": 0.6, "3": 0.4, "4": 0.7, "5": 0.5}
        surface, cat, lad = measure_opportunity_surface(g, yes, overround_thresh=0.02)
        self.assertEqual(surface["categorical_events_priced"], 1)
        self.assertEqual(surface["ladder_events_priced"], 1)
        self.assertAlmostEqual(cat["overround"].iloc[0], 0.2)
        self.assertEqual(int(lad["n_breaks"].iloc[0]), 0)


if __name__ == "__main__":
    unittest.main()
