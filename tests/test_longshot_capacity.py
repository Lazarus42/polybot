from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import longshot_capacity_analysis as lca
import longshot_market_analysis as lma


class TestSweepSeries(unittest.TestCase):
    def test_overlapping_intervals_stack(self):
        s = lca.sweep_series([(0, 10, 5.0), (5, 15, 3.0)])
        self.assertEqual(s, [(0, 5.0), (5, 8.0), (10, 3.0), (15, 0.0)])

    def test_degenerate_intervals_ignored(self):
        self.assertEqual(lca.sweep_series([(5, 5, 1.0), (7, 3, 1.0), (0, 1, 0.0),
                                           (None, 5, 1.0), (0, None, 1.0)]), [])

    def test_same_timestamp_events_merge(self):
        s = lca.sweep_series([(0, 5, 2.0), (5, 10, 4.0)])
        self.assertEqual(s, [(0, 2.0), (5, 4.0), (10, 0.0)])


class TestTimeWeightedStats(unittest.TestCase):
    def test_step_series_stats(self):
        # level 10 for 100s, level 2 for 300s -> mean = (10*100+2*300)/400 = 4.0
        series = [(0, 10.0), (100, 2.0), (400, 0.0)]
        st_ = lca.time_weighted_stats(series)
        self.assertAlmostEqual(st_["mean"], 4.0)
        self.assertEqual(st_["peak"], 10.0)
        self.assertAlmostEqual(st_["median"], 2.0)   # 2 holds 75% of the time
        self.assertAlmostEqual(st_["p90"], 10.0)     # top 25% of time is at 10

    def test_until_extends_final_level(self):
        series = [(0, 4.0)]
        st_ = lca.time_weighted_stats(series, until=100)
        self.assertAlmostEqual(st_["mean"], 4.0)
        self.assertAlmostEqual(st_["median"], 4.0)

    def test_empty(self):
        self.assertEqual(lca.time_weighted_stats([])["peak"], 0.0)


class TestAnalyzeCell(unittest.TestCase):
    def _clip(self, e_t, l_t, cost, e_ask, entry=0.045, pnl=1.0, resolved=True):
        return {"e_t": e_t, "l_t": l_t, "cost": cost, "e_ask": e_ask, "entry": entry,
                "pnl": pnl, "resolved": resolved}

    def test_metrics(self):
        day = 86400.0
        clips = [self._clip(0, day, 10.0, 1000.0),          # B contrib 1000*0.045 = 45
                 self._clip(0, 2 * day, 10.0, 2000.0)]      # B contrib 90
        res = lca.analyze_cell(clips, bankroll=5000.0, until=2 * day, label="t")
        self.assertEqual(res["clips"], 2)
        self.assertAlmostEqual(res["A_deployed"]["peak"], 20.0)
        self.assertAlmostEqual(res["B_ask_depth"]["peak"], 135.0)
        # first day level A=20, second day A=10 -> median 10 or 20 boundary; mean = 15
        self.assertAlmostEqual(res["A_deployed"]["mean"], 15.0)
        self.assertAlmostEqual(res["pnl_per_day"], 1.0)     # 2.0 pnl / 2 days
        self.assertAlmostEqual(res["bankroll_pct_per_day"], 100.0 * 1.0 / 5000.0)
        self.assertAlmostEqual(res["bankroll_pct_per_day_haircut50"],
                               res["bankroll_pct_per_day"] / 2.0)
        self.assertAlmostEqual(res["hold_hours_median"], 36.0)  # (24h + 48h)/2

    def test_missing_ask_fraction(self):
        clips = [self._clip(0, 10, 1.0, None), self._clip(0, 10, 1.0, 100.0)]
        res = lca.analyze_cell(clips, bankroll=5000.0, label="t")
        self.assertAlmostEqual(res["missing_ask_frac"], 0.5)

    def test_invalid_clips_dropped(self):
        clips = [self._clip(None, 10, 1.0, 1.0), self._clip(10, 5, 1.0, 1.0)]
        res = lca.analyze_cell(clips, bankroll=5000.0, label="t")
        self.assertEqual(res["clips"], 0)


class TestCollectClipsCapacityFields(unittest.TestCase):
    def test_clips_expose_entry_exit_and_ask_depth(self):
        rows = [
            {"t": 100.0, "token": "tok1", "config": "longshot", "mid": 0.045, "inv": 0.0,
             "marked_pnl": 0.0, "q_bid_book": 3000.0, "q_ask_book": 4000.0},
            {"t": 160.0, "token": "tok1", "config": "longshot", "mid": 0.045, "inv": 80.0,
             "marked_pnl": 0.0, "q_bid_book": 3000.0, "q_ask_book": 4000.0},
            {"t": 220.0, "token": "tok1", "config": "longshot", "mid": 0.02, "inv": 80.0,
             "marked_pnl": -2.0, "q_bid_book": 100.0, "q_ask_book": 100.0},
        ]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "paper_host_123_1.jsonl.gz"
            with gzip.open(p, "wt") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            clips = lma.collect_clips(str(Path(td) / "paper_*.jsonl.gz"), {}, 0.10, 0.90)
        self.assertEqual(len(clips), 1)
        c = clips[0]
        self.assertEqual(c["e_t"], 160.0)
        self.assertEqual(c["l_t"], 220.0)
        self.assertEqual(c["e_ask"], 4000.0)
        self.assertEqual(c["entry_bucket"], "4-5c")
        self.assertEqual(c["liq_bucket"], "<10k")
        self.assertAlmostEqual(c["cost"], 80.0 * 0.045)


if __name__ == "__main__":
    unittest.main()
