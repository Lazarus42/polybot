from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_token_tags import merge_manifests
from longshot_market_analysis import load_manifest_tags


def _man(path: Path, created: str, token_meta: dict):
    path.write_text(json.dumps({"created": created, "token_meta": token_meta}))


class TestMergeManifests(unittest.TestCase):
    def test_union_later_wins_first_horizon_kept(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            _man(td / "manifest_20260701T000000Z.json", "20260701T000000Z",
                 {"A": {"category": "sports", "horizon_days": 30.0, "reward_daily_est": 10},
                  "B": {"category": "politics", "horizon_days": 5.0}})
            _man(td / "manifest_20260705T000000Z.json", "20260705T000000Z",
                 {"A": {"category": "sports", "horizon_days": 26.0, "reward_daily_est": 12},
                  "C": {"category": "tech", "horizon_days": 2.0}})
            out = merge_manifests([str(td / "manifest_20260701T000000Z.json"),
                                   str(td / "manifest_20260705T000000Z.json")])
        self.assertEqual(out["n_manifests"], 2)
        self.assertEqual(out["n_tokens"], 3)
        a = out["token_meta"]["A"]
        self.assertEqual(a["horizon_days"], 26.0)          # last wins
        self.assertEqual(a["horizon_days_first"], 30.0)    # first kept
        self.assertEqual(a["reward_daily_est"], 12)
        self.assertEqual(a["first_seen"], "20260701T000000Z")
        self.assertEqual(a["last_seen"], "20260705T000000Z")
        self.assertEqual(out["token_meta"]["B"]["last_seen"], "20260701T000000Z")

    def test_corrupt_manifest_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "manifest_bad.json").write_text("{not json")
            _man(td / "manifest_20260701T000000Z.json", "20260701T000000Z",
                 {"A": {"category": "sports"}})
            out = merge_manifests([str(td / "manifest_bad.json"),
                                   str(td / "manifest_20260701T000000Z.json")])
        self.assertEqual(out["n_manifests"], 1)
        self.assertEqual(out["n_tokens"], 1)

    def test_none_fields_do_not_clobber(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            _man(td / "manifest_20260701T000000Z.json", "20260701T000000Z",
                 {"A": {"category": "sports", "horizon_days": 30.0}})
            _man(td / "manifest_20260702T000000Z.json", "20260702T000000Z",
                 {"A": {"category": None, "horizon_days": 29.0}})
            out = merge_manifests([str(td / p.name) for p in sorted(td.iterdir())])
        self.assertEqual(out["token_meta"]["A"]["category"], "sports")


class TestHorizonAtEntry(unittest.TestCase):
    def test_entry_horizon_derived_from_last_seen(self):
        import longshot_market_analysis as lma
        rows = [
            {"t": 1783000000.0, "token": "A", "config": "longshot", "mid": 0.045, "inv": 80.0,
             "marked_pnl": 0.0, "q_bid_book": 100.0, "q_ask_book": 100.0},
        ]
        # last_seen 2026-07-08T00:00Z (epoch 1783555200... use strftime round trip) + 2d horizon
        from datetime import datetime, timezone
        last_seen_epoch = 1783600000.0
        stamp = datetime.fromtimestamp(last_seen_epoch, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        # re-derive the epoch the parser will see (stamp truncates seconds precision)
        stamp_epoch = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc).timestamp()
        tags = {"A": {"last_seen": stamp, "horizon_days": 2.0, "category": "sports"}}
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "paper_h_1_1.jsonl.gz"
            with gzip.open(p, "wt") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            clips = lma.collect_clips(str(Path(td) / "paper_*.jsonl.gz"), tags, 0.10, 0.90)
        self.assertEqual(len(clips), 1)
        expected_days = (stamp_epoch + 2.0 * 86400.0 - 1783000000.0) / 86400.0
        # bucket edges [3, 14, 60]: expected ~8.9d -> '3-14d'
        self.assertEqual(clips[0]["horizon_entry_bucket"],
                         "3-14d" if 3 <= expected_days < 14 else "assert-recheck")

    def test_entry_horizon_unknown_without_tags(self):
        import longshot_market_analysis as lma
        rows = [{"t": 100.0, "token": "A", "config": "longshot", "mid": 0.045, "inv": 80.0,
                 "marked_pnl": 0.0, "q_bid_book": 1.0, "q_ask_book": 1.0}]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "paper_h_1_1.jsonl.gz"
            with gzip.open(p, "wt") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            clips = lma.collect_clips(str(Path(td) / "paper_*.jsonl.gz"), {}, 0.10, 0.90)
        self.assertEqual(clips[0]["horizon_entry_bucket"], "unknown")


class TestTrueEntryRecovery(unittest.TestCase):
    def test_entry_price_recovered_from_marked_pnl(self):
        import longshot_market_analysis as lma
        # fill at 0.045; by the first snapshot the mid has popped to 0.10, so
        # marked_pnl = 80*(0.10-0.045) = 4.4 and the true entry must be recovered as 0.045
        rows = [{"t": 100.0, "token": "A", "config": "longshot", "mid": 0.10, "inv": 80.0,
                 "marked_pnl": 4.4, "q_bid_book": 100.0, "q_ask_book": 100.0}]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "paper_h_1_1.jsonl.gz"
            with gzip.open(p, "wt") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            clips = lma.collect_clips(str(Path(td) / "paper_*.jsonl.gz"), {}, 0.10, 0.90)
        c = clips[0]
        self.assertAlmostEqual(c["entry"], 0.045)
        self.assertAlmostEqual(c["entry_mid_snapshot"], 0.10)
        self.assertAlmostEqual(c["cost"], 80.0 * 0.045)
        self.assertEqual(c["entry_bucket"], "4-5c")      # bucketed on FILL, not the popped mid


class TestLoadTagsGz(unittest.TestCase):
    def test_load_manifest_tags_reads_gz_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            with gzip.open(td / "token_tags.json.gz", "wt") as fh:
                json.dump({"token_meta": {"A": {"category": "tech"}}}, fh)
            tags = load_manifest_tags(str(td / "token_tags.json.gz"))
        self.assertEqual(tags["A"]["category"], "tech")


if __name__ == "__main__":
    unittest.main()
