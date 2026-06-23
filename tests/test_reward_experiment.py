from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import reward_experiment as rx


def _write_gz(path: Path, events: list[dict]) -> None:
    with gzip.open(path, "wt") as w:
        for e in events:
            w.write(json.dumps(e) + "\n")


class Args:
    max_shards = 0
    time_budget = 0.0


class TestReconstruction(unittest.TestCase):
    def test_price_change_add_and_remove_tracked_in_cache(self):
        # token T: book, then add a tighter bid level, then remove it. Samples 60s apart.
        events = [
            {"event_type": "book", "asset_id": "T", "timestamp": "0",
             "bids": [{"price": "0.49", "size": "1000"}],
             "asks": [{"price": "0.51", "size": "1000"}]},
            {"event_type": "price_change", "timestamp": "60000",
             "price_changes": [{"asset_id": "T", "price": "0.495", "size": "5000", "side": "BUY",
                                "best_bid": "0.495", "best_ask": "0.51"}]},
            {"event_type": "price_change", "timestamp": "120000",
             "price_changes": [{"asset_id": "T", "price": "0.495", "size": "0", "side": "BUY",
                                "best_bid": "0.49", "best_ask": "0.51"}]},
        ]
        with tempfile.TemporaryDirectory() as d:
            day = Path(d) / "raw"; day.mkdir()
            _write_gz(day / "book_h_1_1000.jsonl.gz", events)
            cache = Path(d) / "cache"
            token_meta = {"T": {"pool": 1440.0, "min_size": 200.0, "v_cents": 3.0}}
            shards = rx.group_files_by_shard(day)
            rx.build_cache(Args(), shards, token_meta, cache)
            recs = [json.loads(l) for l in gzip.open(cache / "1.jsonl.gz", "rt")]
            self.assertEqual(len(recs), 1)
            samples = recs[0]["s"]
            self.assertEqual(len(samples), 3)            # one per 60s event
            # sample fields: [t, mid, bb, ba, q_bid, q_ask]
            self.assertAlmostEqual(samples[0][2], 0.49)  # initial best bid
            self.assertAlmostEqual(samples[1][2], 0.495)  # tighter bid after add
            self.assertAlmostEqual(samples[2][2], 0.49)  # reverts after remove
            # q_bid at the tighter (closer to mid) sample exceeds the wide one
            self.assertGreater(samples[1][4], samples[0][4])

    def test_below_min_size_book_not_credited(self):
        # all resting size below min_size -> no qualifying bid/ask -> no samples emitted
        events = [
            {"event_type": "book", "asset_id": "T", "timestamp": "0",
             "bids": [{"price": "0.49", "size": "50"}],
             "asks": [{"price": "0.51", "size": "50"}]},
        ]
        with tempfile.TemporaryDirectory() as d:
            day = Path(d) / "raw"; day.mkdir()
            _write_gz(day / "book_h_1_1000.jsonl.gz", events)
            cache = Path(d) / "cache"
            token_meta = {"T": {"pool": 1440.0, "min_size": 200.0, "v_cents": 3.0}}
            rx.build_cache(Args(), rx.group_files_by_shard(day), token_meta, cache)
            recs = [json.loads(l) for l in gzip.open(cache / "1.jsonl.gz", "rt")]
            self.assertEqual(recs, [])


if __name__ == "__main__":
    unittest.main()
