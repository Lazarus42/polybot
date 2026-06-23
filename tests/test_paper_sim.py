from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import paper_sim as ps


class TestPaperSim(unittest.TestCase):
    def _sim(self, out, configs=("neutral", "clv_full")):
        meta = {"T": {"pool": 1440.0, "min_size": 100.0, "v_cents": 3.0, "question": "Will T win?"}}
        return ps.PaperSim(meta, size=200.0, inv_cap_mult=5.0, configs=list(configs),
                           fill_model="prorata", capture_mult=1.0, out_dir=out, rotate_minutes=15.0)

    def test_reconstruct_quote_fill_and_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            sim = self._sim(out)
            sim.process_message({"event_type": "book", "asset_id": "T", "timestamp": "0",
                                 "bids": [{"price": "0.49", "size": "1000"}],
                                 "asks": [{"price": "0.51", "size": "1000"}]})
            # a BUY trade lifting our ask -> we sell (prorata share of the 50-lot)
            sim.process_message({"event_type": "last_trade_price", "asset_id": "T",
                                 "timestamp": "1000", "price": "0.51", "side": "BUY", "size": "50"})
            sim.close()
            q = sim.q[("clv_full", "T")]              # quoters now keyed by (config, token)
            self.assertEqual(q.n_quotes, 1)
            self.assertEqual(len(q.fills), 1)         # the crossing trade filled our ask
            self.assertGreaterEqual(sim.n_snapshots, 1)
            rows = [json.loads(l) for l in gzip.open(next(out.glob("paper_*.jsonl.gz")), "rt")]
            self.assertTrue(rows and rows[0]["token"] == "T")
            # one snapshot row per config per sample
            self.assertEqual({r["config"] for r in rows}, {"neutral", "clv_full"})
            self.assertIn("reward_cum", rows[0])

    def test_dust_below_min_size_no_snapshot(self):
        # all resting depth below min_size -> no qualifying book -> no samples emitted
        with tempfile.TemporaryDirectory() as d:
            sim = self._sim(Path(d))
            sim.process_message({"event_type": "book", "asset_id": "T", "timestamp": "0",
                                 "bids": [{"price": "0.49", "size": "10"}],
                                 "asks": [{"price": "0.51", "size": "10"}]})
            sim.close()
            self.assertEqual(sim.n_snapshots, 0)


if __name__ == "__main__":
    unittest.main()
