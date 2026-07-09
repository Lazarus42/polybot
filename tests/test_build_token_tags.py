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
