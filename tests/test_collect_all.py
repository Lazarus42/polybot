from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from collect_all import assign_new_tokens, shard


class TestAssignNewTokens(unittest.TestCase):
    def test_fills_least_full_shard_first(self):
        # two shards, room 2 and 5 -> first new tokens go to the emptier (idx 1)
        assigns, overflow = assign_new_tokens(["x", "y"], shard_counts=[8, 5], shard_size=10)
        self.assertEqual(overflow, [])
        self.assertIn(1, assigns)
        self.assertEqual(sorted(assigns[1]), ["x", "y"])
        self.assertNotIn(0, assigns)

    def test_overflow_when_all_full(self):
        assigns, overflow = assign_new_tokens(["a", "b", "c"], shard_counts=[10, 10], shard_size=10)
        self.assertEqual(assigns, {})
        self.assertEqual(overflow, ["a", "b", "c"])

    def test_partial_fill_then_overflow(self):
        # one shard with room for exactly 1; second token overflows
        assigns, overflow = assign_new_tokens(["a", "b"], shard_counts=[9], shard_size=10)
        self.assertEqual(assigns, {0: ["a"]})
        self.assertEqual(overflow, ["b"])

    def test_no_shards_all_overflow(self):
        assigns, overflow = assign_new_tokens(["a"], shard_counts=[], shard_size=10)
        self.assertEqual(assigns, {})
        self.assertEqual(overflow, ["a"])

    def test_spreads_across_shards(self):
        # equal room -> tokens spread out rather than piling on one
        assigns, overflow = assign_new_tokens(list("abcd"), shard_counts=[0, 0], shard_size=10)
        self.assertEqual(overflow, [])
        self.assertEqual(len(assigns[0]) + len(assigns[1]), 4)
        self.assertLessEqual(abs(len(assigns[0]) - len(assigns[1])), 1)


class TestShard(unittest.TestCase):
    def test_chunks(self):
        self.assertEqual(shard(list(range(5)), 2), [[0, 1], [2, 3], [4]])
        self.assertEqual(shard([], 2), [])


if __name__ == "__main__":
    unittest.main()
