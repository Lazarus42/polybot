from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from book_mm_backtest import normalize_events, simulate_book_mm


def quote(t, bid, ask, bs=0.0, asz=0.0):
    return {"type": "quote", "t": t, "bid": bid, "ask": ask, "bid_size": bs, "ask_size": asz}


def trade(t, price, side, size):
    return {"type": "trade", "t": t, "price": price, "side": side, "size": size}


class TestNormalize(unittest.TestCase):
    def test_book_uses_max_bid_min_ask_and_ms_timestamp(self):
        # real data: bids ASCENDING (best = max price), timestamp in milliseconds
        raw = [{"event_type": "book", "asset_id": "T", "timestamp": "1782071846000",
                "bids": [{"price": "0.001", "size": "9"}, {"price": "0.48", "size": "100"}],
                "asks": [{"price": "0.52", "size": "80"}, {"price": "0.99", "size": "5"}]},
               {"event_type": "last_trade_price", "asset_id": "T", "timestamp": "1782071847000",
                "price": "0.48", "side": "SELL", "size": "10"}]
        ev = normalize_events(raw)["T"]
        self.assertEqual(ev[0]["type"], "quote")
        self.assertAlmostEqual(ev[0]["bid"], 0.48)   # max bid, not bids[0]
        self.assertAlmostEqual(ev[0]["ask"], 0.52)   # min ask
        self.assertAlmostEqual(ev[0]["t"], 1782071846.0)  # ms -> s
        self.assertEqual(ev[1]["type"], "trade")

    def test_price_change_list_is_parsed(self):
        # 98% of real events: price_change with a per-asset price_changes[] list
        raw = [{"event_type": "price_change", "timestamp": "1782071874618",
                "price_changes": [
                    {"asset_id": "A", "price": "0.2", "size": "10000", "side": "BUY",
                     "best_bid": "0.41", "best_ask": "0.45"},
                    {"asset_id": "B", "price": "0.8", "size": "1", "side": "SELL",
                     "best_bid": "0", "best_ask": "0.001"}]}]
        out = normalize_events(raw)
        self.assertIn("A", out)
        self.assertAlmostEqual(out["A"][0]["bid"], 0.41)
        self.assertAlmostEqual(out["A"][0]["ask"], 0.45)
        self.assertNotIn("B", out)   # best_bid 0 (empty side) -> filtered


class TestBookMM(unittest.TestCase):
    def test_earns_spread_on_uninformed_flow(self):
        # stable mid 0.50, we're at front of queue (size ahead 0); flow alternates hitting
        # bid then ask -> we round-trip the spread
        events = [quote(0, 0.49, 0.51)]
        t = 1
        for _ in range(50):
            events += [trade(t, 0.49, "SELL", 5), quote(t + 0.1, 0.49, 0.51),
                       trade(t + 0.2, 0.51, "BUY", 5), quote(t + 0.3, 0.49, 0.51)]
            t += 1
        r = simulate_book_mm(events, our_size=5, inventory_cap=100, mark_delay_s=0.0)
        self.assertGreater(r["pnl"], 0.0)
        self.assertGreater(r["gross_spread_captured"], 0.0)
        self.assertGreater(r["n_fills"], 50)

    def test_queue_position_blocks_fills(self):
        # size ahead of us is 100; a single 10-lot trade must NOT fill us
        events = [quote(0, 0.49, 0.51, bs=100, asz=100), trade(1, 0.49, "SELL", 10)]
        r = simulate_book_mm(events, our_size=5, inventory_cap=100)
        self.assertEqual(r["n_fills"], 0)

    def test_trend_causes_adverse_selection(self):
        # mid marches up: our asks keep getting lifted (we sell into a rally) -> short
        # inventory marked into a loss, and measured adverse selection is positive
        events = []
        t = 0
        for i in range(60):
            b = 0.30 + i * 0.005
            events += [quote(t, round(b, 4), round(b + 0.02, 4)),
                       trade(t + 0.1, round(b + 0.02, 4), "BUY", 5)]
            t += 1
        r = simulate_book_mm(events, our_size=5, inventory_cap=100, mark_delay_s=2.0)
        self.assertGreater(r["adverse_selection"], 0.0)   # filled before adverse drift
        self.assertLess(r["pnl"], 0.0)

    def test_signal_informed_beats_neutral_in_trend(self):
        # up-trending mid with two-sided flow (SELL hits our bid, BUY lifts our ask).
        # Neutral MM sells into the rally (adverse). A correct UP signal quotes bid-only,
        # accumulating long into the move -> should beat neutral and be profitable.
        events = []
        t = 0
        for i in range(60):
            b = round(0.30 + i * 0.005, 4)
            events += [quote(t, b, round(b + 0.02, 4)),
                       trade(t + 0.1, b, "SELL", 5),                 # hits our bid (we buy)
                       trade(t + 0.2, round(b + 0.02, 4), "BUY", 5)]  # lifts our ask (we sell)
            t += 1
        neutral = simulate_book_mm(events, our_size=5, inventory_cap=100, mark_delay_s=2.0,
                                   signal=0.0, skew_threshold=0.0)
        informed = simulate_book_mm(events, our_size=5, inventory_cap=100, mark_delay_s=2.0,
                                    signal=1.0, skew_threshold=0.5)   # predict up -> bid only
        self.assertGreater(informed["pnl"], neutral["pnl"])
        self.assertGreater(informed["pnl"], 0.0)

    def test_causal_momentum_skew_beats_neutral(self):
        # up-trend with two-sided flow; the CAUSAL momentum forecast (computed inside the sim
        # from trailing mids) should detect the rise and quote bid-only, beating neutral.
        events = []
        t = 0
        for i in range(80):
            b = round(0.30 + i * 0.004, 4)
            events += [quote(t, b, round(b + 0.02, 4)),
                       trade(t + 0.1, b, "SELL", 5),
                       trade(t + 0.2, round(b + 0.02, 4), "BUY", 5)]
            t += 1
        neutral = simulate_book_mm(events, 5, 100, mark_delay_s=2.0, momentum_window=0.0)
        informed = simulate_book_mm(events, 5, 100, mark_delay_s=2.0,
                                    momentum_window=10.0, skew_threshold=0.003)
        self.assertGreater(informed["pnl"], neutral["pnl"])

    def test_quoting_days_tracked(self):
        # quotes span ~1 day (86400s); we provide liquidity throughout
        events = [quote(0, 0.49, 0.51), trade(43200, 0.49, "SELL", 5), quote(86400, 0.49, 0.51)]
        r = simulate_book_mm(events, 5, 100)
        self.assertAlmostEqual(r["quoting_days"], 1.0, places=3)

    def test_no_crossing_no_fills(self):
        # trades print inside the spread; never cross our quotes
        events = [quote(0, 0.45, 0.55), trade(1, 0.50, "BUY", 100), trade(2, 0.50, "SELL", 100)]
        r = simulate_book_mm(events, our_size=10, inventory_cap=100)
        self.assertEqual(r["n_fills"], 0)
        self.assertEqual(r["pnl"], 0.0)


if __name__ == "__main__":
    unittest.main()
