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


class TestBookMMAdvanced(unittest.TestCase):
    def test_new_params_off_match_legacy(self):
        # all additive components default-off must reproduce the legacy result EXACTLY
        events = [quote(0, 0.49, 0.51)]
        t = 1
        for _ in range(40):
            events += [trade(t, 0.49, "SELL", 5), quote(t + 0.1, 0.49, 0.51),
                       trade(t + 0.2, 0.51, "BUY", 5), quote(t + 0.3, 0.49, 0.51)]
            t += 1
        legacy = simulate_book_mm(events, 5, 100, mark_delay_s=1.0, momentum_window=5.0,
                                  skew_threshold=0.003)
        extended = simulate_book_mm(events, 5, 100, mark_delay_s=1.0, momentum_window=5.0,
                                    skew_threshold=0.003, debounce_trades=0, inv_skew=0.0,
                                    vol_window=0.0, vol_spread_coeff=0.0, tox_threshold=0.0)
        self.assertEqual(legacy, extended)

    def test_debounce_beats_raw_momentum_under_bounce(self):
        # mid trends UP but the QUOTE mid carries heavy bid-ask bounce; trades are clean.
        # Raw-mid momentum is fooled by the bounce; debounced (trade-VWAP) momentum tracks the
        # real trend and quotes bid-only into the rally -> should not do worse than raw momentum.
        events = []
        t = 0
        for i in range(120):
            base = 0.30 + i * 0.003
            bounce = 0.02 if i % 2 == 0 else -0.02      # alternating bounce in the quoted mid
            mid_b = base + bounce
            events += [quote(t, round(mid_b - 0.01, 4), round(mid_b + 0.01, 4)),
                       trade(t + 0.1, round(base, 4), "SELL", 5),       # clean trade prints
                       trade(t + 0.2, round(base + 0.01, 4), "BUY", 5)]
            t += 1
        raw = simulate_book_mm(events, 5, 100, mark_delay_s=2.0,
                               momentum_window=8.0, skew_threshold=0.004)
        deb = simulate_book_mm(events, 5, 100, mark_delay_s=2.0,
                               momentum_window=8.0, skew_threshold=0.004, debounce_trades=6)
        # debouncing strips the bid-ask bounce: the signal is meaningfully steadier than raw...
        self.assertLess(deb["mean_abs_signal"], raw["mean_abs_signal"])
        # ...while still detecting the real uptrend (quotes one-sided a non-trivial fraction)
        self.assertGreater(deb["one_sided_quote_frac"], 0.3)

    def test_inventory_skew_curbs_one_sided_accumulation(self):
        # persistent SELL flow hits our bid; without skew we keep buying to the cap. With
        # inventory skew, the bid slides away as we get long -> far fewer one-sided fills.
        events = [quote(0, 0.49, 0.51)]
        t = 1
        for _ in range(30):
            events += [trade(t, 0.49, "SELL", 5), quote(t + 0.1, 0.49, 0.51)]
            t += 1
        noskew = simulate_book_mm(events, 5, 100, inv_skew=0.0)
        skew = simulate_book_mm(events, 5, 100, inv_skew=0.10)
        self.assertLess(skew["n_fills"], noskew["n_fills"])

    def test_vol_widening_reduces_fills(self):
        # volatile mid (large swings inside the vol window); flow prints at the touch. Widening
        # pushes our quotes off the touch, so the touch-priced trades stop crossing us.
        events = []
        t = 0
        for i in range(40):
            mid_b = 0.50 + (0.06 if i % 2 == 0 else -0.06)   # high realized vol
            bb, ba = round(mid_b - 0.01, 4), round(mid_b + 0.01, 4)
            events += [quote(t, bb, ba), trade(t + 0.1, bb, "SELL", 5),
                       trade(t + 0.2, ba, "BUY", 5)]
            t += 1
        tight = simulate_book_mm(events, 5, 100, vol_window=20.0, vol_spread_coeff=0.0)
        wide = simulate_book_mm(events, 5, 100, vol_window=20.0, vol_spread_coeff=2.0)
        self.assertLess(wide["n_fills"], tight["n_fills"])

    def test_toxicity_gate_steps_off_after_adverse_fill(self):
        # steady up-rally; BUY flow keeps lifting our ask (we sell into the rise = adverse).
        # The toxicity gate should detect the post-fill adverse move and step off the ask,
        # cutting the number of adverse sells.
        events = []
        t = 0
        for i in range(60):
            b = round(0.30 + i * 0.01, 4)
            events += [quote(t, b, round(b + 0.02, 4)),
                       trade(t + 0.1, round(b + 0.02, 4), "BUY", 5)]   # lifts our ask
            t += 1
        nogate = simulate_book_mm(events, 5, 100, mark_delay_s=2.0,
                                  tox_threshold=0.0)
        gated = simulate_book_mm(events, 5, 100, mark_delay_s=2.0,
                                 tox_threshold=0.005, tox_window=5.0, tox_cooldown=5.0)
        self.assertLess(gated["n_fills"], nogate["n_fills"])
        self.assertGreater(gated["pnl"], nogate["pnl"])   # fewer adverse sells -> less loss


class TestFillModel(unittest.TestCase):
    def test_prorata_fills_where_fifo_queue_blocks(self):
        # 900 resting ahead, a single 500 trade: FIFO never reaches us; prorata still gets a share
        events = [quote(0, 0.49, 0.51, bs=900, asz=900), trade(1, 0.49, "SELL", 500)]
        fifo = simulate_book_mm(events, our_size=100, inventory_cap=1000, fill_model="fifo")
        pro = simulate_book_mm(events, our_size=100, inventory_cap=1000, fill_model="prorata")
        self.assertEqual(fifo["n_fills"], 0)
        self.assertEqual(pro["n_fills"], 1)        # share = 100/(100+900)=0.1 -> 50 lots

    def test_prorata_capture_mult_haircut(self):
        # round-trip flow; capture_mult halves the filled volume -> ~half the pnl
        events = [quote(0, 0.49, 0.51, bs=100, asz=100)]
        t = 1
        for _ in range(40):
            events += [trade(t, 0.49, "SELL", 100), quote(t + 0.1, 0.49, 0.51, bs=100, asz=100),
                       trade(t + 0.2, 0.51, "BUY", 100), quote(t + 0.3, 0.49, 0.51, bs=100, asz=100)]
            t += 1
        full = simulate_book_mm(events, 100, 1000, mark_delay_s=0.0, fill_model="prorata", capture_mult=1.0)
        half = simulate_book_mm(events, 100, 1000, mark_delay_s=0.0, fill_model="prorata", capture_mult=0.5)
        self.assertGreater(full["pnl"], 0.0)
        self.assertGreater(half["pnl"], 0.0)
        self.assertGreater(full["pnl"], half["pnl"])

    def test_prorata_share_grows_when_we_are_larger(self):
        # same flow, deeper our_size relative to competing depth -> bigger fill -> more spread
        events = [quote(0, 0.49, 0.51, bs=200, asz=200), trade(1, 0.49, "SELL", 1000)]
        small = simulate_book_mm(events, 100, 100000, fill_model="prorata")
        big = simulate_book_mm(events, 800, 100000, fill_model="prorata")
        self.assertGreater(big["gross_spread_captured"], small["gross_spread_captured"])


class TestRewardAccrual(unittest.TestCase):
    # depth sample format: [t, mid, bb, ba, q_bid_book, q_ask_book]
    SAMPLES = [[0.5, 0.50, 0.49, 0.51, 444.0, 444.0],
               [1.5, 0.50, 0.49, 0.51, 444.0, 444.0],
               [2.5, 0.50, 0.49, 0.51, 444.0, 444.0]]
    QUOTES = [quote(0, 0.49, 0.51), quote(1, 0.49, 0.51), quote(2, 0.49, 0.51)]

    def _run(self, size, **kw):
        return simulate_book_mm(self.QUOTES, our_size=size, inventory_cap=1000,
                                depth_samples=self.SAMPLES, reward_pool=1440.0,
                                reward_min_size=200.0, reward_v_cents=3.0, **kw)

    def test_reward_off_by_default(self):
        r = simulate_book_mm(self.QUOTES, 1000, 1000)   # no depth_samples
        self.assertEqual(r["reward"], 0.0)

    def test_below_min_size_earns_no_reward(self):
        self.assertEqual(self._run(20)["reward"], 0.0)        # 20 < min_size 200

    def test_two_sided_earns_reward(self):
        r = self._run(1000)
        # capture ~0.5/sample (our ~444.4 vs book 444), per_min = 1440/1440 = 1, 3 samples -> ~1.5
        self.assertAlmostEqual(r["reward"], 1.5, places=2)

    def test_stepping_off_reduces_reward(self):
        both = self._run(1000)                                 # neutral two-sided
        one_sided = self._run(1000, signal=1.0, skew_threshold=0.5)  # ask stepped off
        self.assertLess(one_sided["reward"], both["reward"])
        self.assertGreater(one_sided["reward"], 0.0)


class TestCaptureShareCap(unittest.TestCase):
    def test_share_capped(self):
        from quoter import Quoter
        # empty competing book -> our share would be 100%; cap to 10%
        q = Quoter(our_size=1000, inventory_cap=10000, reward_pool=1440.0,
                   reward_min_size=100.0, reward_v_cents=3.0, max_capture_share=0.10)
        q.on_quote(0.0, 0.49, 0.51, 100, 100)
        q.credit_sample(0.50, 0.0, 0.0)        # per_min=1, raw share=1.0 -> capped 0.10
        self.assertAlmostEqual(q.reward, 0.10, places=4)

    def test_uncapped_default(self):
        from quoter import Quoter
        q = Quoter(our_size=1000, inventory_cap=10000, reward_pool=1440.0,
                   reward_min_size=100.0, reward_v_cents=3.0)   # default max_capture_share=1.0
        q.on_quote(0.0, 0.49, 0.51, 100, 100)
        q.credit_sample(0.50, 0.0, 0.0)
        self.assertAlmostEqual(q.reward, 1.0, places=4)         # full share, no cap


class TestLatencyPickoff(unittest.TestCase):
    def _run(self, latency, cancel=0.0):
        from quoter import Quoter
        q = Quoter(our_size=100, inventory_cap=1000, fill_model="prorata",
                   quote_latency=latency, cancel_on_move=cancel)
        q.on_quote(0.0, 0.49, 0.51, 0, 0)        # rest bid 0.49 (mid 0.50)
        q.on_quote(2.0, 0.44, 0.46, 0, 0)        # market drops to mid 0.45
        q.on_trade(2.2, 0.49, "SELL", 100)       # informed sell hits 0.49 in the latency window
        return q

    def test_latency_causes_pickoff(self):
        self.assertEqual(len(self._run(0.0).fills), 0)   # fast: already re-quoted to 0.44, no fill
        self.assertEqual(len(self._run(0.5).fills), 1)   # slow: filled at the stale 0.49 (picked off)

    def test_cancel_on_move_defends(self):
        # same 0.5s latency, but cancel-on-move yanks the stale quote when mid drifts > 2c
        self.assertEqual(len(self._run(0.5, cancel=0.02).fills), 0)


if __name__ == "__main__":
    unittest.main()
