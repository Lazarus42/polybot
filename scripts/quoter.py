#!/usr/bin/env python3
"""Shared streaming market-making QUOTER state machine.

This is the SINGLE source of truth for quoting/fill/reward logic, fed an event stream one event
at a time. The backtest (`book_mm_backtest.simulate_book_mm`) drives it over a captured tape; the
live paper simulator (`paper_sim.py`) drives it over the real-time WebSocket — so what we
forward-test is provably the same code we backtested (no drift). The equivalence is locked by the
existing `tests/test_book_mm.py` suite, which runs entirely through this class.

Causal by construction: every decision uses only past events. See `simulate_book_mm`'s docstring
for the meaning of each parameter (debounce / inventory skew / vol spread / toxicity gate / fill
model / reward). Reward is credited via `credit_sample(...)`, called by the driver whenever a
per-minute book sample is available (replayed from a list in backtest, computed live in paper sim).
"""
from __future__ import annotations

import bisect
from collections import deque

from reward_model import order_score as _osc, q_min as _qm


class Quoter:
    def __init__(self, our_size: float, inventory_cap: float, maker_fee: float = 0.0,
                 mark_delay_s: float = 60.0, improve: bool = False, tick: float = 0.01,
                 signal: float = 0.0, skew_threshold: float = 0.0, momentum_window: float = 0.0,
                 debounce_trades: int = 0, inv_skew: float = 0.0,
                 vol_window: float = 0.0, vol_spread_coeff: float = 0.0,
                 tox_threshold: float = 0.0, tox_window: float = 0.0, tox_cooldown: float = 0.0,
                 reward_pool: float = 0.0, reward_min_size: float = 0.0, reward_v_cents: float = 0.0,
                 fill_model: str = "fifo", capture_mult: float = 1.0, max_capture_share: float = 1.0,
                 quote_latency: float = 0.0, cancel_on_move: float = 0.0,
                 max_hold_seconds: float = 0.0, quote_offset: float = 0.0,
                 min_mid: float = 0.0, max_mid: float = 1.0, liq_outside_band: bool = False,
                 stop_loss_cents: float = 0.0):
        self.our_size = our_size; self.inventory_cap = inventory_cap
        self.maker_fee = maker_fee; self.mark_delay_s = mark_delay_s
        self.improve = improve; self.tick = tick
        self.skew_threshold = skew_threshold; self.momentum_window = momentum_window
        self.use_momentum = momentum_window > 0.0
        self.debounce_trades = debounce_trades; self.inv_skew = inv_skew
        self.vol_window = vol_window; self.vol_spread_coeff = vol_spread_coeff
        self.tox_threshold = tox_threshold; self.tox_window = tox_window; self.tox_cooldown = tox_cooldown
        self.reward_pool = reward_pool; self.reward_min_size = reward_min_size
        self.reward_v_cents = reward_v_cents; self._per_min = reward_pool / 1440.0
        self.fill_model = fill_model; self.capture_mult = capture_mult
        self.max_capture_share = max_capture_share   # cap modeled reward share (competition fills in)
        # latency model: our LIVE quote lags the DESIRED quote by quote_latency seconds (so fast
        # flow can pick off a stale order); cancel_on_move>0 instantly pulls the live quote when the
        # mid has moved more than that from where the quote was set (the fast cancel-on-move defense).
        self.quote_latency = quote_latency; self.cancel_on_move = cancel_on_move
        # risk control: if a position is carried longer than this, actively cross out of it (so we
        # never hold inventory into a resolution, where it settles at $0/$1 not mid). 0 = never.
        self.max_hold_seconds = max_hold_seconds
        # depth control: rest each side this far (in PRICE units, e.g. 0.012 = 1.2c) from the mid
        # instead of pegging to the touch. This is the s* from optimal_spread that trades reward
        # harvesting against adverse selection. 0 = legacy touch-pegged behaviour (unchanged).
        self.quote_offset = quote_offset
        # price-band gate (resolution guard): only quote when min_mid <= mid <= max_mid. Outside
        # the band the reward is tiny and the tail (snap to 0/1) is huge, so we stand aside.
        self.min_mid = min_mid; self.max_mid = max_mid
        # when True, actively liquidate inventory (cross out) the moment the mid leaves the band,
        # instead of only ceasing to quote — so we don't ride a position into resolution.
        self.liq_outside_band = liq_outside_band
        # stop-loss: if the open position is more than this many cents/share underwater (mid vs our
        # average entry), liquidate it. 0 = off. Cuts losers without waiting for the max-hold clock.
        self.stop_loss_cents = stop_loss_cents
        self.avg_entry = 0.0             # volume-weighted entry price of the OPEN position
        self.inv_since = None            # time inventory first became non-zero (for max-hold)
        self.flat_cost = 0.0; self.n_flats = 0   # spread paid crossing out of stale inventory
        # runtime state
        self.cur_signal = signal
        self.want_bid = self.cur_signal >= -skew_threshold
        self.want_ask = self.cur_signal <= skew_threshold
        self.best_bid = self.best_ask = None
        self.our_bid = self.our_ask = None
        self.q_bid_ahead = self.q_ask_ahead = 0.0
        self.cur_bid_depth = self.cur_ask_depth = 0.0
        self.inv = self.cash = self.fees = self.gross_spread = self.reward = 0.0
        self.fills: list[tuple[float, int, float]] = []
        self.mids: list[tuple[float, float]] = []
        self.ref_hist: deque = deque()
        self.trade_hist: deque = deque()
        self.vol_hist: deque = deque()
        self.vol_sum = self.vol_sumsq = 0.0
        self.n_quotes = self.n_quote_ev = self.n_onesided = 0
        self.abs_signal_sum = 0.0
        self.first_quote_t = self.last_quote_t = None
        self.last_bid_fill = self.last_ask_fill = None
        self.bid_off_until = self.ask_off_until = -1.0
        self.desired: deque = deque()    # (t, bid, ask, bid_size, ask_size, mid) pending quotes
        self.live_mid = None             # mid when the current LIVE quote was set (for cancel-on-move)

    def mid(self):
        return ((self.best_bid + self.best_ask) / 2
                if (self.best_bid is not None and self.best_ask is not None) else None)

    def _debounced_ref(self, m):
        if self.debounce_trades <= 0 or not self.trade_hist:
            return m
        num = sum(p * s for p, s in self.trade_hist)
        den = sum(s for _, s in self.trade_hist)
        return num / den if den > 0 else m

    def _recent_vol(self):
        n = len(self.vol_hist)
        if self.vol_window <= 0.0 or n < 2:
            return 0.0
        var = (self.vol_sumsq - self.vol_sum * self.vol_sum / n) / (n - 1)
        return var ** 0.5 if var > 0 else 0.0

    def on_quote(self, t, bid, ask, bid_size=0.0, ask_size=0.0):
        self.n_quotes += 1
        self.best_bid, self.best_ask = bid, ask
        m = self.mid()
        if m is not None:
            self.mids.append((t, m))
            if self.vol_window > 0.0:
                self.vol_hist.append((t, m)); self.vol_sum += m; self.vol_sumsq += m * m
                while self.vol_hist and self.vol_hist[0][0] < t - self.vol_window:
                    _, om = self.vol_hist.popleft(); self.vol_sum -= om; self.vol_sumsq -= om * om
            if self.use_momentum:
                ref = self._debounced_ref(m)
                self.ref_hist.append((t, ref))
                while len(self.ref_hist) > 1 and self.ref_hist[0][0] < t - self.momentum_window:
                    self.ref_hist.popleft()
                self.cur_signal = ref - self.ref_hist[0][1]
                self.want_bid = self.cur_signal >= -self.skew_threshold
                self.want_ask = self.cur_signal <= self.skew_threshold
                self.n_quote_ev += 1
                self.abs_signal_sum += abs(self.cur_signal)
                if self.want_bid != self.want_ask:
                    self.n_onesided += 1
        if self.tox_threshold > 0.0 and m is not None:
            if self.last_bid_fill is not None and t - self.last_bid_fill[0] <= self.tox_window \
                    and m < self.last_bid_fill[1] - self.tox_threshold:
                self.bid_off_until = t + self.tox_cooldown
            if self.last_ask_fill is not None and t - self.last_ask_fill[0] <= self.tox_window \
                    and m > self.last_ask_fill[1] + self.tox_threshold:
                self.ask_off_until = t + self.tox_cooldown
        in_band = m is not None and self.min_mid <= m <= self.max_mid
        if self.liq_outside_band and m is not None and not in_band and self.inv != 0:
            self._do_flatten()           # price went extreme -> dump before it resolves to 0/1
        self._maybe_stop()               # stop-loss: cut the position if too far underwater
        bid_open = self.want_bid and t >= self.bid_off_until and in_band
        ask_open = self.want_ask and t >= self.ask_off_until and in_band
        if bid_open or ask_open:
            if self.first_quote_t is None:
                self.first_quote_t = t
            self.last_quote_t = t
        if self.quote_offset > 0.0 and m is not None:
            # rest s* from the mid (the reward/adverse-selection optimum), not at the touch. This
            # may sit inside the spread (improving, when s* < half the touch spread) or behind the
            # touch (deeper, when s* is wider); both are valid passive placements and the price-
            # crossing fill model handles the resulting fill rate. Reward is scored on this same
            # mid-distance via credit_sample, so quote_offset IS the s fed to the equation.
            nb = m - self.quote_offset
            na = m + self.quote_offset
        else:
            nb = self.best_bid + self.tick if self.improve else self.best_bid
            na = self.best_ask - self.tick if self.improve else self.best_ask
        if nb >= na:
            nb, na = self.best_bid, self.best_ask
        if self.vol_spread_coeff > 0.0:
            widen = self.vol_spread_coeff * self._recent_vol()
            nb -= widen; na += widen
        if self.inv_skew != 0.0 and self.inventory_cap > 0:
            off = self.inv_skew * (self.inv / self.inventory_cap)
            nb -= off; na -= off
        nb = min(max(nb, self.tick), 1.0 - self.tick)
        na = min(max(na, self.tick), 1.0 - self.tick)
        if nb >= na:
            nb, na = self.best_bid, self.best_ask
        # the quote we WANT now (becomes live after quote_latency seconds)
        des_bid = nb if bid_open else None
        des_ask = na if ask_open else None
        self.desired.append((t, des_bid, des_ask, bid_size or 0.0, ask_size or 0.0, m))
        self._promote(t)
        # fast cancel-on-move: yank a now-stale live quote the instant the mid has drifted past
        # the threshold from where it was set (cancels are ~free/instant; re-quoting still lags).
        if self.cancel_on_move > 0.0 and m is not None and self.live_mid is not None \
                and abs(m - self.live_mid) > self.cancel_on_move:
            self.our_bid = self.our_ask = None
        self._maybe_flatten(t)           # force-exit stale inventory on time alone (no trade needed)

    def _promote(self, t):
        """Apply the most recent DESIRED quote that is now older than quote_latency -> LIVE."""
        applied = None
        while self.desired and self.desired[0][0] <= t - self.quote_latency:
            applied = self.desired.popleft()
        if applied is None:
            return
        _, db, da, dbs, das, dmid = applied
        if db is not None:
            if self.our_bid != db:
                self.our_bid, self.q_bid_ahead = db, (0.0 if self.improve else dbs)
            self.cur_bid_depth = dbs
        else:
            self.our_bid = None
        if da is not None:
            if self.our_ask != da:
                self.our_ask, self.q_ask_ahead = da, (0.0 if self.improve else das)
            self.cur_ask_depth = das
        else:
            self.our_ask = None
        self.live_mid = dmid

    def on_trade(self, t, price, side, size):
        self._promote(t)                 # live quote reflects its lagged state at trade time
        m = self.mid()
        if self.debounce_trades > 0:
            self.trade_hist.append((price, size))
            while len(self.trade_hist) > self.debounce_trades:
                self.trade_hist.popleft()
        if side == "SELL" and self.our_bid is not None and price <= self.our_bid + 1e-12 \
                and self.inv < self.inventory_cap:
            if self.fill_model == "prorata":
                tot = self.our_size + self.cur_bid_depth
                share = self.our_size / tot if tot > 0 else 1.0
                fillable = min(self.our_size, size * share * self.capture_mult, self.inventory_cap - self.inv)
            elif self.q_bid_ahead >= size:
                self.q_bid_ahead -= size; fillable = 0.0
            else:
                fillable = min(self.our_size, size - self.q_bid_ahead, self.inventory_cap - self.inv)
                self.q_bid_ahead = 0.0
            if fillable > 0:
                self._record_entry(+fillable, self.our_bid)
                self.inv += fillable; self.cash -= fillable * self.our_bid
                self.fees += self.maker_fee * fillable * self.our_bid
                if m is not None:
                    self.gross_spread += fillable * (m - self.our_bid)
                    self.fills.append((t, +1, m)); self.last_bid_fill = (t, m)
        elif side == "BUY" and self.our_ask is not None and price >= self.our_ask - 1e-12 \
                and self.inv > -self.inventory_cap:
            if self.fill_model == "prorata":
                tot = self.our_size + self.cur_ask_depth
                share = self.our_size / tot if tot > 0 else 1.0
                fillable = min(self.our_size, size * share * self.capture_mult, self.inventory_cap + self.inv)
            elif self.q_ask_ahead >= size:
                self.q_ask_ahead -= size; fillable = 0.0
            else:
                fillable = min(self.our_size, size - self.q_ask_ahead, self.inventory_cap + self.inv)
                self.q_ask_ahead = 0.0
            if fillable > 0:
                self._record_entry(-fillable, self.our_ask)
                self.inv -= fillable; self.cash += fillable * self.our_ask
                self.fees += self.maker_fee * fillable * self.our_ask
                if m is not None:
                    self.gross_spread += fillable * (self.our_ask - m)
                    self.fills.append((t, -1, m)); self.last_ask_fill = (t, m)
        # track how long we've been carrying a position, and force-exit if it's too old
        self.inv_since = None if self.inv == 0 else (self.inv_since or t)
        self._maybe_stop()
        self._maybe_flatten(t)

    def _record_entry(self, dq, price):
        """Maintain the open position's volume-weighted entry price as fills arrive.
        dq is signed (+buy / -sell). Adding to the position averages in; reducing leaves the
        entry as-is; flipping the sign resets it to the new fill price."""
        prev = self.inv
        new = prev + dq
        if prev == 0 or (prev > 0) == (dq > 0):                # opening / adding same direction
            denom = abs(prev) + abs(dq)
            self.avg_entry = (abs(prev) * self.avg_entry + abs(dq) * price) / denom if denom else price
        elif new == 0:                                        # closed flat
            self.avg_entry = 0.0
        elif (new > 0) != (prev > 0):                         # flipped sides
            self.avg_entry = price
        # pure reduction (same sign, smaller): keep avg_entry

    def _do_flatten(self):
        """Cross the spread to exit the whole position now, paying the half-spread (the realistic
        cost of getting flat). Shared by the max-hold timer, extreme-price, and stop-loss exits."""
        if self.inv == 0:
            return
        m = self.mid()
        if m is None or self.best_bid is None or self.best_ask is None:
            return
        half = (self.best_ask - self.best_bid) / 2
        self.cash += self.inv * (m - (half if self.inv > 0 else -half))   # sell at bid / buy at ask
        self.flat_cost += abs(self.inv) * half       # the spread we paid to get flat
        self.n_flats += 1
        self.inv = 0.0; self.inv_since = None; self.avg_entry = 0.0

    def _maybe_stop(self):
        """Liquidate if the open position is more than stop_loss_cents underwater (mid vs entry)."""
        if self.stop_loss_cents <= 0 or self.inv == 0 or self.avg_entry == 0:
            return
        m = self.mid()
        if m is None:
            return
        loss_c = ((self.avg_entry - m) if self.inv > 0 else (m - self.avg_entry)) * 100.0
        if loss_c >= self.stop_loss_cents - 1e-9:
            self._do_flatten()

    def _maybe_flatten(self, t):
        """Force-exit a position held longer than max_hold_seconds (time-based resolution guard)."""
        if self.max_hold_seconds <= 0 or self.inv == 0 or self.inv_since is None:
            return
        if t - self.inv_since < self.max_hold_seconds:
            return
        self._do_flatten()

    def credit_sample(self, s_mid, q_bid_book, q_ask_book):
        """Accrue one per-minute reward sample from the CURRENT live quote placement."""
        s_bid = s_ask = 0.0
        if self.our_bid is not None and self.our_size >= self.reward_min_size:
            s_bid = _osc(self.reward_v_cents, (s_mid - self.our_bid) * 100.0) * self.our_size
        if self.our_ask is not None and self.our_size >= self.reward_min_size:
            s_ask = _osc(self.reward_v_cents, (self.our_ask - s_mid) * 100.0) * self.our_size
        our_qmin = _qm(s_bid, s_ask, s_mid)
        tot_qmin = _qm(q_bid_book + s_bid, q_ask_book + s_ask, s_mid)
        if tot_qmin > 0:
            share = min(our_qmin / tot_qmin, self.max_capture_share)   # competition caps real share
            self.reward += self._per_min * share

    def adverse_selection(self):
        adverse = 0.0
        mid_ts = [tt for tt, _ in self.mids]
        for (t, d, m0) in self.fills:
            i = bisect.bisect_left(mid_ts, t + self.mark_delay_s)
            future = self.mids[i][1] if i < len(self.mids) else (self.mids[-1][1] if self.mids else m0)
            adverse += -d * (future - m0) * self.our_size
        return adverse

    def finalize(self) -> dict:
        """Flatten residual inventory at mid -/+ half-spread and return the result dict."""
        m = self.mid()
        cash = self.cash
        if self.inv != 0 and m is not None:
            half = (self.best_ask - self.best_bid) / 2
            cash += self.inv * (m - (half if self.inv > 0 else -half))
        pnl = cash - self.fees
        return {
            "pnl": float(pnl), "gross_spread_captured": float(self.gross_spread),
            "adverse_selection": float(self.adverse_selection()), "fees": float(self.fees),
            "n_fills": len(self.fills), "n_quotes": int(self.n_quotes),
            "one_sided_quote_frac": (self.n_onesided / self.n_quote_ev) if self.n_quote_ev else 0.0,
            "mean_abs_signal": (self.abs_signal_sum / self.n_quote_ev) if self.n_quote_ev else 0.0,
            "quoting_days": ((self.last_quote_t - self.first_quote_t) / 86400.0)
            if (self.first_quote_t is not None and self.last_quote_t is not None) else 0.0,
            "reward": float(self.reward),
            "flatten_cost": float(self.flat_cost), "n_flatten": int(self.n_flats),
        }
