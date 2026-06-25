#!/usr/bin/env python3
"""Real-time PAPER market-making simulator.

Runs the SAME `Quoter` we backtested against a live (or replayed) Polymarket book, maintaining
virtual quotes / inventory / P&L and accruing modeled in-band reward each minute. Lands immutable
snapshots to disk (gzipped JSONL, rotated) for S3 upload — the land-raw-transform-later principle:
paper-sim is a transform over the live event stream, decoupled from the raw collector.

It does NOT place real orders. It answers "what would our strategy have quoted, filled, and been
paid, on fresh out-of-sample data" — closing the backtest's idealizations EXCEPT the two only real
capital can settle: (1) our presence would change the book, and (2) actual reward payout requires
actually resting orders. Use `reconcile_rewards()` against the Markets API once real orders exist.

Modes:
  --replay FILE.jsonl[.gz]   feed a captured collector file (testable offline; what CI/dev uses)
  --live                     connect to the CLOB market WebSocket (same endpoint as the collector)

Per token we hold a reconstructed book + a Quoter. Every --sample-seconds we score the live
in-band competing depth (reward_model.side_score), credit the Quoter, and write a snapshot row:
intended quotes, modeled reward share, virtual inventory, marked P&L, cumulative reward.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quoter import Quoter  # noqa: E402
from reward_model import side_score  # noqa: E402
from optimal_spread import const_eta, exp_lambda, solve_optimal_spread  # noqa: E402

SAMPLE_SECONDS = 60.0
MIDS_MAXLEN = 2000        # cap each quoter's mid history (paper-sim never needs full markout history)

# Placeholder fill-rate / adverse-selection assumptions for the depth_optimal strategy's s*.
# These are PARAMETRIC stand-ins (exponential fill decay A*e^{-k*s}, constant toxicity eta0 in
# cents/share) until per-market lambda(s)/eta(s) are fit from the tape (stale_order_pool.py).
# Replace `depth_params_for(token)` with the fitted curves once available.
# Measured from data/pull/dt=2026-06-22 (the "other" bucket = ~whole universe): fill rate barely
# decays with depth (k~0.012) and touch toxicity eta0~0.29c. Placeholder guesses were a=20,k=1.0.
DEPTH_PARAMS = dict(a=0.007, k=0.012, eta0=0.286, size=1.0)


def _is_ephemeral(question: str) -> bool:
    """True for ultra-short-lived markets (the 5-minute crypto 'up or down' churn) — bad for
    market-making (they resolve faster than we can refresh and are pure resolution risk)."""
    q = (question or "").lower()
    return "up or down" in q or "updown" in q


def optimal_offset_cents(v_cents: float, per_min_reward: float, size: float = 1.0) -> float:
    """s* (in CENTS) balancing reward harvesting vs adverse selection for one token.

    The equation is in CENTS-PER-SHARE / min, so every term must be normalised per resting share:
    r0 = reward at the touch in those units = per_min_reward ($/min) * 100 (c/$) / size (shares).
    The trading term uses size=1 (already per-share). Returns 0 (-> touch placement) for a
    degenerate band or a sub-tick optimum.
    """
    if v_cents <= 0 or size <= 0:
        return 0.0
    r0 = max(per_min_reward, 0.0) * 100.0 / size
    res = solve_optimal_spread(v=v_cents, r0=r0,
                               lam=exp_lambda(DEPTH_PARAMS["a"], DEPTH_PARAMS["k"]),
                               eta=const_eta(DEPTH_PARAMS["eta0"]), size=1.0)
    s = res["s_star"]
    return s if s >= 0.1 else 0.0      # below a tenth of a cent: treat as touch (no offset)


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _ts(raw):
    v = _f(raw)
    if v is None:
        return 0.0
    return v / 1000.0 if v > 1e11 else v


def load_token_meta(manifest: Path | None) -> dict:
    out = {}
    if manifest and manifest.exists():
        man = json.loads(manifest.read_text())
        for tok, m in (man.get("token_meta") or {}).items():
            v = _f(m.get("rewards_max_spread")) or 0.0
            out[str(tok)] = {"pool": _f(m.get("reward_daily_est")) or 0.0,
                             "min_size": _f(m.get("rewards_min_size")) or 0.0,
                             "v_cents": v if v > 1 else v * 100.0,
                             "question": m.get("question", "")}
    return out


CONFIGS = {
    "neutral": {},
    "raw_momentum": dict(momentum_window=300, skew_threshold=0.005),
    "clv_debounce": dict(momentum_window=300, skew_threshold=0.005, debounce_trades=10),
    "clv_full": dict(momentum_window=300, skew_threshold=0.005, debounce_trades=10, inv_skew=0.01,
                     vol_window=300, vol_spread_coeff=0.5, tox_threshold=0.01, tox_window=60,
                     tox_cooldown=60),
    # rest at s* (computed per-token from optimal_spread) instead of the touch, and SKEW on
    # inventory rather than pulling a side (no tox gate -> Q_min never collapses). The "_optimal"
    # marker tells _ensure to inject a per-token quote_offset; it is not a Quoter kwarg.
    "depth_optimal": dict(inv_skew=0.01, _optimal=True),
    # resolution-guarded variants: same as neutral but only quote inside a mid band. Run these
    # head-to-head to find the band that best protects net_if_flat from the snap-to-0/1 tail.
    "band_10_90": dict(min_mid=0.10, max_mid=0.90),
    "band_20_80": dict(min_mid=0.20, max_mid=0.80),
    "band_30_70": dict(min_mid=0.30, max_mid=0.70),
    "depth_optimal_guarded": dict(inv_skew=0.01, _optimal=True, min_mid=0.10, max_mid=0.90),
    # --- tail-control experiment: cap inventory / bail faster / lean against position ---
    "tight_inv": dict(inv_cap_mult=0.25),                 # carry at most 1/4 the inventory
    "fast_flat": dict(max_hold_minutes=10),               # bail out of a position after 10 min
    "neutral_flat30": dict(max_hold_minutes=30),          # neutral but clear inventory every 30 min
    "fast_flat_5": dict(max_hold_minutes=5),              # even faster bail, to bracket the timing
    "skew_revert": dict(inv_skew=0.02),                   # lean quotes to shed inventory
    "tail_guard": dict(inv_cap_mult=0.25, max_hold_minutes=10, inv_skew=0.02,
                       min_mid=0.10, max_mid=0.90),       # all three + stand aside near resolution
    # --- extreme-price liquidation: dump the position the moment the price leaves the band ---
    "exit_extreme": dict(min_mid=0.10, max_mid=0.90, liq_outside_band=True),
    "exit_extreme_tight": dict(min_mid=0.15, max_mid=0.85, liq_outside_band=True),
    "exit_extreme_fast": dict(min_mid=0.10, max_mid=0.90, liq_outside_band=True,
                              max_hold_minutes=10),        # extreme-dump + 10-min bail combined
    # --- stop-loss: cut a position once it's X cents/share underwater (vs our average entry) ---
    "stop_2c": dict(stop_loss_cents=2.0),
    "stop_3c": dict(stop_loss_cents=3.0),
    "stop_5c": dict(stop_loss_cents=5.0),
    "fast_flat_stop3": dict(max_hold_minutes=10, stop_loss_cents=3.0),  # winner so far + stop-loss
    # all defenses together: 10-min bail + stop quoting outside [0.10,0.90] + liquidate on exit
    "guarded_flat": dict(max_hold_minutes=10, min_mid=0.10, max_mid=0.90, liq_outside_band=True),
    # --- take-profit SWEEP: lock winners at +X, each ALSO carrying the symmetric guards (10-min bail
    # + price band [0.10,0.90] with liquidation) so we never quote/hold at illiquid extremes and
    # losers get cut too. Only the take-profit threshold varies, isolating its effect. ---
    "take_profit_1c": dict(take_profit_cents=1.0, max_hold_minutes=10,
                           min_mid=0.10, max_mid=0.90, liq_outside_band=True),
    "take_profit_2c": dict(take_profit_cents=2.0, max_hold_minutes=10,
                           min_mid=0.10, max_mid=0.90, liq_outside_band=True),
    "take_profit_3c": dict(take_profit_cents=3.0, max_hold_minutes=10,
                           min_mid=0.10, max_mid=0.90, liq_outside_band=True),
    "take_profit_5c": dict(take_profit_cents=5.0, max_hold_minutes=10,
                           min_mid=0.10, max_mid=0.90, liq_outside_band=True),
    # --- SYMMETRIC price exit sweep: lock winner at +X AND cut loser at -X (matched stop/take),
    # plus the same band + 10-min guards. Only the ±X threshold varies. ---
    "tp_sl_1c": dict(take_profit_cents=1.0, stop_loss_cents=1.0, max_hold_minutes=10,
                     min_mid=0.10, max_mid=0.90, liq_outside_band=True),
    "tp_sl_2c": dict(take_profit_cents=2.0, stop_loss_cents=2.0, max_hold_minutes=10,
                     min_mid=0.10, max_mid=0.90, liq_outside_band=True),
    "tp_sl_3c": dict(take_profit_cents=3.0, stop_loss_cents=3.0, max_hold_minutes=10,
                     min_mid=0.10, max_mid=0.90, liq_outside_band=True),
    "tp_sl_5c": dict(take_profit_cents=5.0, stop_loss_cents=5.0, max_hold_minutes=10,
                     min_mid=0.10, max_mid=0.90, liq_outside_band=True),
    "guarded_revolver": dict(take_profit_cents=3.0, stop_loss_cents=3.0, max_hold_minutes=10,
                             min_mid=0.10, max_mid=0.90, liq_outside_band=True),  # == tp_sl_3c
    # CRYPTO maker: quotes ONLY the 5-min up/down markets, with hard guards so we exit well before
    # each resolution — short max-hold, tight stop/take, band liquidation. Tests whether the (often
    # richer) crypto reward pools beat their fast-resolution adverse selection. Returns-per-risk play.
    "crypto_mm": dict(allow_ephemeral=True, max_hold_minutes=2, stop_loss_cents=2.0,
                      take_profit_cents=2.0, min_mid=0.10, max_mid=0.90, liq_outside_band=True),
}


class PaperSim:
    """Holds per-token reconstructed book + Quoter; emits per-minute snapshots."""

    def __init__(self, token_meta: dict, size: float, inv_cap_mult: float, configs: list[str],
                 fill_model: str, capture_mult: float, out_dir: Path, rotate_minutes: float,
                 capital: float = 0.0, max_capture_share: float = 1.0,
                 quote_latency: float = 0.0, cancel_on_move: float = 0.0,
                 max_hold_seconds: float = 0.0, min_roc: float = 0.0, auto_min_roc: bool = False,
                 max_roc: float = 0.0):
        self.size = size; self.configs = configs
        # size <= 0  => MIN-QUALIFY mode: rest each market at its minimum reward-eligible clip
        # (rewards_min_size), not a flat size — so the capital spreads across far more markets.
        self.min_qualify = size <= 0
        self.inv_cap_mult = inv_cap_mult
        self.inv_cap = max(size, 0.0) * inv_cap_mult   # fixed-mode fallback; per-token in _ensure
        self.kw = {c: dict(CONFIGS[c]) for c in configs}   # all configs run in parallel, same feed
        self.fill_model = fill_model; self.capture_mult = capture_mult
        self.max_capture_share = max_capture_share
        self.quote_latency = quote_latency; self.cancel_on_move = cancel_on_move
        self.max_hold_seconds = max_hold_seconds
        # CAPITAL: each strategy runs its OWN revolving bankroll, starting at `capital`. It deploys
        # into markets best-reward-ROC-first, only while it has free cash AND the market clears the
        # `min_roc` hurdle (reward-$/$/day) — so we never deploy past the point of diminishing
        # returns. Bankroll grows with reward + realized P&L; freed cash redeploys on refresh.
        self.start_capital = capital
        self.min_roc = min_roc
        self.auto_min_roc = auto_min_roc      # if set, recompute min_roc each refresh to exactly fill capital
        self.max_roc = max_roc                # skip markets ABOVE this ROC (the toxic high-reward ones)
        self.meta = token_meta
        self.allowed: dict[str, set] = {c: set() for c in configs}    # config -> tokens it quotes
        self.sizes: dict[str, dict] = {c: {} for c in configs}        # config -> {token: size}
        self.q: dict[tuple, Quoter] = {}      # (config, token) -> Quoter; created lazily in _ensure
        self.toks: set = set()                # any token at least one config quotes
        # realized results of markets a strategy has EXITED (resolved/dropped) — folded in here so we
        # can free their quoters without losing their reward/P&L from the running bankroll.
        self.realized: dict[str, dict] = {c: {"reward": 0.0, "trade": 0.0, "flat_cost": 0.0,
                                              "n_flat": 0, "fills": 0} for c in configs}
        self.deployed_clip: dict[str, float] = {c: 0.0 for c in configs}   # capital committed per config
        # if any strategy opts into ephemeral (5-min crypto) markets, we must subscribe to them too
        self.any_ephemeral = any(CONFIGS[c].get("allow_ephemeral") for c in configs)
        import threading  # noqa: PLC0415
        self._lock = threading.RLock()        # reentrant: serialize refresh/heartbeat vs message handling
        self.bids: dict[str, dict] = defaultdict(dict)
        self.asks: dict[str, dict] = defaultdict(dict)
        self.last_sample: dict[str, float] = {}
        self._w = None; self._w_opened = 0.0; self._w_path = None
        self.n_snapshots = 0; self.msgs_in = 0
        self._dbg_ev: dict = {}               # DIAG: event-type counts seen on the feed
        self._dbg_reject = None               # DIAG: (token, in_meta, in_allowed) of first reject
        self._dbg_pc = [0, 0, 0]              # DIAG: pc tokens [total, in_meta, in_allocated]
        self._dbg_err = None                  # DIAG: first per-event exception (repr + last frame)
        self.reallocate(token_meta)           # initial per-config allocation
        self.out_dir = out_dir; out_dir.mkdir(parents=True, exist_ok=True)
        self.rotate_s = rotate_minutes * 60.0

    def _size_for(self, m: dict) -> float:
        """Per-market resting size: the minimum reward-eligible clip (min mode) or fixed --size."""
        if self.min_qualify:
            return m.get("min_size", 0.0) or 0.0
        return self.size

    def _bankroll(self, c: str) -> float:
        """A strategy's current equity: starting cash + reward + realized/marked trading P&L,
        including markets it has already exited (folded into self.realized)."""
        r = self.realized[c]
        eq = self.start_capital + r["reward"] + r["trade"]
        for t in self.toks:
            q = self.q.get((c, t))
            if q is None:
                continue
            mid = q.mids[-1][1] if q.mids else 0.0
            eq += q.reward - q.fees + q.cash + q.inv * mid
        return eq

    def reallocate(self, token_meta: dict) -> list:
        """Refresh the manifest and free capital from resolved markets. Allocation itself is
        DEMAND-DRIVEN (see _ensure): we only commit capital to a market when we actually observe it
        streaming AND it's in the manifest — so we never quote a stale/dead market. This call:
          1. swaps in the new manifest,
          2. drops markets that resolved (gone from the manifest) — flatten, fold realized P&L,
             release their capital,
        and returns the eligible (durable, reward-paying) manifest tokens to SUBSCRIBE to."""
        with self._lock:
            return self._reallocate(token_meta)

    def _reallocate(self, token_meta: dict) -> list:
        self.meta = token_meta
        for c in self.configs:
            for t in list(self.allowed.get(c, set())):
                if t not in token_meta:                            # resolved -> exit and free capital
                    q = self.q.pop((c, t), None)
                    if q is not None:
                        q._do_flatten()
                        r = self.realized[c]
                        r["reward"] += q.reward
                        r["trade"] += q.cash - q.fees              # inv is 0 after flatten
                        r["flat_cost"] += q.flat_cost
                        r["n_flat"] += q.n_flats
                        r["fills"] += len(q.fills)
                    self.allowed[c].discard(t)
                    self.deployed_clip[c] -= self.sizes[c].pop(t, 0.0)
        self.toks = {t for (_c, t) in self.q}
        # subscribe to every eligible reward market; capital is committed lazily as they stream.
        # Include 5-min crypto only if some strategy opts in (allow_ephemeral).
        self.universe = sorted(t for t, m in token_meta.items()
                               if m.get("pool", 0) > 0 and m.get("v_cents", 0) > 0
                               and (self.any_ephemeral or not _is_ephemeral(m.get("question", ""))))
        if self.auto_min_roc and self.start_capital > 0:
            self.min_roc = self._capital_filling_cutoff(token_meta)
            print(f"[auto-min-roc] cutoff={self.min_roc:.3f} (fills ${self.start_capital:.0f} "
                  f"with the highest-ROC markets)", flush=True)
        self._log_roc(token_meta)
        return self.universe

    def _capital_filling_cutoff(self, token_meta: dict) -> float:
        """The ROC hurdle that just fills start_capital with the highest-ROC eligible markets:
        sort markets by reward-$/$/day descending, accumulate their clips until the budget is used,
        and return the ROC of the marginal market. Deploys the full budget into the BEST markets."""
        cand = []
        for _t, m in token_meta.items():
            if m.get("pool", 0) <= 0 or m.get("v_cents", 0) <= 0:
                continue
            if not self.any_ephemeral and _is_ephemeral(m.get("question", "")):
                continue
            s = self._size_for(m)
            if s <= 0:
                continue
            cand.append((m["pool"] * self.max_capture_share / s, s))
        cand.sort(reverse=True)
        cum, cutoff = 0.0, 0.0
        for r, s in cand:
            cutoff = r
            cum += s
            if cum >= self.start_capital:
                break
        return cutoff

    def _log_roc(self, token_meta: dict) -> None:
        """Print the reward-ROC distribution (modeled $reward/$/day) for durable vs crypto markets,
        so we can pick a sensible --min-roc and see whether crypto pools are actually richer."""
        def roc(m):
            s = self._size_for(m)
            return (m["pool"] * self.max_capture_share / s) if s > 0 else 0.0
        dur, cry = [], []
        for _t, m in token_meta.items():
            if m.get("pool", 0) <= 0 or m.get("v_cents", 0) <= 0:
                continue
            (cry if _is_ephemeral(m.get("question", "")) else dur).append(roc(m))
        dur.sort(); cry.sort()
        def pc(a, p):
            return a[min(len(a) - 1, int(len(a) * p))] if a else 0.0
        if dur:
            print(f"[roc] durable n={len(dur)} p50={pc(dur,.5):.3f} p90={pc(dur,.9):.3f} "
                  f"p99={pc(dur,.99):.3f} max={dur[-1]:.3f}", flush=True)
        if cry:
            print(f"[roc] crypto  n={len(cry)} p50={pc(cry,.5):.3f} p90={pc(cry,.9):.3f} "
                  f"p99={pc(cry,.99):.3f} max={cry[-1]:.3f}", flush=True)

    def _ensure(self, tok: str) -> bool:
        """Demand-driven allocation: only markets IN THE MANIFEST (m is not None) get quoted, and a
        strategy commits capital to one the first time it's seen streaming — if the market is durable,
        clears the reward-ROC hurdle, and the strategy still has free capital. Guarantees we only ever
        trade markets we're actually capturing."""
        m = self.meta.get(tok)
        if not m or m["v_cents"] <= 0:
            return False                                  # not in the manifest -> never quote it
        eph = _is_ephemeral(m.get("question", ""))        # 5-min crypto up/down
        size = self._size_for(m)
        if size <= 0:
            return False
        roc = (m["pool"] * self.max_capture_share) / size     # est reward $/$/day
        any_active = False
        for c in self.configs:
            # crypto strategies (allow_ephemeral) quote ONLY ephemeral markets; others ONLY durable
            if eph != bool(self.kw[c].get("allow_ephemeral")):
                continue
            if tok not in self.allowed[c]:
                # consider committing capital to this freshly-seen market
                if roc < self.min_roc or (self.max_roc > 0 and roc > self.max_roc):
                    continue                              # below the floor or above the toxic ceiling
                if self.start_capital > 0:                 # capital<=0 => unlimited (quote everything)
                    avail = self.start_capital + self.realized[c]["reward"] + self.realized[c]["trade"]
                    if self.deployed_clip[c] + size > avail:
                        continue                          # no free capital for this strategy
                self.allowed[c].add(tok); self.sizes[c][tok] = size; self.deployed_clip[c] += size
            any_active = True
            if (c, tok) in self.q:
                continue
            kw = dict(self.kw[c])
            kw.pop("allow_ephemeral", None)               # not a Quoter kwarg
            if kw.pop("_optimal", False):
                s_star_c = optimal_offset_cents(m["v_cents"], m["pool"] / 1440.0, size)
                kw["quote_offset"] = s_star_c / 100.0   # cents -> price units
            mult = kw.pop("inv_cap_mult") if "inv_cap_mult" in kw else self.inv_cap_mult
            inv_cap = size * mult
            mh = kw.pop("max_hold_minutes") * 60.0 if "max_hold_minutes" in kw else self.max_hold_seconds
            self.q[(c, tok)] = Quoter(size, inv_cap, fill_model=self.fill_model,
                                      capture_mult=self.capture_mult, reward_pool=m["pool"],
                                      reward_min_size=m["min_size"], reward_v_cents=m["v_cents"],
                                      max_capture_share=self.max_capture_share,
                                      quote_latency=self.quote_latency,
                                      cancel_on_move=self.cancel_on_move,
                                      max_hold_seconds=mh, mids_maxlen=MIDS_MAXLEN, **kw)
        if any_active:
            self.toks.add(tok)
        return any_active

    def _close_current(self):
        """Close the open spool and rename .tmp -> .jsonl.gz so the uploader only ever sees a
        COMPLETE file (the open file stays *.tmp, which the uploader skips — no mid-write corruption)."""
        if self._w:
            self._w.close()
            if self._w_path and self._w_path.endswith(".tmp"):
                try:
                    os.replace(self._w_path, self._w_path[:-4])
                except OSError:
                    pass
        self._w = None; self._w_path = None

    def _writer(self, t):
        if self._w is None or (time.time() - self._w_opened) > self.rotate_s:
            self._close_current()
            host = os.uname().nodename.replace("_", "-")
            # epoch in the name (like the collector) so the S3 uploader's date logic works uniformly;
            # write to *.tmp while open, renamed on close so the uploader never grabs a partial file.
            final = self.out_dir / f"paper_{host}_{os.getpid()}_{int(time.time())}.jsonl.gz"
            self._w_path = str(final) + ".tmp"
            self._w = gzip.open(self._w_path, "wt")
            self._w_opened = time.time()
        return self._w

    def _maybe_sample(self, tok: str, t: float):
        m = self.meta.get(tok)
        if tok not in self.toks or not m or t - self.last_sample.get(tok, -1e9) < SAMPLE_SECONDS:
            return
        bb_l = [(p, s) for p, s in self.bids[tok].items() if s > 0]
        ba_l = [(p, s) for p, s in self.asks[tok].items() if s > 0]
        mn = [p for p, s in bb_l if s >= m["min_size"]]
        mx = [p for p, s in ba_l if s >= m["min_size"]]
        if not mn or not mx:
            return
        bb, ba = max(mn), min(mx)
        mid = (bb + ba) / 2.0
        q1 = side_score(bb_l, mid, m["v_cents"], m["min_size"])   # competing depth — config-independent
        q2 = side_score(ba_l, mid, m["v_cents"], m["min_size"])
        self.last_sample[tok] = t
        w = self._writer(t)
        for c in self.configs:                                   # one snapshot row per config in this market
            q = self.q.get((c, tok))
            if q is None:
                continue
            rw_before = q.reward
            q.credit_sample(mid, q1, q2)
            marked = q.cash + q.inv * mid - q.fees
            w.write(json.dumps({
                "t": round(t, 1), "token": tok, "config": c, "size": q.our_size,
                "mid": round(mid, 4), "our_bid": q.our_bid, "our_ask": q.our_ask,
                "inv": round(q.inv, 2), "marked_pnl": round(marked, 4),
                "reward_cum": round(q.reward, 4), "reward_step": round(q.reward - rw_before, 5),
                "q_bid_book": round(q1, 1), "q_ask_book": round(q2, 1), "n_fills": len(q.fills),
            }) + "\n")
        w.flush()                 # flush gzip buffer to disk so the .tmp is readable + crash-durable
        self.n_snapshots += 1

    def _agg(self, cfg: str) -> dict:
        qs = [self.q[(cfg, t)] for t in self.toks if (cfg, t) in self.q]
        r = self.realized[cfg]                      # results from markets already exited
        reward = r["reward"] + sum(q.reward for q in qs)
        trade = r["trade"] + sum(q.cash - q.fees + (q.inv * q.mids[-1][1] if q.mids else 0.0) for q in qs)
        # cost to flatten the CURRENT (live) inventory now (|inv| * half-spread)
        liq = sum(abs(q.inv) * (q.best_ask - q.best_bid) / 2
                  for q in qs if q.best_bid is not None and q.best_ask is not None)
        deployed = sum(self.sizes[cfg].get(t, 0.0) for t in self.allowed.get(cfg, ()))
        return {"reward": reward, "trade": trade, "net": reward + trade,
                "bankroll": self.start_capital + reward + trade, "n_markets": len(self.allowed.get(cfg, ())),
                "deployed": deployed,
                "fills": r["fills"] + sum(len(q.fills) for q in qs), "inv": sum(abs(q.inv) for q in qs),
                "liq_now": liq, "net_if_flat": reward + trade - liq,
                "flat_cost": r["flat_cost"] + sum(q.flat_cost for q in qs),
                "n_flat": r["n_flat"] + sum(q.n_flats for q in qs)}

    def heartbeat(self) -> str:
        with self._lock:
            return self._heartbeat()

    def _heartbeat(self) -> str:
        head = f"[hb] msgs={self.msgs_in} markets={len(self.toks)} snapshots={self.n_snapshots}"
        lines = [head, f"   DIAG ev={self._dbg_ev} pc[total,in_meta,allocated]={self._dbg_pc} "
                       f"meta={len(self.meta)} quoters={len(self.q)} err={self._dbg_err}"]
        for c in self.configs:                       # one line per strategy — head-to-head
            a = self._agg(c)
            # bankroll = revolving equity (start + reward + P&L); mkts = markets it currently quotes
            lines.append(f"   {c:16} bankroll=${a['bankroll']:.2f} reward=${a['reward']:.4f} "
                         f"net_if_flat=${a['net_if_flat']:.4f} mkts={a['n_markets']} "
                         f"deployed=${a['deployed']:.0f} |inv|={a['inv']:.0f} "
                         f"flat=${a['flat_cost']:.4f}({a['n_flat']})")
        return "\n".join(lines)

    def process_message(self, payload):
        """Handle one collector WS payload (book / price_change / last_trade_price). Idempotent.
        Locked so a concurrent universe refresh can't mutate quoters mid-iteration."""
        with self._lock:
            self._process_message(payload)

    def _process_message(self, payload):
        self.msgs_in += 1
        msgs = payload if isinstance(payload, list) else [payload]
        for e in msgs:
            try:
                self._handle_event(e)
            except Exception as exc:  # noqa: BLE001  surface (don't let the WS swallow it silently)
                if self._dbg_err is None:
                    import traceback
                    self._dbg_err = repr(exc) + " | " + traceback.format_exc().splitlines()[-2][:120]

    def _handle_event(self, e):
        if True:
            et = e.get("event_type") or e.get("type")
            self._dbg_ev[et] = self._dbg_ev.get(et, 0) + 1            # DIAG: event-type counts
            t = _ts(e.get("timestamp"))
            if et == "book":
                tok = str(e.get("asset_id") or "")
                if not self._ensure(tok):
                    if self._dbg_reject is None:                      # DIAG: capture first rejection
                        self._dbg_reject = (tok, tok in self.meta,
                                            any(tok in self.allowed[c] for c in self.configs))
                    return
                self.bids[tok] = {pp: ss for pp, ss in
                                  ((_f(b.get("price")), _f(b.get("size"))) for b in (e.get("bids") or []))
                                  if pp is not None and ss}
                self.asks[tok] = {pp: ss for pp, ss in
                                  ((_f(a.get("price")), _f(a.get("size"))) for a in (e.get("asks") or []))
                                  if pp is not None and ss}
                self._feed_quote(tok, t)
                self._maybe_sample(tok, t)
            elif et == "price_change":
                for pc in (e.get("price_changes") or []):
                    tok = str(pc.get("asset_id") or "")
                    self._dbg_pc[0] += 1                              # DIAG: total pc tokens seen
                    if tok in self.meta:
                        self._dbg_pc[1] += 1                         # ... that are in the manifest
                        if any(tok in self.allowed[c] for c in self.configs):
                            self._dbg_pc[2] += 1                     # ... that we allocated
                    if not self._ensure(tok):
                        if self._dbg_reject is None:                  # DIAG: capture first pc rejection
                            self._dbg_reject = ("pc", tok, tok in self.meta,
                                                any(tok in self.allowed[c] for c in self.configs))
                        continue
                    price, sz = _f(pc.get("price")), _f(pc.get("size"))
                    side = str(pc.get("side") or "").upper()
                    if price is None or sz is None:
                        continue
                    book = self.bids[tok] if side == "BUY" else self.asks[tok]
                    if sz <= 0:
                        book.pop(price, None)
                    else:
                        book[price] = sz
                    self._feed_quote(tok, t)
                    self._maybe_sample(tok, t)
            elif et == "last_trade_price":
                tok = str(e.get("asset_id") or "")
                p, s = _f(e.get("price")), _f(e.get("size"))
                side = str(e.get("side") or "").upper()
                if self._ensure(tok) and p is not None and s and side in ("BUY", "SELL"):
                    for c in self.configs:                       # every strategy that quotes this market
                        q = self.q.get((c, tok))
                        if q is not None:
                            q.on_trade(t, p, side, s)

    def _feed_quote(self, tok: str, t: float):
        bb_l = [(p, s) for p, s in self.bids[tok].items() if s > 0]
        ba_l = [(p, s) for p, s in self.asks[tok].items() if s > 0]
        if not bb_l or not ba_l:
            return
        bb = max(bb_l, key=lambda x: x[0]); ba = min(ba_l, key=lambda x: x[0])
        if bb[0] < ba[0]:
            for c in self.configs:                               # every strategy that quotes this market
                q = self.q.get((c, tok))
                if q is not None:
                    q.on_quote(t, bb[0], ba[0], bb[1], ba[1])

    def close(self):
        self._close_current()

    def summary(self) -> dict:
        with self._lock:
            return self._summary()

    def _summary(self) -> dict:
        out = {"size": ("min_qualify" if self.min_qualify else self.size),
               "start_capital": self.start_capital, "markets_subscribed": len(self.toks),
               "min_roc": self.min_roc, "snapshots": self.n_snapshots, "by_config": {}}
        for c in self.configs:
            a = self._agg(c)
            deployed = sum(self.sizes[c].get(t, 0.0) for t in self.allowed.get(c, ()))
            out["by_config"][c] = {"bankroll": round(a["bankroll"], 2), "reward": round(a["reward"], 3),
                                   "trade_pnl": round(a["trade"], 3), "net": round(a["net"], 3),
                                   "net_if_flat": round(a["net_if_flat"], 3), "n_markets": a["n_markets"],
                                   "capital_deployed_est": round(deployed, 0),
                                   "flatten_cost": round(a["flat_cost"], 3), "n_flatten": a["n_flat"],
                                   "return_pct": round(a["net"] / self.start_capital * 100, 3) if self.start_capital > 0 else None}
        out["best_net"] = max(self.configs, key=lambda c: self._agg(c)["net"]) if self.toks else None
        return out


def run_replay(sim: PaperSim, paths):
    """Replay one or more captured files in chronological order (single sim, single close)."""
    import glob as _glob
    import re as _re
    if isinstance(paths, (str, Path)):
        paths = [paths]
    files = []
    for p in paths:
        files.extend(_glob.glob(str(p)))
    # order by the epoch embedded in the collector filename so the tape is chronological
    def _epoch(p):
        m = _re.search(r"_(\d+)\.jsonl", os.path.basename(p))
        return int(m.group(1)) if m else 0
    files.sort(key=_epoch)
    for path in files:
        opener = gzip.open if path.endswith(".gz") else open
        try:
            with opener(path, "rt") as fh:
                for line in fh:
                    if line.strip():
                        try:
                            sim.process_message(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except (OSError, EOFError) as e:
            print(f"skip {os.path.basename(path)}: {e}", flush=True)
    sim.close()


def run_live(sim: PaperSim, tokens: list[str], minutes: float,
             manifest_dir: Path | None = None, refresh_minutes: float = 0.0):
    """Connect to the CLOB market WS and stream into the sim. Mirrors the collector's connection
    (app-level PING keepalive — the server's protocol ping/pong is unreliable). Live-only; not
    exercised in CI. Requires `websocket-client`.

    If refresh_minutes>0 and a manifest_dir is given, a background thread periodically reloads the
    newest manifest, re-runs sim.reallocate() (so freshly created markets are added and resolved
    ones drop / their freed capital redeploys), and forces a reconnect to resubscribe."""
    import threading  # noqa: PLC0415
    import websocket  # noqa: PLC0415
    url = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    stop_at = time.time() + minutes * 60.0
    state = {"tokens": list(tokens), "ws": None}

    def refresher():
        while time.time() < stop_at:
            time.sleep(max(refresh_minutes, 1.0) * 60.0)
            try:
                mans = sorted(manifest_dir.glob("manifest_*.json"))
                if not mans:
                    continue
                new_tokens = sim.reallocate(load_token_meta(mans[-1]))
                if set(new_tokens) != set(state["tokens"]):
                    added = len(set(new_tokens) - set(state["tokens"]))
                    dropped = len(set(state["tokens"]) - set(new_tokens))
                    state["tokens"] = new_tokens
                    print(f"paper-sim universe refresh: {len(new_tokens)} markets "
                          f"(+{added}/-{dropped})", flush=True)
                    if state["ws"] is not None:
                        state["ws"].close()   # reconnect -> on_open resubscribes to new_tokens
            except Exception as exc:  # noqa: BLE001
                print("paper-sim refresh failed:", repr(exc), flush=True)

    def heartbeater():
        # print on a fixed timer, independent of message flow, so silence never hides a live process;
        # the msgs= counter then reveals whether the FEED has stalled vs the process being stuck.
        while time.time() < stop_at:
            time.sleep(60.0)
            try:
                print(sim.heartbeat(), flush=True)
            except Exception as exc:  # noqa: BLE001
                print("paper-sim heartbeat error:", repr(exc), flush=True)

    if refresh_minutes and manifest_dir:
        threading.Thread(target=refresher, daemon=True).start()
    threading.Thread(target=heartbeater, daemon=True).start()

    def on_open(ws):
        ws.send(json.dumps({"type": "market", "assets_ids": sorted(state["tokens"])}))
        print(f"paper-sim subscribed to {len(state['tokens'])} tokens", flush=True)

    def on_message(ws, msg):
        try:
            sim.process_message(json.loads(msg))
        except json.JSONDecodeError:
            pass  # non-JSON keepalive ("PONG")
        if time.time() > stop_at:
            ws.close()

    while time.time() < stop_at:
        ws = websocket.WebSocketApp(url, on_open=on_open, on_message=on_message)
        state["ws"] = ws
        try:
            ws.run_forever(ping_interval=10, ping_timeout=8)
        except Exception as exc:  # noqa: BLE001
            print("paper-sim reconnecting after:", exc, flush=True)
            time.sleep(2)
    sim.close()


def reconcile_rewards(tokens: list[str]) -> dict:
    """STUB: pull actual per-epoch reward allocations from the Markets API for `tokens` and compare
    to our modeled reward. Only meaningful once REAL orders have rested (paper orders are not in the
    book, so no real payout accrues). Endpoint: GET gamma-api / markets reward allocations."""
    raise NotImplementedError("reconciliation requires real resting orders; wire to Markets API")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--replay", type=str, nargs="+",
                   help="captured collector jsonl(.gz) file(s)/glob(s) to replay offline, in time order")
    g.add_argument("--live", action="store_true", help="stream the live CLOB WebSocket")
    ap.add_argument("--manifest", type=Path, default=None, help="manifest_*.json for token reward params")
    ap.add_argument("--manifest-dir", type=Path, default=None)
    ap.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS),
                    help="strategies to run in parallel on the same feed (head-to-head). Default: all.")
    ap.add_argument("--size", type=float, default=200.0)
    ap.add_argument("--capital", type=float, default=5000.0,
                    help="total capital budget; quotes top floor(capital/size) markets by pool (0=unlimited)")
    ap.add_argument("--inv-cap-mult", type=float, default=1.0,
                    help="inventory cap per market = mult*size; 1 keeps held position within the budget")
    ap.add_argument("--fill-model", choices=["prorata", "fifo"], default="prorata")
    ap.add_argument("--capture-mult", type=float, default=1.0)
    ap.add_argument("--max-capture-share", type=float, default=0.10,
                    help="cap modeled reward share per market (competition fills in; 1.0=uncapped)")
    ap.add_argument("--quote-latency", type=float, default=0.2,
                    help="seconds our live quote lags the book (stale-order pickoff risk); 0=instant")
    ap.add_argument("--cancel-on-move", type=float, default=0.01,
                    help="cancel a resting quote when mid drifts more than this (price units); 0=off")
    ap.add_argument("--max-hold-minutes", type=float, default=120.0,
                    help="force-exit any position carried longer than this (resolution-risk control); 0=never")
    ap.add_argument("--output-dir", type=Path, default=Path("reports/paper_sim"))
    ap.add_argument("--rotate-minutes", type=float, default=15.0)
    ap.add_argument("--minutes", type=float, default=240.0, help="live run duration")
    ap.add_argument("--refresh-minutes", type=float, default=0.0,
                    help="live: reload the newest manifest every N min and resubscribe (0=off)")
    ap.add_argument("--min-roc", type=float, default=0.0,
                    help="min reward-$/$/day to deploy into a market (capacity hurdle; 0=deploy any)")
    ap.add_argument("--auto-min-roc", action="store_true",
                    help="auto-set the ROC hurdle each refresh to exactly fill capital with the best markets")
    ap.add_argument("--max-roc", type=float, default=0.0,
                    help="skip markets ABOVE this reward-ROC (the toxic high-reward ones); 0=no cap")
    ap.add_argument("--tokens", nargs="*", default=[], help="token ids for --live")
    ap.add_argument("--tokens-file", type=Path, default=None)
    args = ap.parse_args()

    manifest = args.manifest or (sorted(args.manifest_dir.glob("manifest_*.json"))[-1]
                                 if args.manifest_dir else None)
    token_meta = load_token_meta(manifest)
    sim = PaperSim(token_meta, args.size, args.inv_cap_mult, args.configs,
                   args.fill_model, args.capture_mult, args.output_dir, args.rotate_minutes,
                   capital=args.capital, max_capture_share=args.max_capture_share,
                   quote_latency=args.quote_latency, cancel_on_move=args.cancel_on_move,
                   max_hold_seconds=args.max_hold_minutes * 60.0, min_roc=args.min_roc,
                   auto_min_roc=args.auto_min_roc, max_roc=args.max_roc)
    size_desc = "min-qualify" if args.size <= 0 else str(args.size)
    print(f"paper-sim: configs={args.configs} size={size_desc} capital=${args.capital:,.0f}/strategy "
          f"min_roc={args.min_roc} -> {len(sim.universe)} eligible markets to watch; capital committed "
          f"on first sight; reward-tokens={sum(1 for m in token_meta.values() if m['pool']>0)}", flush=True)

    # close the spool cleanly on systemd stop/restart (rename .tmp -> .jsonl.gz so it ships, not lost)
    import signal  # noqa: PLC0415

    def _shutdown(*_):
        sim.close()
        (args.output_dir / "paper_sim_summary.json").write_text(json.dumps(sim.summary(), indent=2) + "\n")
        os._exit(0)
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    if args.replay:
        run_replay(sim, args.replay)
    else:
        tokens = list(args.tokens)
        if args.tokens_file and args.tokens_file.exists():
            tokens += [ln.strip() for ln in args.tokens_file.read_text().splitlines() if ln.strip()]
        if not tokens:
            tokens = list(sim.universe)   # all eligible manifest markets; capital committed on first sight
        run_live(sim, tokens, args.minutes,
                 manifest_dir=args.manifest_dir, refresh_minutes=args.refresh_minutes)

    s = sim.summary()
    (args.output_dir / "paper_sim_summary.json").write_text(json.dumps(s, indent=2) + "\n")
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
