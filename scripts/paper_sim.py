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

# Placeholder fill-rate / adverse-selection assumptions for the depth_optimal strategy's s*.
# These are PARAMETRIC stand-ins (exponential fill decay A*e^{-k*s}, constant toxicity eta0 in
# cents/share) until per-market lambda(s)/eta(s) are fit from the tape (stale_order_pool.py).
# Replace `depth_params_for(token)` with the fitted curves once available.
# Measured from data/pull/dt=2026-06-22 (the "other" bucket = ~whole universe): fill rate barely
# decays with depth (k~0.012) and touch toxicity eta0~0.29c. Placeholder guesses were a=20,k=1.0.
DEPTH_PARAMS = dict(a=0.007, k=0.012, eta0=0.286, size=1.0)


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
}


class PaperSim:
    """Holds per-token reconstructed book + Quoter; emits per-minute snapshots."""

    def __init__(self, token_meta: dict, size: float, inv_cap_mult: float, configs: list[str],
                 fill_model: str, capture_mult: float, out_dir: Path, rotate_minutes: float,
                 capital: float = 0.0, max_capture_share: float = 1.0,
                 quote_latency: float = 0.0, cancel_on_move: float = 0.0,
                 max_hold_seconds: float = 0.0):
        self.meta = token_meta
        self.size = size; self.configs = configs
        self.kw = {c: dict(CONFIGS[c]) for c in configs}   # all configs run in parallel, same feed
        self.fill_model = fill_model; self.capture_mult = capture_mult
        self.max_capture_share = max_capture_share
        self.quote_latency = quote_latency; self.cancel_on_move = cancel_on_move
        self.max_hold_seconds = max_hold_seconds
        self.inv_cap = size * inv_cap_mult
        # CAPITAL BUDGET: two-sided resting collateral is ~$1*size per market, so a fixed budget
        # funds floor(capital/size) markets — we quote the top-N by reward pool (the targetable
        # signal), matching how a real $capital book would be deployed.
        self.capital = capital
        self.allowed = None
        if capital and capital > 0:
            n = max(1, int(capital / size))
            # only markets we can actually qualify for: our clip must clear min_incentive_size,
            # else we score ZERO reward there. Rank the qualifiable ones by pool.
            elig = [t for t, m in token_meta.items() if m["pool"] > 0 and m["min_size"] <= size]
            ranked = sorted(elig, key=lambda t: token_meta[t]["pool"], reverse=True)
            self.allowed = set(ranked[:n])
        self.out_dir = out_dir; out_dir.mkdir(parents=True, exist_ok=True)
        self.rotate_s = rotate_minutes * 60.0
        self.bids: dict[str, dict] = defaultdict(dict)
        self.asks: dict[str, dict] = defaultdict(dict)
        self.q: dict[tuple, Quoter] = {}      # (config, token) -> Quoter (all configs, same markets)
        self.toks: set = set()                # tokens we've instantiated quoters for
        self.last_sample: dict[str, float] = {}
        self._w = None; self._w_opened = 0.0; self._w_path = None
        self.n_snapshots = 0; self.msgs_in = 0

    def _ensure(self, tok: str) -> bool:
        """Ensure a Quoter exists for every config on this token; False if token isn't eligible."""
        m = self.meta.get(tok)
        if not m or m["v_cents"] <= 0:
            return False
        if self.allowed is not None and tok not in self.allowed:
            return False   # outside the capital budget's top-N markets
        if tok not in self.toks:
            for c in self.configs:
                kw = dict(self.kw[c])
                if kw.pop("_optimal", False):
                    # per-token s*: r0 = this token's per-minute reward slice at the touch
                    s_star_c = optimal_offset_cents(m["v_cents"], m["pool"] / 1440.0, self.size)
                    kw["quote_offset"] = s_star_c / 100.0   # cents -> price units
                # per-config overrides for the inventory/flatten experiment (else use the globals)
                inv_cap = self.size * kw.pop("inv_cap_mult") if "inv_cap_mult" in kw else self.inv_cap
                mh = kw.pop("max_hold_minutes") * 60.0 if "max_hold_minutes" in kw else self.max_hold_seconds
                self.q[(c, tok)] = Quoter(self.size, inv_cap, fill_model=self.fill_model,
                                          capture_mult=self.capture_mult, reward_pool=m["pool"],
                                          reward_min_size=m["min_size"], reward_v_cents=m["v_cents"],
                                          max_capture_share=self.max_capture_share,
                                          quote_latency=self.quote_latency,
                                          cancel_on_move=self.cancel_on_move,
                                          max_hold_seconds=mh, **kw)
            self.toks.add(tok)
        return True

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
        for c in self.configs:                                   # one snapshot row per config
            q = self.q[(c, tok)]
            rw_before = q.reward
            q.credit_sample(mid, q1, q2)
            marked = q.cash + q.inv * mid - q.fees
            w.write(json.dumps({
                "t": round(t, 1), "token": tok, "config": c, "size": self.size,
                "mid": round(mid, 4), "our_bid": q.our_bid, "our_ask": q.our_ask,
                "inv": round(q.inv, 2), "marked_pnl": round(marked, 4),
                "reward_cum": round(q.reward, 4), "reward_step": round(q.reward - rw_before, 5),
                "q_bid_book": round(q1, 1), "q_ask_book": round(q2, 1), "n_fills": len(q.fills),
            }) + "\n")
        w.flush()                 # flush gzip buffer to disk so the .tmp is readable + crash-durable
        self.n_snapshots += 1

    def _agg(self, cfg: str) -> dict:
        qs = [self.q[(cfg, t)] for t in self.toks]
        reward = sum(q.reward for q in qs)
        trade = sum(q.cash - q.fees + (q.inv * q.mids[-1][1] if q.mids else 0.0) for q in qs)
        # cost to flatten the CURRENT inventory now (|inv| * half-spread) — continuous, not lumpy
        liq = sum(abs(q.inv) * (q.best_ask - q.best_bid) / 2
                  for q in qs if q.best_bid is not None and q.best_ask is not None)
        return {"reward": reward, "trade": trade, "net": reward + trade,
                "fills": sum(len(q.fills) for q in qs), "inv": sum(abs(q.inv) for q in qs),
                "liq_now": liq, "net_if_flat": reward + trade - liq,
                "flat_cost": sum(q.flat_cost for q in qs), "n_flat": sum(q.n_flats for q in qs)}

    def heartbeat(self) -> str:
        head = f"[hb] msgs={self.msgs_in} markets={len(self.toks)} snapshots={self.n_snapshots}"
        lines = [head]
        for c in self.configs:                       # one line per strategy — head-to-head
            a = self._agg(c)
            # net_if_flat = what you'd keep liquidating now; liq = exit cost of current inventory;
            # flat = spread already paid on forced exits
            lines.append(f"   {c:13} reward=${a['reward']:.4f} net_if_flat=${a['net_if_flat']:.4f} "
                         f"|inv|={a['inv']:.0f} liq=${a['liq_now']:.4f} flat=${a['flat_cost']:.4f}({a['n_flat']})")
        return "\n".join(lines)

    def process_message(self, payload):
        """Handle one collector WS payload (book / price_change / last_trade_price). Idempotent."""
        self.msgs_in += 1
        msgs = payload if isinstance(payload, list) else [payload]
        for e in msgs:
            et = e.get("event_type") or e.get("type")
            t = _ts(e.get("timestamp"))
            if et == "book":
                tok = str(e.get("asset_id") or "")
                if not self._ensure(tok):
                    continue
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
                    if not self._ensure(tok):
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
                    for c in self.configs:                       # every strategy sees the same trade
                        self.q[(c, tok)].on_trade(t, p, side, s)

    def _feed_quote(self, tok: str, t: float):
        bb_l = [(p, s) for p, s in self.bids[tok].items() if s > 0]
        ba_l = [(p, s) for p, s in self.asks[tok].items() if s > 0]
        if not bb_l or not ba_l:
            return
        bb = max(bb_l, key=lambda x: x[0]); ba = min(ba_l, key=lambda x: x[0])
        if bb[0] < ba[0]:
            for c in self.configs:                               # every strategy sees the same book
                self.q[(c, tok)].on_quote(t, bb[0], ba[0], bb[1], ba[1])

    def close(self):
        self._close_current()

    def summary(self) -> dict:
        deployed = len(self.toks) * self.size          # ~$1*size resting collateral per market
        out = {"size": self.size, "capital_budget": self.capital, "markets_quoted": len(self.toks),
               "capital_deployed_est": round(deployed, 0), "snapshots": self.n_snapshots,
               "by_config": {}}
        for c in self.configs:
            a = self._agg(c)
            out["by_config"][c] = {"reward": round(a["reward"], 3), "trade_pnl": round(a["trade"], 3),
                                   "net": round(a["net"], 3), "liq_now": round(a["liq_now"], 3),
                                   "net_if_flat": round(a["net_if_flat"], 3),
                                   "flatten_cost": round(a["flat_cost"], 3), "n_flatten": a["n_flat"],
                                   "roc_on_budget": round(a["net_if_flat"] / self.capital, 5) if self.capital > 0 else None}
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


def run_live(sim: PaperSim, tokens: list[str], minutes: float):
    """Connect to the CLOB market WS and stream into the sim. Mirrors the collector's connection
    (app-level PING keepalive — the server's protocol ping/pong is unreliable). Live-only; not
    exercised in CI. Requires `websocket-client`."""
    import websocket  # noqa: PLC0415
    url = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    stop_at = time.time() + minutes * 60.0
    last_hb = [time.time()]

    def on_open(ws):
        ws.send(json.dumps({"type": "market", "assets_ids": sorted(tokens)}))
        print(f"paper-sim subscribed to {len(tokens)} tokens", flush=True)

    def on_message(ws, msg):
        try:
            sim.process_message(json.loads(msg))
        except json.JSONDecodeError:
            pass  # non-JSON keepalive ("PONG")
        if time.time() - last_hb[0] > 60.0:        # heartbeat to the journal every minute
            print(sim.heartbeat(), flush=True); last_hb[0] = time.time()
        if time.time() > stop_at:
            ws.close()

    while time.time() < stop_at:
        ws = websocket.WebSocketApp(url, on_open=on_open, on_message=on_message)
        # app-level PING every ~10s in a side thread would go here (see collect_clob_book.run_collector)
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
                   max_hold_seconds=args.max_hold_minutes * 60.0)
    budget_n = (int(args.capital / args.size) if args.capital > 0 else None)
    print(f"paper-sim: configs={args.configs} size={args.size} capital=${args.capital:,.0f} "
          f"-> top {budget_n if budget_n else 'ALL'} markets by pool; "
          f"reward-tokens={sum(1 for m in token_meta.values() if m['pool']>0)}", flush=True)

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
            # only subscribe to the budgeted top-N markets (or all reward markets if unlimited)
            tokens = sorted(sim.allowed) if sim.allowed is not None \
                else [t for t, m in token_meta.items() if m["pool"] > 0]
        run_live(sim, tokens, args.minutes)

    s = sim.summary()
    (args.output_dir / "paper_sim_summary.json").write_text(json.dumps(s, indent=2) + "\n")
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
