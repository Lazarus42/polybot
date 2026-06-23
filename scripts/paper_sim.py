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

SAMPLE_SECONDS = 60.0


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
    "clv_full": dict(momentum_window=300, skew_threshold=0.005, debounce_trades=10, inv_skew=0.01,
                     vol_window=300, vol_spread_coeff=0.5, tox_threshold=0.01, tox_window=60,
                     tox_cooldown=60),
    "neutral": {},
}


class PaperSim:
    """Holds per-token reconstructed book + Quoter; emits per-minute snapshots."""

    def __init__(self, token_meta: dict, size: float, inv_cap_mult: float, config: str,
                 fill_model: str, capture_mult: float, out_dir: Path, rotate_minutes: float,
                 capital: float = 0.0):
        self.meta = token_meta
        self.size = size; self.config = config
        self.kw = dict(CONFIGS[config])
        self.fill_model = fill_model; self.capture_mult = capture_mult
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
        self.q: dict[str, Quoter] = {}
        self.last_sample: dict[str, float] = {}
        self._w = None; self._w_opened = 0.0; self._w_path = None
        self.n_snapshots = 0; self.msgs_in = 0

    def _quoter(self, tok: str) -> Quoter | None:
        m = self.meta.get(tok)
        if not m or m["v_cents"] <= 0:
            return None
        if self.allowed is not None and tok not in self.allowed:
            return None   # outside the capital budget's top-N markets
        if tok not in self.q:
            self.q[tok] = Quoter(self.size, self.inv_cap, fill_model=self.fill_model,
                                 capture_mult=self.capture_mult, reward_pool=m["pool"],
                                 reward_min_size=m["min_size"], reward_v_cents=m["v_cents"], **self.kw)
        return self.q[tok]

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
        q = self.q.get(tok); m = self.meta.get(tok)
        if not q or not m or t - self.last_sample.get(tok, -1e9) < SAMPLE_SECONDS:
            return
        bb_l = [(p, s) for p, s in self.bids[tok].items() if s > 0]
        ba_l = [(p, s) for p, s in self.asks[tok].items() if s > 0]
        mn = [p for p, s in bb_l if s >= m["min_size"]]
        mx = [p for p, s in ba_l if s >= m["min_size"]]
        if not mn or not mx:
            return
        bb, ba = max(mn), min(mx)
        mid = (bb + ba) / 2.0
        q1 = side_score(bb_l, mid, m["v_cents"], m["min_size"])
        q2 = side_score(ba_l, mid, m["v_cents"], m["min_size"])
        rw_before = q.reward
        q.credit_sample(mid, q1, q2)
        self.last_sample[tok] = t
        marked = q.cash + q.inv * mid - q.fees
        w = self._writer(t)
        w.write(json.dumps({
            "t": round(t, 1), "token": tok, "config": self.config, "size": self.size,
            "mid": round(mid, 4), "our_bid": q.our_bid, "our_ask": q.our_ask,
            "inv": round(q.inv, 2), "marked_pnl": round(marked, 4),
            "reward_cum": round(q.reward, 4), "reward_step": round(q.reward - rw_before, 5),
            "q_bid_book": round(q1, 1), "q_ask_book": round(q2, 1), "n_fills": len(q.fills),
        }) + "\n")
        w.flush()                 # flush gzip buffer to disk so the .tmp is readable + crash-durable
        self.n_snapshots += 1

    def heartbeat(self) -> str:
        fills = sum(len(q.fills) for q in self.q.values())
        reward = sum(q.reward for q in self.q.values())
        quoting = sum(1 for q in self.q.values() if q.our_bid is not None or q.our_ask is not None)
        # trading P&L = cash from fills - fees + inventory marked at the live mid (adverse shows here)
        trade_pnl = sum(q.cash - q.fees + (q.inv * q.mids[-1][1] if q.mids else 0.0)
                        for q in self.q.values())
        inv = sum(abs(q.inv) for q in self.q.values())
        return (f"[hb] msgs={self.msgs_in} markets={len(self.q)} quoting_now={quoting} "
                f"snapshots={self.n_snapshots} fills={fills} reward=${reward:.4f} "
                f"trade_pnl=${trade_pnl:.4f} net=${reward + trade_pnl:.4f} |inv|={inv:.2f}")

    def process_message(self, payload):
        """Handle one collector WS payload (book / price_change / last_trade_price). Idempotent."""
        self.msgs_in += 1
        msgs = payload if isinstance(payload, list) else [payload]
        for e in msgs:
            et = e.get("event_type") or e.get("type")
            t = _ts(e.get("timestamp"))
            if et == "book":
                tok = str(e.get("asset_id") or "")
                if not self._quoter(tok):
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
                    if not self._quoter(tok):
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
                q = self._quoter(tok)
                p, s = _f(e.get("price")), _f(e.get("size"))
                side = str(e.get("side") or "").upper()
                if q and p is not None and s and side in ("BUY", "SELL"):
                    q.on_trade(t, p, side, s)

    def _feed_quote(self, tok: str, t: float):
        bb_l = [(p, s) for p, s in self.bids[tok].items() if s > 0]
        ba_l = [(p, s) for p, s in self.asks[tok].items() if s > 0]
        if not bb_l or not ba_l:
            return
        bb = max(bb_l, key=lambda x: x[0]); ba = min(ba_l, key=lambda x: x[0])
        if bb[0] < ba[0]:
            self.q[tok].on_quote(t, bb[0], ba[0], bb[1], ba[1])

    def close(self):
        self._close_current()

    def summary(self) -> dict:
        tot_reward = sum(q.reward for q in self.q.values())
        tot_marked = sum(q.cash - q.fees for q in self.q.values())  # ex-inventory mark
        deployed = len(self.q) * self.size            # ~$1*size resting collateral per market
        net = tot_reward + tot_marked
        s = {"config": self.config, "size": self.size, "capital_budget": self.capital,
             "markets_quoted": len(self.q), "capital_deployed_est": round(deployed, 0),
             "snapshots": self.n_snapshots, "total_reward": round(tot_reward, 2),
             "total_trading_cash": round(tot_marked, 2), "net": round(net, 2)}
        if deployed > 0:
            s["roc_on_deployed"] = round(net / deployed, 4)
        if self.capital > 0:
            s["roc_on_budget"] = round(net / self.capital, 4)
        return s


def run_replay(sim: PaperSim, path: Path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            if line.strip():
                try:
                    sim.process_message(json.loads(line))
                except json.JSONDecodeError:
                    continue
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
    g.add_argument("--replay", type=Path, help="captured collector jsonl(.gz) to replay offline")
    g.add_argument("--live", action="store_true", help="stream the live CLOB WebSocket")
    ap.add_argument("--manifest", type=Path, default=None, help="manifest_*.json for token reward params")
    ap.add_argument("--manifest-dir", type=Path, default=None)
    ap.add_argument("--config", choices=list(CONFIGS), default="clv_full")
    ap.add_argument("--size", type=float, default=200.0)
    ap.add_argument("--capital", type=float, default=5000.0,
                    help="total capital budget; quotes top floor(capital/size) markets by pool (0=unlimited)")
    ap.add_argument("--inv-cap-mult", type=float, default=5.0)
    ap.add_argument("--fill-model", choices=["prorata", "fifo"], default="prorata")
    ap.add_argument("--capture-mult", type=float, default=1.0)
    ap.add_argument("--output-dir", type=Path, default=Path("reports/paper_sim"))
    ap.add_argument("--rotate-minutes", type=float, default=15.0)
    ap.add_argument("--minutes", type=float, default=240.0, help="live run duration")
    ap.add_argument("--tokens", nargs="*", default=[], help="token ids for --live")
    ap.add_argument("--tokens-file", type=Path, default=None)
    args = ap.parse_args()

    manifest = args.manifest or (sorted(args.manifest_dir.glob("manifest_*.json"))[-1]
                                 if args.manifest_dir else None)
    token_meta = load_token_meta(manifest)
    sim = PaperSim(token_meta, args.size, args.inv_cap_mult, args.config,
                   args.fill_model, args.capture_mult, args.output_dir, args.rotate_minutes,
                   capital=args.capital)
    budget_n = (int(args.capital / args.size) if args.capital > 0 else None)
    print(f"paper-sim: config={args.config} size={args.size} capital=${args.capital:,.0f} "
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
