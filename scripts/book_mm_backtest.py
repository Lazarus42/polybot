#!/usr/bin/env python3
"""Book-aware passive market-making backtest over CAPTURED CLOB L2 data.

This is the honest MM test the trade-only version couldn't be: we quote at the REAL touch
(from the live book), respect queue position, fill only when real taker flow crosses our
quote, mark inventory at the REAL mid, and measure adverse selection directly (how the mid
moves against us after each fill). Consumes the JSONL written by collect_clob_book.py.

Key outputs: realized pnl, gross spread captured, measured adverse-selection cost, fees, and
inventory stats. `pnl = gross_spread - adverse_selection - fees` is the whole MM question; the
trade-only sim could only estimate adverse selection, here we observe it.

`normalize_events` and `simulate_book_mm` are pure and unit-tested on synthetic streams.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _ts(raw_ts) -> float:
    """Parse a timestamp; convert Unix milliseconds to seconds."""
    v = _f(raw_ts)
    if v is None:
        return 0.0
    return v / 1000.0 if v > 1e11 else v   # >1e11 => milliseconds


def _emit_quote(by_token, tok, t, bb, ba, bs=0.0, asz=0.0):
    if tok and bb is not None and ba is not None and 0.0 < bb < ba < 1.0:
        by_token[tok].append({"type": "quote", "t": t, "bid": bb, "ask": ba,
                              "bid_size": bs or 0.0, "ask_size": asz or 0.0})


def normalize_events(raw: list[dict]) -> dict[str, list[dict]]:
    """Group raw captured CLOB events by token and normalize to quote/trade events by time.

    Matches the ACTUAL WebSocket schema: `price_change` carries a `price_changes[]` list with
    per-asset best_bid/best_ask; `book` lists levels ascending (best bid = max price);
    `best_bid_ask` and `last_trade_price` are top-level per asset; timestamps are ms.
    quote: {type:'quote', t, bid, ask, bid_size, ask_size}; trade: {type:'trade', t, price, side, size}.
    """
    by_token: dict[str, list[dict]] = defaultdict(list)
    for e in raw:
        et = e.get("event_type") or e.get("type")
        t = _ts(e.get("timestamp"))
        if et == "book":
            tok = str(e.get("asset_id") or "")
            bids = [(_f(b.get("price")), _f(b.get("size"))) for b in (e.get("bids") or [])]
            asks = [(_f(a.get("price")), _f(a.get("size"))) for a in (e.get("asks") or [])]
            bids = [(p, s) for p, s in bids if p is not None]
            asks = [(p, s) for p, s in asks if p is not None]
            if bids and asks:
                bb = max(bids, key=lambda x: x[0])      # best bid = highest buy price
                ba = min(asks, key=lambda x: x[0])      # best ask = lowest sell price
                _emit_quote(by_token, tok, t, bb[0], ba[0], bb[1], ba[1])
        elif et == "price_change":
            for pc in (e.get("price_changes") or []):
                _emit_quote(by_token, str(pc.get("asset_id") or ""), t,
                            _f(pc.get("best_bid")), _f(pc.get("best_ask")))
        elif et == "best_bid_ask":
            _emit_quote(by_token, str(e.get("asset_id") or ""), t,
                        _f(e.get("best_bid")), _f(e.get("best_ask")))
        elif et == "last_trade_price":
            tok = str(e.get("asset_id") or "")
            p, s = _f(e.get("price")), _f(e.get("size"))
            side = str(e.get("side") or "").upper()
            if tok and p is not None and s and side in ("BUY", "SELL"):
                by_token[tok].append({"type": "trade", "t": t, "price": p, "side": side, "size": s})
    for tok in by_token:
        by_token[tok].sort(key=lambda x: x["t"])
    return by_token


def simulate_book_mm(events: list[dict], our_size: float, inventory_cap: float,
                     maker_fee: float = 0.0, mark_delay_s: float = 60.0,
                     improve: bool = False, tick: float = 0.01,
                     signal: float = 0.0, skew_threshold: float = 0.0,
                     momentum_window: float = 0.0,
                     # --- additive predictive components (all default OFF == legacy behavior) ---
                     debounce_trades: int = 0,
                     inv_skew: float = 0.0,
                     vol_window: float = 0.0, vol_spread_coeff: float = 0.0,
                     tox_threshold: float = 0.0, tox_window: float = 0.0,
                     tox_cooldown: float = 0.0,
                     # --- in-band reward accrual (default OFF; needs depth_samples + meta) ---
                     depth_samples: list | None = None,
                     reward_pool: float = 0.0, reward_min_size: float = 0.0,
                     reward_v_cents: float = 0.0,
                     # --- fill realism (default 'fifo' == legacy) ---
                     fill_model: str = "fifo", capture_mult: float = 1.0,
                     # --- depth control: rest s* from mid instead of the touch (0 == legacy) ---
                     quote_offset: float = 0.0,
                     # --- price-band gate (resolution guard); default [0,1] == quote everywhere ---
                     min_mid: float = 0.0, max_mid: float = 1.0,
                     liq_outside_band: bool = False, stop_loss_cents: float = 0.0,
                     take_profit_cents: float = 0.0) -> dict[str, Any]:
    """Replay one token's normalized event stream as a passive two-sided MM.

    With `skew_threshold` > 0 the MM becomes SIGNAL-INFORMED: it quotes the bid only when the
    forecast isn't strongly down and the ask only when it isn't strongly up. So when the
    forecast predicts a rise it provides liquidity only on the buy side (accumulating long into
    the move) and stops offering — earning the spread while the forecast steers it away from
    the adverse side.

    The forecast is either a fixed `signal`, or — when `momentum_window` > 0 — computed
    CAUSALLY inside the sim as trailing reference-price momentum: signal = ref(now) - ref(now -
    window), using only past prices (the dominant CLV feature). skew_threshold=0 = plain
    symmetric MM.

    Optional predictive components (each default-off so the legacy path is byte-identical):
      * `debounce_trades` > 0 -> the momentum REFERENCE is the size-weighted VWAP of the last
        `debounce_trades` trades (a debounced mid that strips bid-ask bounce — the actual CLV
        feature in forward_return_predictability), instead of the raw quote mid.
      * `inv_skew` -> inventory-aware quoting: shift BOTH quotes by inv_skew*(inv/cap) so when
        long we lower bid+ask to mean-revert the position (point 4 of the MM checklist).
      * `vol_window`/`vol_spread_coeff` -> widen both quotes by vol_spread_coeff*recent_vol
        (stdev of mids over vol_window) when the market is volatile.
      * `tox_threshold`/`tox_window`/`tox_cooldown` -> reactive toxicity gate: if a fill is
        followed within tox_window by an adverse mid move > tox_threshold, step off that side
        for tox_cooldown seconds (informed flow ran us over).

    REWARD: with `depth_samples` (per-minute [t, mid, bb, ba, q_bid_book, q_ask_book] for this
    token) plus `reward_pool`/`reward_min_size`/`reward_v_cents`, the sim credits Polymarket
    liquidity reward at each sample from its LIVE quote placement and resting size `our_size`:
    `reward += (pool/1440) * Q_min(self)/Q_min(self+book)`. This is config- AND size-specific —
    stepping off a side, quoting below min_size, or skewing away from mid all reduce it.

    FILL MODEL: 'fifo' (default, legacy) fills behind the queue size resting when we joined, then
    captures ALL subsequent crossing flow at our price — optimistic at large size. 'prorata'
    instead shares each crossing trade with the live competing depth at our level:
    our_fill = trade_size * our_size/(our_size + competing_depth) * capture_mult (capped by our
    size, inventory room). This stops us from assuming we take 100% of flow once front-of-queue;
    `capture_mult` < 1 is an extra haircut for queue-jumping / latency / partial presence.
    """
    from quoter import Quoter  # noqa: PLC0415  (shared streaming state machine — single source of truth)
    q = Quoter(our_size, inventory_cap, maker_fee=maker_fee, mark_delay_s=mark_delay_s,
               improve=improve, tick=tick, signal=signal, skew_threshold=skew_threshold,
               momentum_window=momentum_window, debounce_trades=debounce_trades, inv_skew=inv_skew,
               vol_window=vol_window, vol_spread_coeff=vol_spread_coeff, tox_threshold=tox_threshold,
               tox_window=tox_window, tox_cooldown=tox_cooldown, reward_pool=reward_pool,
               reward_min_size=reward_min_size, reward_v_cents=reward_v_cents,
               fill_model=fill_model, capture_mult=capture_mult, quote_offset=quote_offset,
               min_mid=min_mid, max_mid=max_mid, liq_outside_band=liq_outside_band,
               stop_loss_cents=stop_loss_cents, take_profit_cents=take_profit_cents)
    rw_on = bool(depth_samples) and reward_pool > 0 and reward_v_cents > 0
    smp = 0

    def credit_due(up_to_t):
        nonlocal smp
        while rw_on and smp < len(depth_samples) and depth_samples[smp][0] <= up_to_t:
            _, s_mid, _, _, q1b, q2b = depth_samples[smp]; smp += 1
            q.credit_sample(s_mid, q1b, q2b)   # uses pre-event quote state (causal)

    for e in events:
        credit_due(e["t"])
        if e["type"] == "quote":
            q.on_quote(e["t"], e["bid"], e["ask"], e.get("bid_size", 0.0), e.get("ask_size", 0.0))
        else:
            q.on_trade(e["t"], e["price"], e["side"], e["size"])
    credit_due(float("inf"))
    return q.finalize()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", type=Path, required=True, help="JSONL from collect_clob_book.py")
    ap.add_argument("--output-dir", type=Path, default=Path("reports/book_mm"))
    ap.add_argument("--size", type=float, default=20.0, help="Quote size (contracts) per side.")
    ap.add_argument("--inventory-cap", type=float, default=100.0)
    ap.add_argument("--maker-fee", type=float, default=0.0)
    ap.add_argument("--mark-delay-s", type=float, default=60.0)
    ap.add_argument("--improve", action="store_true", help="Quote one tick inside the touch (jump the queue).")
    ap.add_argument("--skew-threshold", type=float, default=0.005,
                    help="Signal-informed MM: quote one-sided when |momentum| exceeds this (price units).")
    ap.add_argument("--momentum-window", type=float, default=300.0,
                    help="Seconds of trailing mids for the causal momentum forecast (0 = neutral only).")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="manifest_*.json from the collector; supplies per-token reward pools.")
    ap.add_argument("--capture-share", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05, 0.10],
                    help="Fraction of each market's real daily reward pool you capture (swept).")
    args = ap.parse_args()

    import gzip  # noqa: PLC0415
    opener = gzip.open if str(args.capture).endswith(".gz") else open
    with opener(args.capture, "rt") as fh:
        raw = [json.loads(line) for line in fh if line.strip()]
    by_token = normalize_events(raw)

    # per-token real daily reward pool ($/day) from the manifest snapshot
    token_pool: dict[str, float] = {}
    if args.manifest and args.manifest.exists():
        man = json.loads(args.manifest.read_text())
        for tok, meta in (man.get("token_meta") or {}).items():
            token_pool[str(tok)] = float(meta.get("reward_daily_est") or 0.0)

    def run(momentum_window: float, skew: float) -> dict:
        total = defaultdict(float); n_tok = 0; qw = 0.0; os_q = 0.0; sig = 0.0
        reward_pool_days = 0.0   # sum over eligible tokens of (daily pool $ x quoting days)
        for tok, events in by_token.items():
            if sum(1 for e in events if e["type"] == "trade") < 5:
                continue
            r = simulate_book_mm(events, args.size, args.inventory_cap, args.maker_fee,
                                 args.mark_delay_s, args.improve,
                                 skew_threshold=skew, momentum_window=momentum_window)
            n_tok += 1
            for k in ("pnl", "gross_spread_captured", "adverse_selection", "fees", "n_fills"):
                total[k] += r[k]
            qw += r["n_quotes"]; os_q += r["one_sided_quote_frac"] * r["n_quotes"]
            sig += r["mean_abs_signal"] * r["n_quotes"]
            reward_pool_days += token_pool.get(tok, 0.0) * r["quoting_days"]
        d = {"tokens": n_tok, "total_pnl": round(total["pnl"], 2),
             "gross_spread_captured": round(total["gross_spread_captured"], 2),
             "adverse_selection": round(total["adverse_selection"], 2),
             "fees": round(total["fees"], 2), "n_fills": int(total["n_fills"]),
             "reward_pool_dollar_days": round(reward_pool_days, 1)}
        # pnl after capturing a fraction of the REAL daily reward pool (swept)
        d["pnl_at_capture_share"] = {f"{s:.0%}": round(total["pnl"] + s * reward_pool_days, 2)
                                     for s in args.capture_share}
        # breakeven: fraction of the pool needed to flip MM positive
        d["breakeven_capture_share"] = (round(-total["pnl"] / reward_pool_days, 4)
                                        if reward_pool_days > 0 and total["pnl"] < 0 else 0.0)
        if momentum_window > 0:
            d["one_sided_quote_frac"] = round(os_q / qw, 3) if qw else 0.0
            d["mean_abs_momentum"] = round(sig / qw, 5) if qw else 0.0
        return d

    neutral = run(0.0, 0.0)
    informed = run(args.momentum_window, args.skew_threshold)
    out = {"neutral_mm": neutral, "momentum_informed_mm": informed,
           "pnl_improvement": round(informed["total_pnl"] - neutral["total_pnl"], 2),
           "adverse_reduction": round(neutral["adverse_selection"] - informed["adverse_selection"], 2)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "book_mm_summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print("\nread: pnl_at_capture_share adds your share of the REAL daily reward pools (from the"
          " manifest) x quoting time. breakeven_capture_share = fraction of the pool needed to flip")
    print("MM positive. Pools are ~$2k/day/market, so even a low single-digit % share likely dominates"
          " the spread/adverse-selection cents — that's the whole MM thesis.")


if __name__ == "__main__":
    main()
