#!/usr/bin/env python3
"""Capture the LIVE Polymarket CLOB L2 order book to build our own historical dataset.

The free CLOB API only serves the *current* book (no historical depth), so to backtest
market-making we record the live feed going forward. This subscribes to the CLOB WebSocket
`market` channel for a set of tokens and appends every event to a JSONL file with a local
receive timestamp. The recorded stream (book snapshots + price changes + trades) is exactly
what `book_mm_backtest.py` replays.

Run in the project venv (needs network + `websocket-client`):
    pip install websocket-client requests
    # discover ~40 liquid tokens and capture for an hour into reports/clob_capture/:
    python scripts/collect_clob_book.py --discover 40 --minutes 60
    # or capture a specific token list:
    python scripts/collect_clob_book.py --tokens 7211... 5119... --minutes 1440

Leave it running (a screen/tmux session, or --minutes large). The longer it runs, the more
of a backtestable book history you accumulate.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

# crypto / recurring-up-down markets to exclude (they move but are short-lived noise)
_CRYPTO_PAT = re.compile(
    r"bitcoin|btc|ethereum|\beth\b|solana|\bsol\b|dogecoin|doge|\bxrp\b|ripple|cardano|"
    r"litecoin|crypto|up.?or.?down|updown|\bavax\b|\bbnb\b|chainlink|polygon|-\d+m-|-\d+min-",
    re.I)


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _market_tokens(m: dict) -> list[str]:
    toks = m.get("clobTokenIds")
    try:
        toks = json.loads(toks) if isinstance(toks, str) else (toks or [])
    except json.JSONDecodeError:
        return []
    return [str(t) for t in toks if t]


def _is_crypto(m: dict) -> bool:
    text = f"{m.get('slug', '')} {m.get('question', '')} {m.get('seriesSlug', '')}"
    return bool(_CRYPTO_PAT.search(text))


def select_target_markets(markets: list[dict], n_markets: int = 40, min_liquidity: float = 5000.0,
                          exclude_crypto: bool = True) -> list[str]:
    """Pick a mix of LIQUID and VOLATILE markets (so the capture exercises both regimes) and
    return their token ids. Volatility is proxied by |oneDayPriceChange| then 24h turnover.
    Pure (no I/O) so it is unit-tested."""
    cand = []
    for m in markets:
        if exclude_crypto and _is_crypto(m):
            continue
        liq = _num(m.get("liquidityNum") or m.get("liquidity"))
        if liq < min_liquidity:
            continue
        toks = _market_tokens(m)
        if not toks:
            continue
        cand.append({"slug": m.get("slug", ""), "tokens": toks, "liq": liq,
                     "move": abs(_num(m.get("oneDayPriceChange"))),
                     "turn": _num(m.get("volume24hr")) / max(liq, 1.0)})
    by_liq = sorted(cand, key=lambda c: c["liq"], reverse=True)
    by_vol = sorted(cand, key=lambda c: (c["move"], c["turn"]), reverse=True)
    half = n_markets // 2
    selected, seen = [], set()
    for pool, k in ((by_liq, half), (by_vol, n_markets - half)):
        cnt = 0
        for c in pool:
            if c["slug"] in seen:
                continue
            seen.add(c["slug"]); selected.append(c); cnt += 1
            if cnt >= k:
                break
    tokens: list[str] = []
    for c in selected:
        tokens.extend(c["tokens"])
    return tokens


def _event_is_crypto(ev: dict) -> bool:
    text = f"{ev.get('slug', '')} {ev.get('title', '')} {ev.get('seriesSlug', '')}"
    text += " " + " ".join(m.get("question", "") for m in (ev.get("markets") or []))
    return bool(_CRYPTO_PAT.search(text))


_TOP_CATEGORIES = {"sports", "politics", "crypto", "economics", "pop-culture", "science",
                   "business", "tech", "world", "soccer", "nfl", "nba", "elections", "geopolitics"}


def _market_has_rewards(m: dict) -> bool:
    """A market is in the maker-rewards program if it has a positive max-spread reward band."""
    if _num(m.get("rewardsMaxSpread")) > 0:
        return True
    cr = m.get("clobRewards")
    return bool(cr) and cr != []


def _reward_daily_estimate(clob_rewards) -> float:
    """Best-effort daily reward pool from the clobRewards field (shape varies; sum likely keys)."""
    total = 0.0
    items = clob_rewards if isinstance(clob_rewards, list) else [clob_rewards]
    for it in items:
        if not isinstance(it, dict):
            continue
        for key in ("rewardsDailyRate", "rewardsAmount", "dailyRate", "amount", "rate"):
            if key in it:
                total += _num(it[key])
                break
    return total


def _market_reward_params(m: dict) -> dict:
    """Persist the actual reward + microstructure params for a market (snapshot at discovery)."""
    return {"rewards_max_spread": _num(m.get("rewardsMaxSpread")),
            "rewards_min_size": _num(m.get("rewardsMinSize")),
            "reward_daily_est": _reward_daily_estimate(m.get("clobRewards")),
            "clob_rewards": m.get("clobRewards"),
            "maker_fee": _num(m.get("makerBaseFee")), "taker_fee": _num(m.get("takerBaseFee")),
            "spread": _num(m.get("spread")), "best_bid": _num(m.get("bestBid")),
            "best_ask": _num(m.get("bestAsk")), "min_tick": _num(m.get("orderPriceMinTickSize")),
            "question": m.get("question", "")}


def _event_meta(ev: dict) -> dict:
    """Strategy-relevant taxonomy tags for one event (category, exclusivity, horizon, ...)."""
    from datetime import datetime, timezone  # noqa: PLC0415
    tags = [t.get("slug", "") for t in (ev.get("tags") or []) if isinstance(t, dict)]
    category = next((t for t in tags if t in _TOP_CATEGORIES), (tags[0] if tags else "other"))
    horizon = None
    if ev.get("endDate"):
        try:
            end = datetime.fromisoformat(str(ev["endDate"]).replace("Z", "+00:00"))
            horizon = round((end - datetime.now(timezone.utc)).total_seconds() / 86400, 1)
        except (ValueError, TypeError):
            pass
    return {"category": category, "neg_risk": bool(ev.get("negRisk") or ev.get("enableNegRisk")),
            "horizon_days": horizon, "competitive": round(_num(ev.get("competitive")), 3)}


def select_target_event_records(events: list[dict], n_events: int = 30, min_liquidity: float = 5000.0,
                                exclude_crypto: bool = True, max_markets_per_event: int = 15,
                                one_token_per_market: bool = True) -> list[dict]:
    """Select whole EVENTS (a third by liquidity, a third by recent movement, a third
    multi-outcome with >=3 live markets) so the capture serves every strategy. Per event, keep
    only the most-liquid `max_markets_per_event` markets (so 128-candidate giants don't eat the
    token budget; small events stay complete), and one token per market by default (the YES/NO
    pair is redundant). Returns selected event records. Pure / tested."""
    cand = []
    for ev in events:
        if exclude_crypto and _event_is_crypto(ev):
            continue
        live = [m for m in (ev.get("markets") or [])
                if not m.get("closed") and _market_tokens(m)]
        live.sort(key=lambda m: _num(m.get("liquidityNum") or m.get("liquidity")), reverse=True)
        live = live[:max_markets_per_event]
        toks: list[str] = []
        token_meta: dict[str, dict] = {}
        liq = move = 0.0
        n_reward = 0
        for m in live:
            mt = _market_tokens(m)
            chosen = mt[:1] if one_token_per_market else mt
            toks += chosen
            rp = _market_reward_params(m)
            for t in chosen:
                token_meta[t] = rp
            liq += _num(m.get("liquidityNum") or m.get("liquidity"))
            move = max(move, abs(_num(m.get("oneDayPriceChange"))))
            n_reward += _market_has_rewards(m)
        if not toks or liq < min_liquidity:
            continue
        cand.append({"slug": ev.get("slug", ""), "title": ev.get("title", ""),
                     "tokens": toks, "token_meta": token_meta, "liq": round(liq),
                     "move": round(move, 4), "n_live": len(live), "rewards": n_reward > 0,
                     **_event_meta(ev)})
    # strategy-aligned buckets: rewards-eligible (MM), liquid (MM/CLV + capacity),
    # volatile (forecast/adverse-selection), basket (negRisk multi-outcome)
    by_reward = sorted([c for c in cand if c["rewards"]], key=lambda c: c["liq"], reverse=True)
    by_liq = sorted(cand, key=lambda c: c["liq"], reverse=True)
    by_move = sorted(cand, key=lambda c: c["move"], reverse=True)
    by_basket = sorted([c for c in cand if c["neg_risk"] and c["n_live"] >= 3],
                       key=lambda c: c["liq"], reverse=True)
    q = max(1, n_events // 4)
    selected, seen = [], set()
    for pool, k, label in ((by_reward, q, "rewards"), (by_liq, q, "liquid"),
                           (by_move, q, "volatile"), (by_basket, n_events - 3 * q, "basket")):
        cnt = 0
        for c in pool:
            if c["slug"] in seen:
                continue
            seen.add(c["slug"]); selected.append({**c, "bucket": label}); cnt += 1
            if cnt >= k:
                break
    return selected


def select_target_events(events: list[dict], n_events: int = 30, min_liquidity: float = 5000.0,
                         exclude_crypto: bool = True, max_tokens: int = 200) -> list[str]:
    """Token-id union of the selected events (deduped, capped). Pure / unit-tested."""
    tokens: list[str] = []
    for c in select_target_event_records(events, n_events, min_liquidity, exclude_crypto):
        tokens.extend(c["tokens"])
    return list(dict.fromkeys(tokens))[:max_tokens]


def discover_target_records(n_events: int, min_liquidity: float = 5000.0, exclude_crypto: bool = True,
                            max_markets_per_event: int = 15, one_token_per_market: bool = True) -> list[dict]:
    """Fetch active EVENTS and return the selected event records (with slug/title)."""
    import requests  # noqa: PLC0415

    url = "https://gamma-api.polymarket.com/events"
    params = {"active": "true", "closed": "false", "archived": "false", "limit": 500,
              "order": "volume24hr", "ascending": "false"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return select_target_event_records(r.json(), n_events, min_liquidity, exclude_crypto,
                                       max_markets_per_event, one_token_per_market)


def run_collector(tokens: list[str], out_dir: Path, minutes: float, max_gb: float = 12.0) -> None:
    """Stream the market feed to gzipped, UTC-daily-rotated JSONL, stopping at `minutes` or
    when the capture directory reaches `max_gb`."""
    import gzip  # noqa: PLC0415
    import websocket  # noqa: PLC0415  (websocket-client)

    out_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + minutes * 60.0
    max_bytes = max_gb * 1e9
    state = {"count": 0, "day": None, "fh": None, "bytes_since_check": 0}

    def dir_bytes() -> float:
        return sum(f.stat().st_size for f in out_dir.glob("book_*.jsonl.gz"))

    def file_for_today():
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        if day != state["day"]:
            if state["fh"]:
                state["fh"].close()
            state["day"] = day
            state["fh"] = gzip.open(out_dir / f"book_{day}.jsonl.gz", "at")
            print(f"rotated to book_{day}.jsonl.gz")
        return state["fh"]

    def record(raw: str) -> None:
        recv = datetime.now(timezone.utc).isoformat()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return
        fh = file_for_today()
        for evt in (payload if isinstance(payload, list) else [payload]):
            evt["_recv_ts"] = recv
            line = json.dumps(evt) + "\n"
            fh.write(line)
            state["count"] += 1
            state["bytes_since_check"] += len(line)
        fh.flush()

    def on_open(ws):
        ws.send(json.dumps({"type": "market", "assets_ids": tokens, "custom_feature_enabled": True}))
        print(f"subscribed to {len(tokens)} tokens -> {out_dir} (cap {max_gb} GB)")

    def on_message(ws, message):
        record(message)
        # periodic disk-budget check (every ~5 MB of new data)
        if state["bytes_since_check"] > 5e6:
            state["bytes_since_check"] = 0
            if dir_bytes() > max_bytes:
                print(f"reached {max_gb} GB cap; stopping.")
                ws.close()
                return
        if time.time() > deadline:
            ws.close()

    def on_error(ws, error):
        print("ws error:", error)

    url = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    while time.time() < deadline and dir_bytes() <= max_bytes:
        try:
            ws = websocket.WebSocketApp(url, on_open=on_open, on_message=on_message, on_error=on_error)
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as exc:  # network drop -> reconnect
            print("reconnecting after:", exc)
            time.sleep(3)
    if state["fh"]:
        state["fh"].close()
    print(f"done. wrote {state['count']} events; dir size {dir_bytes()/1e9:.2f} GB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", nargs="*", default=[], help="Explicit token ids to capture.")
    ap.add_argument("--discover", type=int, default=0,
                    help="Auto-pick this many EVENTS (liquid + volatile + multi-outcome mix).")
    ap.add_argument("--max-tokens", type=int, default=250, help="Cap total tokens captured.")
    ap.add_argument("--max-markets-per-event", type=int, default=15,
                    help="Keep only the most-liquid N markets per event (caps giant events).")
    ap.add_argument("--both-tokens", action="store_true",
                    help="Capture both YES/NO tokens per market (default: token1 only — they mirror).")
    ap.add_argument("--min-liquidity", type=float, default=5000.0)
    ap.add_argument("--include-crypto", action="store_true",
                    help="Keep crypto / recurring up-down markets (excluded by default).")
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--max-gb", type=float, default=12.0, help="Stop when the capture dir reaches this.")
    ap.add_argument("--output-dir", type=Path, default=Path("reports/clob_capture"))
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the selected events (slug/title/bucket) and exit — verify before capturing.")
    args = ap.parse_args()

    records: list[dict] = []
    tokens = list(args.tokens)
    if args.discover:
        records = discover_target_records(args.discover, args.min_liquidity,
                                          exclude_crypto=not args.include_crypto,
                                          max_markets_per_event=args.max_markets_per_event,
                                          one_token_per_market=not args.both_tokens)
        for c in records:
            tokens.extend(c["tokens"])
    tokens = list(dict.fromkeys(tokens))[: args.max_tokens]

    if args.dry_run:
        print(f"{len(records)} events selected, {len(tokens)} tokens (capped {args.max_tokens}):\n")
        for c in sorted(records, key=lambda x: x["bucket"]):
            flags = ("R" if c.get("rewards") else "-") + ("N" if c.get("neg_risk") else "-")
            hz = c.get("horizon_days")
            print(f"  [{c['bucket']:8}] {flags} {c.get('category','?'):10} "
                  f"liq=${c['liq']:>9,} move={c['move']:.3f} n={c['n_live']} "
                  f"hz={hz if hz is not None else '?':>5}d  {c['slug'][:48]}")
        from collections import Counter
        print("\nby category:", dict(Counter(c.get("category", "?") for c in records)))
        print("reward-eligible events:", sum(c.get("rewards", False) for c in records),
              "| negRisk events:", sum(c.get("neg_risk", False) for c in records))
        return
    if not tokens:
        raise SystemExit("No tokens. Pass --tokens or --discover N.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    token_meta: dict[str, dict] = {}
    for c in records:
        token_meta.update(c.get("token_meta", {}))
    manifest = {"created": stamp, "n_tokens": len(tokens), "tokens": tokens,
                "token_meta": token_meta,   # per-token reward params + microstructure (gamma snapshot)
                "events": [{k: c.get(k) for k in ("bucket", "slug", "title", "category", "rewards",
                                                  "neg_risk", "horizon_days", "competitive", "liq",
                                                  "move", "n_live", "tokens")}
                           for c in records]}
    (args.output_dir / f"manifest_{stamp}.json").write_text(json.dumps(manifest, indent=2))
    print(f"capturing {len(tokens)} tokens from {len(records)} events "
          f"(readable manifest: manifest_{stamp}.json)")
    run_collector(tokens, args.output_dir, args.minutes, args.max_gb)


if __name__ == "__main__":
    main()
