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
import os
import re
import socket
import time
from datetime import datetime, timezone
from pathlib import Path


def tokens_to_add(current: set[str], discovered: list[str]) -> list[str]:
    """New tokens from a re-discovery pass that we are not already subscribed to (order-preserving)."""
    seen = set(current)
    out = []
    for t in discovered:
        t = str(t)
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _due(last_ts: float, now: float, interval_seconds: float) -> bool:
    return interval_seconds > 0 and (now - last_ts) >= interval_seconds

# recurring ultra-short series = a pure-noise firehose; ALWAYS excluded regardless of --include-crypto
_RECURRING_PAT = re.compile(r"up.?or.?down|updown|-\d+m-|-\d+min-|-\d+h-|-\d+hr-", re.I)
# substantive crypto keywords; excluded only when exclude_crypto is set (kept in raw otherwise)
_CRYPTO_PAT = re.compile(
    r"bitcoin|\bbtc\b|ethereum|\beth\b|solana|\bsol\b|dogecoin|doge|\bxrp\b|ripple|cardano|"
    r"litecoin|\bcrypto\b|\bavax\b|\bbnb\b|chainlink|polygon", re.I)


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


def _market_price(m: dict) -> float:
    """Representative price for a market outcome (ask = cost to buy the longshot leg)."""
    return _num(m.get("bestAsk")) or _num(m.get("lastTradePrice"))


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


def _event_excluded(ev: dict, exclude_crypto: bool) -> bool:
    """Recurring ultra-short series are always excluded; substantive crypto only if requested."""
    text = f"{ev.get('slug', '')} {ev.get('title', '')} {ev.get('seriesSlug', '')}"
    text += " " + " ".join(m.get("question", "") for m in (ev.get("markets") or []))
    if _RECURRING_PAT.search(text):
        return True
    return exclude_crypto and bool(_CRYPTO_PAT.search(text))


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
                                one_token_per_market: bool = True, longshot_max_price: float = 0.08) -> list[dict]:
    """Select whole EVENTS in strategy-aligned buckets (rewards/liquid/volatile/basket/longshot)
    so the capture serves every strategy. Per event we keep the most-liquid
    `max_markets_per_event` markets PLUS any cheap 'longshot' legs (price <= `longshot_max_price`),
    so 128-candidate giants don't eat the token budget but the penny tails are still captured for
    longshot strategies. One token per market by default (YES/NO mirror). Pure / tested."""
    cand = []
    for ev in events:
        if _event_excluded(ev, exclude_crypto):
            continue
        live = [m for m in (ev.get("markets") or [])
                if not m.get("closed") and _market_tokens(m)]
        live.sort(key=lambda m: _num(m.get("liquidityNum") or m.get("liquidity")), reverse=True)
        cheap = [m for m in live if 0.005 <= _market_price(m) <= longshot_max_price]
        chosen, ids = [], set()
        for m in live[:max_markets_per_event] + cheap:    # top-liquid + the penny tails
            mid = str(m.get("id") or id(m))
            if mid not in ids:
                ids.add(mid); chosen.append(m)
        toks: list[str] = []
        token_meta: dict[str, dict] = {}
        liq = move = 0.0
        n_reward = 0
        for m in chosen:
            mt = _market_tokens(m)
            sel = mt[:1] if one_token_per_market else mt
            toks += sel
            rp = _market_reward_params(m)
            for t in sel:
                token_meta[t] = rp
            liq += _num(m.get("liquidityNum") or m.get("liquidity"))
            move = max(move, abs(_num(m.get("oneDayPriceChange"))))
            n_reward += _market_has_rewards(m)
        if not toks or liq < min_liquidity:
            continue
        cand.append({"slug": ev.get("slug", ""), "title": ev.get("title", ""),
                     "tokens": toks, "token_meta": token_meta, "liq": round(liq),
                     "move": round(move, 4), "n_live": len(chosen), "n_longshot": len(cheap),
                     "rewards": n_reward > 0, **_event_meta(ev)})
    # strategy-aligned buckets
    by_reward = sorted([c for c in cand if c["rewards"]], key=lambda c: c["liq"], reverse=True)
    by_liq = sorted(cand, key=lambda c: c["liq"], reverse=True)
    by_move = sorted(cand, key=lambda c: c["move"], reverse=True)
    by_basket = sorted([c for c in cand if c["neg_risk"] and c["n_live"] >= 3],
                       key=lambda c: c["liq"], reverse=True)
    by_longshot = sorted([c for c in cand if c["n_longshot"] >= 2],
                         key=lambda c: c["n_longshot"], reverse=True)
    q = max(1, n_events // 5)
    selected, seen = [], set()
    for pool, k, label in ((by_reward, q, "rewards"), (by_liq, q, "liquid"),
                           (by_move, q, "volatile"), (by_basket, q, "basket"),
                           (by_longshot, n_events - 4 * q, "longshot")):
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


def all_tokens_from_events(events: list[dict], min_liquidity: float = 1000.0,
                           exclude_crypto: bool = True, one_token_per_market: bool = True,
                           longshot_max_price: float = 0.08) -> tuple[list[str], dict]:
    """EVERY active, non-excluded market's token(s) + per-token meta — for full-universe
    collection (no bucketing / no per-event cap). Keeps a market if it clears the (low)
    liquidity floor OR is a tradeable penny longshot. Pure / unit-tested."""
    tokens: list[str] = []
    meta: dict[str, dict] = {}
    for ev in events:
        if _event_excluded(ev, exclude_crypto):
            continue
        ev_meta = _event_meta(ev)
        for m in (ev.get("markets") or []):
            if m.get("closed") or not _market_tokens(m):
                continue
            liq = _num(m.get("liquidityNum") or m.get("liquidity"))
            price = _market_price(m)
            if liq < min_liquidity and not (0.005 <= price <= longshot_max_price):
                continue
            sel = _market_tokens(m)[:1] if one_token_per_market else _market_tokens(m)
            rp = {**_market_reward_params(m), **ev_meta}
            for t in sel:
                if t not in meta:
                    tokens.append(t)
                    meta[t] = rp
    return tokens, meta


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


def resolved_asset_ids(payload) -> list[str]:
    """Winning asset ids from any market_resolved events in a WS payload (for pruning)."""
    out = []
    for evt in (payload if isinstance(payload, list) else [payload]):
        if isinstance(evt, dict) and (evt.get("event_type") or evt.get("type")) == "market_resolved":
            wid = evt.get("winning_asset_id") or evt.get("asset_id")
            if wid:
                out.append(str(wid))
    return out


def run_collector(tokens: list[str], out_dir: Path, minutes: float, rotate_minutes: float = 15.0,
                  rediscover_minutes: float = 30.0, rediscover_fn=None, manifest_path: Path = None,
                  token_meta: dict | None = None, max_local_gb: float = 5.0,
                  max_subscriptions: int = 500, add_file: Path = None, status_file: Path = None) -> None:
    """Stream the market feed to gzipped JSONL, rotating a new SPOOL file every `rotate_minutes`
    (closed files are renamed .tmp -> .jsonl.gz so an uploader can ship+delete them), and every
    `rediscover_minutes` re-running `rediscover_fn` to dynamically SUBSCRIBE to newly-listed
    markets without dropping existing ones. Runs until `minutes` elapse (use a huge value under
    systemd). `max_local_gb` is a safety stop if the uploader is failing and spool grows."""
    import gzip  # noqa: PLC0415
    import threading  # noqa: PLC0415
    import websocket  # noqa: PLC0415  (websocket-client)

    out_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + minutes * 60.0
    host = socket.gethostname().split(".")[0]
    st = {"count": 0, "fh": None, "tmp": None, "last_rotate": 0.0,
          "last_rediscover": time.time(), "last_add": time.time(),
          "tokens": set(map(str, tokens)), "meta": dict(token_meta or {})}

    def spool_bytes() -> float:
        return sum(f.stat().st_size for f in out_dir.glob("book_*.jsonl.gz*"))

    def write_status():
        # report our live subscribed set so the fleet launcher knows real occupancy (zero-gap add)
        if status_file:
            tmp = status_file.with_suffix(status_file.suffix + ".tmp")
            tmp.write_text("\n".join(sorted(st["tokens"])))
            tmp.replace(status_file)

    def poll_add(ws):
        # the launcher appends newly-listed tokens here; subscribe them live (no restart, no gap)
        if not add_file or not _due(st["last_add"], time.time(), 20):
            return
        st["last_add"] = time.time()
        if not add_file.exists():
            return
        try:
            lines = [ln.strip() for ln in add_file.read_text().splitlines() if ln.strip()]
            add_file.write_text("")  # consume; a missed line self-heals next launcher cycle
        except OSError:
            return
        room = max(0, max_subscriptions - len(st["tokens"]))
        add = tokens_to_add(st["tokens"], lines)[:room]
        if add:
            ws.send(json.dumps({"assets_ids": add, "operation": "subscribe"}))
            st["tokens"].update(add)
            print(f"+{len(add)} tokens via add-file (now {len(st['tokens'])})")

    def open_new():
        epoch = int(time.time())
        # include pid so multiple shard processes never collide on a filename
        st["tmp"] = out_dir / f"book_{host}_{os.getpid()}_{epoch}.jsonl.gz.tmp"
        st["fh"] = gzip.open(st["tmp"], "at")
        st["last_rotate"] = time.time()

    def rotate():
        if st["fh"]:
            st["fh"].close()
            final = st["tmp"].with_suffix("")          # drop .tmp -> *.jsonl.gz (uploadable)
            st["tmp"].rename(final)
        open_new()
        write_status()

    def record(raw: str) -> list[str]:
        recv = datetime.now(timezone.utc).isoformat()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if st["fh"] is None:
            open_new()
        for evt in (payload if isinstance(payload, list) else [payload]):
            evt["_recv_ts"] = recv
            st["fh"].write(json.dumps(evt) + "\n")
            st["count"] += 1
        st["fh"].flush()
        return resolved_asset_ids(payload)

    def prune(ws, resolved: list[str]) -> None:
        gone = [t for t in resolved if t in st["tokens"]]
        if gone:
            ws.send(json.dumps({"assets_ids": gone, "operation": "unsubscribe"}))
            st["tokens"].difference_update(gone)
            write_status()
            print(f"pruned {len(gone)} resolved tokens (now {len(st['tokens'])})")

    def maybe_rediscover(ws):
        if rediscover_fn is None or not _due(st["last_rediscover"], time.time(), rediscover_minutes * 60):
            return
        st["last_rediscover"] = time.time()
        try:
            records = rediscover_fn()
        except Exception as exc:
            print("rediscover failed:", exc)
            return
        discovered = [t for c in records for t in c.get("tokens", [])]
        room = max(0, max_subscriptions - len(st["tokens"]))
        add = tokens_to_add(st["tokens"], discovered)[:room]   # respect the WS subscription cap
        if add:
            ws.send(json.dumps({"assets_ids": add, "operation": "subscribe"}))
            st["tokens"].update(add)
            for c in records:
                st["meta"].update(c.get("token_meta", {}))
            if manifest_path:
                manifest_path.write_text(json.dumps(
                    {"updated": datetime.now(timezone.utc).isoformat(), "n_tokens": len(st["tokens"]),
                     "tokens": sorted(st["tokens"]), "token_meta": st["meta"]}, indent=2))
            print(f"re-discovery: +{len(add)} new tokens (now {len(st['tokens'])})")

    def on_open(ws):
        ws.send(json.dumps({"type": "market", "assets_ids": sorted(st["tokens"]),
                            "custom_feature_enabled": True}))
        write_status()
        print(f"subscribed to {len(st['tokens'])} tokens -> {out_dir}")

        def keepalive():
            # Polymarket expects an APP-LEVEL "PING" string; its server does not reliably answer
            # WebSocket protocol pings, so relying on those falsely trips "ping/pong timed out"
            # (especially across many connections). Server replies "PONG" (non-JSON -> dropped).
            while getattr(ws, "keep_running", False):
                time.sleep(10)
                try:
                    ws.send("PING")
                except Exception:
                    break
        threading.Thread(target=keepalive, daemon=True).start()

    def on_message(ws, message):
        resolved = record(message)
        if resolved:
            prune(ws, resolved)
        if _due(st["last_rotate"], time.time(), rotate_minutes * 60):
            rotate()
        maybe_rediscover(ws)
        poll_add(ws)
        if time.time() > deadline or spool_bytes() > max_local_gb * 1e9:
            ws.close()

    url = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    while time.time() < deadline and spool_bytes() <= max_local_gb * 1e9:
        try:
            ws = websocket.WebSocketApp(url, on_open=on_open, on_message=on_message,
                                        on_error=lambda w, e: print("ws error:", e))
            ws.run_forever()   # app-level PING keepalive (above) instead of protocol ping/pong
        except Exception as exc:
            print("reconnecting after:", exc)
            time.sleep(3)
    rotate()  # close + finalize the open spool file
    print(f"done. wrote {st['count']} events; {len(st['tokens'])} tokens")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", nargs="*", default=[], help="Explicit token ids to capture.")
    ap.add_argument("--tokens-file", type=Path, default=None,
                    help="Newline-delimited token ids (used by the collect-all fleet for a shard).")
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
    ap.add_argument("--minutes", type=float, default=60.0,
                    help="Run duration; use a huge value under systemd to run indefinitely.")
    ap.add_argument("--rotate-minutes", type=float, default=15.0,
                    help="Close a spool file this often so the uploader can ship + delete it.")
    ap.add_argument("--rediscover-minutes", type=float, default=30.0,
                    help="Re-scan gamma this often and dynamically subscribe to newly-listed markets.")
    ap.add_argument("--max-subscriptions", type=int, default=500,
                    help="Hard cap on concurrent token subscriptions (WS limit); resolved markets are pruned.")
    ap.add_argument("--max-local-gb", type=float, default=5.0,
                    help="Safety stop if the spool grows past this (uploader failing).")
    ap.add_argument("--output-dir", type=Path, default=Path("reports/clob_capture"))
    ap.add_argument("--add-file", type=Path, default=None,
                    help="Inbox file the collect-all fleet appends new tokens to; subscribed live (zero-gap).")
    ap.add_argument("--status-file", type=Path, default=None,
                    help="This shard writes its live subscribed token set here for the fleet launcher.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the selected events (slug/title/bucket) and exit — verify before capturing.")
    args = ap.parse_args()

    records: list[dict] = []
    tokens = list(args.tokens)
    if args.tokens_file and args.tokens_file.exists():
        tokens += [ln.strip() for ln in args.tokens_file.read_text().splitlines() if ln.strip()]
    if args.discover:
        records = discover_target_records(args.discover, args.min_liquidity,
                                          exclude_crypto=not args.include_crypto,
                                          max_markets_per_event=args.max_markets_per_event,
                                          one_token_per_market=not args.both_tokens)
        for c in records:
            tokens.extend(c["tokens"])
    tokens = list(dict.fromkeys(tokens))
    if args.discover and not (args.tokens or args.tokens_file):
        tokens = tokens[: args.max_tokens]   # cap discovery only; explicit token lists pass through

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
        print("by bucket:", dict(Counter(c.get("bucket", "?") for c in records)))
        print("reward-eligible events:", sum(c.get("rewards", False) for c in records),
              "| negRisk events:", sum(c.get("neg_risk", False) for c in records),
              "| penny longshot legs:", sum(c.get("n_longshot", 0) for c in records))
        return
    if not tokens:
        raise SystemExit("No tokens. Pass --tokens / --tokens-file / --discover N.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = None
    token_meta: dict[str, dict] = {}
    if records:   # discovery mode owns a manifest; shard children (tokens-file) do not
        for c in records:
            token_meta.update(c.get("token_meta", {}))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        manifest = {"created": stamp, "n_tokens": len(tokens), "tokens": tokens,
                    "token_meta": token_meta,
                    "events": [{k: c.get(k) for k in ("bucket", "slug", "title", "category", "rewards",
                                                      "neg_risk", "horizon_days", "competitive", "liq",
                                                      "move", "n_live", "n_longshot", "tokens")}
                               for c in records]}
        manifest_path = args.output_dir / f"manifest_{stamp}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"capturing {len(tokens)} tokens from {len(records)} events "
              f"(readable manifest: {manifest_path.name})")
    else:
        print(f"capturing {len(tokens)} explicit tokens -> {args.output_dir}")

    def rediscover():
        return discover_target_records(args.discover, args.min_liquidity,
                                       exclude_crypto=not args.include_crypto,
                                       max_markets_per_event=args.max_markets_per_event,
                                       one_token_per_market=not args.both_tokens)

    run_collector(tokens, args.output_dir, args.minutes, args.rotate_minutes,
                  args.rediscover_minutes, rediscover_fn=(rediscover if args.discover else None),
                  manifest_path=manifest_path, token_meta=token_meta, max_local_gb=args.max_local_gb,
                  max_subscriptions=args.max_subscriptions,
                  add_file=args.add_file, status_file=args.status_file)


if __name__ == "__main__":
    main()
