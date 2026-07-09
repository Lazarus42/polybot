#!/usr/bin/env python3
"""OBSERVATION-ONLY arbitrage monitor for Polymarket. Never places orders.

Single-shot (designed for cron every 5 min): enumerates active events from the Gamma API,
polls top-of-book from the CLOB REST API, and measures two structural arb families:

  1. negRisk baskets (mutually-exclusive multi-outcome events, >=3 open markets):
       - buy-the-basket:  sum of best ASKs across every outcome's YES token < $1  -> negrisk_buy
       - sell-the-basket: sum of best BIDs across every outcome's YES token > $1  -> negrisk_sell
     A basket is only valid if EVERY leg has a live ask (buy) / bid (sell).
  2. YES/NO parity on ordinary binary markets:
       - YES ask + NO ask < $1 -> parity_buy;  YES bid + NO bid > $1 -> parity_sell

Detected opportunities (deviation >= threshold, default 0.5c) append one JSON line each to
reports/arb_monitor/arb_YYYYMMDD.jsonl (top 50 per run, ranked by deviation * depth). Every run
ALSO appends one summary line to reports/arb_monitor/scan_YYYYMMDD.jsonl so zero-opportunity
periods are measurable (absence of evidence logged as evidence of absence).

Numbers are raw top-of-book (Polymarket is fee-free); depth bound = min over legs of
(touch size * price) in USD — slippage modeling can be layered later. Token ids are kept as
STRINGS everywhere (they are ~77-digit integers). Pure math/parsing is separated from I/O for
unit testing (tests/test_arb_monitor.py). stdlib + requests only. All network failures non-fatal.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
CLOB_BOOKS_URL = "https://clob.polymarket.com/books"
USER_AGENT = "polymarket-exp-arb-monitor/1.0 (research; observation-only)"
DEFAULT_THRESHOLD = 0.005          # 0.5 cents deviation from $1
MAX_OPPS_PER_RUN = 50              # cap on logged opportunities, ranked by deviation * depth
_EPS = 1e-12                       # float slack so an exactly-at-threshold deviation qualifies


# --------------------------------------------------------------------------------------------
# Pure core (no I/O) — everything below down to the I/O marker is unit-tested.
# --------------------------------------------------------------------------------------------

def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def top_of_book(book: dict) -> dict:
    """Best bid/ask of one CLOB book as {"bid": (price, size) | None, "ask": ... | None}.
    Does NOT assume the API's level ordering: best bid = max price, best ask = min price.
    Levels with non-positive price or size are ignored."""
    def _levels(side):
        out = []
        for lv in (book.get(side) or []):
            p, s = _num(lv.get("price")), _num(lv.get("size"))
            if p > 0 and s > 0:
                out.append((p, s))
        return out

    bids, asks = _levels("bids"), _levels("asks")
    return {"bid": max(bids, key=lambda x: x[0]) if bids else None,
            "ask": min(asks, key=lambda x: x[0]) if asks else None}


def basket_opportunities(legs: list[dict], threshold: float = DEFAULT_THRESHOLD) -> list[dict]:
    """Structural-arb check on a basket of legs, each {"token": str, "bid": (p,s)|None,
    "ask": (p,s)|None}. Returns 0-2 raw opportunities (side buy and/or sell).

    Buy the basket (pay sum of asks, worth $1 at resolution): valid ONLY if every leg has a
    live ask; arb if 1 - sum(asks) >= threshold. Sell the basket: every leg needs a live bid;
    arb if sum(bids) - 1 >= threshold. Depth bound = min over legs of touch size * price (USD).
    YES/NO parity is the 2-leg special case of the same identity."""
    out: list[dict] = []
    if not legs:
        return out
    for side, quote, dev_of in (("buy", "ask", lambda s: 1.0 - s),
                                ("sell", "bid", lambda s: s - 1.0)):
        if not all(l.get(quote) for l in legs):
            continue                                    # a missing quote invalidates the basket
        total = sum(l[quote][0] for l in legs)
        dev = dev_of(total)
        if dev < threshold - _EPS:
            continue
        out.append({
            "side": side,
            "n_legs": len(legs),
            "sum": round(total, 6),
            "deviation": round(dev, 6),
            "min_touch_depth_usd": round(min(l[quote][0] * l[quote][1] for l in legs), 2),
            "legs": [{"token": l["token"], "price": l[quote][0], "size": l[quote][1]}
                     for l in legs],
        })
    return out


def rank_opportunities(opps: list[dict], max_n: int = MAX_OPPS_PER_RUN) -> list[dict]:
    """Keep the top `max_n` opportunities ranked by deviation * depth (the $ that could
    actually be extracted at the touch), descending."""
    return sorted(opps, key=lambda o: o["deviation"] * o["min_touch_depth_usd"],
                  reverse=True)[:max_n]


def _market_tokens(m: dict) -> list[str]:
    """clobTokenIds arrives as a JSON string of huge integers — keep every id as a STRING."""
    toks = m.get("clobTokenIds")
    try:
        toks = json.loads(toks) if isinstance(toks, str) else (toks or [])
    except json.JSONDecodeError:
        return []
    return [str(t) for t in toks if t]


def _yes_index(m: dict) -> int:
    """Index of the YES outcome in the market's token list (Gamma `outcomes` aligns with
    clobTokenIds). Defaults to 0, Polymarket's standard ordering."""
    outcomes = m.get("outcomes")
    try:
        outcomes = json.loads(outcomes) if isinstance(outcomes, str) else (outcomes or [])
    except json.JSONDecodeError:
        return 0
    for i, o in enumerate(outcomes):
        if str(o).strip().lower() == "yes":
            return i
    return 0


def _market_open(m: dict) -> bool:
    """Tradeable right now: not closed, active, order book on, accepting orders (absent
    fields are treated as permissive — the book poll is the ground truth anyway)."""
    if m.get("closed"):
        return False
    if m.get("active") is False:
        return False
    if m.get("enableOrderBook") is False:
        return False
    return m.get("acceptingOrders") is not False


def _is_negrisk(ev: dict, markets: list[dict]) -> bool:
    return bool(ev.get("negRisk") or ev.get("enableNegRisk")
                or any(m.get("negRisk") for m in markets))


def extract_targets(events: list[dict]) -> tuple[list[dict], list[dict]]:
    """From Gamma events, build the two check lists (pure / testable):
      negrisk: [{"slug": event_slug, "yes_tokens": [token, ...]}]  — one YES token per open
               outcome market, only for negRisk events with >= 3 open markets, and only if
               EVERY open market yielded a usable YES token (a partial basket is not a basket).
      binary:  [{"slug": market_slug, "yes": token, "no": token}] — every open 2-token market
               (negRisk members included: YES/NO parity holds for them too).
    Input event order (gamma is volume-ranked) is preserved so deadline truncation drops the
    low-volume tail first."""
    negrisk: list[dict] = []
    binary: list[dict] = []
    for ev in events:
        markets = ev.get("markets") or []
        open_markets = [m for m in markets if _market_open(m)]
        yes_tokens: list[str] = []
        basket_ok = True
        for m in open_markets:
            toks = _market_tokens(m)
            if len(toks) != 2:
                basket_ok = False
                continue
            yi = _yes_index(m)
            binary.append({"slug": m.get("slug", ""), "yes": toks[yi], "no": toks[1 - yi]})
            yes_tokens.append(toks[yi])
        if (_is_negrisk(ev, markets) and basket_ok and len(yes_tokens) >= 3):
            negrisk.append({"slug": ev.get("slug", ""), "yes_tokens": yes_tokens})
    return negrisk, binary


def ordered_token_set(negrisk: list[dict], binary: list[dict]) -> list[str]:
    """Unique tokens to poll, preserving discovery (volume) order so a runtime deadline cuts
    the least-important tail. Binary covers most negRisk legs already; the negrisk pass just
    catches any leg not emitted as a binary market."""
    seen: set[str] = set()
    ordered: list[str] = []
    for b in binary:
        for t in (b["yes"], b["no"]):
            if t not in seen:
                seen.add(t)
                ordered.append(t)
    for e in negrisk:
        for t in e["yes_tokens"]:
            if t not in seen:
                seen.add(t)
                ordered.append(t)
    return ordered


def evaluate(negrisk: list[dict], binary: list[dict], books: dict[str, dict], ts: str,
             threshold: float = DEFAULT_THRESHOLD) -> tuple[list[dict], int, int]:
    """Run both checks against fetched books. A candidate is only 'checked' (and countable in
    the scan summary) if every one of its tokens has a fetched book. Returns
    (opportunities, n_negrisk_events_checked, n_binary_markets_checked)."""
    tob = {t: top_of_book(b) for t, b in books.items()}
    opps: list[dict] = []
    n_events = 0
    for e in negrisk:
        if not all(t in tob for t in e["yes_tokens"]):
            continue
        n_events += 1
        legs = [{"token": t, **tob[t]} for t in e["yes_tokens"]]
        for o in basket_opportunities(legs, threshold):
            opps.append({"ts": ts, "kind": f"negrisk_{o.pop('side')}", "slug": e["slug"], **o})
    n_binary = 0
    for b in binary:
        if b["yes"] not in tob or b["no"] not in tob:
            continue
        n_binary += 1
        legs = [{"token": b["yes"], **tob[b["yes"]]}, {"token": b["no"], **tob[b["no"]]}]
        for o in basket_opportunities(legs, threshold):
            opps.append({"ts": ts, "kind": f"parity_{o.pop('side')}", "slug": b["slug"], **o})
    return opps, n_events, n_binary


def make_scan_record(ts: str, n_events_checked: int, n_binary_checked: int,
                     opportunities: list[dict], runtime_s: float) -> dict:
    """One always-written summary row per run — zero-opportunity scans are data too."""
    return {"ts": ts,
            "n_events_checked": n_events_checked,
            "n_binary_checked": n_binary_checked,
            "n_opportunities": len(opportunities),
            "max_deviation": round(max((o["deviation"] for o in opportunities), default=0.0), 6),
            "runtime_s": round(runtime_s, 2)}


# --------------------------------------------------------------------------------------------
# I/O — network (read-only endpoints; this tool NEVER places orders) and jsonl appends.
# --------------------------------------------------------------------------------------------

def _get_with_retries(session, url: str, *, params=None, json_body=None, deadline: float,
                      attempts: int = 3, timeout: float = 20.0):
    """GET (params) or POST (json_body) with exponential backoff. Returns parsed JSON or None;
    never raises. Respects the run deadline."""
    for i in range(attempts):
        if time.monotonic() > deadline:
            return None
        try:
            if json_body is not None:
                r = session.post(url, json=json_body, timeout=timeout)
            else:
                r = session.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001 — every network failure must be non-fatal
            wait = 2.0 ** i
            print(f"  request failed ({exc.__class__.__name__}: {exc}); "
                  f"retry in {wait:.0f}s" if i + 1 < attempts else
                  f"  request failed ({exc.__class__.__name__}); giving up on this call")
            if i + 1 < attempts and time.monotonic() + wait <= deadline:
                time.sleep(wait)
    return None


def fetch_all_events(session, deadline: float, page_limit: int = 100,
                     max_events: int | None = None) -> list[dict]:
    """Paginate gamma /events for active, open events (same pattern as collect_all.py:
    100/page by offset until a short/empty page, volume-ranked so truncation is benign)."""
    out: list[dict] = []
    offset = 0
    while time.monotonic() < deadline:
        evs = _get_with_retries(session, GAMMA_EVENTS_URL, deadline=deadline,
                                params={"active": "true", "closed": "false",
                                        "archived": "false", "limit": page_limit,
                                        "offset": offset, "order": "volume24hr",
                                        "ascending": "false"})
        if not evs:
            break
        out.extend(evs)
        if max_events is not None and len(out) >= max_events:
            return out[:max_events]
        if len(evs) < page_limit:
            break
        offset += page_limit
        time.sleep(0.2)
    return out


def fetch_books(session, tokens: list[str], deadline: float, batch_size: int = 50,
                sleep_s: float = 0.5) -> dict[str, dict]:
    """POST batches of {"token_id": ...} to /books. Gentle: small batches, sleep between them,
    stop at the deadline. Failed batches are skipped (their candidates just aren't checked)."""
    books: dict[str, dict] = {}
    for i in range(0, len(tokens), batch_size):
        if time.monotonic() > deadline:
            print(f"  deadline hit after {i}/{len(tokens)} tokens; truncating book poll")
            break
        batch = tokens[i:i + batch_size]
        data = _get_with_retries(session, CLOB_BOOKS_URL, deadline=deadline,
                                 json_body=[{"token_id": t} for t in batch])
        if isinstance(data, list):
            for b in data:
                if isinstance(b, dict) and b.get("asset_id"):
                    books[str(b["asset_id"])] = b
        if i + batch_size < len(tokens):
            time.sleep(sleep_s)
    return books


def append_jsonl(path: Path, records: list[dict]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for rec in records:
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=Path("reports/arb_monitor"))
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="Minimum deviation from $1 to log (default 0.005 = 0.5c).")
    ap.add_argument("--max-events", type=int, default=None,
                    help="Cap gamma events fetched (smoke testing).")
    ap.add_argument("--max-runtime", type=float, default=180.0,
                    help="Hard wall-clock cap in seconds (default 180 for a 5-min cron slot).")
    ap.add_argument("--batch-size", type=int, default=50, help="Tokens per /books request.")
    ap.add_argument("--sleep", type=float, default=0.5, help="Seconds between /books batches.")
    args = ap.parse_args()

    import requests  # noqa: PLC0415 — matches project style; keeps import cost out of tests

    t0 = time.monotonic()
    deadline = t0 + args.max_runtime
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    day = now.strftime("%Y%m%d")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})

    events = fetch_all_events(session, deadline, max_events=args.max_events)
    negrisk, binary = extract_targets(events)
    tokens = ordered_token_set(negrisk, binary)
    print(f"[{ts}] {len(events)} events -> {len(negrisk)} negRisk baskets, "
          f"{len(binary)} binary markets, {len(tokens)} tokens to poll")

    books = fetch_books(session, tokens, deadline,
                        batch_size=args.batch_size, sleep_s=args.sleep)
    opps, n_events_checked, n_binary_checked = evaluate(
        negrisk, binary, books, ts, threshold=args.threshold)

    runtime_s = time.monotonic() - t0
    scan = make_scan_record(ts, n_events_checked, n_binary_checked, opps, runtime_s)
    append_jsonl(args.output_dir / f"arb_{day}.jsonl", rank_opportunities(opps))
    append_jsonl(args.output_dir / f"scan_{day}.jsonl", [scan])

    print(f"[{ts}] checked {n_events_checked} baskets + {n_binary_checked} binaries "
          f"({len(books)}/{len(tokens)} books) -> {len(opps)} opportunities "
          f"(max_dev={scan['max_deviation']}) in {runtime_s:.1f}s")


if __name__ == "__main__":
    main()
