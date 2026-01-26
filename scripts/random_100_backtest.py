#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class FirstHit:
    timestamp: str
    price: float
    side: str  # token1/token2


def load_markets(path: Path) -> Dict[str, Dict[str, str]]:
    markets: Dict[str, Dict[str, str]] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            market_id = row.get("id")
            if not market_id:
                continue
            markets[market_id] = row
    return markets


def find_first_hits(path: Path, threshold: float) -> Tuple[Dict[str, FirstHit], List[str]]:
    first_hits: Dict[str, FirstHit] = {}
    market_ids: List[str] = []
    seen = set()
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            market_id = row.get("market_id")
            if not market_id:
                continue
            if market_id not in seen:
                seen.add(market_id)
                market_ids.append(market_id)
            if market_id in first_hits:
                continue
            price_str = row.get("price")
            if price_str is None:
                continue
            try:
                price = float(price_str)
            except ValueError:
                continue
            if price < threshold:
                continue
            ts = row.get("timestamp") or ""
            side = row.get("nonusdc_side") or ""
            first_hits[market_id] = FirstHit(timestamp=ts, price=price, side=side)
    return first_hits, market_ids


def make_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_market_resolution(
    session: requests.Session, gamma_base: str, market_id: str
) -> Tuple[Optional[str], Dict[str, str]]:
    url = f"{gamma_base.rstrip('/')}/markets/{market_id}"
    resp = session.get(url, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        return None, {}
    # Try direct resolved outcome
    for key in ("resolvedOutcome", "resolved_outcome", "resolution", "result", "winner"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip(), data
    # Try outcomePrices at resolution (1.0 indicates winner)
    outcomes = data.get("outcomes")
    prices = data.get("outcomePrices")
    if isinstance(outcomes, list) and isinstance(prices, list) and len(outcomes) == len(prices):
        for idx, price in enumerate(prices):
            try:
                if float(price) >= 0.99:
                    return str(outcomes[idx]), data
            except (TypeError, ValueError):
                continue
    return None, data


def normalize(text: str) -> str:
    return text.strip().lower()


def map_outcome_to_token(
    resolved_outcome: str, market_row: Dict[str, str]
) -> Optional[str]:
    answer1 = market_row.get("answer1") or ""
    answer2 = market_row.get("answer2") or ""
    if not answer1 or not answer2:
        return None
    if normalize(resolved_outcome) == normalize(answer1):
        return "token1"
    if normalize(resolved_outcome) == normalize(answer2):
        return "token2"
    return None


def compute_pnl(entry_price: float, won: bool, fee_rate: float) -> float:
    shares = 1.0 / entry_price
    if won:
        gross = shares * 1.0
        fee_paid = gross * fee_rate
        return gross - 1.0 - fee_paid
    return -1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Random 100 market backtest.")
    parser.add_argument("--trades-csv", default="archive/processed/trades.csv")
    parser.add_argument("--markets-csv", default="archive/markets.csv")
    parser.add_argument("--output", default="reports/random_100_pnl.csv")
    parser.add_argument("--summary", default="reports/random_100_summary.json")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--fee-rate", type=float, default=0.0)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma-base", default="https://gamma-api.polymarket.com")
    args = parser.parse_args()

    trades_path = Path(args.trades_csv)
    markets_path = Path(args.markets_csv)
    output_path = Path(args.output)
    summary_path = Path(args.summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    markets = load_markets(markets_path)
    first_hits, market_ids = find_first_hits(trades_path, args.threshold)

    random.seed(args.seed)
    if len(market_ids) < args.sample_size:
        sample_ids = market_ids
    else:
        sample_ids = random.sample(market_ids, args.sample_size)

    session = make_session()
    results: List[Dict[str, object]] = []
    pnl_values: List[float] = []
    counters = {
        "sample_size": len(sample_ids),
        "with_first_hit": 0,
        "with_resolution": 0,
        "with_pnl": 0,
        "missing_market_row": 0,
        "missing_resolution": 0,
        "missing_outcome_mapping": 0,
    }

    for market_id in sample_ids:
        market_row = markets.get(market_id)
        if not market_row:
            counters["missing_market_row"] += 1
        resolved_outcome = None
        resolved_source = None
        try:
            resolved_outcome, _ = fetch_market_resolution(session, args.gamma_base, market_id)
            resolved_source = "gamma"
        except Exception:
            resolved_outcome = None
            resolved_source = "gamma_error"

        winner_token = None
        if resolved_outcome and market_row:
            winner_token = map_outcome_to_token(resolved_outcome, market_row)

        if resolved_outcome:
            counters["with_resolution"] += 1
        else:
            counters["missing_resolution"] += 1

        if resolved_outcome and not winner_token:
            counters["missing_outcome_mapping"] += 1

        hit = first_hits.get(market_id)
        if hit:
            counters["with_first_hit"] += 1

        pnl = None
        won = None
        if hit and winner_token:
            won = hit.side == winner_token
            pnl = compute_pnl(hit.price, won, args.fee_rate)
            pnl_values.append(pnl)
            counters["with_pnl"] += 1

        results.append(
            {
                "market_id": market_id,
                "question": market_row.get("question") if market_row else None,
                "answer1": market_row.get("answer1") if market_row else None,
                "answer2": market_row.get("answer2") if market_row else None,
                "entry_ts": hit.timestamp if hit else None,
                "entry_price": hit.price if hit else None,
                "entry_side": hit.side if hit else None,
                "resolved_outcome": resolved_outcome,
                "winner_token": winner_token,
                "won": won,
                "pnl": pnl,
                "resolution_source": resolved_source,
            }
        )

        time.sleep(0.05)

    # Write results
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    # Summary
    mean = sum(pnl_values) / len(pnl_values) if pnl_values else None
    median = None
    if pnl_values:
        sorted_pnl = sorted(pnl_values)
        mid = len(sorted_pnl) // 2
        if len(sorted_pnl) % 2 == 0:
            median = (sorted_pnl[mid - 1] + sorted_pnl[mid]) / 2
        else:
            median = sorted_pnl[mid]

    summary = {
        "counters": counters,
        "pnl_count": len(pnl_values),
        "pnl_mean": mean,
        "pnl_median": median,
    }
    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
