#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass
class MarketMeta:
    answer1: str
    answer2: str
    question: str


@dataclass
class FirstHit:
    timestamp: str
    price: float
    side: str  # token1/token2


def normalize(text: str) -> str:
    return text.strip().lower()


def map_outcome_to_token(outcome: str, answer1: str, answer2: str) -> Optional[str]:
    if normalize(outcome) == normalize(answer1):
        return "token1"
    if normalize(outcome) == normalize(answer2):
        return "token2"
    return None


def load_markets(markets_path: Path) -> Dict[str, MarketMeta]:
    markets: Dict[str, MarketMeta] = {}
    with markets_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            market_id = row.get("id")
            if not market_id:
                continue
            markets[market_id] = MarketMeta(
                answer1=row.get("answer1") or "",
                answer2=row.get("answer2") or "",
                question=row.get("question") or "",
            )
    return markets


def load_resolutions(path: Path) -> Dict[str, str]:
    resolutions: Dict[str, str] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            market_id = row.get("market_id")
            resolution = row.get("resolution")
            if not market_id or not resolution:
                continue
            if resolution not in ("Yes", "No"):
                continue
            resolutions[market_id] = resolution
    return resolutions


def find_first_hits(
    trades_path: Path, target_ids: Set[str], threshold: float
) -> Dict[str, FirstHit]:
    hits: Dict[str, FirstHit] = {}
    remaining = set(target_ids)
    with trades_path.open("r", newline="") as f:
        f.readline()  # header
        for line in f:
            if not remaining:
                break
            parts = line.rstrip("\n").split(",", 8)
            if len(parts) < 8:
                continue
            market_id = parts[1]
            if market_id not in remaining:
                continue
            try:
                price = float(parts[7])
            except ValueError:
                continue
            if price < threshold:
                continue
            hits[market_id] = FirstHit(timestamp=parts[0], price=price, side=parts[4])
            remaining.remove(market_id)
    return hits


def compute_pnl(entry_price: float, won: bool, fee_rate: float) -> float:
    shares = 1.0 / entry_price
    if won:
        gross = shares * 1.0
        fee_paid = gross * fee_rate
        return gross - 1.0 - fee_paid
    return -1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="90% first-hit backtest using resolutions file.")
    parser.add_argument("--resolutions-csv", default="polymarket_resolutions.csv")
    parser.add_argument("--markets-csv", default="archive/markets.csv")
    parser.add_argument("--trades-csv", default="archive/processed/trades.csv")
    parser.add_argument("--output", default="reports/resolutions_pnl.csv")
    parser.add_argument("--summary", default="reports/resolutions_summary.json")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--fee-rate", type=float, default=0.0)
    args = parser.parse_args()

    resolutions = load_resolutions(Path(args.resolutions_csv))
    markets = load_markets(Path(args.markets_csv))
    target_ids = set(resolutions.keys())

    hits = find_first_hits(Path(args.trades_csv), target_ids, args.threshold)

    results: List[Dict[str, object]] = []
    pnl_values: List[float] = []
    wins = 0
    losses = 0

    for market_id, resolution in resolutions.items():
        meta = markets.get(market_id)
        if not meta:
            continue
        winner_token = map_outcome_to_token(resolution, meta.answer1, meta.answer2)
        hit = hits.get(market_id)
        won = None
        pnl = None
        if hit and winner_token:
            won = hit.side == winner_token
            pnl = compute_pnl(hit.price, won, args.fee_rate)
            pnl_values.append(pnl)
            if won:
                wins += 1
            else:
                losses += 1

        results.append(
            {
                "market_id": market_id,
                "question": meta.question,
                "answer1": meta.answer1,
                "answer2": meta.answer2,
                "resolved_outcome": resolution,
                "winner_token": winner_token,
                "entry_ts": hit.timestamp if hit else None,
                "entry_price": hit.price if hit else None,
                "entry_side": hit.side if hit else None,
                "won": won,
                "pnl": pnl,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    pnl_mean = sum(pnl_values) / len(pnl_values) if pnl_values else None
    pnl_median = None
    if pnl_values:
        sorted_pnl = sorted(pnl_values)
        mid = len(sorted_pnl) // 2
        if len(sorted_pnl) % 2 == 0:
            pnl_median = (sorted_pnl[mid - 1] + sorted_pnl[mid]) / 2
        else:
            pnl_median = sorted_pnl[mid]

    summary = {
        "markets_with_yes_no_resolution": len(resolutions),
        "with_first_hit": len(hits),
        "with_pnl": len(pnl_values),
        "win_rate": wins / (wins + losses) if (wins + losses) else None,
        "lose_rate": losses / (wins + losses) if (wins + losses) else None,
        "pnl_mean": pnl_mean,
        "pnl_median": pnl_median,
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
