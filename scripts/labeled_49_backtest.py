#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class LabelRow:
    market_id: str
    question: str
    answer1: str
    answer2: str
    resolved_outcome: str
    winner_token: Optional[str]


@dataclass
class FirstHit:
    timestamp: str
    price: float
    side: str  # token1/token2


def normalize(text: str) -> str:
    return text.strip().lower()


def map_outcome_to_token(resolved_outcome: str, answer1: str, answer2: str) -> Optional[str]:
    if not answer1 or not answer2:
        return None
    if normalize(resolved_outcome) == normalize(answer1):
        return "token1"
    if normalize(resolved_outcome) == normalize(answer2):
        return "token2"
    return None


def load_labels(path: Path) -> Dict[str, LabelRow]:
    labels: Dict[str, LabelRow] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            resolved_outcome = row.get("resolved_outcome") or ""
            if not resolved_outcome:
                continue
            market_id = row.get("market_id") or ""
            if not market_id:
                continue
            answer1 = row.get("answer1") or ""
            answer2 = row.get("answer2") or ""
            winner_token = map_outcome_to_token(resolved_outcome, answer1, answer2)
            labels[market_id] = LabelRow(
                market_id=market_id,
                question=row.get("question") or "",
                answer1=answer1,
                answer2=answer2,
                resolved_outcome=resolved_outcome,
                winner_token=winner_token,
            )
    return labels


def find_hits(
    trades_path: Path, target_ids: set[str], threshold: float, mode: str
) -> Dict[str, FirstHit]:
    hits: Dict[str, FirstHit] = {}
    remaining = set(target_ids)
    with trades_path.open("r", newline="") as f:
        header = f.readline()
        for line in f:
            if mode == "first" and not remaining:
                break
            # Fast split; trades.csv has no quoted commas
            parts = line.rstrip("\n").split(",", 8)
            if len(parts) < 8:
                continue
            market_id = parts[1]
            if market_id not in target_ids:
                continue
            try:
                price = float(parts[7])
            except ValueError:
                continue
            if price < threshold:
                continue
            ts = parts[0]
            side = parts[4]
            if mode == "first":
                if market_id in hits:
                    continue
                hits[market_id] = FirstHit(timestamp=ts, price=price, side=side)
                remaining.discard(market_id)
            else:
                # keep overwriting; last qualifying hit wins
                hits[market_id] = FirstHit(timestamp=ts, price=price, side=side)
    return hits


def compute_pnl(entry_price: float, won: bool, fee_rate: float) -> float:
    shares = 1.0 / entry_price
    if won:
        gross = shares * 1.0
        fee_paid = gross * fee_rate
        return gross - 1.0 - fee_paid
    return -1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest labeled markets (90% first-hit).")
    parser.add_argument("--labels-csv", default="archive/processed/random_100_markets.csv")
    parser.add_argument("--trades-csv", default="archive/processed/trades.csv")
    parser.add_argument("--output", default="reports/labeled_pnl.csv")
    parser.add_argument("--summary", default="reports/labeled_summary.json")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--fee-rate", type=float, default=0.0)
    parser.add_argument("--mode", choices=["first", "last"], default="first")
    args = parser.parse_args()

    labels_path = Path(args.labels_csv)
    trades_path = Path(args.trades_csv)
    output_path = Path(args.output)
    summary_path = Path(args.summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    labels = load_labels(labels_path)
    target_ids = set(labels.keys())
    hits = find_hits(trades_path, target_ids, args.threshold, args.mode)

    results: List[Dict[str, object]] = []
    pnl_values: List[float] = []

    for market_id, label in labels.items():
        hit = hits.get(market_id)
        winner_token = label.winner_token
        won = None
        pnl = None
        if hit and winner_token:
            won = hit.side == winner_token
            pnl = compute_pnl(hit.price, won, args.fee_rate)
            pnl_values.append(pnl)

        results.append(
            {
                "market_id": market_id,
                "question": label.question,
                "answer1": label.answer1,
                "answer2": label.answer2,
                "resolved_outcome": label.resolved_outcome,
                "winner_token": winner_token,
                "entry_ts": hit.timestamp if hit else None,
                "entry_price": hit.price if hit else None,
                "entry_side": hit.side if hit else None,
                "won": won,
                "pnl": pnl,
            }
        )

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        for row in results:
            writer.writerow(row)

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
        "labeled_markets": len(labels),
        "with_first_hit": len(hits),
        "with_pnl": len(pnl_values),
        "pnl_mean": mean,
        "pnl_median": median,
    }
    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
