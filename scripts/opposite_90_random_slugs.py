#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def normalize(text: str) -> str:
    return text.strip().lower()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Opposite 90% strategy on a slug list (Yes/No resolutions only)."
    )
    parser.add_argument("--slugs", default="archive/processed/random_4000_slugs.txt")
    parser.add_argument("--resolutions", default="polymarket_resolutions.csv")
    parser.add_argument("--markets", default="archive/markets.csv")
    parser.add_argument("--trades", default="archive/processed/trades.csv")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--fee-rate", type=float, default=0.0)
    parser.add_argument("--output", default="reports/opposite_90_random_4000_pnl.csv")
    parser.add_argument("--summary", default="reports/opposite_90_random_4000_summary.json")
    args = parser.parse_args()

    slugs_path = Path(args.slugs)
    res_path = Path(args.resolutions)
    markets_path = Path(args.markets)
    trades_path = Path(args.trades)

    with slugs_path.open("r") as f:
        target_slugs = set(s.strip() for s in f if s.strip())

    slug_to_resolution = {}
    slug_to_mid = {}
    with res_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            slug = row.get("slug")
            res = row.get("resolution")
            mid = row.get("market_id")
            if not slug or slug not in target_slugs:
                continue
            if res not in ("Yes", "No"):
                continue
            if not mid:
                continue
            slug_to_resolution[slug] = res
            slug_to_mid[slug] = str(mid)

    mid_to_answers = {}
    with markets_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row.get("id")
            if not mid:
                continue
            mid_to_answers[mid] = (
                row.get("answer1") or "",
                row.get("answer2") or "",
                row.get("question") or "",
            )

    mid_to_resolution = {}
    for slug, mid in slug_to_mid.items():
        res = slug_to_resolution.get(slug)
        if res:
            mid_to_resolution[mid] = res

    threshold = args.threshold
    remaining = set(mid_to_resolution.keys())
    first_hits = {}
    with trades_path.open("r", newline="") as f:
        f.readline()
        for line in f:
            if not remaining:
                break
            parts = line.rstrip("\n").split(",", 8)
            if len(parts) < 8:
                continue
            mid = parts[1]
            if mid not in remaining:
                continue
            try:
                price = float(parts[7])
            except ValueError:
                continue
            if price < threshold:
                continue
            first_hits[mid] = (parts[0], price, parts[4])
            remaining.remove(mid)

    results = []
    pnl_values = []
    wins = 0
    losses = 0

    for mid, res in mid_to_resolution.items():
        answers = mid_to_answers.get(mid)
        if not answers:
            continue
        a1, a2, question = answers
        winner_token = None
        if normalize(res) == normalize(a1):
            winner_token = "token1"
        elif normalize(res) == normalize(a2):
            winner_token = "token2"
        if not winner_token:
            continue
        hit = first_hits.get(mid)
        if not hit:
            continue
        ts, price, side = hit
        opp_price = 1.0 - price
        if opp_price <= 0:
            continue
        opp_won = side != winner_token
        if opp_won:
            gross = 1.0 / opp_price
            fee_paid = gross * args.fee_rate
            pnl = gross - 1.0 - fee_paid
        else:
            pnl = -1.0
        pnl_values.append(pnl)
        if opp_won:
            wins += 1
        else:
            losses += 1
        results.append(
            {
                "market_id": mid,
                "question": question,
                "resolved_outcome": res,
                "entry_ts": ts,
                "entry_price": price,
                "entry_side": side,
                "opp_entry_price": opp_price,
                "opp_won": opp_won,
                "pnl": pnl,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if results:
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            for row in results:
                writer.writerow(row)

    pnl_values_sorted = sorted(pnl_values)
    mean = sum(pnl_values) / len(pnl_values) if pnl_values else None
    median = None
    if pnl_values:
        midx = len(pnl_values_sorted) // 2
        if len(pnl_values_sorted) % 2 == 0:
            median = (pnl_values_sorted[midx - 1] + pnl_values_sorted[midx]) / 2
        else:
            median = pnl_values_sorted[midx]

    summary = {
        "target_slugs": len(target_slugs),
        "yes_no_resolutions_in_targets": len(mid_to_resolution),
        "with_first_hit": len(first_hits),
        "with_pnl": len(pnl_values),
        "win_rate": wins / (wins + losses) if (wins + losses) else None,
        "lose_rate": losses / (wins + losses) if (wins + losses) else None,
        "pnl_mean": mean,
        "pnl_median": median,
        "pnl_min": min(pnl_values) if pnl_values else None,
        "pnl_max": max(pnl_values) if pnl_values else None,
    }

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
