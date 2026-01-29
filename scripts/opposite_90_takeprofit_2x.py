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
        description="Inverse 90% strategy with 2x take-profit based on first trade >= 2x."
    )
    parser.add_argument("--slugs", required=True)
    parser.add_argument("--resolutions", required=True)
    parser.add_argument("--markets", default="archive/markets.csv")
    parser.add_argument("--trades", default="archive/processed/trades.csv")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--fee-rate", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
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

    # Pass: find first >= threshold trade (signal) and then first trade on opposite side
    threshold = args.threshold
    signal = {}  # mid -> (ts, price, side)
    opposite = {}  # mid -> (ts, price, side) actual opposite-side trade after signal
    remaining = set(mid_to_resolution.keys())

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
            ts = parts[0]
            side = parts[4]

            if mid not in signal:
                if price >= threshold:
                    signal[mid] = (ts, price, side)
            else:
                if mid in opposite:
                    continue
                sig_side = signal[mid][2]
                if side != sig_side:
                    opposite[mid] = (ts, price, side)
                    remaining.remove(mid)

    # Pass: find first trade >= 2x entry price on the opposite side after entry
    takeprofit = {}  # mid -> (ts, price)
    remaining_tp = set(opposite.keys())
    with trades_path.open("r", newline="") as f:
        f.readline()
        for line in f:
            if not remaining_tp:
                break
            parts = line.rstrip("\n").split(",", 8)
            if len(parts) < 8:
                continue
            mid = parts[1]
            if mid not in remaining_tp:
                continue
            try:
                price = float(parts[7])
            except ValueError:
                continue
            ts = parts[0]
            side = parts[4]
            entry = opposite.get(mid)
            if not entry:
                continue
            entry_side = entry[2]
            if side != entry_side:
                continue
            entry_price = entry[1]
            if price >= 2.0 * entry_price:
                takeprofit[mid] = (ts, price)
                remaining_tp.remove(mid)

    results = []
    pnl_values = []
    wins = 0
    losses = 0
    exits = 0
    holds = 0

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

        sig = signal.get(mid)
        opp = opposite.get(mid)
        if not sig or not opp:
            continue

        sig_ts, sig_price, sig_side = sig
        opp_ts, opp_price, opp_side = opp

        shares = 1.0 / opp_price
        tp = takeprofit.get(mid)
        if tp:
            # sell at 2x entry price (take-profit threshold)
            exit_price = 2.0 * opp_price
            gross = shares * exit_price
            fee_paid = gross * args.fee_rate
            pnl = gross - 1.0 - fee_paid
            exits += 1
            exit_type = "take_profit"
        else:
            # hold to resolution
            holds += 1
            opp_won = opp_side == winner_token
            if opp_won:
                gross = shares * 1.0
                fee_paid = gross * args.fee_rate
                pnl = gross - 1.0 - fee_paid
            else:
                pnl = -1.0
            exit_type = "resolution"

        pnl_values.append(pnl)
        if pnl > 0:
            wins += 1
        else:
            losses += 1

        results.append(
            {
                "market_id": mid,
                "question": question,
                "resolved_outcome": res,
                "signal_ts": sig_ts,
                "signal_price": sig_price,
                "signal_side": sig_side,
                "opp_entry_ts": opp_ts,
                "opp_entry_price": opp_price,
                "opp_entry_side": opp_side,
                "exit_type": exit_type,
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
        "with_signal": len(signal),
        "with_opposite_trade": len(opposite),
        "with_take_profit": exits,
        "with_resolution_hold": holds,
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
