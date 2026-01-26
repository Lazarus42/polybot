#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple


def parse_iso_ts(value: str) -> datetime:
    if value.endswith("Z"):
        value = value.replace("Z", "+00:00")
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def find_first_hit(
    csv_path: Path, market_id: str, threshold: float
) -> Optional[Tuple[datetime, float, str]]:
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("market_id") != market_id:
                continue
            price_str = row.get("price")
            if price_str is None:
                continue
            try:
                price = float(price_str)
            except ValueError:
                continue
            if price >= threshold:
                ts = parse_iso_ts(row["timestamp"])
                side = row.get("nonusdc_side") or ""
                return ts, price, side
    return None


def compute_pnl(entry_price: float, won: bool, fee_rate: float) -> Tuple[float, float]:
    shares = 1.0 / entry_price
    if won:
        gross = shares * 1.0
        fee_paid = gross * fee_rate
        pnl = gross - 1.0 - fee_paid
    else:
        fee_paid = 0.0
        pnl = -1.0
    return pnl, fee_paid


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-market 90% first-hit backtest.")
    parser.add_argument("--csv", default="archive/processed/trades.csv")
    parser.add_argument("--market-id", default="240380")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--fee-rate", type=float, default=0.0)
    parser.add_argument("--resolution-side", choices=["token0", "token1"], default=None)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    hit = find_first_hit(csv_path, args.market_id, args.threshold)
    if not hit:
        print("No trade hit threshold for market:", args.market_id)
        return

    ts, price, side = hit
    print("market_id:", args.market_id)
    print("first_hit_ts:", ts.isoformat())
    print("entry_price:", price)
    print("entry_side:", side)

    if args.resolution_side is None:
        print("resolution_side: (not provided)")
        print("pnl: (requires resolution_side)")
        return

    won = side == args.resolution_side
    pnl, fee_paid = compute_pnl(price, won, args.fee_rate)
    print("resolution_side:", args.resolution_side)
    print("won:", won)
    print("fee_paid:", fee_paid)
    print("pnl:", pnl)


if __name__ == "__main__":
    main()
