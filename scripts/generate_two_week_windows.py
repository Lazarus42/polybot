#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate slug lists for N random two-week windows."
    )
    parser.add_argument("--markets", default="archive/markets.csv")
    parser.add_argument("--out-dir", default="archive/processed/windows")
    parser.add_argument("--windows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    markets_path = Path(args.markets)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load markets with closedTime + slug + binary
    rows = []
    with markets_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not (row.get("answer1") and row.get("answer2")):
                continue
            closed = row.get("closedTime")
            slug = row.get("market_slug")
            if not closed or not slug:
                continue
            try:
                normalized = closed.replace("Z", "+00:00")
                if normalized.endswith("+00"):
                    normalized = normalized[:-3] + "+00:00"
                dt = datetime.fromisoformat(normalized)
            except ValueError:
                continue
            rows.append((dt, slug))

    if not rows:
        raise SystemExit("No eligible markets found.")

    rows.sort(key=lambda x: x[0])
    min_dt = rows[0][0]
    max_dt = rows[-1][0]

    random.seed(args.seed)
    for i in range(args.windows):
        span_days = (max_dt - min_dt).days
        start_offset = random.randint(0, max(0, span_days - 14))
        start = min_dt + timedelta(days=start_offset)
        end = start + timedelta(days=14)
        slugs = [slug for dt, slug in rows if start <= dt <= end]

        out_path = out_dir / f"window_{i:03d}_slugs.txt"
        out_path.write_text("\n".join(slugs))

        print(
            f"window {i:03d}: {start.isoformat()} → {end.isoformat()} "
            f"({len(slugs)} slugs) -> {out_path}"
        )


if __name__ == "__main__":
    main()
