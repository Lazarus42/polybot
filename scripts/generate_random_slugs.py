#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random resolved binary market slugs.")
    parser.add_argument("--count", type=int, default=4000)
    parser.add_argument("--markets", default="archive/markets.csv")
    parser.add_argument("--output", default="archive/processed/random_slugs.txt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    markets_path = Path(args.markets)
    output_path = Path(args.output)

    sample_size = args.count
    sample: list[str] = []
    count = 0
    random.seed(args.seed)

    with markets_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            answer1 = (row.get("answer1") or "").strip()
            answer2 = (row.get("answer2") or "").strip()
            closed = (row.get("closedTime") or "").strip()
            slug = (row.get("market_slug") or "").strip()
            if not answer1 or not answer2:
                continue
            if not closed:
                continue
            if not slug:
                continue
            count += 1
            if len(sample) < sample_size:
                sample.append(slug)
            else:
                j = random.randint(1, count)
                if j <= sample_size:
                    sample[j - 1] = slug

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        f.write("\n".join(sample))

    print("eligible", count)
    print("wrote", len(sample), "to", output_path)


if __name__ == "__main__":
    main()
