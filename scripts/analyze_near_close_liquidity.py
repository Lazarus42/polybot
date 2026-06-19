#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


def percentile(values: list[float], quantile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def distribution(values: Iterable[float]) -> dict[str, Optional[float] | int]:
    data = list(values)
    return {
        "count": len(data),
        "total": sum(data),
        "min": min(data) if data else None,
        "p10": percentile(data, 0.10),
        "p25": percentile(data, 0.25),
        "median": percentile(data, 0.50),
        "p75": percentile(data, 0.75),
        "p90": percentile(data, 0.90),
        "p95": percentile(data, 0.95),
        "p99": percentile(data, 0.99),
        "max": max(data) if data else None,
        "mean": sum(data) / len(data) if data else None,
    }


def raw_timestamp(iso_timestamp: str) -> str:
    return datetime.fromisoformat(iso_timestamp).strftime("%Y-%m-%dT%H:%M:%S.%f")


def load_market_volumes(path: Path) -> dict[str, float]:
    result = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                result[row["id"]] = float(row["volume"])
            except (KeyError, TypeError, ValueError):
                continue
    return result


def load_signals(path: Path) -> tuple[list[dict[str, str]], dict[tuple[str, str, str, str], list[int]]]:
    rows = []
    targets: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["clock"] != "end_date":
                continue
            index = len(rows)
            rows.append(row)
            key = (
                raw_timestamp(row["entry_time"]),
                row["market_id"],
                row["bought_side"],
                row["entry_price"],
            )
            targets[key].append(index)
    return rows, targets


def add_fill_sizes(
    rows: list[dict[str, str]],
    targets: dict[tuple[str, str, str, str], list[int]],
    trades_path: Path,
) -> dict[str, int]:
    # Price strings can differ in their final floating-point digit after the
    # backtest CSV round trip, so use timestamp/market/side plus a tolerance.
    by_identity: dict[
        tuple[str, str, str],
        list[tuple[float, tuple[str, str, str, str], list[int]]],
    ] = defaultdict(list)
    for key, indexes in targets.items():
        timestamp, market_id, side, price = key
        by_identity[(timestamp, market_id, side)].append((float(price), key, indexes))

    unmatched_targets = set(targets)
    lines_scanned = 0
    with trades_path.open("rb", buffering=16 * 1024 * 1024) as handle:
        handle.readline()
        for raw_line in handle:
            lines_scanned += 1
            parts = raw_line.split(b",", 10)
            if len(parts) < 10:
                continue
            try:
                identity = (
                    parts[0].decode("ascii"),
                    parts[1].decode("ascii"),
                    parts[4].decode("ascii"),
                )
            except UnicodeDecodeError:
                continue
            matches = by_identity.get(identity)
            if not matches:
                continue
            try:
                fill_price = float(parts[7])
                fill_usd = float(parts[8])
                fill_shares = float(parts[9])
            except ValueError:
                continue
            for target_price, target_key, indexes in matches:
                if abs(fill_price - target_price) > 1e-12:
                    continue
                if target_key not in unmatched_targets:
                    continue
                for index in indexes:
                    rows[index]["signal_fill_usd"] = str(fill_usd)
                    rows[index]["signal_fill_shares"] = str(fill_shares)
                unmatched_targets.remove(target_key)
            if not unmatched_targets:
                break
    return {
        "fill_rows_scanned": lines_scanned,
        "unique_signal_fills_requested": len(targets),
        "unique_signal_fills_matched": len(targets) - len(unmatched_targets),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze volume and fill sizes for near-close signals.")
    parser.add_argument("--signals", type=Path, default=Path("reports/near_close/trades.csv"))
    parser.add_argument("--markets", type=Path, default=Path("archive/markets.csv"))
    parser.add_argument("--trades", type=Path, default=Path("archive/processed/trades.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/near_close"))
    args = parser.parse_args()

    rows, targets = load_signals(args.signals)
    volumes = load_market_volumes(args.markets)
    diagnostics = add_fill_sizes(rows, targets, args.trades)

    grouped: dict[tuple[int, float], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["window_minutes"]), float(row["threshold"]))].append(row)

    results = []
    for (minutes, threshold), selected in sorted(grouped.items()):
        market_ids = {row["market_id"] for row in selected}
        market_values = [volumes[market_id] for market_id in market_ids if market_id in volumes]
        fill_values = [float(row["signal_fill_usd"]) for row in selected if row.get("signal_fill_usd")]
        share_values = [float(row["signal_fill_shares"]) for row in selected if row.get("signal_fill_shares")]
        results.append(
            {
                "window_minutes": minutes,
                "threshold": threshold,
                "markets": len(market_ids),
                "historical_market_volume_usd": distribution(market_values),
                "qualifying_fill_size_usd": distribution(fill_values),
                "qualifying_fill_size_shares": distribution(share_values),
                "markets_below_volume_usd": {
                    "1000": sum(value < 1_000 for value in market_values),
                    "10000": sum(value < 10_000 for value in market_values),
                    "100000": sum(value < 100_000 for value in market_values),
                },
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    enriched_path = args.output_dir / "trades_with_liquidity.csv"
    with enriched_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0]) + [name for name in ("signal_fill_usd", "signal_fill_shares") if name not in rows[0]]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "definitions": {
            "historical_market_volume_usd": "Cumulative market volume in archive/markets.csv; not capital currently resting in the book.",
            "qualifying_fill_size_usd": "Executed size of the archived fill that triggered the strategy; not total executable depth.",
            "order_book_depth": "Unavailable because the archive has no historical order-book snapshots.",
        },
        "diagnostics": diagnostics,
        "results": results,
    }
    report_path = args.output_dir / "liquidity_summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {report_path} and {enriched_path}")


if __name__ == "__main__":
    main()
