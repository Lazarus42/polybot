#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import duckdb


def parse_time(value: str):
    value = value.strip()
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    if normalized.endswith("+00"):
        normalized += ":00"
    try:
        return datetime.fromisoformat(normalized).replace(tzinfo=None)
    except ValueError:
        return None


def load_metadata(markets_path: Path) -> dict[str, dict[str, str]]:
    with markets_path.open(newline="", encoding="utf-8") as handle:
        return {row["id"]: row for row in csv.DictReader(handle) if row.get("id")}


def load_resolved_binary_markets(markets_path: Path, resolutions_dir: Path) -> list[tuple]:
    metadata = load_metadata(markets_path)
    resolved = {}
    for path in sorted(resolutions_dir.glob("*_resolutions.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") != "RESOLVED":
                    continue
                market_id = (row.get("market_id") or "").strip()
                resolution = (row.get("resolution") or "").strip()
                meta = metadata.get(market_id)
                if not market_id or not resolution or not meta:
                    continue
                answer1 = (meta.get("answer1") or "").strip()
                answer2 = (meta.get("answer2") or "").strip()
                if resolution.casefold() == answer1.casefold():
                    winner_side = "token1"
                elif resolution.casefold() == answer2.casefold():
                    winner_side = "token2"
                else:
                    continue
                record = (
                    int(market_id),
                    row.get("slug") or meta.get("market_slug") or "",
                    row.get("market_title") or meta.get("question") or "",
                    answer1,
                    answer2,
                    winner_side,
                    parse_time(meta.get("createdAt") or ""),
                    parse_time(row.get("endDate") or ""),
                    parse_time(meta.get("closedTime") or ""),
                    float(meta["volume"]) if meta.get("volume") else None,
                )
                previous = resolved.get(market_id)
                if previous is not None and previous != record:
                    raise ValueError(f"Conflicting resolution metadata for market {market_id}")
                resolved[market_id] = record
    return list(resolved.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact resolved-market fill data for underdog tests.")
    parser.add_argument("--markets", type=Path, default=Path("archive/markets.csv"))
    parser.add_argument("--resolutions", type=Path, default=Path("reports/windows"))
    parser.add_argument("--trades", type=Path, default=Path("archive/processed/trades.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("archive/processed/underdog_events"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="8GB")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fills_path = (args.output_dir / "fills_sorted.parquet").resolve()
    markets_path = (args.output_dir / "markets.parquet").resolve()
    manifest_path = args.output_dir / "manifest.json"

    markets = load_resolved_binary_markets(args.markets, args.resolutions)
    connection = duckdb.connect()
    connection.execute(f"SET threads = {args.threads}")
    connection.execute(f"SET memory_limit = '{args.memory_limit}'")
    connection.execute("SET preserve_insertion_order = false")
    connection.execute("PRAGMA enable_progress_bar")
    connection.execute(
        """
        CREATE TABLE eligible_markets (
            market_id BIGINT,
            slug VARCHAR,
            question VARCHAR,
            answer1 VARCHAR,
            answer2 VARCHAR,
            winner_side VARCHAR,
            created_at TIMESTAMP,
            end_date TIMESTAMP,
            closed_time TIMESTAMP,
            historical_volume DOUBLE
        )
        """
    )
    connection.executemany("INSERT INTO eligible_markets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", markets)

    trades_sql_path = str(args.trades.resolve()).replace("'", "''")
    fills_sql_path = str(fills_path).replace("'", "''")
    connection.execute(
        f"""
        COPY (
            SELECT
                t.timestamp,
                t.market_id,
                t.nonusdc_side AS side,
                t.price,
                t.usd_amount,
                t.token_amount,
                t.transactionHash AS transaction_hash,
                COUNT(*)::UTINYINT AS source_row_count
            FROM read_csv_auto('{trades_sql_path}', header = true) t
            INNER JOIN eligible_markets m USING (market_id)
            WHERE t.price > 0 AND t.price < 1
              AND t.nonusdc_side IN ('token1', 'token2')
              AND (m.closed_time IS NULL OR t.timestamp <= m.closed_time)
            GROUP BY ALL
            ORDER BY market_id, timestamp, transaction_hash, side, price
        ) TO '{fills_sql_path}' (
            FORMAT PARQUET,
            COMPRESSION ZSTD,
            ROW_GROUP_SIZE 250000
        )
        """
    )

    markets_sql_path = str(markets_path).replace("'", "''")
    connection.execute(
        f"""
        COPY (
            SELECT m.*, firsts.first_trade_time
            FROM eligible_markets m
            LEFT JOIN (
                SELECT market_id, MIN(timestamp) AS first_trade_time
                FROM read_parquet('{fills_sql_path}')
                GROUP BY market_id
            ) firsts USING (market_id)
        ) TO '{markets_sql_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    fill_count = connection.execute(f"SELECT COUNT(*) FROM read_parquet('{fills_sql_path}')").fetchone()[0]
    market_count = connection.execute(f"SELECT COUNT(*) FROM read_parquet('{markets_sql_path}')").fetchone()[0]
    markets_with_fills = connection.execute(
        f"SELECT COUNT(*) FROM read_parquet('{markets_sql_path}') WHERE first_trade_time IS NOT NULL"
    ).fetchone()[0]
    manifest = {
        "format_version": 1,
        "source_trades": str(args.trades),
        "source_markets": str(args.markets),
        "source_resolutions": str(args.resolutions),
        "market_count": market_count,
        "markets_with_fills": markets_with_fills,
        "deduplicated_fill_count": fill_count,
        "fills_bytes": fills_path.stat().st_size,
        "markets_bytes": markets_path.stat().st_size,
        "deduplication": "Exact duplicate processed rows are collapsed; source_row_count preserves multiplicity.",
        "sort_order": ["market_id", "timestamp", "transaction_hash", "side", "price"],
        "price_filter": "0 < price < 1",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
