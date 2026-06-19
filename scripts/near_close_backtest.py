#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional


UTC = timezone.utc


@dataclass(frozen=True)
class Market:
    market_id: str
    slug: str
    question: str
    winner_side: str
    end_ts: Optional[int]
    close_ts: Optional[int]


def parse_time(value: str) -> Optional[int]:
    value = value.strip()
    if not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        if normalized.endswith("+00"):
            normalized += ":00"
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return int(parsed.timestamp())
    except ValueError:
        return None


def load_market_metadata(path: Path) -> Dict[str, dict[str, str]]:
    result: Dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            market_id = (row.get("id") or "").strip()
            if market_id:
                result[market_id] = row
    return result


def resolution_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
    else:
        yield from sorted(path.glob("*_resolutions.csv"))


def load_markets(markets_path: Path, resolutions_path: Path) -> tuple[Dict[str, Market], dict]:
    metadata = load_market_metadata(markets_path)
    markets: Dict[str, Market] = {}
    conflicts = 0
    rows_seen = 0

    for path in resolution_files(resolutions_path):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows_seen += 1
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

                market = Market(
                    market_id=market_id,
                    slug=(row.get("slug") or meta.get("market_slug") or "").strip(),
                    question=(row.get("market_title") or meta.get("question") or "").strip(),
                    winner_side=winner_side,
                    end_ts=parse_time(row.get("endDate") or ""),
                    close_ts=parse_time(meta.get("closedTime") or ""),
                )
                previous = markets.get(market_id)
                if previous and previous != market:
                    conflicts += 1
                    continue
                markets[market_id] = market

    diagnostics = {
        "resolution_rows_seen": rows_seen,
        "deduplicated_binary_markets": len(markets),
        "conflicting_duplicate_rows": conflicts,
        "markets_with_end_date": sum(m.end_ts is not None for m in markets.values()),
        "markets_with_close_time": sum(m.close_ts is not None for m in markets.values()),
    }
    return markets, diagnostics


def iso_from_epoch(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat()


def run(
    markets: Dict[str, Market],
    trades_path: Path,
    windows: list[int],
    thresholds: list[float],
    max_lines: Optional[int],
) -> tuple[list[dict], dict]:
    # key: (clock, minutes, threshold, market_id)
    entries: Dict[tuple[str, int, float, str], dict] = {}
    lines_seen = 0
    eligible_fills = 0

    with trades_path.open("rb", buffering=16 * 1024 * 1024) as handle:
        handle.readline()
        for raw_line in handle:
            lines_seen += 1
            if max_lines is not None and lines_seen > max_lines:
                break
            parts = raw_line.split(b",", 8)
            if len(parts) < 8:
                continue
            try:
                market_id = parts[1].decode("ascii")
            except UnicodeDecodeError:
                continue
            market = markets.get(market_id)
            if market is None:
                continue
            try:
                trade_ts = parse_time(parts[0].decode("ascii"))
                side = parts[4].decode("ascii")
                price = float(parts[7])
            except (UnicodeDecodeError, ValueError):
                continue
            if trade_ts is None or side not in ("token1", "token2") or not 0 < price < 1:
                continue

            for clock, boundary in (("end_date", market.end_ts), ("closed_time_lookahead", market.close_ts)):
                if boundary is None or trade_ts > boundary:
                    continue
                seconds_left = boundary - trade_ts
                for minutes in windows:
                    if seconds_left > minutes * 60:
                        continue
                    for threshold in thresholds:
                        if price < threshold:
                            continue
                        key = (clock, minutes, threshold, market_id)
                        if key in entries:
                            continue
                        eligible_fills += 1
                        won = side == market.winner_side
                        entries[key] = {
                            "clock": clock,
                            "window_minutes": minutes,
                            "threshold": threshold,
                            "market_id": market_id,
                            "slug": market.slug,
                            "question": market.question,
                            "boundary_time": iso_from_epoch(boundary),
                            "entry_time": iso_from_epoch(trade_ts),
                            "seconds_left": seconds_left,
                            "bought_side": side,
                            "winner_side": market.winner_side,
                            "entry_price": price,
                            "won": won,
                            "pnl_per_dollar": (1.0 / price - 1.0) if won else -1.0,
                        }

    return list(entries.values()), {
        "fill_rows_scanned": lines_seen,
        "signals_recorded": eligible_fills,
        "scan_was_limited": max_lines is not None,
    }


def summarize(rows: list[dict], windows: list[int], thresholds: list[float]) -> list[dict]:
    summaries = []
    for clock in ("end_date", "closed_time_lookahead"):
        for minutes in windows:
            for threshold in thresholds:
                selected = [
                    row for row in rows
                    if row["clock"] == clock
                    and row["window_minutes"] == minutes
                    and row["threshold"] == threshold
                ]
                wins = sum(bool(row["won"]) for row in selected)
                pnl = sum(float(row["pnl_per_dollar"]) for row in selected)
                stake = len(selected)
                summaries.append(
                    {
                        "clock": clock,
                        "window_minutes": minutes,
                        "threshold": threshold,
                        "trades": stake,
                        "wins": wins,
                        "losses": stake - wins,
                        "win_rate": wins / stake if stake else None,
                        "total_staked": stake,
                        "total_pnl": pnl,
                        "roi": pnl / stake if stake else None,
                    }
                )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest high-probability fills near market end.")
    parser.add_argument("--markets", type=Path, default=Path("archive/markets.csv"))
    parser.add_argument("--resolutions", type=Path, default=Path("reports/windows"))
    parser.add_argument("--trades", type=Path, default=Path("archive/processed/trades.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/near_close"))
    parser.add_argument("--windows", default="30,60")
    parser.add_argument("--thresholds", default="0.98,0.99")
    parser.add_argument("--max-lines", type=int)
    args = parser.parse_args()

    windows = sorted({int(value) for value in args.windows.split(",")})
    thresholds = sorted({float(value) for value in args.thresholds.split(",")})
    markets, load_diagnostics = load_markets(args.markets, args.resolutions)
    rows, scan_diagnostics = run(markets, args.trades, windows, thresholds, args.max_lines)
    summaries = summarize(rows, windows, thresholds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trades_output = args.output_dir / "trades.csv"
    with trades_output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0]) if rows else ["clock", "window_minutes", "threshold"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["clock"], row["window_minutes"], row["threshold"], row["entry_time"])))

    result = {
        "assumptions": {
            "fees": 0,
            "stake_per_trade": 1,
            "entry": "first archived fill at or above threshold within window",
            "threshold_comparison": "greater than or equal",
            "closed_time_warning": "retrospective upper bound with look-ahead bias; not deployable",
        },
        "diagnostics": {**load_diagnostics, **scan_diagnostics},
        "results": summaries,
    }
    summary_output = args.output_dir / "summary.json"
    summary_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote {trades_output} and {summary_output}")


if __name__ == "__main__":
    main()
