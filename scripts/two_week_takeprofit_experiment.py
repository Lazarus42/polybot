#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class MarketMeta:
    slug: str
    answer1: str
    answer2: str
    closed_time: datetime


@dataclass
class CacheRow:
    market_id: str
    signal_price: float
    signal_side: str
    opp_entry_price: float
    opp_entry_side: str
    tp_hit: bool


def normalize(text: str) -> str:
    return text.strip().lower()


def load_resolutions(path: Path, slugs_filter: Optional[set[str]] = None) -> Dict[str, str]:
    slug_to_res = {}
    slug_to_mid = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            slug = row.get("slug")
            res = row.get("resolution")
            mid = row.get("market_id")
            if not slug or not mid:
                continue
            if slugs_filter is not None and slug not in slugs_filter:
                continue
            if res not in ("Yes", "No"):
                continue
            slug_to_res[slug] = res
            slug_to_mid[slug] = str(mid)
    # return market_id -> resolution
    return {slug_to_mid[s]: slug_to_res[s] for s in slug_to_res}


def load_markets(path: Path) -> Dict[str, MarketMeta]:
    markets: Dict[str, MarketMeta] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row.get("id")
            if not mid:
                continue
            answer1 = row.get("answer1") or ""
            answer2 = row.get("answer2") or ""
            slug = row.get("market_slug") or ""
            closed = row.get("closedTime") or ""
            if not answer1 or not answer2 or not slug or not closed:
                continue
            try:
                normalized = closed.replace("Z", "+00:00")
                if normalized.endswith("+00"):
                    normalized = normalized[:-3] + "+00:00"
                closed_dt = datetime.fromisoformat(normalized)
            except ValueError:
                continue
            markets[mid] = MarketMeta(
                slug=slug, answer1=answer1, answer2=answer2, closed_time=closed_dt
            )
    return markets


def build_cache(
    trades_path: Path, target_ids: set[str], threshold: float
) -> Dict[str, CacheRow]:
    signal: Dict[str, Tuple[float, str]] = {}
    opposite: Dict[str, Tuple[float, str]] = {}
    tp_hit: Dict[str, bool] = {}

    with trades_path.open("r", newline="") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split(",", 8)
            if len(parts) < 8:
                continue
            mid = parts[1]
            if mid not in target_ids:
                continue
            try:
                price = float(parts[7])
            except ValueError:
                continue
            side = parts[4]

            if mid not in signal:
                if price >= threshold:
                    signal[mid] = (price, side)
                continue

            if mid not in opposite:
                sig_side = signal[mid][1]
                if side != sig_side:
                    opposite[mid] = (price, side)
                continue

            # already have opposite, check take-profit
            if tp_hit.get(mid):
                continue
            opp_price, opp_side = opposite[mid]
            if side == opp_side and price >= 2.0 * opp_price:
                tp_hit[mid] = True

    cache: Dict[str, CacheRow] = {}
    for mid in target_ids:
        if mid in signal and mid in opposite:
            s_price, s_side = signal[mid]
            o_price, o_side = opposite[mid]
            cache[mid] = CacheRow(
                market_id=mid,
                signal_price=s_price,
                signal_side=s_side,
                opp_entry_price=o_price,
                opp_entry_side=o_side,
                tp_hit=bool(tp_hit.get(mid, False)),
            )
    return cache


def save_cache(path: Path, cache: Dict[str, CacheRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "market_id",
                "signal_price",
                "signal_side",
                "opp_entry_price",
                "opp_entry_side",
                "tp_hit",
            ],
        )
        writer.writeheader()
        for row in cache.values():
            writer.writerow(
                {
                    "market_id": row.market_id,
                    "signal_price": row.signal_price,
                    "signal_side": row.signal_side,
                    "opp_entry_price": row.opp_entry_price,
                    "opp_entry_side": row.opp_entry_side,
                    "tp_hit": int(row.tp_hit),
                }
            )


def load_cache(path: Path) -> Dict[str, CacheRow]:
    cache: Dict[str, CacheRow] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row.get("market_id")
            if not mid:
                continue
            cache[mid] = CacheRow(
                market_id=mid,
                signal_price=float(row["signal_price"]),
                signal_side=row["signal_side"],
                opp_entry_price=float(row["opp_entry_price"]),
                opp_entry_side=row["opp_entry_side"],
                tp_hit=row.get("tp_hit") in ("1", "true", "True"),
            )
    return cache


def compute_pnl(
    cache_row: CacheRow, winner_token: str, fee_rate: float
) -> float:
    shares = 1.0 / cache_row.opp_entry_price
    if cache_row.tp_hit:
        # exit at 2x entry
        gross = shares * (2.0 * cache_row.opp_entry_price)
        fee_paid = gross * fee_rate
        return gross - 1.0 - fee_paid
    # hold to resolution
    if cache_row.opp_entry_side == winner_token:
        gross = shares * 1.0
        fee_paid = gross * fee_rate
        return gross - 1.0 - fee_paid
    return -1.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 100 random two-week windows of inverse 90c take-profit strategy."
    )
    parser.add_argument("--markets", default="archive/markets.csv")
    parser.add_argument("--resolutions", required=True)
    parser.add_argument("--trades", default="archive/processed/trades.csv")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--fee-rate", type=float, default=0.02)
    parser.add_argument("--windows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", default="reports/takeprofit_cache.csv")
    parser.add_argument("--output", default="reports/two_week_takeprofit_windows.csv")
    parser.add_argument("--summary", default="reports/two_week_takeprofit_summary.json")
    args = parser.parse_args()

    markets = load_markets(Path(args.markets))
    resolutions = load_resolutions(Path(args.resolutions))

    # Only consider markets with resolutions and metadata
    target_ids = {mid for mid in resolutions if mid in markets}
    if not target_ids:
        raise SystemExit("No markets with resolutions + metadata found.")

    cache_path = Path(args.cache)
    if cache_path.exists():
        cache = load_cache(cache_path)
    else:
        cache = build_cache(Path(args.trades), target_ids, args.threshold)
        save_cache(cache_path, cache)

    # Determine min/max closed_time from eligible markets
    closed_times = [markets[mid].closed_time for mid in target_ids]
    min_dt = min(closed_times)
    max_dt = max(closed_times)

    random.seed(args.seed)
    windows = []
    for _ in range(args.windows):
        # sample start uniformly
        span_days = (max_dt - min_dt).days
        start_offset = random.randint(0, max(0, span_days - 14))
        start = min_dt + timedelta(days=start_offset)
        end = start + timedelta(days=14)
        windows.append((start, end))

    rows = []
    total_profits = []
    for idx, (start, end) in enumerate(windows):
        eligible = [
            mid
            for mid in target_ids
            if start <= markets[mid].closed_time <= end and mid in cache
        ]
        pnl_values = []
        wins = 0
        losses = 0
        for mid in eligible:
            meta = markets[mid]
            res = resolutions[mid]
            winner_token = None
            if normalize(res) == normalize(meta.answer1):
                winner_token = "token1"
            elif normalize(res) == normalize(meta.answer2):
                winner_token = "token2"
            if not winner_token:
                continue
            pnl = compute_pnl(cache[mid], winner_token, args.fee_rate)
            pnl_values.append(pnl)
            if pnl > 0:
                wins += 1
            else:
                losses += 1
        total_profit = sum(pnl_values) if pnl_values else 0.0
        total_profits.append(total_profit)
        rows.append(
            {
                "window_idx": idx,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "markets": len(eligible),
                "trades": len(pnl_values),
                "win_rate": wins / (wins + losses) if (wins + losses) else None,
                "mean_pnl": sum(pnl_values) / len(pnl_values) if pnl_values else None,
                "median_pnl": (sorted(pnl_values)[len(pnl_values)//2] if pnl_values else None),
                "total_profit": total_profit,
            }
        )

    # Write per-window stats
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    total_profits_sorted = sorted(total_profits)
    summary = {
        "windows": args.windows,
        "fee_rate": args.fee_rate,
        "min_window_profit": min(total_profits) if total_profits else None,
        "max_window_profit": max(total_profits) if total_profits else None,
        "mean_window_profit": sum(total_profits) / len(total_profits) if total_profits else None,
        "median_window_profit": (
            total_profits_sorted[len(total_profits_sorted) // 2]
            if total_profits else None
        ),
        "cache_path": str(cache_path),
        "per_window_csv": str(out_path),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
