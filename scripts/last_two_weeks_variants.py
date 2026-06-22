#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class MarketMeta:
    answer1: str
    answer2: str
    closed_time: datetime


@dataclass
class MarketState:
    signal_price: float
    signal_side: str
    signal_ts: datetime
    entry_price: float
    entry_side: str
    opp_trades: int = 0
    opp_prices_first_k: List[float] = field(default_factory=list)
    tp2x: bool = False
    tp3x: bool = False
    last_price_before_cutoff: Optional[float] = None


def normalize(text: str) -> str:
    return text.strip().lower()


def load_resolutions(path: Path, slugs_filter: set[str]) -> Dict[str, str]:
    slug_to_res = {}
    slug_to_mid = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            slug = row.get("slug")
            res = row.get("resolution")
            mid = row.get("market_id")
            if not slug or slug not in slugs_filter:
                continue
            if res not in ("Yes", "No"):
                continue
            if not mid:
                continue
            slug_to_res[slug] = res
            slug_to_mid[slug] = str(mid)
    return {slug_to_mid[s]: slug_to_res[s] for s in slug_to_res}


def load_markets(path: Path) -> Dict[str, MarketMeta]:
    markets: Dict[str, MarketMeta] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row.get("id")
            if not mid:
                continue
            a1 = row.get("answer1") or ""
            a2 = row.get("answer2") or ""
            closed = row.get("closedTime") or ""
            if not a1 or not a2 or not closed:
                continue
            try:
                closed_dt = datetime.fromisoformat(closed.replace("Z", "+00:00"))
            except ValueError:
                continue
            markets[mid] = MarketMeta(answer1=a1, answer2=a2, closed_time=closed_dt)
    return markets


def compute_pnl_exit(exit_price: float, entry_price: float, fee_rate: float) -> float:
    shares = 1.0 / entry_price
    gross = shares * exit_price
    fee_paid = gross * fee_rate
    return gross - 1.0 - fee_paid


def compute_pnl_resolution(entry_price: float, win: bool, fee_rate: float) -> float:
    if not win:
        return -1.0
    gross = (1.0 / entry_price) * 1.0
    fee_paid = gross * fee_rate
    return gross - 1.0 - fee_paid


def summarize(pnls: List[float]) -> Dict[str, float]:
    if not pnls:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None, "total": 0.0}
    pnls_sorted = sorted(pnls)
    mid = len(pnls_sorted) // 2
    median = (
        (pnls_sorted[mid - 1] + pnls_sorted[mid]) / 2
        if len(pnls_sorted) % 2 == 0
        else pnls_sorted[mid]
    )
    return {
        "count": len(pnls),
        "mean": sum(pnls) / len(pnls),
        "median": median,
        "min": min(pnls),
        "max": max(pnls),
        "total": sum(pnls),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run last-two-weeks variants backtest.")
    parser.add_argument("--slugs", required=True)
    parser.add_argument("--resolutions", required=True)
    parser.add_argument("--markets", default="archive/markets.csv")
    parser.add_argument("--trades", default="archive/processed/trades.csv")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--fee-rate", type=float, default=0.02)
    parser.add_argument("--time-stop-hours", type=int, default=48)
    parser.add_argument("--momentum-k", type=int, default=3)
    parser.add_argument("--momentum-min-pct", type=float, default=0.05)
    parser.add_argument("--liquidity-min-trades", type=int, default=5)
    parser.add_argument("--signal-max", type=float, default=0.97)
    parser.add_argument("--min-hours-to-close", type=int, default=24)
    parser.add_argument("--summary", default="reports/last_two_weeks_variants_summary.json")
    args = parser.parse_args()

    slugs_path = Path(args.slugs)
    with slugs_path.open("r") as f:
        target_slugs = set(s.strip() for s in f if s.strip())

    markets = load_markets(Path(args.markets))
    resolutions = load_resolutions(Path(args.resolutions), target_slugs)

    target_ids = {mid for mid in resolutions if mid in markets}
    if not target_ids:
        raise SystemExit("No markets with resolutions + metadata found.")

    cutoff_time = {
        mid: markets[mid].closed_time - timedelta(hours=args.time_stop_hours)
        for mid in target_ids
    }

    state: Dict[str, MarketState] = {}
    signal_found: Dict[str, tuple] = {}

    # Scan trades once to build state
    with Path(args.trades).open("r", newline="") as f:
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
            ts = parts[0]
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

            if mid not in signal_found:
                if price >= args.threshold:
                    signal_found[mid] = (price, side, dt)
                continue

            if mid not in state:
                sig_price, sig_side, sig_ts = signal_found[mid]
                if side != sig_side:
                    state[mid] = MarketState(
                        signal_price=sig_price,
                        signal_side=sig_side,
                        signal_ts=sig_ts,
                        entry_price=price,
                        entry_side=side,
                    )
                continue

            st = state[mid]
            if side != st.entry_side:
                continue

            # track opposite side trade stats after entry
            st.opp_trades += 1
            if len(st.opp_prices_first_k) < max(args.momentum_k, 5):
                st.opp_prices_first_k.append(price)

            if not st.tp2x and price >= 2.0 * st.entry_price:
                st.tp2x = True
            if not st.tp3x and price >= 3.0 * st.entry_price:
                st.tp3x = True

            # update last price before cutoff
            if dt <= cutoff_time[mid]:
                st.last_price_before_cutoff = price

    # Strategy variants
    variants = {
        "base_takeprofit_2x": [],
        "takeprofit_time_stop": [],
        "momentum_gate": [],
        "momentum_gate_strict": [],
        "min_time_to_close": [],
        "liquidity_filter": [],
        "scaled_exits_2x_3x": [],
        "price_gap_filter": [],
    }

    for mid, st in state.items():
        meta = markets[mid]
        res = resolutions[mid]
        winner_token = None
        if normalize(res) == normalize(meta.answer1):
            winner_token = "token1"
        elif normalize(res) == normalize(meta.answer2):
            winner_token = "token2"
        if not winner_token:
            continue

        opp_win = st.entry_side == winner_token

        # Base takeprofit 2x
        if st.tp2x:
            pnl = compute_pnl_exit(2.0 * st.entry_price, st.entry_price, args.fee_rate)
        else:
            pnl = compute_pnl_resolution(st.entry_price, opp_win, args.fee_rate)
        variants["base_takeprofit_2x"].append(pnl)

        # Takeprofit + time stop (exit at last price before cutoff if no tp2x)
        if st.tp2x:
            pnl = compute_pnl_exit(2.0 * st.entry_price, st.entry_price, args.fee_rate)
        elif st.last_price_before_cutoff is not None:
            pnl = compute_pnl_exit(st.last_price_before_cutoff, st.entry_price, args.fee_rate)
        else:
            pnl = compute_pnl_resolution(st.entry_price, opp_win, args.fee_rate)
        variants["takeprofit_time_stop"].append(pnl)

        # Momentum-gated entry
        if st.opp_prices_first_k:
            max_first_k = max(st.opp_prices_first_k)
            if max_first_k >= st.entry_price * (1.0 + args.momentum_min_pct):
                if st.tp2x:
                    pnl = compute_pnl_exit(
                        2.0 * st.entry_price, st.entry_price, args.fee_rate
                    )
                else:
                    pnl = compute_pnl_resolution(st.entry_price, opp_win, args.fee_rate)
                variants["momentum_gate"].append(pnl)

        # Stricter momentum gate (hard-coded: K=5, +10%)
        if st.opp_prices_first_k:
            max_first_k = max(st.opp_prices_first_k)
            if len(st.opp_prices_first_k) >= 5 and max_first_k >= st.entry_price * 1.10:
                if st.tp2x:
                    pnl = compute_pnl_exit(
                        2.0 * st.entry_price, st.entry_price, args.fee_rate
                    )
                else:
                    pnl = compute_pnl_resolution(st.entry_price, opp_win, args.fee_rate)
                variants["momentum_gate_strict"].append(pnl)

        # Liquidity filter
        if st.opp_trades >= args.liquidity_min_trades:
            if st.tp2x:
                pnl = compute_pnl_exit(2.0 * st.entry_price, st.entry_price, args.fee_rate)
            else:
                pnl = compute_pnl_resolution(st.entry_price, opp_win, args.fee_rate)
            variants["liquidity_filter"].append(pnl)

        # Scaled exits: half at 2x, half at 3x, remainder to resolution
        shares = 1.0 / st.entry_price
        gross = 0.0
        if st.tp2x:
            gross += (shares * 0.5) * (2.0 * st.entry_price)
            if st.tp3x:
                gross += (shares * 0.5) * (3.0 * st.entry_price)
            else:
                # remaining half to resolution
                if opp_win:
                    gross += (shares * 0.5) * 1.0
        else:
            if opp_win:
                gross += shares * 1.0
        fee_paid = gross * args.fee_rate
        pnl = gross - 1.0 - fee_paid
        variants["scaled_exits_2x_3x"].append(pnl)

        # Price-gap filter on signal price
        if st.signal_price <= args.signal_max:
            if st.tp2x:
                pnl = compute_pnl_exit(2.0 * st.entry_price, st.entry_price, args.fee_rate)
            else:
                pnl = compute_pnl_resolution(st.entry_price, opp_win, args.fee_rate)
            variants["price_gap_filter"].append(pnl)

        # Minimum time-to-close filter (signal at least N hours before close)
        if meta.closed_time - st.signal_ts >= timedelta(hours=args.min_hours_to_close):
            if st.tp2x:
                pnl = compute_pnl_exit(2.0 * st.entry_price, st.entry_price, args.fee_rate)
            else:
                pnl = compute_pnl_resolution(st.entry_price, opp_win, args.fee_rate)
            variants["min_time_to_close"].append(pnl)

    summary = {
        "target_slugs": len(target_slugs),
        "eligible_markets": len(state),
        "fee_rate": args.fee_rate,
        "variants": {name: summarize(pnls) for name, pnls in variants.items()},
    }

    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
