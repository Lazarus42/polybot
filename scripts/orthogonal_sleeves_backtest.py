#!/usr/bin/env python3
"""Generate residual-compatible signals for structural orthogonal sleeves.

This script looks for price dislocations that do not depend on the existing
single-market underdog families:

1. same-market complete-set underround: token1 + token2 costs less than $1;
2. duplicate-market gaps: buy cheap Yes in one market and No in the expensive
   duplicate;
3. monotonic ladder violations: for threshold ladders where one leg implies
   another, buy the broader Yes and the narrower No when the basket costs < $1.

Signals are written in the same loose component format consumed by
walk_forward_residual_portfolio.py via --external-signals. Realized returns are
audited against winner_side, so noisy duplicate/ladder grouping is penalized rather
than treated as riskless.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from realistic_underdog_account import write_csv


def parse_float_sweep(values: list[str]) -> list[float]:
    out: list[float] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                out.append(float(part))
    return sorted(set(out))


def synthetic_market_id(kind: str, left: int, right: int | None, lead_hours: float) -> int:
    text = f"{kind}|{left}|{right or 0}|{lead_hours:g}"
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=6).hexdigest()
    return 8_000_000_000 + int(digest, 16) % 900_000_000


def normalize_text(value: str) -> str:
    text = str(value).lower()
    text = text.replace("’", "'")
    text = re.sub(r"(\d+)p(?:t)?(\d+)", r"\1.\2", text)
    text = re.sub(r"[^a-z0-9.+%-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_markets(data_dir: Path, min_historical_volume: float, max_markets: int) -> pd.DataFrame:
    con = duckdb.connect()
    path = str((data_dir / "markets.parquet").resolve()).replace("'", "''")
    query = f"""
        SELECT market_id, slug, question, answer1, answer2, winner_side,
               end_date, closed_time, historical_volume
        FROM read_parquet('{path}')
        WHERE winner_side IN ('token1', 'token2')
    """
    if min_historical_volume > 0:
        query += f" AND coalesce(historical_volume, 0) >= {float(min_historical_volume)}"
    query += " ORDER BY coalesce(historical_volume, 0) DESC"
    if max_markets > 0:
        query += f" LIMIT {int(max_markets)}"
    df = con.execute(query).fetchdf()
    con.close()
    df["close"] = pd.to_datetime(df["closed_time"].fillna(df["end_date"]), utc=True, errors="coerce")
    df = df.dropna(subset=["close"]).copy()
    df["question_norm"] = df["question"].map(normalize_text)
    df["answer1_norm"] = df["answer1"].map(normalize_text)
    df["answer2_norm"] = df["answer2"].map(normalize_text)
    return df


def load_fills(data_dir: Path, market_ids: list[int]) -> pd.DataFrame:
    con = duckdb.connect()
    con.execute("CREATE TEMP TABLE sel(market_id BIGINT)")
    con.executemany("INSERT INTO sel VALUES (?)", [(int(m),) for m in market_ids])
    path = str((data_dir / "fills_sorted.parquet").resolve()).replace("'", "''")
    fills = con.execute(
        f"""
        SELECT f.market_id, f.timestamp, f.side, f.price, f.usd_amount
        FROM read_parquet('{path}') AS f JOIN sel USING (market_id)
        ORDER BY f.market_id, f.timestamp
        """
    ).fetchdf()
    con.close()
    fills["timestamp"] = pd.to_datetime(fills["timestamp"], utc=True)
    fills["side"] = fills["side"].astype(str)
    return fills


def last_side(g: pd.DataFrame, cutoff: pd.Timestamp, side: str, stale: pd.Timedelta | None) -> dict[str, Any] | None:
    window = g[(g["timestamp"] <= cutoff) & (g["side"] == side)]
    if stale is not None:
        window = window[window["timestamp"] >= cutoff - stale]
    if window.empty:
        return None
    row = window.iloc[-1]
    price = float(row["price"])
    usd = float(row["usd_amount"])
    if not math.isfinite(price) or price <= 0.0 or price >= 1.0:
        return None
    return {"price": price, "usd": usd if math.isfinite(usd) else 0.0, "timestamp": pd.Timestamp(row["timestamp"])}


def basket_capacity(prices: list[float], usd_amounts: list[float], participation: float) -> float:
    caps = [participation * usd / price for price, usd in zip(prices, usd_amounts) if price > 0 and usd > 0]
    return min(caps) if len(caps) == len(prices) and caps else 0.0


def price_regime(price: float) -> str:
    if price <= 0.05:
        return "price_01_05"
    if price <= 0.15:
        return "price_06_15"
    if price <= 0.30:
        return "price_16_30"
    if price <= 0.49:
        return "price_31_49"
    return "price_50_99"


def liquidity_regime(capital: float) -> str:
    if capital <= 2:
        return "liq_0_2"
    if capital <= 10:
        return "liq_2_10"
    if capital <= 50:
        return "liq_10_50"
    if capital <= 250:
        return "liq_50_250"
    return "liq_250_plus"


def complete_set_families(price1: float, price2: float, capacity_capital: float) -> list[str]:
    min_price = min(price1, price2)
    max_price = max(price1, price2)
    balance = "balanced" if max_price <= 0.65 else "skewed"
    return [
        "same_market_all",
        f"same_market_{price_regime(min_price)}",
        f"same_market_{balance}",
        f"same_market_{liquidity_regime(capacity_capital)}",
    ]


def pair_families(base_family: str, yes_price: float, no_price: float, capacity_capital: float) -> list[str]:
    return [
        base_family,
        f"{base_family}_{price_regime(min(yes_price, no_price))}",
        f"{base_family}_{liquidity_regime(capacity_capital)}",
    ]


def append_signal(
    rows: list[dict[str, Any]],
    *,
    kind: str,
    family: str,
    threshold: float,
    lead_hours: float,
    timestamp: pd.Timestamp,
    market_id: int,
    other_market_id: int | None,
    cost: float,
    payoff: float,
    trigger_edge: float,
    capacity_units: float,
    leg_count: int,
    metadata: dict[str, Any],
) -> None:
    if cost <= 0.0 or cost >= 1.0 or capacity_units <= 0.0:
        return
    if trigger_edge < threshold:
        return
    profit = payoff - cost
    unit_return = profit / cost
    threshold_bp = int(round(threshold * 10000))
    rows.append({
        "timestamp": timestamp,
        "market_id": synthetic_market_id(kind, market_id, other_market_id, lead_hours),
        "source_market_id": market_id,
        "other_market_id": other_market_id,
        "strategy": f"{kind}_{family}_{threshold_bp:04d}bp",
        "exit_rule": f"lead_{lead_hours:g}h_structural_basket",
        "sleeve": "orthogonal",
        "category": f"orthogonal_{kind}",
        "horizon_days": lead_hours / 24.0,
        "unit_return": unit_return,
        "edge": trigger_edge,
        "realized_profit_per_unit": profit,
        "basket_cost": cost,
        "basket_payoff": payoff,
        "entry_fill_usd": capacity_units * cost,
        "leg_count": leg_count,
        "threshold": threshold,
        "lead_hours": lead_hours,
        **metadata,
    })


def complete_set_signals(
    markets: pd.DataFrame,
    fills_by_market: dict[int, pd.DataFrame],
    lead_hours: list[float],
    thresholds: list[float],
    stale: pd.Timedelta | None,
    participation: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in markets.itertuples(index=False):
        mid = int(row.market_id)
        g = fills_by_market.get(mid)
        if g is None:
            continue
        for lead in lead_hours:
            cutoff = pd.Timestamp(row.close) - pd.Timedelta(hours=lead)
            one = last_side(g, cutoff, "token1", stale)
            two = last_side(g, cutoff, "token2", stale)
            if one is None or two is None:
                continue
            cost = one["price"] + two["price"]
            payoff = 1.0
            trigger_edge = 1.0 - cost
            cap = basket_capacity([one["price"], two["price"]], [one["usd"], two["usd"]], participation)
            families = complete_set_families(one["price"], two["price"], cap * cost)
            for threshold in thresholds:
                for family in families:
                    append_signal(
                        rows,
                        kind="complete_set_underround",
                        family=family,
                        threshold=threshold,
                        lead_hours=lead,
                        timestamp=cutoff,
                        market_id=mid,
                        other_market_id=None,
                    cost=cost,
                    payoff=payoff,
                    trigger_edge=trigger_edge,
                    capacity_units=cap,
                        leg_count=2,
                        metadata={
                            "question": row.question,
                            "price_token1": one["price"],
                            "price_token2": two["price"],
                        },
                    )
    return rows


def duplicate_pairs(markets: pd.DataFrame, max_group_size: int) -> list[tuple[pd.Series, pd.Series, str]]:
    pairs = []
    grouped = markets.groupby(["question_norm", "answer1_norm", "answer2_norm"], sort=False)
    for key, group in grouped:
        if len(group) < 2 or len(group) > max_group_size:
            continue
        ordered = group.sort_values("historical_volume", ascending=False)
        records = [r for _, r in ordered.iterrows()]
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                pairs.append((records[i], records[j], "exact_question"))
    return pairs


LADDER_PATTERNS = [
    ("more_than", "decreasing", re.compile(r"\b(more than|by more than|over|above|at least|minimum of)\s+\$?([0-9]+(?:\.[0-9]+)?)")),
    ("less_than", "increasing", re.compile(r"\b(less than|under|below|fewer than|at most|maximum of)\s+\$?([0-9]+(?:\.[0-9]+)?)")),
    ("plus_suffix", "decreasing", re.compile(r"\b([0-9]+(?:\.[0-9]+)?)\s*\+")),
]


def ladder_key(question: str, slug: str, answer1: str) -> tuple[str, str, str, float] | None:
    qtext = normalize_text(question)
    stext = normalize_text(slug)
    for family, direction, pattern in LADDER_PATTERNS:
        text = qtext if pattern.search(qtext) else stext
        match = pattern.search(text)
        if not match:
            continue
        num_text = match.group(2) if family != "plus_suffix" else match.group(1)
        try:
            threshold = float(num_text)
        except ValueError:
            continue
        base = pattern.sub(" {x} ", text)
        base = re.sub(r"\s+", " ", base).strip()
        # Include answer1 so "candidate gets X" ladders do not merge across outcomes.
        return family, direction, f"{base}|answer1:{normalize_text(answer1)}", threshold
    return None


def ladder_pairs(markets: pd.DataFrame, max_group_size: int) -> list[tuple[pd.Series, pd.Series, str]]:
    enriched = []
    for _, row in markets.iterrows():
        parsed = ladder_key(str(row["question"]), str(row["slug"]), str(row["answer1"]))
        if parsed is None:
            continue
        family, direction, key, threshold = parsed
        item = row.copy()
        item["_ladder_family"] = family
        item["_ladder_direction"] = direction
        item["_ladder_key"] = key
        item["_ladder_threshold"] = threshold
        enriched.append(item)
    if not enriched:
        return []
    df = pd.DataFrame(enriched)
    pairs = []
    for (_, direction, key), group in df.groupby(["_ladder_family", "_ladder_direction", "_ladder_key"], sort=False):
        if len(group) < 2 or len(group) > max_group_size:
            continue
        ordered = group.sort_values("_ladder_threshold")
        records = [r for _, r in ordered.iterrows()]
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                low, high = records[i], records[j]
                if direction == "decreasing":
                    broad, narrow = low, high
                else:
                    broad, narrow = high, low
                pairs.append((broad, narrow, str(broad["_ladder_family"])))
    return pairs


def pair_signals(
    pairs: list[tuple[pd.Series, pd.Series, str]],
    kind: str,
    markets_by_id: dict[int, pd.Series],
    fills_by_market: dict[int, pd.DataFrame],
    lead_hours: list[float],
    thresholds: list[float],
    stale: pd.Timedelta | None,
    participation: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left, right, family in pairs:
        left_id = int(left["market_id"])
        right_id = int(right["market_id"])
        left_fills = fills_by_market.get(left_id)
        right_fills = fills_by_market.get(right_id)
        if left_fills is None or right_fills is None:
            continue
        for lead in lead_hours:
            cutoff = min(pd.Timestamp(left["close"]), pd.Timestamp(right["close"])) - pd.Timedelta(hours=lead)
            left_yes = last_side(left_fills, cutoff, "token1", stale)
            right_yes = last_side(right_fills, cutoff, "token1", stale)
            left_no = last_side(left_fills, cutoff, "token2", stale)
            right_no = last_side(right_fills, cutoff, "token2", stale)
            if left_yes is None or right_yes is None or left_no is None or right_no is None:
                continue

            if kind == "duplicate_gap":
                # Try both directions; buy cheap Yes and No on expensive duplicate.
                candidates = [
                    (left, right, left_yes, right_no, left_yes["price"] + right_no["price"]),
                    (right, left, right_yes, left_no, right_yes["price"] + left_no["price"]),
                ]
            else:
                # left=broad, right=narrow. Buy broad Yes and narrow No.
                candidates = [(left, right, left_yes, right_no, left_yes["price"] + right_no["price"])]

            for buy_yes, buy_no, yes_quote, no_quote, cost in candidates:
                yes_win = str(buy_yes["winner_side"]) == "token1"
                no_win = str(buy_no["winner_side"]) == "token2"
                payoff = float(yes_win) + float(no_win)
                # Ex-ante trigger assumes the parsed structural relation is true; realized
                # payoff below audits noisy duplicate/ladder grouping without leaking it
                # into signal inclusion.
                trigger_edge = 1.0 - cost
                cap = basket_capacity([yes_quote["price"], no_quote["price"]], [yes_quote["usd"], no_quote["usd"]], participation)
                for threshold in thresholds:
                    for expanded_family in pair_families(family, yes_quote["price"], no_quote["price"], cap * cost):
                        append_signal(
                            rows,
                            kind=kind,
                            family=expanded_family,
                            threshold=threshold,
                            lead_hours=lead,
                            timestamp=cutoff,
                            market_id=int(buy_yes["market_id"]),
                            other_market_id=int(buy_no["market_id"]),
                        cost=cost,
                        payoff=payoff,
                        trigger_edge=trigger_edge,
                        capacity_units=cap,
                            leg_count=2,
                            metadata={
                                "yes_question": buy_yes["question"],
                                "no_question": buy_no["question"],
                                "yes_price": yes_quote["price"],
                                "no_price": no_quote["price"],
                            },
                        )
    return rows


def summarize(signals: pd.DataFrame) -> list[dict[str, Any]]:
    if signals.empty:
        return []
    df = signals.copy()
    df["month"] = pd.to_datetime(df["timestamp"], utc=True).dt.strftime("%Y-%m")
    rows = []
    for (strategy, exit_rule), g in df.groupby(["strategy", "exit_rule"], sort=False):
        profit_per_dollar = g["unit_return"].to_numpy(dtype=float)
        rows.append({
            "strategy": strategy,
            "exit_rule": exit_rule,
            "months": int(g["month"].nunique()),
            "signals": int(len(g)),
            "mean_unit_return": float(np.mean(profit_per_dollar)),
            "median_unit_return": float(np.median(profit_per_dollar)),
            "positive_rate": float(np.mean(profit_per_dollar > 0)),
            "mean_entry_fill_usd": float(g["entry_fill_usd"].mean()),
            "total_capacity_usd": float(g["entry_fill_usd"].sum()),
            "mean_edge": float(g["edge"].mean()),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("archive/processed/underdog_events"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/orthogonal_sleeves"))
    parser.add_argument("--lead-hours", type=float, nargs="+", default=[2.0, 24.0, 72.0, 168.0])
    parser.add_argument("--thresholds", nargs="+", default=["0.0025", "0.005", "0.01", "0.02", "0.05"])
    parser.add_argument("--max-staleness-hours", type=float, default=48.0)
    parser.add_argument("--participation-fraction", type=float, default=0.10)
    parser.add_argument("--min-historical-volume", type=float, default=0.0)
    parser.add_argument("--max-markets", type=int, default=0)
    parser.add_argument("--max-duplicate-group-size", type=int, default=8)
    parser.add_argument("--max-ladder-group-size", type=int, default=12)
    parser.add_argument("--min-entry-fill", type=float, default=0.25,
                        help="Drop structural baskets with less executable capacity than this.")
    parser.add_argument("--max-unit-return", type=float, default=5.0,
                        help="Drop extreme unit-return outliers; use 0 to disable.")
    parser.add_argument("--families", nargs="+",
                        default=["complete_set_underround", "duplicate_gap", "ladder_violation"],
                        choices=["complete_set_underround", "duplicate_gap", "ladder_violation"])
    args = parser.parse_args()

    thresholds = parse_float_sweep(args.thresholds)
    stale = pd.Timedelta(hours=args.max_staleness_hours) if args.max_staleness_hours > 0 else None
    markets = load_markets(args.data_dir, args.min_historical_volume, args.max_markets)
    fills = load_fills(args.data_dir, markets["market_id"].astype(int).tolist())
    fills_by_market = {int(mid): g for mid, g in fills.groupby("market_id", sort=False)}
    markets_by_id = {int(row.market_id): row for row in markets.itertuples(index=False)}

    frames: list[pd.DataFrame] = []
    diagnostics: dict[str, Any] = {
        "markets": int(len(markets)),
        "families": args.families,
        "lead_hours": args.lead_hours,
        "thresholds": thresholds,
    }
    if "complete_set_underround" in args.families:
        rows = complete_set_signals(markets, fills_by_market, args.lead_hours, thresholds, stale, args.participation_fraction)
        diagnostics["complete_set_signals"] = len(rows)
        frames.append(pd.DataFrame(rows))
    if "duplicate_gap" in args.families:
        pairs = duplicate_pairs(markets, args.max_duplicate_group_size)
        rows = pair_signals(pairs, "duplicate_gap", markets_by_id, fills_by_market, args.lead_hours, thresholds, stale, args.participation_fraction)
        diagnostics["duplicate_pairs"] = len(pairs)
        diagnostics["duplicate_signals"] = len(rows)
        frames.append(pd.DataFrame(rows))
    if "ladder_violation" in args.families:
        pairs = ladder_pairs(markets, args.max_ladder_group_size)
        rows = pair_signals(pairs, "ladder_violation", markets_by_id, fills_by_market, args.lead_hours, thresholds, stale, args.participation_fraction)
        diagnostics["ladder_pairs"] = len(pairs)
        diagnostics["ladder_signals"] = len(rows)
        frames.append(pd.DataFrame(rows))

    nonempty = [f for f in frames if not f.empty]
    signals = pd.concat(nonempty, ignore_index=True, sort=False) if nonempty else pd.DataFrame()
    if not signals.empty:
        signals = signals[signals["entry_fill_usd"].astype(float) >= args.min_entry_fill].copy()
        if args.max_unit_return and args.max_unit_return > 0:
            signals = signals[signals["unit_return"].astype(float) <= args.max_unit_return].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    signals.to_csv(args.output_dir / "orthogonal_signals.csv", index=False)
    summary = summarize(signals)
    write_csv(args.output_dir / "orthogonal_summary.csv", summary)
    (args.output_dir / "summary.json").write_text(json.dumps({
        **diagnostics,
        "signals": int(len(signals)),
        "summary_rows": len(summary),
        "files": ["orthogonal_signals.csv", "orthogonal_summary.csv"],
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), **diagnostics, "signals": int(len(signals))}, indent=2))


if __name__ == "__main__":
    main()
