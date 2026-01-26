from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from .api import (
    PolymarketClient,
    derive_outcome_from_market,
    extract_resolution_time,
    extract_start_time,
    parse_outcomes,
    parse_outcome_prices,
)
from .utils import parse_date_utc, parse_json_maybe


@dataclass
class BacktestConfig:
    start_date: datetime
    end_date: datetime
    fee_rate: float
    fidelity: int
    output_dir: str
    gamma_base: str
    clob_base: str
    us_api_base: str
    us_api_key: Optional[str] = None
    us_api_secret: Optional[str] = None
    us_api_passphrase: Optional[str] = None


@dataclass
class TradeRecord:
    market_id: str
    slug: str
    title: str
    category: str
    resolution_ts: int
    resolution_field: str
    entry_ts: int
    entry_price: float
    bought_outcome: str
    resolution_outcome: str
    shares: float
    fee_paid: float
    pnl: float


def _first_hit(history: List[Tuple[int, float]], threshold: float) -> Optional[Tuple[int, float]]:
    for ts, price in history:
        if price >= threshold:
            return ts, price
    return None


def _normalize_outcome(value: str) -> str:
    return value.strip().upper()


def _select_resolution_outcome(
    client: PolymarketClient, market: Dict[str, Any]
) -> Tuple[Optional[str], str]:
    slug = market.get("slug") or ""
    outcome = client.get_settlement_outcome(slug)
    if outcome:
        return _normalize_outcome(outcome), "us_api_settlement"
    derived = derive_outcome_from_market(market)
    if derived:
        return _normalize_outcome(derived), "gamma_outcomePrices"
    return None, "missing"


def _extract_category(market: Dict[str, Any]) -> str:
    category = market.get("category") or market.get("categorySlug")
    if category:
        return str(category)
    event = market.get("event")
    if isinstance(event, dict):
        category = event.get("category")
        if category:
            return str(category)
    return "unknown"


def run_backtest(config: BacktestConfig) -> Dict[str, Any]:
    client = PolymarketClient(
        gamma_base=config.gamma_base,
        clob_base=config.clob_base,
        us_api_base=config.us_api_base,
        us_api_key=config.us_api_key,
        us_api_secret=config.us_api_secret,
        us_api_passphrase=config.us_api_passphrase,
    )

    start_ts = int(config.start_date.replace(tzinfo=timezone.utc).timestamp())
    end_ts = int(config.end_date.replace(tzinfo=timezone.utc).timestamp())

    trades: List[TradeRecord] = []
    issues: List[Dict[str, Any]] = []
    counters = {
        "markets_seen": 0,
        "binary_markets": 0,
        "in_window": 0,
        "triggered": 0,
        "no_trigger": 0,
        "skipped_missing_resolution": 0,
        "skipped_missing_history": 0,
        "skipped_tie": 0,
        "skipped_missing_outcome": 0,
    }

    for market in client.list_closed_markets():
        counters["markets_seen"] += 1

        outcomes = parse_outcomes(market)
        if not outcomes or len(outcomes) != 2:
            continue

        clob_token_ids = parse_json_maybe(market.get("clobTokenIds"))
        if not isinstance(clob_token_ids, list) or len(clob_token_ids) != 2:
            continue

        counters["binary_markets"] += 1

        res_info = extract_resolution_time(market)
        if not res_info:
            issues.append(
                {
                    "market_id": market.get("id"),
                    "slug": market.get("slug"),
                    "title": market.get("question") or market.get("title"),
                    "issue": "missing_resolution_time",
                }
            )
            counters["skipped_missing_resolution"] += 1
            continue

        res_field, resolution_ts = res_info
        if resolution_ts < start_ts or resolution_ts > end_ts:
            continue

        counters["in_window"] += 1

        start_time = extract_start_time(market)
        if start_time is None:
            start_time = max(resolution_ts - 365 * 86400, start_ts - 365 * 86400)
        history_yes = client.get_prices_history(
            clob_token_ids[0], start_time, resolution_ts, config.fidelity
        )
        history_no = client.get_prices_history(
            clob_token_ids[1], start_time, resolution_ts, config.fidelity
        )
        if not history_yes or not history_no:
            issues.append(
                {
                    "market_id": market.get("id"),
                    "slug": market.get("slug"),
                    "title": market.get("question") or market.get("title"),
                    "issue": "missing_price_history",
                }
            )
            counters["skipped_missing_history"] += 1
            continue

        hit_yes = _first_hit(history_yes, 0.9)
        hit_no = _first_hit(history_no, 0.9)
        if not hit_yes and not hit_no:
            counters["no_trigger"] += 1
            continue

        if hit_yes and hit_no and hit_yes[0] == hit_no[0]:
            issues.append(
                {
                    "market_id": market.get("id"),
                    "slug": market.get("slug"),
                    "title": market.get("question") or market.get("title"),
                    "issue": "tie_at_90_percent",
                    "timestamp": hit_yes[0],
                }
            )
            counters["skipped_tie"] += 1
            continue

        if hit_no and (not hit_yes or hit_no[0] < hit_yes[0]):
            entry_ts, entry_price = hit_no
            bought_idx = 1
        else:
            entry_ts, entry_price = hit_yes  # type: ignore[misc]
            bought_idx = 0

        bought_outcome = _normalize_outcome(outcomes[bought_idx])
        resolution_outcome, outcome_source = _select_resolution_outcome(client, market)
        if not resolution_outcome:
            issues.append(
                {
                    "market_id": market.get("id"),
                    "slug": market.get("slug"),
                    "title": market.get("question") or market.get("title"),
                    "issue": "missing_resolution_outcome",
                    "source": outcome_source,
                }
            )
            counters["skipped_missing_outcome"] += 1
            continue

        shares = 1.0 / entry_price
        if bought_outcome == resolution_outcome:
            gross = shares * 1.0
            fee_paid = gross * config.fee_rate
            pnl = gross - 1.0 - fee_paid
        else:
            fee_paid = 0.0
            pnl = -1.0

        counters["triggered"] += 1

        trades.append(
            TradeRecord(
                market_id=str(market.get("id") or ""),
                slug=str(market.get("slug") or ""),
                title=str(market.get("question") or market.get("title") or ""),
                category=_extract_category(market),
                resolution_ts=resolution_ts,
                resolution_field=res_field,
                entry_ts=entry_ts,
                entry_price=entry_price,
                bought_outcome=bought_outcome,
                resolution_outcome=resolution_outcome,
                shares=shares,
                fee_paid=fee_paid,
                pnl=pnl,
            )
        )

    return {
        "config": {
            "start_date": config.start_date.isoformat(),
            "end_date": config.end_date.isoformat(),
            "fee_rate": config.fee_rate,
            "fidelity": config.fidelity,
            "gamma_base": config.gamma_base,
            "clob_base": config.clob_base,
            "us_api_base": config.us_api_base,
        },
        "counters": counters,
        "trades": trades,
        "issues": issues,
    }


def load_config_from_env(env: Dict[str, str]) -> BacktestConfig:
    start = parse_date_utc(env.get("PM_START_DATE", "2025-07-25"))
    end = parse_date_utc(env.get("PM_END_DATE", "2026-01-25")) + timedelta(days=1) - timedelta(
        seconds=1
    )
    fee_rate = float(env.get("PM_FEE_RATE", "0.02"))
    fidelity = int(env.get("PM_FIDELITY", "5"))
    output_dir = env.get("PM_OUTPUT_DIR", "reports")
    gamma_base = env.get("PM_GAMMA_BASE", "https://gamma-api.polymarket.com")
    clob_base = env.get("PM_CLOB_BASE", "https://clob.polymarket.com")
    us_api_base = env.get("PM_US_API_BASE", "https://api.polymarket.us")
    return BacktestConfig(
        start_date=start,
        end_date=end,
        fee_rate=fee_rate,
        fidelity=fidelity,
        output_dir=output_dir,
        gamma_base=gamma_base,
        clob_base=clob_base,
        us_api_base=us_api_base,
        us_api_key=env.get("PM_US_API_KEY"),
        us_api_secret=env.get("PM_US_API_SECRET"),
        us_api_passphrase=env.get("PM_US_API_PASSPHRASE"),
    )
