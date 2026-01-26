from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .utils import parse_json_maybe, parse_iso_utc, safe_float


def _make_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class PolymarketClient:
    def __init__(
        self,
        gamma_base: str,
        clob_base: str,
        us_api_base: str,
        us_api_key: Optional[str] = None,
        us_api_secret: Optional[str] = None,
        us_api_passphrase: Optional[str] = None,
    ) -> None:
        self.gamma_base = gamma_base.rstrip("/")
        self.clob_base = clob_base.rstrip("/")
        self.us_api_base = us_api_base.rstrip("/")
        self.us_api_key = us_api_key
        self.us_api_secret = us_api_secret
        self.us_api_passphrase = us_api_passphrase
        self.session = _make_session()

    def list_closed_markets(self, limit: int = 100) -> Iterable[Dict[str, Any]]:
        offset = 0
        while True:
            params = {"closed": "true", "limit": limit, "offset": offset}
            resp = self.session.get(f"{self.gamma_base}/markets", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            for item in data:
                yield item
            offset += limit

    def get_prices_history(
        self,
        clob_token_id: str,
        start_ts: int,
        end_ts: int,
        fidelity: int,
    ) -> List[Tuple[int, float]]:
        params = {
            "market": clob_token_id,
            "startTs": start_ts,
            "endTs": end_ts,
            "fidelity": fidelity,
        }
        resp = self.session.get(f"{self.clob_base}/prices-history", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        history = data.get("history") if isinstance(data, dict) else data
        results: List[Tuple[int, float]] = []
        if not history:
            return results
        for entry in history:
            ts = entry.get("t") if isinstance(entry, dict) else None
            price = entry.get("p") if isinstance(entry, dict) else None
            if ts is None or price is None:
                continue
            price_val = safe_float(price)
            if price_val is None:
                continue
            results.append((int(ts), price_val))
        results.sort(key=lambda item: item[0])
        return results

    def get_settlement_outcome(self, slug: str) -> Optional[str]:
        if not slug:
            return None
        url = f"{self.us_api_base}/v1/markets/{slug}/settlement"
        headers = {}
        if self.us_api_key:
            headers["API-KEY"] = self.us_api_key
        if self.us_api_secret:
            headers["API-SECRET"] = self.us_api_secret
        if self.us_api_passphrase:
            headers["API-PASSPHRASE"] = self.us_api_passphrase
        try:
            resp = self.session.get(url, headers=headers, timeout=30)
            if resp.status_code == 401:
                return None
            resp.raise_for_status()
        except requests.RequestException:
            return None
        data = resp.json()
        if isinstance(data, dict):
            for key in ("outcome", "result", "resolution", "resolvedOutcome"):
                value = data.get(key)
                if isinstance(value, str):
                    return value.upper()
        return None


def parse_outcomes(market: Dict[str, Any]) -> Optional[List[str]]:
    outcomes = parse_json_maybe(market.get("outcomes"))
    if isinstance(outcomes, list):
        return [str(x) for x in outcomes]
    return None


def parse_outcome_prices(market: Dict[str, Any]) -> Optional[List[float]]:
    outcome_prices = parse_json_maybe(market.get("outcomePrices"))
    if isinstance(outcome_prices, list):
        parsed = []
        for item in outcome_prices:
            val = safe_float(item)
            if val is None:
                return None
            parsed.append(val)
        return parsed
    return None


def derive_outcome_from_market(market: Dict[str, Any]) -> Optional[str]:
    outcome_prices = parse_outcome_prices(market)
    outcomes = parse_outcomes(market)
    if not outcome_prices or not outcomes or len(outcome_prices) != len(outcomes):
        return None
    for idx, price in enumerate(outcome_prices):
        if price >= 0.99:
            return outcomes[idx].upper()
    return None


def extract_resolution_time(market: Dict[str, Any]) -> Optional[Tuple[str, int]]:
    fields = [
        "resolvedTime",
        "resolvedTimeIso",
        "resolutionTime",
        "resolutionTimeIso",
        "closedTime",
        "closedTimeIso",
        "endDate",
        "endDateIso",
    ]
    for field in fields:
        value = market.get(field)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            return field, int(value)
        if isinstance(value, str):
            parsed = parse_iso_utc(value)
            if parsed:
                return field, int(parsed.timestamp())
    return None


def extract_start_time(market: Dict[str, Any]) -> Optional[int]:
    fields = [
        "createdTime",
        "createdTimeIso",
        "createdAt",
        "createdAtIso",
        "startDate",
        "startDateIso",
    ]
    for field in fields:
        value = market.get(field)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            parsed = parse_iso_utc(value)
            if parsed:
                return int(parsed.timestamp())
    return None
