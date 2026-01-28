#!/usr/bin/env python3
"""
Fetch Polymarket market resolutions for slugs read from a newline-separated .txt file.

Usage:
  python scripts/build_resolutions.py --slugs path/to/slugs.txt --output path/to/out.csv
"""

import argparse
import csv
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import requests


BASE = "https://gamma-api.polymarket.com"
SLEEP_S = 0.15
TIMEOUT_S = 20


def load_slugs(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        slugs = [line.strip() for line in f]
    return [s for s in slugs if s and not s.startswith("#")]


def parse_maybe_json_list(x: Any) -> Optional[List[Any]]:
    if x is None:
        return None
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                v = json.loads(s)
                return v if isinstance(v, list) else None
            except json.JSONDecodeError:
                return None
    return None


def to_float_list(xs: List[Any]) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for v in xs:
        try:
            out.append(float(v))
        except Exception:
            out.append(None)
    return out


def infer_resolution_from_payouts(
    outcomes: List[Any], prices: List[Optional[float]]
) -> Tuple[Optional[str], str]:
    if not outcomes or not prices:
        return None, "missing outcomes/prices"

    ones = [i for i, p in enumerate(prices) if p == 1.0]

    if len(ones) == 1:
        i = ones[0]
        if i < len(outcomes):
            return str(outcomes[i]), ""
        return f"index_{i}", "winner index beyond outcomes length"

    if len(ones) > 1:
        return None, f"multiple payouts==1: idx={ones}"

    return None, "no payout==1 found"


def fetch_market_by_slug(
    session: requests.Session, slug: str
) -> Tuple[Optional[Dict[str, Any]], str]:
    url = f"{BASE}/markets/slug/{slug}"
    try:
        r = session.get(url, timeout=TIMEOUT_S)
    except requests.RequestException as e:
        return None, f"request_error: {e}"

    if r.status_code == 404:
        return None, "404_not_found"
    if r.status_code != 200:
        return None, f"http_{r.status_code}"

    try:
        data = r.json()
    except ValueError:
        return None, "json_decode_error"

    if not isinstance(data, dict) or not data:
        return None, "empty_payload"

    return data, ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build resolutions CSV from slug list.")
    parser.add_argument("--slugs", required=True, help="Path to slug list (.txt)")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    args = parser.parse_args()

    slugs = load_slugs(args.slugs)
    if not slugs:
        print("No slugs found in file.")
        return

    session = requests.Session()
    session.headers.update({"User-Agent": "polymarket-resolution-fetch/1.0"})

    rows: List[Dict[str, Any]] = []

    for slug in slugs:
        m, err = fetch_market_by_slug(session, slug)

        if m is None:
            rows.append(
                {
                    "slug": slug,
                    "status": "INVALID",
                    "resolution": "",
                    "market_title": "",
                    "market_id": "",
                    "closed": "",
                    "endDate": "",
                    "resolutionSource": "",
                    "notes": err,
                }
            )
            time.sleep(SLEEP_S)
            continue

        title = m.get("question") or m.get("title") or ""
        market_id = m.get("id") or ""
        closed = m.get("closed")
        end_date = m.get("endDate") or ""
        resolution_source = m.get("resolutionSource") or m.get("resolvedBy") or ""

        if closed is False or closed is None:
            rows.append(
                {
                    "slug": slug,
                    "status": "UNRESOLVED",
                    "resolution": "",
                    "market_title": title,
                    "market_id": market_id,
                    "closed": closed,
                    "endDate": end_date,
                    "resolutionSource": resolution_source,
                    "notes": "",
                }
            )
            time.sleep(SLEEP_S)
            continue

        outcomes = parse_maybe_json_list(m.get("outcomes")) or []
        prices_raw = parse_maybe_json_list(m.get("outcomePrices")) or []
        prices = to_float_list(prices_raw)

        winner, note = infer_resolution_from_payouts(outcomes, prices)

        rows.append(
            {
                "slug": slug,
                "status": "RESOLVED" if winner else "RESOLVED_UNKNOWN",
                "resolution": winner or "",
                "market_title": title,
                "market_id": market_id,
                "closed": closed,
                "endDate": end_date,
                "resolutionSource": resolution_source,
                "notes": note,
            }
        )

        time.sleep(SLEEP_S)

    fieldnames = [
        "slug",
        "status",
        "resolution",
        "market_title",
        "market_id",
        "closed",
        "endDate",
        "resolutionSource",
        "notes",
    ]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Done. Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
