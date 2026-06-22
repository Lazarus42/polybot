#!/usr/bin/env python3
"""Recover event groups from markets and measure the cross-market opportunity surface.

The pipeline currently treats every binary market as its own event (`event_cluster_id`
== `market_id`, 100% singletons), so cross-market relational strategies can only see the
within-market token1+token2=1 constraint. Most markets are actually outcomes of a larger
event: in the tradeable universe ~52% of markets share a `ticker` with >=1 other market.

This builds proper event groups and measures where their prices violate the no-arbitrage
constraints — the actual opportunity surface for the one edge type the efficiency tests
found is NOT fully priced (cross-market relational).

Grouping (pure, `build_event_groups`): event_id = ticker, falling back to a slug-stem for
markets with a missing/unique ticker. Each group is classified:
  - `ladder`      : >=3 markets with parseable, distinct numeric thresholds and a
                    directional word (above/over/>=/below/under) -> YES price must be
                    monotone in the threshold.
  - `categorical` : a multi-outcome "who wins" event -> YES prices should sum to ~1
                    (sum>1 overround = sell basket; sum<1 underround = buy complete set).
  - `singleton`   : alone in its group (no cross-market relation).

Constraint math (pure, tested): `sum_to_one_overround`, `ladder_monotonicity_breaks`.

Prices: per market we need a representative YES (token1) probability. `load_yes_prices_duckdb`
reads the fill tape (needs the project venv) and takes the median token1 price per market.
Run grouping anywhere; run the violation measurement in the venv.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_NUM = re.compile(r"\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(k|m|bn|billion|million|thousand)?", re.I)
_MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "bn": 1e9, "billion": 1e9, "": 1.0}
_DIRECTIONAL = re.compile(r"\b(above|over|greater|more than|at least|>=|>|below|under|less than|<=|<|reach|hit|exceed)\b", re.I)


def parse_threshold(text: str) -> float | None:
    """Extract the first numeric threshold (with k/m/bn units) from a question string."""
    if not isinstance(text, str):
        return None
    m = _NUM.search(text.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", "")) * _MULT[(m.group(2) or "").lower()]
    except (ValueError, KeyError):
        return None


def _skeleton_tokens(question: str) -> set[str]:
    """Tokenize a question with numbers masked to <NUM>, for template comparison."""
    q = str(question).lower()
    q = re.sub(r"\$?\d[\d,]*(?:\.\d+)?\s*%?", "<NUM>", q)
    return set(re.findall(r"<NUM>|[a-z]+", q))


def template_exclusivity_score(questions: list[str], min_share: float = 0.8) -> float:
    """Fraction of a typical question that is a SHARED skeleton across the group.

    True mutually-exclusive sets (price/temp buckets, "Will <X> win <same event>?",
    rate-decision buckets) differ only in a number or a single entity, so they share most
    of their tokens -> high score. Thematic bundles of independent markets (a UFC card's
    separate fights, an NFL week's separate games) share almost nothing -> low score.
    """
    toksets = [_skeleton_tokens(q) for q in questions if isinstance(q, str) and q]
    if len(toksets) < 2:
        return 0.0
    from collections import Counter
    cnt: Counter = Counter()
    for ts in toksets:
        cnt.update(ts)
    n = len(toksets)
    shared = {t for t, c in cnt.items() if c >= min_share * n and t != "<NUM>"}
    mean_len = float(np.mean([len(ts) for ts in toksets]))
    return len(shared) / max(mean_len, 1.0)


def classify_group(questions: list[str], min_ladder: int = 3,
                   exclusivity_thresh: float = 0.5) -> str:
    """Classify an event group: singleton / ladder / categorical / thematic.

    `categorical` (sum-to-one applies) requires a shared question template, so thematic
    groupings of independent markets are split off as `thematic` and excluded from the
    no-arbitrage measurement.
    """
    if len(questions) <= 1:
        return "singleton"
    thrs = [parse_threshold(q) for q in questions]
    n_thr = sum(t is not None for t in thrs)
    n_distinct = len({t for t in thrs if t is not None})
    directional = sum(bool(_DIRECTIONAL.search(q or "")) for q in questions)
    if (n_thr >= max(min_ladder, int(0.7 * len(questions)))
            and n_distinct >= min_ladder and directional >= max(2, int(0.5 * len(questions)))):
        return "ladder"
    if template_exclusivity_score(questions) >= exclusivity_thresh:
        return "categorical"
    return "thematic"


def slug_stem(slug: str) -> str:
    s = str(slug)
    s = re.sub(r"-(yes|no|over|under|above|below|\d{4,}|\d+(?:k|m|bn)?)$", "", s, flags=re.I)
    return s


def cohort_by_overlap(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Split a set of [start, end] live-intervals into cohorts of overlapping markets.

    Markets are in the same cohort iff their live intervals form a connected overlapping
    chain. A simultaneous categorical (all candidates live at once) collapses to one
    cohort; a recurring time-series (disjoint 15-min windows sharing a ticker) splits into
    one cohort per window. This is what de-merges the artifactual mega-tickers.
    """
    starts = np.asarray(starts, float)
    ends = np.asarray(ends, float)
    order = np.argsort(starts, kind="mergesort")
    cohort = np.empty(len(starts), dtype=int)
    c = -1
    cur_end = -np.inf
    for k in order:
        if starts[k] <= cur_end:                 # overlaps the running chain
            cohort[k] = c
            cur_end = max(cur_end, ends[k])
        else:
            c += 1
            cohort[k] = c
            cur_end = ends[k]
    return cohort


def _to_epoch(series: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(series, utc=True, errors="coerce")
    arr = dt.to_numpy(dtype="datetime64[ns]")               # tz dropped, values in UTC
    out = arr.astype("int64").astype(float) / 1e9
    out[np.isnat(arr)] = np.nan
    return out


def build_event_groups(markets: pd.DataFrame, min_ladder: int = 3,
                       use_time: bool = True) -> pd.DataFrame:
    """Map each market to an event_id + event_type. Expects columns id, question,
    market_slug, ticker. If `use_time` and createdAt/closedTime are present, ticker groups
    are further split into temporally co-live cohorts (and a per-cohort snapshot time and
    live-interval are attached for contemporaneous price evaluation)."""
    m = markets.copy()
    m["id"] = m["id"].astype(str)
    tick = m["ticker"].astype(str).replace({"": np.nan, "nan": np.nan})
    tick_counts = tick.value_counts()
    multi_tickers = set(tick_counts[tick_counts > 1].index)
    stem = m["market_slug"].map(slug_stem)
    stem_counts = stem.value_counts()
    multi_stems = set(stem_counts[stem_counts > 1].index)

    def base_event(row_tick, row_stem, row_id):
        if isinstance(row_tick, str) and row_tick in multi_tickers:
            return f"tk:{row_tick}"
        if row_stem in multi_stems:
            return f"st:{row_stem}"
        return f"id:{row_id}"

    m["base_event"] = [base_event(t, s, i) for t, s, i in zip(tick, stem, m["id"])]

    have_time = use_time and "createdAt" in m.columns and "closedTime" in m.columns
    if have_time:
        m["start_ts"] = _to_epoch(m["createdAt"])
        m["end_ts"] = _to_epoch(m["closedTime"])
        # markets with unknown dates can't be cohorted -> own cohort
        m["event_id"] = m["base_event"]
        for ev, g in m.groupby("base_event"):
            if len(g) <= 1 or g[["start_ts", "end_ts"]].isna().any(axis=None):
                continue
            ch = cohort_by_overlap(g["start_ts"].to_numpy(), g["end_ts"].to_numpy())
            m.loc[g.index, "event_id"] = [f"{ev}#{c}" for c in ch]
        # per-cohort contemporaneous snapshot = midpoint of the common-overlap window
        snap = {}
        for ev, g in m.groupby("event_id"):
            s, e = g["start_ts"].to_numpy(), g["end_ts"].to_numpy()
            if np.isnan(s).any() or np.isnan(e).any():
                snap[ev] = np.nan
                continue
            lo, hi = np.nanmax(s), np.nanmin(e)          # all members live in [lo, hi]
            snap[ev] = (lo + hi) / 2 if hi >= lo else float(np.median((s + e) / 2))
        m["snapshot_ts"] = m["event_id"].map(snap)
    else:
        m["event_id"] = m["base_event"]
        m["snapshot_ts"] = np.nan

    qmap = m.groupby("event_id")["question"].apply(list)
    types = {ev: classify_group(qs, min_ladder) for ev, qs in qmap.items()}
    m["event_type"] = m["event_id"].map(types)
    m["threshold"] = m["question"].map(parse_threshold)
    m["event_size"] = m.groupby("event_id")["id"].transform("size")
    cols = ["id", "event_id", "event_type", "threshold", "event_size", "question", "ticker"]
    if have_time:
        cols += ["snapshot_ts", "start_ts", "end_ts"]
    return m[cols]


def sum_to_one_overround(yes_prices: list[float]) -> float:
    """Categorical event: sum of mutually-exclusive YES prices minus 1.
    >0 overround (basket too expensive); <0 underround (complete set cheap)."""
    ys = [p for p in yes_prices if p is not None and np.isfinite(p)]
    return float(sum(ys) - 1.0) if ys else float("nan")


def ladder_monotonicity_breaks(thresholds: list[float], yes_prices: list[float]) -> dict[str, float]:
    """Ladder event: YES('> threshold') must be non-increasing as threshold rises.
    Returns the number of monotonicity breaks and the largest violating gap."""
    pairs = [(t, p) for t, p in zip(thresholds, yes_prices)
             if t is not None and p is not None and np.isfinite(t) and np.isfinite(p)]
    pairs.sort(key=lambda x: x[0])
    breaks = 0
    max_gap = 0.0
    for (t0, p0), (t1, p1) in zip(pairs, pairs[1:]):
        if p1 > p0 + 1e-9:                      # higher strike priced higher = arb
            breaks += 1
            max_gap = max(max_gap, p1 - p0)
    return {"n_breaks": breaks, "max_gap": float(max_gap), "n_rungs": len(pairs)}


def load_yes_prices_asof_duckdb(fills_path: Path, snaps: pd.DataFrame,
                                max_stale_hours: float = 48.0) -> dict[str, float]:
    """Contemporaneous YES price per market: the last token1 trade at or before the
    market's event snapshot time (fallback 1 - last token2). `snaps` has columns
    market_id (str) and snapshot_ts (epoch seconds). Uses an ASOF join so every market in
    an event is priced AT THE SAME instant. A price is dropped if the last trade is older
    than `max_stale_hours` before the snapshot (a stale quote is not a real probability),
    so the sum-to-one is over genuinely live, contemporaneous outcomes.
    """
    import duckdb  # noqa: PLC0415

    con = duckdb.connect()
    con.execute("SET threads = 4")
    p = str(fills_path.resolve()).replace("'", "''")
    q = snaps.dropna(subset=["snapshot_ts"]).copy()
    q["market_id"] = q["market_id"].astype("int64")
    con.register("snaps", q[["market_id", "snapshot_ts"]])
    con.execute(f"""
        CREATE TEMP TABLE tape AS
        SELECT market_id, epoch(timestamp) AS ts, price,
               CASE WHEN side = 'token1' THEN 1 ELSE 0 END AS is_yes
        FROM read_parquet('{p}') WHERE price > 0 AND price < 1
    """)
    max_age = max_stale_hours * 3600.0

    def asof(is_yes: int) -> pd.DataFrame:
        return con.execute(f"""
            SELECT s.market_id, t.price AS px, (s.snapshot_ts - t.ts) AS age
            FROM snaps s ASOF LEFT JOIN (SELECT * FROM tape WHERE is_yes = {is_yes}) t
            ON s.market_id = t.market_id AND t.ts <= s.snapshot_ts
        """).fetch_df()
    yes_df = asof(1).rename(columns={"px": "yes", "age": "yes_age"})
    no_df = asof(0).rename(columns={"px": "no", "age": "no_age"})
    merged = yes_df.merge(no_df, on="market_id", how="outer")
    out: dict[str, float] = {}
    for _, r in merged.iterrows():
        if pd.notna(r["yes"]) and pd.notna(r["yes_age"]) and r["yes_age"] <= max_age:
            out[str(int(r["market_id"]))] = float(r["yes"])
        elif pd.notna(r["no"]) and pd.notna(r["no_age"]) and r["no_age"] <= max_age:
            out[str(int(r["market_id"]))] = 1.0 - float(r["no"])
    return out


def measure_opportunity_surface(groups: pd.DataFrame, yes_prices: dict[str, float],
                                overround_thresh: float = 0.02,
                                min_coverage: float = 0.9) -> dict[str, Any]:
    """Aggregate violations across event groups given a YES price per market.

    `min_coverage`: a categorical event is only measured when we have a fresh price for at
    least this fraction of its outcomes — otherwise the sum-to-one is over an incomplete
    partition (missing outcomes) and is meaningless. This is the gate that removes the
    dominant artifact.
    """
    groups = groups.copy()
    groups["yes"] = groups["id"].map(yes_prices)
    cat_rows, lad_rows = [], []
    for ev, g in groups.groupby("event_id"):
        et = g["event_type"].iloc[0]
        n_priced = int(g["yes"].notna().sum())
        coverage = n_priced / len(g)
        if et == "categorical" and n_priced >= 2 and coverage >= min_coverage:
            over = sum_to_one_overround(g["yes"].tolist())
            cat_rows.append({"event_id": ev, "n": n_priced, "coverage": round(coverage, 3),
                             "overround": over})
        elif et == "ladder" and n_priced >= 2 and coverage >= min_coverage:
            res = ladder_monotonicity_breaks(g["threshold"].tolist(), g["yes"].tolist())
            lad_rows.append({"event_id": ev, "coverage": round(coverage, 3), **res})
    cat = pd.DataFrame(cat_rows)
    lad = pd.DataFrame(lad_rows)
    out: dict[str, Any] = {
        "categorical_events_priced": int(len(cat)),
        "ladder_events_priced": int(len(lad)),
    }
    if len(cat):
        viol = cat["overround"].abs() > overround_thresh
        out["categorical_violation_rate"] = float(viol.mean())
        out["categorical_mean_abs_overround"] = float(cat["overround"].abs().mean())
        out["categorical_underround_frac"] = float((cat["overround"] < -overround_thresh).mean())
    if len(lad):
        out["ladder_with_break_rate"] = float((lad["n_breaks"] > 0).mean())
        out["ladder_mean_max_gap"] = float(lad["max_gap"].mean())
    return out, cat, lad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markets", type=Path, default=Path("archive/markets.csv"))
    ap.add_argument("--fills", type=Path, default=Path("archive/processed/underdog_events/fills_sorted.parquet"))
    ap.add_argument("--output-dir", type=Path, default=Path("reports/event_groups"))
    ap.add_argument("--restrict-to-fills", action="store_true",
                    help="Trim the saved event_groups.csv listing to the tradeable universe. Does "
                         "NOT affect the violation measurement, which always uses full partitions.")
    ap.add_argument("--measure-prices", action="store_true",
                    help="Join YES prices from the fills tape and measure violations (needs duckdb).")
    ap.add_argument("--overround-thresh", type=float, default=0.02)
    ap.add_argument("--min-coverage", type=float, default=0.9,
                    help="Only measure an event when this fraction of its outcomes is freshly priced "
                         "(else the sum-to-one is over an incomplete partition).")
    ap.add_argument("--max-stale-hours", type=float, default=48.0,
                    help="Drop a market's snapshot price if its last trade is older than this.")
    args = ap.parse_args()

    markets = pd.read_csv(args.markets, low_memory=False)
    groups = build_event_groups(markets)   # FULL partitions (do not drop members)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    if args.measure_prices:
        # measure on the FULL partition; the coverage gate handles outcomes we can't price.
        snaps = groups[["id", "snapshot_ts"]].rename(columns={"id": "market_id"}) \
            if "snapshot_ts" in groups.columns else None
        if snaps is None:
            raise SystemExit("No snapshot_ts (need createdAt/closedTime in markets for contemporaneous pricing).")
        yes = load_yes_prices_asof_duckdb(args.fills, snaps, args.max_stale_hours)
        surface, cat, lad = measure_opportunity_surface(
            groups, yes, args.overround_thresh, args.min_coverage)
        summary["opportunity_surface"] = surface
        cat.to_csv(args.output_dir / "categorical_overround.csv", index=False)
        lad.to_csv(args.output_dir / "ladder_breaks.csv", index=False)

    # saved listing (optionally trimmed to tradeable universe for readability)
    out_groups = groups
    if args.restrict_to_fills:
        import duckdb
        p = str(args.fills.resolve()).replace("'", "''")
        traded = {str(x) for x in duckdb.connect().execute(
            f"SELECT DISTINCT market_id FROM read_parquet('{p}')").fetch_df()["market_id"]}
        out_groups = groups[groups["id"].isin(traded)]
    out_groups.to_csv(args.output_dir / "event_groups.csv", index=False)
    type_counts = groups.drop_duplicates("event_id")["event_type"].value_counts().to_dict()
    summary.update({
        "events": int(groups["event_id"].nunique()),
        "event_type_counts": {k: int(v) for k, v in type_counts.items()},
        "markets_in_multi_event": int((groups["event_size"] > 1).sum()),
    })
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
