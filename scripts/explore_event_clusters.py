#!/usr/bin/env python3
"""Explore slug-based event clustering for cross-market (meta) strategies.

The single-market pipeline treats every binary market independently. Cross-market
strategies (sum-of-probabilities overround fades, cheapest-leg value) need to know
which markets belong to the SAME event. The source data has no reliable event_id, but
market slugs usually share a stem within an event.

This script reads markets.parquet, derives candidate event groups from slugs under a
few heuristics, and validates them against resolutions: a genuine mutually-exclusive
event (e.g. "who wins") should have exactly one winning leg. It then writes a
market->event mapping plus diagnostics so we can judge clustering quality before
building any cross-market signals.

Runs in the project venv (needs duckdb to read parquet):
    .venv/bin/python scripts/explore_event_clusters.py
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import duckdb
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("duckdb required to read markets.parquet; run inside the project venv") from exc


def load_markets(data_dir: Path) -> pd.DataFrame:
    path = str((data_dir / "markets.parquet").resolve()).replace("'", "''")
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{path}')").fetchdf()
    con.close()
    return df


def pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    # fuzzy contains
    for c in df.columns:
        cl = c.lower()
        if any(key in cl for key in candidates):
            return c
    return None


def slug_drop_last(slug: str) -> str:
    parts = str(slug).split("-")
    return "-".join(parts[:-1]) if len(parts) > 1 else str(slug)


def slug_first_k(slug: str, k: int) -> str:
    return "-".join(str(slug).split("-")[:k])


def slug_strip_year(slug: str) -> str:
    # drop trailing date-ish / numeric tokens often used to distinguish outcomes
    return re.sub(r"(-\d{1,4}){1,3}$", "", str(slug))


def slug_suffix_k(slug: str, k: int) -> str:
    # mutually-exclusive legs often share a question SUFFIX with the entity swapped early
    parts = str(slug).split("-")
    return "-".join(parts[-k:]) if len(parts) > k else str(slug)


_STOP = {"will", "the", "be", "a", "an", "to", "of", "in", "on", "by", "at", "is",
         "for", "and", "or", "vs", "win", "wins"}
_MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december", "jan", "feb", "mar", "apr",
           "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"}


def slug_tokens(slug: str) -> set[str]:
    toks = set()
    for t in str(slug).split("-"):
        if not t or t.isdigit() or t in _STOP or t in _MONTHS:
            continue
        toks.add(t)
    return toks


def cluster_closetime_jaccard(df: pd.DataFrame, slug_col: str, end_col: str,
                              threshold: float, max_group: int) -> np.ndarray:
    """Cluster markets that share a close time AND high slug-token overlap.

    Sharing a close time separates recurring series (resolve at different times) from
    genuine mutually-exclusive events (legs resolve together). Within each close-time
    group, union markets whose token-set Jaccard >= threshold. Oversized groups (likely
    series/junk) are left unclustered to keep it O(group^2)-bounded.
    """
    import itertools
    df = df.reset_index(drop=True)
    tok_list = df[slug_col].apply(slug_tokens).tolist()
    result = np.array([""] * len(df), dtype=object)
    counter = 0
    for end_val, grp in df.groupby(end_col, dropna=False):
        idxs = list(grp.index)
        if len(idxs) > max_group:
            for i in idxs:
                result[i] = f"solo_{i}"
            continue
        parent = {i: i for i in idxs}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for a, b in itertools.combinations(idxs, 2):
            ta, tb = tok_list[a], tok_list[b]
            if not ta or not tb:
                continue
            if len(ta & tb) / len(ta | tb) >= threshold:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
        comp: dict[int, list[int]] = {}
        for i in idxs:
            comp.setdefault(find(i), []).append(i)
        for members in comp.values():
            cid = f"evt_{counter}"
            counter += 1
            for i in members:
                result[i] = cid
    return result


def evaluate(df: pd.DataFrame, key_col: str, winner_bool: pd.Series | None, label: str) -> dict:
    sizes = df.groupby(key_col).size()
    multi = sizes[sizes >= 2]
    out = {
        "heuristic": label,
        "n_events": int(sizes.size),
        "n_multi_leg_events": int(multi.size),
        "markets_in_multi_leg": int(multi.sum()),
        "median_event_size": float(sizes.median()),
        "max_event_size": int(sizes.max()),
    }
    if winner_bool is not None:
        wins_per = df.assign(_w=winner_bool.values).groupby(key_col)["_w"].sum()
        wm = wins_per.loc[multi.index]
        out["multi_leg_exactly_one_winner"] = int((wm == 1).sum())
        # The tradeable universe: >=3 legs with exactly one winner = clean mutually-exclusive.
        big = sizes[sizes >= 3].index
        wb = wins_per.loc[big]
        out["events_3plus_one_winner"] = int((wb == 1).sum())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("archive/processed/underdog_events"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/event_clusters"))
    parser.add_argument("--first-k", type=int, default=5, help="Token count for the first-k-tokens heuristic.")
    parser.add_argument("--suffix-k", type=int, default=5, help="Token count for the suffix heuristic.")
    parser.add_argument("--jaccard-threshold", type=float, default=0.4, help="Token-overlap threshold for close-time clustering (lower merges multi-token entities like full names).")
    parser.add_argument("--max-group", type=int, default=400, help="Skip close-time groups larger than this (likely series/junk).")
    parser.add_argument("--min-leg-volume", type=float, default=0.0)
    parser.add_argument("--chosen", default="closetime_jaccard",
                        choices=["drop_last", "first_k", "strip_year", "suffix_k", "closetime_jaccard"],
                        help="Heuristic used for the written market->event mapping.")
    args = parser.parse_args()

    df = load_markets(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("markets.parquet columns:", list(df.columns))
    print("rows:", len(df))

    mid_col = pick_column(df, ["market_id", "marketid", "id"])
    slug_col = pick_column(df, ["slug"])
    q_col = pick_column(df, ["question", "title"])
    end_col = pick_column(df, ["scheduled_end", "end", "close", "deadline"])
    win_col = pick_column(df, ["winning_side", "winner", "result", "resolved", "outcome", "won"])
    vol_col = pick_column(df, ["historical_volume", "volume"])
    print(f"detected -> market_id:{mid_col} slug:{slug_col} question:{q_col} end:{end_col} winner:{win_col} volume:{vol_col}")
    if slug_col is None:
        raise SystemExit("No slug column found; cannot cluster by slug.")

    if vol_col and args.min_leg_volume > 0:
        df = df[df[vol_col].fillna(0) >= args.min_leg_volume]

    print("\nsample slugs:")
    for s in df[slug_col].dropna().head(15):
        print("  ", s)
    tok_counts = df[slug_col].dropna().apply(lambda s: len(str(s).split("-")))
    print("\nslug token-count distribution:")
    print(tok_counts.describe(percentiles=[.25, .5, .75, .9]).round(2).to_string())

    # Derive a boolean "this leg won" if we can interpret the winner column.
    winner_bool = None
    if win_col is not None:
        col = df[win_col]
        if col.dtype == bool:
            winner_bool = col.fillna(False)
        elif np.issubdtype(col.dtype, np.number):
            winner_bool = (col.fillna(0) > 0)
        else:
            sval = col.astype(str).str.lower()
            winner_bool = sval.isin(["yes", "true", "1", "win", "won", "token1"])
        print(f"\ninterpreted winner column '{win_col}': {int(winner_bool.sum())} winning legs of {len(df)}")

    df = df.reset_index(drop=True)
    if winner_bool is not None:
        winner_bool = winner_bool.reset_index(drop=True)
    df["_drop_last"] = df[slug_col].apply(slug_drop_last)
    df["_first_k"] = df[slug_col].apply(lambda s: slug_first_k(s, args.first_k))
    df["_strip_year"] = df[slug_col].apply(slug_strip_year)
    df["_suffix_k"] = df[slug_col].apply(lambda s: slug_suffix_k(s, args.suffix_k))
    if end_col is not None:
        print("\nclustering by close time + token overlap (this can take a moment)...", flush=True)
        df["_closetime_jaccard"] = cluster_closetime_jaccard(df, slug_col, end_col, args.jaccard_threshold, args.max_group)
    else:
        df["_closetime_jaccard"] = df["_drop_last"]
        print("\nno close-time column detected; closetime_jaccard falls back to drop_last")

    reports = []
    for key_col, label in [("_drop_last", "drop_last"), ("_first_k", f"first_{args.first_k}"),
                           ("_strip_year", "strip_year"), ("_suffix_k", f"suffix_{args.suffix_k}"),
                           ("_closetime_jaccard", "closetime_jaccard")]:
        reports.append(evaluate(df, key_col, winner_bool, label))
    rep_df = pd.DataFrame(reports)
    print("\n=== clustering quality by heuristic (events_3plus_one_winner = clean tradeable events) ===")
    print(rep_df.to_string(index=False))

    # Show example multi-leg clusters for the chosen heuristic.
    chosen_col = {"drop_last": "_drop_last", "first_k": "_first_k", "strip_year": "_strip_year",
                  "suffix_k": "_suffix_k", "closetime_jaccard": "_closetime_jaccard"}[args.chosen]
    sizes = df.groupby(chosen_col).size().sort_values(ascending=False)
    multi = sizes[sizes >= 2]
    print(f"\n=== sample multi-leg events ({args.chosen}) — up to 12 ===")
    for key in list(multi.index)[:12]:
        members = df[df[chosen_col] == key]
        qs = members[q_col].head(6).tolist() if q_col else members[slug_col].head(6).tolist()
        nwin = int(winner_bool[members.index].sum()) if winner_bool is not None else -1
        print(f"\n[{key}]  legs={len(members)} winners={nwin}")
        for q in qs:
            print("   -", q)

    # Write the market->event mapping for the chosen heuristic.
    mapping = df[[mid_col, slug_col, chosen_col]].rename(columns={mid_col: "market_id", slug_col: "slug", chosen_col: "event_id"})
    if winner_bool is not None:
        mapping["won"] = winner_bool.values
    mapping.to_csv(args.output_dir / "market_event_map.csv", index=False)
    rep_df.to_csv(args.output_dir / "clustering_quality.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps({
        "chosen_heuristic": args.chosen,
        "columns_detected": {"market_id": mid_col, "slug": slug_col, "question": q_col, "end": end_col, "winner": win_col, "volume": vol_col},
        "reports": reports,
        "files": ["market_event_map.csv", "clustering_quality.csv"],
    }, indent=2) + "\n", encoding="utf-8")
    print("\nwrote:", args.output_dir / "market_event_map.csv")


if __name__ == "__main__":
    main()
