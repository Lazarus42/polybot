#!/usr/bin/env python3
"""Merge every collector manifest into ONE compact token-tags artifact for analysis joins.

Why: each manifest_*.json is ~8.5 MB and only a point-in-time snapshot of the universe, so an
analysis that loads a couple of manifests tags only the tokens alive at those instants — that is
the "~50% of positions unknown on every manifest axis" coverage gap. The union over ALL manifests
tags (nearly) every token the sim ever quoted, and the merged artifact is ~2 MB gzipped instead of
~4 GB of raw manifests — cheap to ship to a laptop.

Output shape is drop-in compatible with `load_manifest_tags` (top-level "token_meta"), gzipped.
Per token we keep the analysis fields only, plus:
  - horizon_days        from the LAST manifest containing the token (short-biased: days-to-resolve
                        shrinks over a token's life)
  - horizon_days_first  from the FIRST manifest (long-biased) — bracket the truth with both
  - first_seen/last_seen manifest stamps (data-coverage debugging)

Run in-region on the EC2 box (manifests are kept locally in reports/clob_capture/):
    python scripts/build_token_tags.py \
        --manifest-glob 'reports/clob_capture/manifest_*.json' \
        --out reports/clob_capture/token_tags.json.gz
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
from pathlib import Path

KEEP = ("category", "neg_risk", "horizon_days", "competitive", "spread",
        "reward_daily_est", "rewards_min_size", "rewards_max_spread", "question")


def merge_manifests(paths: list[str]) -> dict:
    """Union token_meta across manifests in filename order (stamps sort chronologically).
    Later values win per field; horizon_days_first/first_seen keep the earliest sighting."""
    merged: dict[str, dict] = {}
    n_ok = 0
    for p in sorted(paths):
        try:
            man = json.loads(Path(p).read_text())
        except Exception:
            continue                                   # partial/corrupt manifest: skip, non-fatal
        stamp = man.get("created") or Path(p).stem.replace("manifest_", "")
        tm = man.get("token_meta") or {}
        if not tm:
            continue
        n_ok += 1
        for tok, m in tm.items():
            tok = str(tok)
            d = merged.get(tok)
            if d is None:
                d = merged[tok] = {"first_seen": stamp,
                                   "horizon_days_first": m.get("horizon_days")}
            for k in KEEP:
                v = m.get(k)
                if v is not None:
                    d[k] = v
            d["last_seen"] = stamp
    return {"n_manifests": n_ok, "n_tokens": len(merged), "token_meta": merged}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest-glob", required=True)
    ap.add_argument("--out", type=Path, required=True, help="output .json.gz path")
    args = ap.parse_args()

    paths = glob.glob(args.manifest_glob)
    if not paths:
        raise SystemExit(f"no manifests match {args.manifest_glob}")
    out = merge_manifests(paths)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as fh:
        json.dump(out, fh)
    tmp.rename(args.out)                               # atomic-ish: uploader never sees a partial
    print(f"merged {out['n_manifests']} manifests -> {out['n_tokens']} tokens -> {args.out} "
          f"({args.out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
