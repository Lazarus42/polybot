#!/usr/bin/env python3
"""Run the book-aware MM backtest over a SHARDED, GZIPPED S3 day capture.

`book_mm_backtest.py:main()` was written for the single-file laptop capture: it loads one JSONL
into one list. The S3 collector (`collect_all.py`) lands a day as ~hundreds of gzipped shard
files `raw/dt=YYYY-MM-DD/book_<host>_<pid>_<epoch>.jsonl.gz`, where each `<pid>` is one shard
owning a DISJOINT set of tokens and each `<epoch>` is a time-rotation of that shard. Loading the
whole day into RAM would OOM.

Since tokens never cross shards, we process ONE SHARD AT A TIME (read all its rotation files ->
normalize_events -> simulate each token -> accumulate), which is exact and bounds memory to a
single shard. We reuse `normalize_events` and `simulate_book_mm` from book_mm_backtest unchanged,
and reproduce that file's aggregation (`run()` / output schema) verbatim so results are identical
to the single-file path, just over the full sharded day.

Usage:
  python scripts/run_book_mm_day.py --day-dir data/pull/dt=2026-06-22 \
      --manifest-dir data/pull/manifests [--max-shards N] [same MM flags as book_mm_backtest]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

# reuse the tested, pure core (ensure this script's dir is importable first)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from book_mm_backtest import normalize_events, simulate_book_mm  # noqa: E402

SHARD_RE = re.compile(r"_(\d+)_(\d+)\.jsonl\.gz$")  # _<pid>_<epoch>.jsonl.gz


def group_files_by_shard(day_dir: Path) -> dict[str, list[Path]]:
    """Group *.jsonl.gz under day_dir by shard pid; sort each shard's files by epoch (time)."""
    shards: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for p in sorted(day_dir.glob("*.jsonl.gz")):
        m = SHARD_RE.search(p.name)
        if not m:
            # unknown naming -> treat each file as its own shard (still correct, just less batched)
            shards[p.name].append((0, p))
            continue
        pid, epoch = m.group(1), int(m.group(2))
        shards[pid].append((epoch, p))
    return {pid: [p for _, p in sorted(files)] for pid, files in shards.items()}


def load_shard_by_token(files: list[Path]) -> dict[str, list[dict]]:
    """Build the per-token normalized event map for one shard, file-by-file.

    Each raw `book` record is huge (~150 levels); holding a whole shard's raw events at once
    OOMs. Instead we normalize ONE file at a time (collapsing each book to a tiny quote dict)
    and merge the compact results, so peak memory is a single file's raw events plus the small
    accumulated quote/trade stream. We re-sort each token by time at the end because events for
    a token can span several time-rotation files within the shard.
    """
    by_token: dict[str, list[dict]] = defaultdict(list)
    for f in files:
        raw: list[dict] = []
        with gzip.open(f, "rt") as fh:
            for line in fh:
                if line.strip():
                    try:
                        raw.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        part = normalize_events(raw)
        del raw
        for tok, evs in part.items():
            by_token[tok].extend(evs)
        del part
    for tok in by_token:
        by_token[tok].sort(key=lambda x: x["t"])
    return by_token


def latest_manifest(manifest_dir: Path) -> Path | None:
    cands = sorted(manifest_dir.glob("manifest_*.json"))
    return cands[-1] if cands else None


def load_token_pool(manifest: Path | None) -> dict[str, float]:
    pool: dict[str, float] = {}
    if manifest and manifest.exists():
        man = json.loads(manifest.read_text())
        for tok, meta in (man.get("token_meta") or {}).items():
            pool[str(tok)] = float(meta.get("reward_daily_est") or 0.0)
    return pool


def build_configs(args) -> dict[str, dict]:
    """The strategy matrix run in ONE parse. Each value is kwargs for simulate_book_mm.

    neutral       : plain symmetric maker (baseline).
    raw_momentum  : signal-informed skew on RAW quote-mid momentum (legacy informed MM).
    clv_debounce  : same skew but on the DEBOUNCED-mid (trade-VWAP) CLV signal.
    clv_full      : clv_debounce + inventory skew + volatility-scaled spread + toxicity gate
                    (the full predictive MM: fair-value tilt, inventory + vol + toxicity control).
    """
    mw, skew = args.momentum_window, args.skew_threshold
    return {
        "neutral": {},
        "raw_momentum": {"momentum_window": mw, "skew_threshold": skew},
        "clv_debounce": {"momentum_window": mw, "skew_threshold": skew,
                         "debounce_trades": args.debounce_trades},
        "clv_full": {"momentum_window": mw, "skew_threshold": skew,
                     "debounce_trades": args.debounce_trades, "inv_skew": args.inv_skew,
                     "vol_window": args.vol_window, "vol_spread_coeff": args.vol_spread_coeff,
                     "tox_threshold": args.tox_threshold, "tox_window": args.tox_window,
                     "tox_cooldown": args.tox_cooldown},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day-dir", type=Path, required=True, help="dir of book_*.jsonl.gz for one day")
    ap.add_argument("--manifest-dir", type=Path, default=None, help="dir of manifest_*.json (latest used)")
    ap.add_argument("--manifest", type=Path, default=None, help="explicit manifest file (overrides --manifest-dir)")
    ap.add_argument("--output-dir", type=Path, default=Path("reports/book_mm"))
    ap.add_argument("--max-shards", type=int, default=0, help="process only the first N shards (0 = all)")
    ap.add_argument("--time-budget", type=float, default=38.0,
                    help="stop cleanly after this many seconds (resumable via checkpoint); 0 = no limit")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="just aggregate the existing checkpoint into the summary and exit")
    # MM flags mirror book_mm_backtest
    ap.add_argument("--size", type=float, default=20.0)
    ap.add_argument("--inventory-cap", type=float, default=100.0)
    ap.add_argument("--maker-fee", type=float, default=0.0)
    ap.add_argument("--mark-delay-s", type=float, default=60.0)
    ap.add_argument("--improve", action="store_true")
    ap.add_argument("--skew-threshold", type=float, default=0.005)
    ap.add_argument("--momentum-window", type=float, default=300.0)
    ap.add_argument("--capture-share", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05, 0.10])
    # predictive-component knobs used by the clv_debounce / clv_full configs
    ap.add_argument("--debounce-trades", type=int, default=10, help="trade-VWAP window for the CLV signal")
    ap.add_argument("--inv-skew", type=float, default=0.01, help="inventory lean (price units at full cap)")
    ap.add_argument("--vol-window", type=float, default=300.0)
    ap.add_argument("--vol-spread-coeff", type=float, default=0.5)
    ap.add_argument("--tox-threshold", type=float, default=0.01)
    ap.add_argument("--tox-window", type=float, default=60.0)
    ap.add_argument("--tox-cooldown", type=float, default=60.0)
    args = ap.parse_args()

    manifest = args.manifest or (latest_manifest(args.manifest_dir) if args.manifest_dir else None)
    token_pool = load_token_pool(manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output_dir / "day_checkpoint.jsonl"

    shards = group_files_by_shard(args.day_dir)
    shard_ids = list(shards)
    if args.max_shards:
        shard_ids = shard_ids[: args.max_shards]

    done = set()
    if ckpt_path.exists():
        for line in ckpt_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["shard"])
    todo = [s for s in shard_ids if s not in done]

    if args.aggregate_only:
        aggregate(ckpt_path, args, manifest, len(shards), shard_ids)
        return

    print(f"shards: {len(shards)} total, {len(shard_ids)} selected, {len(done)} done, "
          f"{len(todo)} to go; manifest: {manifest}; reward tokens: {len(token_pool)}", flush=True)

    cfgs = build_configs(args)
    t_start = time.time()
    with ckpt_path.open("a") as ck:
        for i, pid in enumerate(todo):
            if args.time_budget and (time.time() - t_start) > args.time_budget:
                print(f"time budget hit after {i} shards this run; checkpointed, re-run to continue.",
                      flush=True)
                break
            by_token = load_shard_by_token(shards[pid])
            rec = {"shard": pid, "n_tokens": len(by_token)}
            for name, kw in cfgs.items():
                acc = defaultdict(float); n_tok = 0; qw = 0.0; os_q = 0.0; sig = 0.0; rpd = 0.0
                for tok, events in by_token.items():
                    if sum(1 for e in events if e["type"] == "trade") < 5:
                        continue
                    r = simulate_book_mm(events, args.size, args.inventory_cap, args.maker_fee,
                                         args.mark_delay_s, args.improve, **kw)
                    n_tok += 1
                    for k in ("pnl", "gross_spread_captured", "adverse_selection", "fees", "n_fills"):
                        acc[k] += r[k]
                    qw += r["n_quotes"]; os_q += r["one_sided_quote_frac"] * r["n_quotes"]
                    sig += r["mean_abs_signal"] * r["n_quotes"]
                    rpd += token_pool.get(tok, 0.0) * r["quoting_days"]
                rec[name] = {"tokens": n_tok, "pnl": acc["pnl"],
                             "gross_spread_captured": acc["gross_spread_captured"],
                             "adverse_selection": acc["adverse_selection"], "fees": acc["fees"],
                             "n_fills": acc["n_fills"], "reward_pool_days": rpd,
                             "quote_weight": qw, "onesided_weight": os_q, "signal_weight": sig}
            ck.write(json.dumps(rec) + "\n"); ck.flush()
            best = max(cfgs, key=lambda n: rec[n]["pnl"])
            print(f"  shard {pid} done ({i+1}/{len(todo)} this run): "
                  f"neutral pnl={rec['neutral']['pnl']:.2f}; best cfg={best} "
                  f"pnl={rec[best]['pnl']:.2f}; fills={int(rec['neutral']['n_fills'])}", flush=True)

    done2 = set()
    for line in ckpt_path.read_text().splitlines():
        if line.strip():
            done2.add(json.loads(line)["shard"])
    left = [s for s in shard_ids if s not in done2]
    if left:
        print(f"\n{len(left)} shards remaining — re-run the same command to continue.", flush=True)
    else:
        print("\nall selected shards done — aggregating.", flush=True)
        aggregate(ckpt_path, args, manifest, len(shards), shard_ids)


def _finalize(acc: dict, capture_share, momentum: bool) -> dict:
    rpd = acc["reward_pool_days"]
    d = {"tokens": int(acc["tokens"]), "total_pnl": round(acc["pnl"], 2),
         "gross_spread_captured": round(acc["gross_spread_captured"], 2),
         "adverse_selection": round(acc["adverse_selection"], 2),
         "fees": round(acc["fees"], 2), "n_fills": int(acc["n_fills"]),
         "reward_pool_dollar_days": round(rpd, 1)}
    d["pnl_at_capture_share"] = {f"{s:.0%}": round(acc["pnl"] + s * rpd, 2) for s in capture_share}
    d["breakeven_capture_share"] = (round(-acc["pnl"] / rpd, 4)
                                    if rpd > 0 and acc["pnl"] < 0 else 0.0)
    if momentum:
        qw = acc["quote_weight"]
        d["one_sided_quote_frac"] = round(acc["onesided_weight"] / qw, 3) if qw else 0.0
        d["mean_abs_momentum"] = round(acc["signal_weight"] / qw, 5) if qw else 0.0
    return d


def aggregate(ckpt_path: Path, args, manifest, n_shards_total: int, selected: list) -> None:
    cfg_names = list(build_configs(args))
    sums = {name: defaultdict(float) for name in cfg_names}
    n_done = 0
    for line in ckpt_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        n_done += 1
        for name in cfg_names:
            if name in rec:
                for k, v in rec[name].items():
                    sums[name][k] += v
    configs = {name: _finalize(sums[name], args.capture_share, momentum=(name != "neutral"))
               for name in cfg_names}
    base = configs["neutral"]["total_pnl"]
    ranking = sorted(cfg_names, key=lambda n: configs[n]["total_pnl"], reverse=True)
    out = {"day_dir": str(args.day_dir), "manifest": str(manifest),
           "shards_processed": n_done, "shards_selected": len(selected),
           "shards_total": n_shards_total,
           "configs": configs,
           "pnl_vs_neutral": {n: round(configs[n]["total_pnl"] - base, 2) for n in cfg_names},
           "ranking_by_pnl": ranking,
           "best_config": ranking[0]}
    (args.output_dir / "book_mm_day_summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\n" + json.dumps(out, indent=2))
    print("\nread: 'configs' holds each strategy's totals; pnl_vs_neutral is edge over the plain"
          " maker. pnl_at_capture_share adds your share of the REAL daily reward pools (manifest) x"
          " quoting time; breakeven_capture_share = fraction of the pool needed to flip MM positive.")


if __name__ == "__main__":
    main()
