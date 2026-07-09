#!/usr/bin/env python3
"""Aggregate a week (or any span) of paper-sim snapshots into a COMPACT summary.

Purpose: the per-minute snapshot tape (config x token x minute) is millions of rows and must NOT be
fed to an LLM directly. This collapses it to a few small tables (per config, per config x day, per
config x market-category) — a few KB — that a model can reason over cheaply. Run it in-region on the
EC2 box (zero S3 egress) or on a laptop after `aws s3 sync` of just the tiny `paper/` prefix.

Reads gzipped JSONL snapshots (see PAPER_TRADING_CONTEXT.md for the row schema) with DuckDB, joins
manifest tags if present, writes CSVs + a markdown digest to --out.

Robust to the sim restarting (Restart=always): reward is summed from the incremental `reward_step`
(never the cumulative `reward_cum`), and trade P&L is taken as the LAST marked_pnl within each
(config, token, source-file) run segment, then summed — so a mid-week restart neither double-counts
nor zeroes anything.

Usage:
    pip install duckdb            # only dependency
    python scripts/analyze_paper_sim.py \
        --paper 'data/paper/dt=*/paper_*.jsonl.gz' \
        --manifest-glob 'data/manifests/manifest_*.json' \
        --out reports/paper_analysis
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
from pathlib import Path

import duckdb


def parse_until(s):
    """Parse a --until cutoff into an epoch-seconds float. Accepts either raw epoch seconds
    ('1783310400') or an ISO-8601 timestamp WITH offset ('2026-07-06T00:00-04:00'). Rows at or
    after this instant are excluded, so results reflect the world only up to the cutoff."""
    if not s:
        return None
    s = s.strip()
    if s.replace(".", "", 1).isdigit():
        return float(s)
    from datetime import datetime  # noqa: PLC0415
    return datetime.fromisoformat(s).timestamp()


def valid_gzip_files(pattern: str):
    """Return (good, bad) split of a glob: readable non-empty gzip vs empty/truncated/corrupt files.
    A live rotated pipeline leaves the occasional 0-byte or partial tail; skip those, don't crash.
    Note: Python's gzip reads a 0-byte file as an empty stream WITHOUT error, so an empty-content
    check is required on top of the decompress attempt (DuckDB rejects such files as 'not a GZIP')."""
    good, bad = [], []
    for p in sorted(glob.glob(pattern)):
        try:
            with open(p, "rb") as raw:
                if raw.read(2) != b"\x1f\x8b":     # gzip magic; also rejects 0-byte files
                    bad.append(p); continue
            with gzip.open(p, "rb") as fh:
                if not fh.read(1):                 # valid header but no decompressed content
                    bad.append(p); continue
            good.append(p)
        except Exception:
            bad.append(p)
    return good, bad


def build_manifest_tags(manifest_glob: str) -> dict[str, dict]:
    """Merge all manifests into token -> {category, horizon, neg_risk, pool, min_size, question}.
    Later manifests win (tags are stable; last-seen is fine)."""
    tags: dict[str, dict] = {}
    for p in sorted(glob.glob(manifest_glob)):
        try:
            man = json.loads(Path(p).read_text())
        except Exception:
            continue
        for tok, m in (man.get("token_meta") or {}).items():
            tags[str(tok)] = {
                "category": m.get("category") or "unknown",
                "horizon": m.get("horizon") or "unknown",
                "neg_risk": bool(m.get("neg_risk")),
                "pool": float(m.get("reward_daily_est") or 0.0),
                "min_size": float(m.get("rewards_min_size") or 0.0),
                "question": (m.get("question") or "")[:80],
            }
    return tags


def raise_fd_limit():
    """DuckDB opens every snapshot file at once; a full run is ~1000s of files, over macOS's
    default 256-fd soft limit. Bump the soft limit toward the hard limit so the read doesn't
    hit 'Too many open files'."""
    try:
        import resource  # noqa: PLC0415  (POSIX only; fine on macOS/Linux)
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = 100000 if hard in (resource.RLIM_INFINITY, -1) else hard
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except Exception:
        pass


def main() -> None:
    raise_fd_limit()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper", required=True, help="glob for paper_*.jsonl.gz snapshots")
    ap.add_argument("--manifest-glob", default=None, help="glob for manifest_*.json (optional, for tag slices)")
    ap.add_argument("--capital", type=float, default=5000.0, help="budget, for ROC-on-budget")
    ap.add_argument("--liq-haircut-c", type=float, default=1.0,
                    help="cents/share haircut assumed to liquidate leftover inventory (honest net)")
    ap.add_argument("--until", default=None,
                    help="exclude snapshots at/after this cutoff (epoch seconds or ISO-8601 with "
                         "offset). e.g. 1783310400 = Mon 2026-07-06 00:00 EDT")
    ap.add_argument("--since", default=None,
                    help="exclude snapshots BEFORE this (epoch or ISO-8601); restricts to a window")
    ap.add_argument("--out", type=Path, default=Path("reports/paper_analysis"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    until = parse_until(args.until)
    since = parse_until(args.since)

    good, bad = valid_gzip_files(args.paper)
    if bad:
        print(f"skipping {len(bad)} unreadable/partial gzip file(s) (e.g. {Path(bad[0]).name})")
    if not good:
        print("No readable snapshot files matched --paper.")
        return
    files_sql = "[" + ",".join("'" + p.replace("'", "''") + "'" for p in good) + "]"

    conds = []
    if until is not None:
        conds.append(f"t < {until}")
    if since is not None:
        conds.append(f"t >= {since}")
    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    con = duckdb.connect()
    con.execute(f"""
        CREATE VIEW snap AS
        SELECT *, regexp_extract(filename, 'dt=([0-9-]+)', 1) AS dt,
                  regexp_extract(filename, '_([0-9]+)_[0-9]+\\.jsonl', 1) AS pid
        FROM read_json_auto({files_sql}, filename=true, union_by_name=true, ignore_errors=true)
        {where};
    """)
    if until is not None:
        print(f"cutoff: excluding snapshots at/after epoch {until:.0f}")

    # ONE row per (config, token, pid) = one position lifetime. marked_pnl is CUMULATIVE within a pid
    # (a position spans many 15-min files), so we take the LAST value, never sum per file. A new pid
    # (sim restart) is a separate lifetime, correctly summed downstream.
    con.execute("""
        CREATE VIEW seg_last AS
        SELECT config, token, pid,
               arg_max(marked_pnl, t) AS final_pnl,
               arg_max(inv, t)        AS final_inv,
               arg_max(mid, t)        AS final_mid,
               max(t) AS t_last, min(t) AS t_first
        FROM snap GROUP BY config, token, pid;
    """)

    # ---- per-config summary ----
    con.execute(f"""
        CREATE VIEW by_config AS
        WITH rw AS (SELECT config, sum(reward_step) AS reward,
                           count(*) AS rows, count(DISTINCT token) AS tokens,
                           count(DISTINCT dt) AS days
                    FROM snap GROUP BY config),
             tp AS (SELECT config, sum(final_pnl) AS trade_pnl,
                           sum(abs(final_inv) * final_mid) AS inv_notional,
                           sum(abs(final_inv)) AS inv_shares
                    FROM seg_last GROUP BY config)
        SELECT rw.config, rw.days, rw.tokens, rw.rows,
               round(rw.reward, 2)                                   AS reward,
               round(tp.trade_pnl, 2)                                AS trade_pnl,
               round(rw.reward + tp.trade_pnl, 2)                    AS net,
               round(tp.inv_shares * {args.liq_haircut_c} / 100.0, 2) AS est_flatten_cost,
               round(rw.reward + tp.trade_pnl
                     - tp.inv_shares * {args.liq_haircut_c}/100.0, 2) AS net_if_flat,
               round(tp.inv_notional, 2)                             AS resid_inv_notional,
               round(100.0*(rw.reward + tp.trade_pnl
                     - tp.inv_shares*{args.liq_haircut_c}/100.0)
                     / {args.capital} / nullif(rw.days,0), 3)        AS roc_pct_per_day
        FROM rw JOIN tp USING (config)
        ORDER BY net_if_flat DESC;
    """)
    con.execute(f"COPY by_config TO '{args.out}/by_config.csv' (HEADER, DELIMITER ',');")

    # ---- per-config x day ----
    # reward/day = sum of incremental reward_step (safe to sum). trade_pnl/day = the DAILY CHANGE in
    # each position's cumulative marked_pnl (diff of cum across days per token,pid), so a multi-day
    # position is attributed to the days it actually moved -- never the cumulative level re-summed.
    con.execute("""
        CREATE VIEW by_config_day AS
        WITH rw AS (SELECT config, dt, sum(reward_step) AS reward FROM snap GROUP BY config, dt),
             daily AS (SELECT config, token, pid, dt, arg_max(marked_pnl, t) AS cum
                       FROM snap GROUP BY config, token, pid, dt),
             perrow AS (SELECT config, dt,
                               cum - coalesce(lag(cum) OVER (PARTITION BY config, token, pid
                                                             ORDER BY dt), 0) AS day_pnl
                        FROM daily),
             tp AS (SELECT config, dt, sum(day_pnl) AS trade_pnl FROM perrow GROUP BY config, dt)
        SELECT config, dt, round(reward,2) AS reward, round(trade_pnl,2) AS trade_pnl,
               round(reward+trade_pnl,2) AS net
        FROM rw JOIN tp USING (config, dt) ORDER BY dt, net DESC;
    """)
    con.execute(f"COPY by_config_day TO '{args.out}/by_config_day.csv' (HEADER, DELIMITER ',');")

    # ---- per-config x market category (needs manifest) ----
    tag_rows = 0
    if args.manifest_glob:
        tags = build_manifest_tags(args.manifest_glob)
        tag_rows = len(tags)
        if tags:
            con.execute("CREATE TABLE tags(token VARCHAR, category VARCHAR, horizon VARCHAR, "
                        "neg_risk BOOLEAN, pool DOUBLE)")
            con.executemany("INSERT INTO tags VALUES (?,?,?,?,?)",
                            [(t, v["category"], v["horizon"], v["neg_risk"], v["pool"])
                             for t, v in tags.items()])
            con.execute("""
                CREATE VIEW by_config_category AS
                SELECT s.config, coalesce(g.category,'unknown') AS category,
                       count(DISTINCT s.token) AS tokens,
                       round(sum(s.reward_step),2) AS reward
                FROM snap s LEFT JOIN tags g USING (token)
                GROUP BY s.config, category ORDER BY s.config, reward DESC;
            """)
            con.execute(f"COPY by_config_category TO '{args.out}/by_config_category.csv' (HEADER, DELIMITER ',');")

    # ---- compact markdown digest (no pandas dependency: render tables from the cursor directly) ----
    def md_table(sql: str) -> str:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
        for r in rows:
            out.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
        return "\n".join(out)

    span = con.execute("SELECT min(dt) a, max(dt) b, count(DISTINCT dt) d, count(*) n FROM snap").fetchone()
    lines = [
        "# Paper-sim week-1 digest (compact — safe to feed a model)",
        "",
        f"- Span: {span[0]} → {span[1]} ({span[2]} days), {span[3]:,} snapshot rows",
        f"- Cutoff applied: excluded snapshots at/after epoch {until:.0f}" if until is not None else "- No time cutoff (full data)",
        f"- Budget: ${args.capital:,.0f} · liquidation haircut assumed: {args.liq_haircut_c}c/share",
        f"- Manifest tags loaded for {tag_rows} tokens" if args.manifest_glob else "- No manifest tags loaded",
        "",
        "## Per-config (ranked by net_if_flat)",
        "",
        md_table("SELECT * FROM by_config"),
        "",
        "See by_config.csv, by_config_day.csv, by_config_category.csv for full slices.",
        "Metric notes: `net_if_flat` = reward + trade_pnl - est cost to dump leftover inventory; "
        "this is the honest headline. Compare each strategy to its `_ctl`. All figures are modeled "
        "ceilings (no market impact, reward modeled not paid).",
    ]
    (args.out / "digest.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}/digest.md and CSVs. Span {span[0]}..{span[1]} ({span[2]}d, {span[3]:,} rows).")
    print(md_table("SELECT config, days, tokens, reward, trade_pnl, net, net_if_flat, roc_pct_per_day "
                   "FROM by_config"))


if __name__ == "__main__":
    main()
