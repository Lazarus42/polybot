#!/usr/bin/env python3
"""Capture (nearly) ALL active Polymarket markets by sharding across N WebSocket collectors.

A single WebSocket connection takes ~500 tokens, so to cover the full universe we run a fleet:
this launcher paginates gamma for every active market, shards the tokens, and starts one
`collect_clob_book.py` child process per shard (each writing pid-unique gzip spool files that the
S3 uploader ships).

**Zero-gap, no restarts.** Children run for the whole month. Every `--re-enumerate-minutes` the
launcher re-scans gamma and, for any *newly-listed* market not already covered, appends its tokens
to a shard's add-file inbox — the child subscribes them live with no teardown. Each child prunes
its own *resolved* markets on `market_resolved`, so resolutions are never missed to a restart gap.
Shards report their live subscribed set back so the launcher knows true occupancy; if every shard
is full, the launcher spins up an additional shard for the overflow. A child is only ever restarted
if it crashes (its full token set is persisted, so it reloads). Recurring ultra-short crypto is
excluded; substantive markets (incl. penny longshots, kept regardless of liquidity) are captured.

Run under systemd (see deploy/polybot-collect-all.service). Sizing: the full universe is
~13k tokens => ~30 shards; use a t3.large (memory headroom for ~30 child processes; gzip+parse CPU
is light). `--min-liquidity` trims the illiquid mid-priced tail WITHOUT dropping penny longshots.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_clob_book import all_tokens_from_events  # noqa: E402


def fetch_all_events(page_limit: int = 100, max_pages: int = 300) -> list[dict]:
    """Paginate gamma /events for every active, open event. gamma caps `limit` at 100/page, so
    page in 100s by offset until a short/empty page (and stops cleanly at gamma's ~2000 offset
    cap — by then we already hold the whole tradeable universe, ranked by volume)."""
    import requests  # noqa: PLC0415

    out: list[dict] = []
    offset = 0
    for _ in range(max_pages):
        try:
            r = requests.get("https://gamma-api.polymarket.com/events",
                             params={"active": "true", "closed": "false", "archived": "false",
                                     "limit": page_limit, "offset": offset,
                                     "order": "volume24hr", "ascending": "false"}, timeout=30)
            r.raise_for_status()
            evs = r.json()
        except requests.HTTPError as exc:
            print(f"pagination stopped at offset {offset} ({exc.response.status_code}); "
                  f"{len(out)} events collected")
            break
        if not evs:
            break
        out.extend(evs)
        offset += page_limit
        if len(evs) < page_limit:
            break
    return out


def shard(tokens: list[str], size: int) -> list[list[str]]:
    return [tokens[i:i + size] for i in range(0, len(tokens), size)]


def assign_new_tokens(new_tokens: list[str], shard_counts: list[int],
                      shard_size: int) -> tuple[dict[int, list[str]], list[str]]:
    """Pack newly-listed tokens into existing shards with spare room (least-full first); anything
    that doesn't fit becomes overflow for the caller to start new shards with. Pure / testable."""
    counts = list(shard_counts)
    assignments: dict[int, list[str]] = {}
    overflow: list[str] = []
    for tok in new_tokens:
        idx = min(range(len(counts)), key=lambda i: counts[i]) if counts else None
        if idx is not None and counts[idx] < shard_size:
            assignments.setdefault(idx, []).append(tok)
            counts[idx] += 1
        else:
            overflow.append(tok)
    return assignments, overflow


class Fleet:
    """Owns the child processes and their per-shard IPC files (tokens / add-inbox / live-status)."""

    def __init__(self, out: Path, collector: str, args):
        self.out, self.collector, self.args = out, collector, args
        self.procs: dict[int, subprocess.Popen] = {}
        self.assigned: set[str] = set()   # everything we've ever handed to a shard
        self.n = 0

    def _paths(self, i: int):
        return (self.out / f"_shard_{i}.tokens", self.out / f"_shard_{i}.add",
                self.out / f"_shard_{i}.live")

    def launch(self, sh: list[str]) -> int:
        i = self.n
        self.n += 1
        tf, add, live = self._paths(i)
        tf.write_text("\n".join(sh))
        add.write_text("")
        live.write_text("\n".join(sh))   # optimistic until the child reports
        self.procs[i] = self._spawn(i)
        self.assigned.update(sh)
        return i

    def _spawn(self, i: int) -> subprocess.Popen:
        tf, add, live = self._paths(i)
        cmd = [sys.executable, self.collector, "--tokens-file", str(tf), "--output-dir", str(self.out),
               "--rotate-minutes", str(self.args.rotate_minutes), "--minutes", "1e9",
               "--max-subscriptions", str(self.args.shard_size + 60),
               "--add-file", str(add), "--status-file", str(live)]
        return subprocess.Popen(cmd)

    def live_counts(self) -> list[int]:
        counts = []
        for i in range(self.n):
            _, _, live = self._paths(i)
            try:
                counts.append(len([x for x in live.read_text().splitlines() if x.strip()]))
            except OSError:
                counts.append(self.args.shard_size)   # unknown -> treat as full, don't overfill
        return counts

    def live_tokens(self) -> set[str]:
        toks: set[str] = set()
        for i in range(self.n):
            _, _, live = self._paths(i)
            try:
                toks.update(x.strip() for x in live.read_text().splitlines() if x.strip())
            except OSError:
                pass
        return toks

    def add_to_shard(self, i: int, tokens: list[str]) -> None:
        tf, add, _ = self._paths(i)
        with add.open("a") as fh:                      # child consumes + truncates
            fh.write("\n".join(tokens) + "\n")
        existing = [x for x in tf.read_text().splitlines() if x.strip()]
        tf.write_text("\n".join(existing + tokens))    # keep tokens-file authoritative for crash-reload
        self.assigned.update(tokens)

    def supervise(self) -> None:
        for i, p in list(self.procs.items()):
            if p.poll() is not None:
                print(f"shard {i} died (rc={p.returncode}); restarting from its tokens-file")
                self.procs[i] = self._spawn(i)

    def stop(self) -> None:
        for p in self.procs.values():
            p.terminate()
        for p in self.procs.values():
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()


def write_manifest(out: Path, n_events: int, n_tokens: int, n_shards: int, meta: dict) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (out / f"manifest_{stamp}.json").write_text(json.dumps(
        {"created": stamp, "n_events": n_events, "n_tokens": n_tokens,
         "n_shards": n_shards, "token_meta": meta}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=Path("reports/clob_capture"))
    ap.add_argument("--shard-size", type=int, default=450, help="Tokens per WebSocket connection (<500).")
    ap.add_argument("--re-enumerate-minutes", type=float, default=30.0,
                    help="Re-scan gamma and add newly-listed markets live (no restart) this often.")
    ap.add_argument("--rotate-minutes", type=float, default=15.0)
    ap.add_argument("--min-liquidity", type=float, default=1000.0,
                    help="Low floor; drops illiquid MID-priced markets but keeps penny longshots.")
    ap.add_argument("--include-crypto", action="store_true")
    ap.add_argument("--both-tokens", action="store_true")
    ap.add_argument("--minutes", type=float, default=1e9, help="Total run; huge under systemd.")
    ap.add_argument("--dry-run", action="store_true", help="Enumerate + report shard plan, then exit.")
    args = ap.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    collector = str(Path(__file__).resolve().parent / "collect_clob_book.py")
    deadline = time.time() + args.minutes * 60.0

    def enumerate_universe():
        events = fetch_all_events()
        tokens, meta = all_tokens_from_events(events, args.min_liquidity,
                                              exclude_crypto=not args.include_crypto,
                                              one_token_per_market=not args.both_tokens)
        return events, tokens, meta

    events, tokens, meta = enumerate_universe()
    shards = shard(tokens, args.shard_size)
    write_manifest(out, len(events), len(tokens), len(shards), meta)
    print(f"[{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}] {len(events)} events -> "
          f"{len(tokens)} tokens in {len(shards)} shards")

    if args.dry_run:
        from collections import Counter
        print("by category:", dict(Counter(m.get("category", "?") for m in meta.values())))
        return

    fleet = Fleet(out, collector, args)
    for sh in shards:
        fleet.launch(sh)
    print(f"launched {fleet.n} shard collectors (zero-gap: new markets added live, no restarts)")

    while time.time() < deadline:
        # supervise children frequently; re-enumerate on the slower cadence
        next_enum = min(deadline, time.time() + args.re_enumerate_minutes * 60)
        while time.time() < next_enum:
            time.sleep(15)
            fleet.supervise()

        try:
            events, tokens, meta = enumerate_universe()
        except Exception as exc:
            print("re-enumeration failed (keeping current fleet):", exc)
            continue
        new = [t for t in tokens if t not in fleet.assigned and t not in fleet.live_tokens()]
        if not new:
            continue
        assignments, overflow = assign_new_tokens(new, fleet.live_counts(), args.shard_size)
        for i, toks in assignments.items():
            fleet.add_to_shard(i, toks)
        for chunk in shard(overflow, args.shard_size):
            fleet.launch(chunk)
        added = sum(len(v) for v in assignments.values()) + len(overflow)
        print(f"+{added} new markets live ({len(overflow)} overflow -> new shards; "
              f"fleet now {fleet.n} shards)")
        write_manifest(out, len(events), len(tokens), fleet.n, meta)

    fleet.stop()


if __name__ == "__main__":
    main()
