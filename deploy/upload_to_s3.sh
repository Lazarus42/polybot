#!/usr/bin/env bash
# Ship closed spool files to S3 and delete them locally. Run from cron every ~15 min.
# Uses the instance IAM role (no keys). Only moves CLOSED *.jsonl.gz (never the open *.tmp).
# Partitions by the file's own date (from its epoch in the name), so late uploads land right.
#
# Robustness (why this script is defensive):
#  - A flock prevents overlapping cron runs from racing the same files ("path does not exist" errors).
#  - NO `set -e`: a single failed/vanished file must never abort the run. It did — the raw section
#    erroring aborted the script before the PAPER section ran, silently starving paper uploads.
#  - Files are re-checked with `[ -e ]` right before upload: the collector prunes old raw files to
#    stay under its size cap, so a globbed file can disappear before we reach it.
#  - PAPER ships FIRST so the active experiment always lands even when the raw backlog is huge.
#  - Uploads run in PARALLEL (xargs -P) so raw (13-16 GB/day) can keep up with the collector.
set -uo pipefail

BUCKET="${POLYBOT_BUCKET:?set POLYBOT_BUCKET}"
SPOOL="${POLYBOT_SPOOL:-/home/ec2-user/polymarket_exp/reports/clob_capture}"
PAPER="${POLYBOT_PAPER_SPOOL:-}"
PAR_PAPER="${POLYBOT_PAR_PAPER:-4}"
PAR_RAW="${POLYBOT_PAR_RAW:-16}"

# single-run lock: if a previous run is still going, skip this one cleanly (no racing/globbing twins)
LOCK="/tmp/polybot_upload.lock"
exec 9>"$LOCK" || exit 0
flock -n 9 || { echo "$(date -u +%H:%M:%S) upload already running; skip"; exit 0; }

shopt -s nullglob

# one file -> S3, non-fatal. Uses BK / SUB from the environment (set by ship). Exported so the
# xargs child shells can call it directly, which keeps the per-file logic un-quote-mangled.
_ship_one() {
    local f="$1" base epoch dt
    [ -e "$f" ] || return 0                            # vanished (pruned / moved by another run)
    base=$(basename "$f")
    epoch=$(echo "$base" | sed -E 's/.*_([0-9]+)\.jsonl\.gz$/\1/')
    dt=$(date -u -d "@$epoch" +%Y-%m-%d 2>/dev/null || date -u -r "$epoch" +%Y-%m-%d)
    aws s3 mv "$f" "s3://$BK/$SUB/dt=$dt/$base" --only-show-errors \
        || echo "$(date -u +%H:%M:%S) skip $base (upload failed or file gone)"
}
export -f _ship_one

ship() {   # ship <dir> <glob-prefix> <s3-subprefix> <parallelism>
    local dir="$1" pref="$2" par="${4:-1}"
    [ -d "$dir" ] || return 0
    export BK="$BUCKET" SUB="$3"
    ls "$dir"/${pref}_*.jsonl.gz 2>/dev/null \
        | xargs -r -P "$par" -I{} bash -c '_ship_one "$@"' _ {}
}

# 1) PAPER FIRST — small, and it's the active experiment; must never be starved by the raw backlog.
if [ -n "$PAPER" ]; then
    ship "$PAPER" paper paper "$PAR_PAPER"
    s="$PAPER/paper_sim_summary.json"
    [ -e "$s" ] && aws s3 cp "$s" "s3://$BUCKET/paper/$(basename "$s")" --only-show-errors \
        || echo "$(date -u +%H:%M:%S) no paper_sim_summary yet"
fi

# 2) manifests: copy (keep local so the collector keeps updating them), small + idempotent.
for m in "$SPOOL"/manifest_*.json; do
    [ -e "$m" ] || continue
    aws s3 cp "$m" "s3://$BUCKET/manifests/$(basename "$m")" --only-show-errors \
        || echo "$(date -u +%H:%M:%S) skip manifest $(basename "$m")"
done

# 2b) arb monitor: daily-append jsonl (small) — cp-overwrite each run, keep local (still appending).
for a in "${POLYBOT_ARB:-/home/ec2-user/polymarket_exp/reports/arb_monitor}"/*.jsonl; do
    [ -e "$a" ] || continue
    aws s3 cp "$a" "s3://$BUCKET/arb/$(basename "$a")" --only-show-errors \
        || echo "$(date -u +%H:%M:%S) skip arb $(basename "$a")"
done

# 3) raw order book: PARALLEL best-effort drain (large, self-capped/pruned upstream).
ship "$SPOOL" book raw "$PAR_RAW"
