#!/usr/bin/env bash
# Ship closed spool files to S3 and delete them locally. Run from cron every ~15 min.
# Uses the instance IAM role (no keys). Only moves CLOSED *.jsonl.gz (never the open *.tmp).
# Partitions by the file's own date (from its epoch in the name), so late uploads land right.
#
# Robustness (why this script is defensive):
#  - A flock prevents overlapping cron runs from racing the same files. Overlap + a moved-out-from-
#    under-us file was causing "path does not exist" errors.
#  - NO `set -e`: a single failed/vanished file must never abort the run. It did — the raw section
#    erroring aborted the script before the PAPER section ran, silently starving paper uploads.
#  - Files are re-checked with `[ -e ]` right before upload: the collector prunes old raw files to
#    stay under its size cap, so a globbed file can disappear before we reach it.
#  - PAPER ships FIRST so the active experiment always lands even when the raw backlog is huge.
set -uo pipefail

BUCKET="${POLYBOT_BUCKET:?set POLYBOT_BUCKET}"
SPOOL="${POLYBOT_SPOOL:-/home/ec2-user/polymarket_exp/reports/clob_capture}"
PAPER="${POLYBOT_PAPER_SPOOL:-}"

# single-run lock: if a previous run is still going, skip this one cleanly (no racing/globbing twins)
LOCK="/tmp/polybot_upload.lock"
exec 9>"$LOCK" || exit 0
flock -n 9 || { echo "$(date -u +%H:%M:%S) upload already running; skip"; exit 0; }

shopt -s nullglob

ship() {   # ship <dir> <glob-prefix> <s3-subprefix>  — move *.jsonl.gz to s3://BUCKET/sub/dt=.../
    local dir="$1" pref="$2" sub="$3" f base epoch dt
    [ -d "$dir" ] || return 0
    for f in "$dir"/${pref}_*.jsonl.gz; do
        [ -e "$f" ] || continue                       # vanished (pruned / moved by another run)
        base=$(basename "$f")
        epoch=$(echo "$base" | sed -E 's/.*_([0-9]+)\.jsonl\.gz$/\1/')
        dt=$(date -u -d "@$epoch" +%Y-%m-%d 2>/dev/null || date -u -r "$epoch" +%Y-%m-%d)
        aws s3 mv "$f" "s3://$BUCKET/$sub/dt=$dt/$base" --only-show-errors \
            || echo "$(date -u +%H:%M:%S) skip $base (upload failed or file gone)"
    done
}

# 1) PAPER FIRST — small, and it's the active experiment; must never be starved by the raw backlog.
if [ -n "$PAPER" ]; then
    ship "$PAPER" paper paper
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

# 3) raw order book: best-effort (large, self-capped/pruned upstream). Never blocks the above.
ship "$SPOOL" book raw
