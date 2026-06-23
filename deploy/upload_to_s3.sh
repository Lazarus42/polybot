#!/usr/bin/env bash
# Ship closed spool files to S3 and delete them locally. Run from cron every ~15 min.
# Uses the instance IAM role (no keys). Only moves CLOSED *.jsonl.gz (never the open *.tmp).
# Partitions by the file's own date (from its epoch in the name), so late uploads land right.
set -euo pipefail

BUCKET="${POLYBOT_BUCKET:?set POLYBOT_BUCKET}"
SPOOL="${POLYBOT_SPOOL:-/home/ec2-user/polymarket_exp/reports/clob_capture}"

shopt -s nullglob
for f in "$SPOOL"/book_*.jsonl.gz; do
    base=$(basename "$f")
    epoch=$(echo "$base" | sed -E 's/.*_([0-9]+)\.jsonl\.gz$/\1/')
    # GNU date (Linux EC2) first, BSD date fallback
    dt=$(date -u -d "@$epoch" +%Y-%m-%d 2>/dev/null || date -u -r "$epoch" +%Y-%m-%d)
    aws s3 mv "$f" "s3://$BUCKET/raw/dt=$dt/$base" --only-show-errors
done

# manifests: copy (keep local for the collector to keep updating), small + idempotent
for m in "$SPOOL"/manifest_*.json; do
    aws s3 cp "$m" "s3://$BUCKET/manifests/$(basename "$m")" --only-show-errors
done

# paper-sim snapshots (closed *.jsonl.gz) -> paper/dt=YYYY-MM-DD/. Optional; set POLYBOT_PAPER_SPOOL.
PAPER="${POLYBOT_PAPER_SPOOL:-}"
if [ -n "$PAPER" ] && [ -d "$PAPER" ]; then
    for f in "$PAPER"/paper_*.jsonl.gz; do
        base=$(basename "$f")
        epoch=$(echo "$base" | sed -E 's/.*_([0-9]+)\.jsonl\.gz$/\1/')
        dt=$(date -u -d "@$epoch" +%Y-%m-%d 2>/dev/null || date -u -r "$epoch" +%Y-%m-%d)
        aws s3 mv "$f" "s3://$BUCKET/paper/dt=$dt/$base" --only-show-errors
    done
    for s in "$PAPER"/paper_sim_summary.json; do
        [ -e "$s" ] && aws s3 cp "$s" "s3://$BUCKET/paper/$(basename "$s")" --only-show-errors
    done
fi
