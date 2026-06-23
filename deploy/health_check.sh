#!/usr/bin/env bash
# Self-check for the collect-all fleet. Run from cron (e.g. hourly). Emails via SNS ONLY if
# something looks wrong — silence means healthy. Catches app-level problems while the box is alive
# (shards down, spool ballooning, disk full, S3 not growing, ping/pong storm returning).
# Instance death/hang is caught separately by a CloudWatch EC2 status-check alarm -> same SNS topic.
#
# Env (set in the crontab line):
#   POLYBOT_BUCKET   - S3 bucket name
#   POLYBOT_SNS_ARN  - SNS topic ARN to publish alerts to
#   POLYBOT_SPOOL    - spool dir (default below)
#   MIN_SHARDS       - alert if fewer live shards than this (default 28)
#   MAX_SPOOL_MB     - alert if spool dir exceeds this (default 4000 = uploader likely stuck)
#   MAX_DISK_PCT     - alert if root disk past this percent (default 80)
set -uo pipefail

BUCKET="${POLYBOT_BUCKET:?set POLYBOT_BUCKET}"
SNS_ARN="${POLYBOT_SNS_ARN:?set POLYBOT_SNS_ARN}"
SPOOL="${POLYBOT_SPOOL:-/home/ec2-user/polymarket_exp/reports/clob_capture}"
MIN_SHARDS="${MIN_SHARDS:-28}"
MAX_SPOOL_MB="${MAX_SPOOL_MB:-4000}"
MAX_DISK_PCT="${MAX_DISK_PCT:-80}"

problems=()

# 1. shard processes alive
shards=$(pgrep -f collect_clob_book.py | wc -l)
[ "$shards" -lt "$MIN_SHARDS" ] && problems+=("Only $shards shard processes alive (expected >= $MIN_SHARDS).")

# 2. spool draining (not ballooning -> uploader stuck)
spool_mb=$(du -sm "$SPOOL" 2>/dev/null | cut -f1)
[ -n "$spool_mb" ] && [ "$spool_mb" -gt "$MAX_SPOOL_MB" ] && \
  problems+=("Spool dir is ${spool_mb} MB (> ${MAX_SPOOL_MB}); uploads may be failing.")

# 3. disk
disk_pct=$(df --output=pcent / | tail -1 | tr -dc '0-9')
[ -n "$disk_pct" ] && [ "$disk_pct" -gt "$MAX_DISK_PCT" ] && \
  problems+=("Root disk at ${disk_pct}% (> ${MAX_DISK_PCT}%).")

# 4. ping/pong storm returning (false-timeout reconnect loop)
pp=$(journalctl -u polybot-collect-all --since "1 hour ago" 2>/dev/null | grep -c "ping/pong timed out")
[ "$pp" -gt 50 ] && problems+=("$pp 'ping/pong timed out' in the last hour; keepalive may be failing.")

# 5. S3 still growing TODAY (compare object count now vs a saved marker from the previous run)
dt=$(date -u +%Y-%m-%d)
now_objs=$(aws s3 ls "s3://$BUCKET/raw/dt=$dt/" 2>/dev/null | grep -c "book_")
marker="/tmp/polybot_s3_objs_${dt}"
if [ -f "$marker" ]; then
    prev=$(cat "$marker")
    # only meaningful mid-day (after a couple cron cycles); allow equal right after midnight rollover
    [ "$now_objs" -le "$prev" ] && \
      problems+=("S3 object count for $dt not growing (was $prev, now $now_objs); capture or upload stalled.")
fi
echo "$now_objs" > "$marker"

# alert only if something's wrong
if [ "${#problems[@]}" -gt 0 ]; then
    body=$(printf '%s\n' "POLYBOT collector health alert on $(hostname) at $(date -u):" "" "${problems[@]}" \
           "" "Live shards: $shards | spool: ${spool_mb:-?} MB | disk: ${disk_pct:-?}% | s3 objs today: $now_objs")
    aws sns publish --topic-arn "$SNS_ARN" --subject "POLYBOT collector alert" --message "$body" --only-show-errors
fi
