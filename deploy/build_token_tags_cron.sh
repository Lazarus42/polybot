#!/usr/bin/env bash
# Daily cron: rebuild the merged token-tags artifact from all local manifests and ship it to S3.
# Analysis pulls s3://$BUCKET/tags/token_tags.json.gz instead of ~4GB of raw manifests.
set -u
BUCKET="${POLYBOT_BUCKET:-polybot-polymarket-sjgibson}"
REPO="${POLYBOT_REPO:-/home/ec2-user/polymarket_exp}"

cd "$REPO" || exit 1
exec 9>/tmp/token_tags.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) another build running; skip"; exit 0; }

.venv/bin/python scripts/build_token_tags.py \
    --manifest-glob 'reports/clob_capture/manifest_*.json' \
    --out reports/clob_capture/token_tags.json.gz \
  && aws s3 cp reports/clob_capture/token_tags.json.gz \
       "s3://$BUCKET/tags/token_tags.json.gz" --only-show-errors \
  && echo "$(date -u +%FT%TZ) tags built + uploaded"
