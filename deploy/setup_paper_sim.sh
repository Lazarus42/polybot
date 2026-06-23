#!/usr/bin/env bash
# One-command paper-sim setup for the running EC2 collector box. Run as root:
#     sudo bash deploy/setup_paper_sim.sh
# Assumes the collector bootstrap already ran (repo cloned at $APP, .venv present, IAM role with
# S3 write + the 15-min uploader cron installed). This pulls latest code, installs + starts the
# paper-sim systemd service ($5k budget, clv_full, size 200), and adds a paper-spool line to the
# uploader cron so snapshots ship to s3://$BUCKET/paper/dt=.../.
set -euo pipefail

POLYBOT_BUCKET="${POLYBOT_BUCKET:-polybot-polymarket-sjgibson}"
HOME_DIR=/home/ec2-user
APP="$HOME_DIR/polymarket_exp"
PAPER_DIR="$APP/reports/paper_sim"

cd "$APP"
sudo -u ec2-user git pull --ff-only || echo "WARN: git pull skipped (local changes?)"
sudo -u ec2-user .venv/bin/pip install -q --upgrade websocket-client requests

mkdir -p "$PAPER_DIR" && chown ec2-user:ec2-user "$PAPER_DIR"

# install + (re)start the paper-sim service
cp deploy/polybot-paper-sim.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now polybot-paper-sim
systemctl restart polybot-paper-sim   # pick up any code/service change on re-run

# add the paper spool to the uploader cron (idempotent: keep existing collector line, add paper)
CRON_LINE="*/15 * * * * POLYBOT_BUCKET=$POLYBOT_BUCKET POLYBOT_SPOOL=$APP/reports/clob_capture POLYBOT_PAPER_SPOOL=$PAPER_DIR $APP/deploy/upload_to_s3.sh >> $HOME_DIR/upload.log 2>&1"
( crontab -u ec2-user -l 2>/dev/null | grep -v upload_to_s3 || true; echo "$CRON_LINE" ) \
  | crontab -u ec2-user -

echo "paper-sim up. follow:  journalctl -u polybot-paper-sim -f"
echo "snapshots -> $PAPER_DIR  ->  s3://$POLYBOT_BUCKET/paper/dt=.../  (every 15 min)"
systemctl --no-pager --lines=8 status polybot-paper-sim || true
