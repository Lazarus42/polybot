#!/usr/bin/env bash
# EC2 first-boot bootstrap (paste into "User data", or run once via SSH as root).
# Sets up the collector as a systemd service + a 15-min cron uploader to S3.
# EDIT THESE TWO, then launch a free-tier t3.micro/t4g.micro with an IAM role that has
# deploy/s3-write-policy.json attached. The S3 bucket must be in the SAME region (free transfer).
set -euxo pipefail

POLYBOT_BUCKET="polybot-polymarket-sjgibson"
REPO_URL="https://github.com/Lazarus42/polybot.git"
# "all" = full-universe sharded fleet (collect_all, needs t3.small/medium);
# "targeted" = single ~500-token bucketed collector (collect_clob_book, fits t3.micro).
COLLECT_MODE="all"

HOME_DIR=/home/ec2-user
APP="$HOME_DIR/polymarket_exp"

# deps (Amazon Linux 2023; for Ubuntu swap dnf->apt)
dnf install -y python3 python3-pip git cronie || yum install -y python3 python3-pip git cronie
systemctl enable --now crond || true
command -v aws >/dev/null || dnf install -y awscli || pip3 install awscli

sudo -u ec2-user git clone "$REPO_URL" "$APP" || (cd "$APP" && sudo -u ec2-user git pull)
cd "$APP"
sudo -u ec2-user python3 -m venv .venv
sudo -u ec2-user .venv/bin/pip install --upgrade pip websocket-client requests

# uploader env + cron (every 15 min)
chmod +x deploy/upload_to_s3.sh
( crontab -u ec2-user -l 2>/dev/null | grep -v upload_to_s3 || true; \
  echo "*/15 * * * * POLYBOT_BUCKET=$POLYBOT_BUCKET POLYBOT_SPOOL=$APP/reports/clob_capture $APP/deploy/upload_to_s3.sh >> $HOME_DIR/upload.log 2>&1" ) \
  | crontab -u ec2-user -

# collector service (full-universe fleet by default, else targeted single collector)
if [ "$COLLECT_MODE" = "all" ]; then
    SERVICE=polybot-collect-all
else
    SERVICE=polybot-collector
fi
cp "deploy/$SERVICE.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now "$SERVICE"

echo "bootstrap done ($SERVICE). logs: journalctl -u $SERVICE -f ; $HOME_DIR/upload.log"
