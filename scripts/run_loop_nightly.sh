#!/usr/bin/env bash
# Nightly autoalpha research loop — run via cron at 6 AM UTC (11 PM PDT)
set -euo pipefail

LOG="/tmp/autoalpha_nightly_$(date +%Y%m%d).log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="/home/ubuntu/slack-claude-bot/.env"
PYTHON="/home/ubuntu/miniconda3/bin/python"

exec >> "$LOG" 2>&1

echo "=== autoalpha nightly loop: $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="

# Load env vars (SLACK_BOT_TOKEN, SLACK_LOOP_CHANNEL, etc.)
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

cd "$REPO_DIR"

# Post a start notification to Slack (top-level, no thread — so each run is visible)
unset SLACK_LOOP_THREAD_TS

"$PYTHON" scripts/run_loop.py \
    --iterations 50 \
    --budget 999 \
    --model claude-sonnet-4-6 \
    --data data/loop_data.parquet \
    --slack-channel "${SLACK_LOOP_CHANNEL:-}"

echo "=== done: $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
