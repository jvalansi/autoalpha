#!/usr/bin/env bash
# Nightly autoalpha pipeline — run via cron at 6 AM UTC (11 PM PDT)
#   1. Update vault with latest bars
#   2. Run 50 iterations of the LLM research loop (so new signals appear in tonight's report)
#   3. Run paper trading and post PnL + new signals to Slack
set -euo pipefail

LOG="/tmp/autoalpha_nightly_$(date +%Y%m%d).log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="/home/ubuntu/slack-claude-bot/.env"
PYTHON="/home/ubuntu/miniconda3/bin/python"

exec >> "$LOG" 2>&1

echo "=== autoalpha nightly: $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="

# Only pull the env vars this pipeline actually needs. The .env file is shared
# with the slack-claude-bot and exports unrelated secrets (notably a stale
# ANTHROPIC_API_KEY that the `claude` CLI would prefer over the subscription
# auth, causing every loop iteration to abort with "Invalid API key").
if [ -f "$ENV_FILE" ]; then
    for var in SLACK_BOT_TOKEN SLACK_LOOP_CHANNEL; do
        value=$(grep -E "^${var}=" "$ENV_FILE" | head -1 | cut -d= -f2-)
        [ -n "$value" ] && export "$var"="$value"
    done
fi

export PATH="/home/ubuntu/.local/bin:$PATH"

cd "$REPO_DIR"

# Post to channel top-level (no thread) so each night's results are visible
unset SLACK_LOOP_THREAD_TS

echo "--- Step 1: Update vault with latest bars ---"
"$PYTHON" scripts/update_vault.py

echo "--- Step 2: Research loop (50 iterations) ---"
"$PYTHON" scripts/run_loop.py \
    --iterations 50 \
    --budget 999 \
    --model claude-sonnet-4-6 \
    --data data/loop_data.parquet \
    --slack-channel "${SLACK_LOOP_CHANNEL:-}" || echo "WARNING: research loop failed (non-fatal)"

echo "--- Step 3: Paper trading update (includes signals found tonight) ---"
"$PYTHON" scripts/run_paper.py \
    --slack-channel "${SLACK_LOOP_CHANNEL:-}" || echo "WARNING: paper trading failed (non-fatal)"

echo "=== done: $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
