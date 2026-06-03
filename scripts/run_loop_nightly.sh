#!/usr/bin/env bash
# Nightly autoalpha pipeline — run via cron at 6 AM UTC (11 PM PDT)
#   1. Refresh OHLCV cache with latest bars
#   2. Rebuild vault_data.parquet
#   3. Run paper trading and post PnL to Slack
#   4. Run 50 iterations of the LLM research loop
set -euo pipefail

LOG="/tmp/autoalpha_nightly_$(date +%Y%m%d).log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="/home/ubuntu/slack-claude-bot/.env"
PYTHON="/home/ubuntu/miniconda3/bin/python"

exec >> "$LOG" 2>&1

echo "=== autoalpha nightly: $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="

# Load env vars (SLACK_BOT_TOKEN, SLACK_LOOP_CHANNEL, etc.)
# Use grep/export instead of source to avoid issues with special chars in values
if [ -f "$ENV_FILE" ]; then
    while IFS='=' read -r key value; do
        # Skip blank lines and comments
        [[ -z "$key" || "$key" == \#* ]] && continue
        export "$key"="$value"
    done < <(grep -v '^\s*#' "$ENV_FILE" | grep -v '^\s*$')
fi

cd "$REPO_DIR"

# Post to channel top-level (no thread) so each night's results are visible
unset SLACK_LOOP_THREAD_TS

echo "--- Step 1: Refresh OHLCV cache ---"
"$PYTHON" scripts/refresh_cache_recent.py

echo "--- Step 2: Rebuild vault dataset ---"
"$PYTHON" scripts/build_vault_dataset.py

echo "--- Step 3: Paper trading update ---"
"$PYTHON" scripts/run_paper.py \
    --slack-channel "${SLACK_LOOP_CHANNEL:-}"

echo "--- Step 4: Research loop (50 iterations) ---"
"$PYTHON" scripts/run_loop.py \
    --iterations 50 \
    --budget 999 \
    --model claude-sonnet-4-6 \
    --data data/loop_data.parquet \
    --slack-channel "${SLACK_LOOP_CHANNEL:-}"

echo "=== done: $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
