#!/usr/bin/env bash
# Nightly autoalpha pipeline — run via cc-connect cron at 6 AM UTC (11 PM PDT)
#   1. Update vault with latest bars
#   2. Run 50 iterations of the LLM research loop (so new signals appear in tonight's report)
#   3. Run paper trading
#   4. Post the daily report directly to the autoalpha Discord channel via the
#      Bot REST API (no cc-connect session — that posts into a thread, and the
#      auto-upgrade path in cron is also flaky).
set -euo pipefail

LOG="/tmp/autoalpha_nightly_$(date +%Y%m%d).log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="/home/ubuntu/miniconda3/bin/python"
REPORT_FILE="$REPO_DIR/data/last_paper_report.txt"
PNL_FILE="$REPO_DIR/data/paper_pnl.json"
LAST_POSTED_FILE="$REPO_DIR/data/.last_posted_paper_end"
ENV_FILE="/home/ubuntu/.env"

exec >> "$LOG" 2>&1

echo "=== autoalpha nightly: $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="

# Pull only the env vars this pipeline needs (the shared .env exports
# unrelated secrets including a stale ANTHROPIC_API_KEY that breaks the
# claude CLI subscription auth).
if [ -f "$ENV_FILE" ]; then
    for var in DISCORD_BOT_TOKEN; do
        value=$(grep -E "^${var}=" "$ENV_FILE" | head -1 | cut -d= -f2-)
        [ -n "$value" ] && export "$var"="$value"
    done
fi

export PATH="/home/ubuntu/.local/bin:$PATH"

cd "$REPO_DIR"

echo "--- Step 1: Update vault with latest bars ---"
"$PYTHON" scripts/update_vault.py

echo "--- Step 2: Research loop (50 iterations) ---"
"$PYTHON" scripts/run_loop.py \
    --iterations 50 \
    --budget 999 \
    --model claude-sonnet-4-6 \
    --data data/loop_data.parquet || echo "WARNING: research loop failed (non-fatal)"

echo "--- Step 3: Paper trading update (includes signals found tonight) ---"
"$PYTHON" scripts/run_paper.py || echo "WARNING: paper trading failed (non-fatal)"

echo "--- Step 4: Deliver daily report to Discord ---"
# Skip the Discord post if paper_end didn't advance since the last delivery
# (weekends, US market holidays, or vault-fetch failures all leave it unchanged).
CURRENT_PAPER_END=""
if [ -f "$PNL_FILE" ]; then
    CURRENT_PAPER_END=$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1])).get('paper_end',''))" "$PNL_FILE" 2>/dev/null || echo "")
fi
LAST_POSTED=""
[ -f "$LAST_POSTED_FILE" ] && LAST_POSTED=$(cat "$LAST_POSTED_FILE")

if [ ! -f "$REPORT_FILE" ]; then
    echo "WARNING: $REPORT_FILE missing — nothing to deliver"
elif [ -n "$CURRENT_PAPER_END" ] && [ "$CURRENT_PAPER_END" = "$LAST_POSTED" ]; then
    echo "SKIP: paper_end ($CURRENT_PAPER_END) unchanged since last post — no new trading day."
else
    if "$PYTHON" scripts/post_to_discord.py < "$REPORT_FILE"; then
        [ -n "$CURRENT_PAPER_END" ] && printf '%s' "$CURRENT_PAPER_END" > "$LAST_POSTED_FILE"
    else
        echo "WARNING: Discord post failed"
    fi
fi

echo "=== done: $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
