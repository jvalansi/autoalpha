#!/usr/bin/env bash
# Nightly autoalpha pipeline — run via cc-connect cron at 6 AM UTC (11 PM PDT)
#   1. Update vault with latest bars
#   2. Run 50 iterations of the LLM research loop (so new signals appear in tonight's report)
#   3. Run paper trading; post PnL + new signals to the Discord session that owns
#      this cron job (CC_PROJECT / CC_SESSION_KEY are injected by cc-connect cron).
set -euo pipefail

LOG="/tmp/autoalpha_nightly_$(date +%Y%m%d).log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="/home/ubuntu/miniconda3/bin/python"
REPORT_FILE="$REPO_DIR/data/last_paper_report.txt"

exec >> "$LOG" 2>&1

echo "=== autoalpha nightly: $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="

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
if [ -f "$REPORT_FILE" ]; then
    cc-connect send --stdin < "$REPORT_FILE" || echo "WARNING: cc-connect send failed"
else
    echo "WARNING: $REPORT_FILE missing — nothing to deliver"
fi

echo "=== done: $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
