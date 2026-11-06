#!/usr/bin/env bash
# Morning delivery of the report run_loop_nightly.sh built overnight — cc-connect cron
# at 12:00 UTC (4 AM PST / 5 AM PDT), directly via the Bot REST API like the alerts.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/ubuntu/miniconda3/bin/python"
REPORT_FILE="$REPO_DIR/data/last_paper_report.txt"
PNL_FILE="$REPO_DIR/data/paper_pnl.json"
LAST_POSTED_FILE="$REPO_DIR/data/.last_posted_paper_end"
ENV_FILE="/home/ubuntu/.env"

exec >> "/tmp/autoalpha_nightly_$(date +%Y%m%d).log" 2>&1

echo "=== autoalpha report delivery: $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="

value=$(/home/ubuntu/bin/secret DISCORD_BOT_TOKEN "$ENV_FILE")  # resolves 1Password op:// references
[ -n "$value" ] && export DISCORD_BOT_TOKEN="$value"

cd "$REPO_DIR"

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
