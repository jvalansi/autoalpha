#!/usr/bin/env bash
# Live PEAD at the open — cc-connect cron fires at 13:31 and 14:31 UTC Mon-Fri so one
# of them lands at 9:31 ET across DST; run_live_pead.py exits unless it's 9:30-10:00 ET,
# the market is open, and it hasn't already run today. The cron is muted; the nightly
# report covers routine runs, and only failures, risk halts and problem notes post here.
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="/home/ubuntu/.env"
exec >> "/tmp/autoalpha_live_pead_$(date -u +%Y%m%d).log" 2>&1
# Only the vars this job needs (the shared .env holds unrelated secrets)
for var in ALPACA_API_KEY ALPACA_SECRET_KEY ALPACA_BASE_URL FMP_API_KEY DISCORD_BOT_TOKEN; do
    value=$(/home/ubuntu/bin/secret "$var" "$ENV_FILE")  # resolves 1Password op:// references
    [ -n "$value" ] && export "$var"="$value"
done
cd "$REPO_DIR"
echo "=== $(date -u '+%F %T UTC') ==="
timeout --kill-after=30 1200 /home/ubuntu/miniconda3/bin/python scripts/run_live_pead.py
rc=$?
if [ "$rc" -ne 0 ]; then  # the cron is muted, so a failed run must alert on its own
    echo "**autoalpha live PEAD FAILED** (exit $rc) — see /tmp/autoalpha_live_pead_$(date -u +%Y%m%d).log" \
        | /home/ubuntu/miniconda3/bin/python scripts/post_to_discord.py
fi
exit "$rc"
