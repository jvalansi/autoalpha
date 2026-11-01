#!/usr/bin/env bash
# $10k small account pre-market — cc-connect cron fires at 12:15 and 13:15 UTC Mon-Fri so
# one lands in 08:00-09:25 ET across DST; run_live_small.py exits outside that window,
# on non-trading days, and if it already ran today.
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="/home/ubuntu/.env"
exec >> "/tmp/autoalpha_live_small_$(date -u +%Y%m%d).log" 2>&1
# Only the vars this job needs (the shared .env holds unrelated secrets)
for var in ALPACA_10K_API_KEY ALPACA_10K_SECRET_KEY ALPACA_10K_BASE_URL; do
    value=$(grep -E "^${var}=" "$ENV_FILE" | head -1 | cut -d= -f2-)
    [ -n "$value" ] && export "$var"="$value"
done
cd "$REPO_DIR"
echo "=== $(date -u '+%F %T UTC') ==="
timeout --kill-after=30 900 /home/ubuntu/miniconda3/bin/python scripts/run_live_small.py
