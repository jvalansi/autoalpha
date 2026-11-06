#!/usr/bin/env bash
# Nightly autoalpha pipeline — run via cc-connect cron at 6 AM UTC (11 PM PDT)
#   1. Update vault with latest bars
#   2. Run 50 iterations of the LLM research loop (so new signals appear in tonight's report)
#   3. Run paper trading
#   3b. Update Darwinian signal weights from the paper book's alpha returns
# The report it writes (the only routine autoalpha post) goes out in the morning via
# post_nightly_report.sh, directly to the autoalpha Discord channel via the Bot REST API
# (no cc-connect session — that posts into a thread, and the auto-upgrade path in cron
# is also flaky).
set -euo pipefail

LOG="/tmp/autoalpha_nightly_$(date +%Y%m%d).log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="/home/ubuntu/miniconda3/bin/python"
REPORT_FILE="$REPO_DIR/data/last_paper_report.txt"
ENV_FILE="/home/ubuntu/.env"

exec >> "$LOG" 2>&1

# The nightly report is the day's only routine notification, so a step that can cost
# it must alert on its own (the cron is muted).
alert() {
    echo "**autoalpha nightly: $1** — no daily report may follow; see $LOG" \
        | "$PYTHON" scripts/post_to_discord.py || echo "WARNING: Discord alert failed"
}

echo "=== autoalpha nightly: $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="

# Pull only the env vars this pipeline needs (the shared .env exports
# unrelated secrets including a stale ANTHROPIC_API_KEY that breaks the
# claude CLI subscription auth).
if [ -f "$ENV_FILE" ]; then
    for var in DISCORD_BOT_TOKEN FMP_API_KEY ALPACA_10K_API_KEY ALPACA_10K_SECRET_KEY ALPACA_10K_BASE_URL; do
        value=$(/home/ubuntu/bin/secret "$var" "$ENV_FILE")  # resolves 1Password op:// references
        [ -n "$value" ] && export "$var"="$value"
    done
fi

export PATH="/home/ubuntu/.local/bin:$PATH"

cd "$REPO_DIR"

# Each hang-prone step gets a `timeout` guard so a stalled network fetch or a
# wedged claude CLI fails THAT step (non-fatally) instead of hanging the whole
# job until the 180-min cron limit kills it as `exit -1` (as happened 2026-07-14).
# `|| echo` only catches a non-zero exit, not a hang — the timeout is what caps a hang.
echo "--- Step 1: Update vault with latest bars ---"
if ! timeout --kill-after=30 900 "$PYTHON" scripts/update_vault.py; then
    echo "WARNING: vault update failed/timed out (non-fatal)"
    alert "vault update FAILED"
fi

echo "--- Step 2: Research loop — PAUSED 2026-08-27 ---"
# Paused deliberately. The loop admits on PSR > 0.65 vs a fixed market
# benchmark, NOT on the deflated Sharpe PLAN.md specifies, and every batch of
# 50 hypotheses raises E[max SR] for the entire book (~sqrt(2 ln N)) — so
# running it actively raises the bar on the signals already in the library
# while adding new ones that were never held to the stated standard.
# Re-enable once the admission gate is settled (PLAN.md Phase 5, "Admission
# gate rework"). To resume, delete this block and restore the call below.
echo "SKIPPED: research loop paused pending admission-gate rework"
# timeout --kill-after=60 7200 "$PYTHON" scripts/run_loop.py \
#     --iterations 50 \
#     --budget 999 \
#     --model claude-sonnet-4-6 \
#     --data data/loop_data.parquet || echo "WARNING: research loop failed/timed out (non-fatal)"

echo "--- Step 3: Paper trading update (includes signals found tonight) ---"
# Clear last night's report: paper_end is written before the report, so a run dying
# between them would otherwise re-post the old report as the new day's.
rm -f "$REPORT_FILE"
if ! timeout --kill-after=30 900 "$PYTHON" scripts/run_paper.py; then
    echo "WARNING: paper trading failed/timed out (non-fatal)"
    alert "paper run FAILED"
fi

echo "--- Step 3b: Darwinian weight update ---"
# Must run after paper trading: it reads data/paper_pnl.json for the per-signal
# daily returns it residualizes into rolling alpha Sharpe.
timeout --kill-after=30 300 "$PYTHON" scripts/update_weights.py \
    || echo "WARNING: weight update failed/timed out (non-fatal)"

echo "=== done: $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
