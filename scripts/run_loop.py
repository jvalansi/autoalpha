"""Run the Phase 4 LLM research loop.

Usage:
    python scripts/run_loop.py [--iterations N] [--budget USD] [--db PATH]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure the package is importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoalpha.research.loop import ResearchLoop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the autoalpha LLM hypothesis loop")
    parser.add_argument("--iterations", type=int, default=3, help="Max iterations (default 3)")
    parser.add_argument("--budget", type=float, default=2.00, help="Max cost in USD (default 2.00)")
    parser.add_argument("--db", type=str, default=None, help="Path to memory.db (default: research/memory.db)")
    parser.add_argument("--data", type=str, default="data/loop_data.parquet", help="Path to enriched parquet")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6", help="Claude model to use")
    parser.add_argument("--slack-channel", type=str, default=None, help="Slack channel ID for notifications")
    parser.add_argument("--slack-thread-ts", type=str, default=None, help="Slack thread timestamp for notifications")
    args = parser.parse_args()

    if args.slack_channel:
        os.environ["SLACK_LOOP_CHANNEL"] = args.slack_channel
    if args.slack_thread_ts:
        os.environ["SLACK_LOOP_THREAD_TS"] = args.slack_thread_ts

    data_path = str(Path(args.data).resolve())
    if not Path(data_path).exists():
        print(f"ERROR: data file not found: {data_path}")
        print("Run scripts/build_loop_dataset.py first.")
        sys.exit(1)

    db_path = Path(args.db).resolve() if args.db else None

    print(f"Starting loop: max_iterations={args.iterations}  budget=${args.budget:.2f}  model={args.model}")
    print(f"Data: {data_path}")
    print()

    with ResearchLoop(
        data_path=data_path,
        db_path=db_path,
        model=args.model,
        max_iterations=args.iterations,
        max_cost_usd=args.budget,
    ) as loop:
        loop.run()

    print("\nDone.")


if __name__ == "__main__":
    main()
