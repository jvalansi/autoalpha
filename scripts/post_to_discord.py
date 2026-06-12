#!/usr/bin/env python3
"""Post a text message to a Discord channel via the Bot REST API.

Reads message body from stdin. Splits at 2000-char boundaries (Discord's
per-message limit) and sleeps between chunks to stay under the rate limit.

Env vars:
    DISCORD_BOT_TOKEN          (required) bot token with channel send permission
    DISCORD_AUTOALPHA_CHANNEL  (optional) channel ID; defaults to autoalpha channel
"""
import json
import os
import sys
import time
import urllib.request

CHANNEL_ID = os.environ.get("DISCORD_AUTOALPHA_CHANNEL", "1513381043002933318")
CHUNK_SIZE = 2000


def main() -> int:
    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        print("DISCORD_BOT_TOKEN not set", file=sys.stderr)
        return 1

    text = sys.stdin.read()
    if not text.strip():
        print("empty message; nothing to send", file=sys.stderr)
        return 0

    chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    url = f"https://discord.com/api/v10/channels/{CHANNEL_ID}/messages"
    for i, chunk in enumerate(chunks):
        req = urllib.request.Request(
            url,
            data=json.dumps({"content": chunk}).encode(),
            headers={
                "Authorization": f"Bot {token}",
                "Content-Type": "application/json",
                "User-Agent": "DiscordBot (https://github.com/jvalansi/autoalpha, 1.0)",
            },
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
        if i < len(chunks) - 1:
            time.sleep(1.2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
