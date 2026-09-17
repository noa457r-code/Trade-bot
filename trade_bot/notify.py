from __future__ import annotations

import logging

import requests

logger = logging.getLogger("notify")


def send_discord_notification(message: str, webhook_url: str | None) -> None:
    """Posts a message to a Discord webhook. No-op if webhook_url is unset,
    so notifications stay optional - the bot works fine without them.
    Network errors are logged, not raised: a failed notification must never
    interrupt trading.
    """
    if not webhook_url:
        return
    try:
        response = requests.post(webhook_url, json={"content": message}, timeout=10)
        response.raise_for_status()
    except requests.RequestException:
        logger.exception("Failed to send Discord notification")
