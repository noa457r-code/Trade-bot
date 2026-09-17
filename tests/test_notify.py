from unittest.mock import patch

import requests

from trade_bot.notify import send_discord_notification


def test_no_webhook_url_does_not_call_requests():
    with patch("trade_bot.notify.requests.post") as mock_post:
        send_discord_notification("hello", None)
    mock_post.assert_not_called()


def test_webhook_url_posts_message_content():
    with patch("trade_bot.notify.requests.post") as mock_post:
        send_discord_notification("hello", "https://discord.com/api/webhooks/test")
    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    assert kwargs["json"] == {"content": "hello"}


def test_request_failure_is_swallowed_not_raised():
    with patch("trade_bot.notify.requests.post", side_effect=requests.ConnectionError("boom")):
        send_discord_notification("hello", "https://discord.com/api/webhooks/test")  # must not raise
