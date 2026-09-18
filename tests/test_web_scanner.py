from __future__ import annotations

import json
from unittest.mock import patch

from trade_bot.scanner import MarketSnapshot
from trade_bot.web_scanner import SnapshotStore, collect_snapshots


def test_snapshot_store_empty_before_first_set():
    store = SnapshotStore()
    payload = json.loads(store.get_json())
    assert payload["markets"] == []
    assert payload["updated_at"] is None


def test_snapshot_store_round_trips_snapshots():
    store = SnapshotStore()
    snapshots = [MarketSnapshot(symbol="AAPL", price=100.0, daily_change_pct=1.5, status_text="x", market_open=True)]

    store.set(snapshots, "2026-09-19T12:00:00+00:00")
    payload = json.loads(store.get_json())

    assert payload["updated_at"] == "2026-09-19T12:00:00+00:00"
    assert payload["markets"] == [
        {"symbol": "AAPL", "price": 100.0, "daily_change_pct": 1.5, "status_text": "x", "market_open": True}
    ]


def test_collect_snapshots_substitutes_error_marker_without_crashing():
    def fake_fetch(symbol, cfg, stock_client, crypto_client, granularity):
        if symbol == "BROKEN":
            raise RuntimeError("boom")
        return MarketSnapshot(symbol=symbol, price=1.0, daily_change_pct=0.0, status_text="ok", market_open=True)

    with patch("trade_bot.web_scanner.fetch_market_snapshot", side_effect=fake_fetch):
        snapshots = collect_snapshots(["AAPL", "BROKEN"], cfg=None, stock_client=None, crypto_client=None, granularity="H1")

    assert len(snapshots) == 2
    aapl, broken = snapshots
    assert aapl.status_text == "ok"
    assert broken.symbol == "BROKEN"
    assert broken.status_text == "FEHLER"
    assert broken.price is None
