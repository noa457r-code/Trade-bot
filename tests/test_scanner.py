from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from trade_bot.config import StrategyConfig
from trade_bot.scanner import (
    MarketSnapshot,
    compute_daily_change_pct,
    format_table,
    is_market_open,
    signal_status_text,
)


def _make_df(closes: list[float], start: str = "2026-09-15", freq: str = "h") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(closes), freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": [10.0] * len(closes)},
        index=idx,
    )


def test_daily_change_uses_previous_calendar_days_last_close():
    # Day 1 (2026-09-15) closes at 100, day 2 (2026-09-16) currently at 110.
    idx = pd.to_datetime([
        "2026-09-15 20:00", "2026-09-15 21:00", "2026-09-15 22:00",
        "2026-09-16 08:00", "2026-09-16 09:00", "2026-09-16 10:00",
    ], utc=True)
    df = pd.DataFrame({"close": [98, 99, 100, 101, 105, 110]}, index=idx)

    result = compute_daily_change_pct(df)

    assert result == (110 / 100 - 1) * 100


def test_daily_change_returns_none_with_fewer_than_two_bars():
    df = _make_df([100.0])
    assert compute_daily_change_pct(df) is None


def test_daily_change_falls_back_to_first_bar_when_only_one_day_present():
    df = _make_df([100.0, 102.0, 105.0])  # all same calendar day
    result = compute_daily_change_pct(df)
    assert result == (105 / 100 - 1) * 100


def test_market_open_always_true_for_crypto_symbol():
    now = datetime.now(timezone.utc)
    stale_df = _make_df([100.0], start=(now - timedelta(hours=10)).isoformat())
    assert is_market_open("BTC/USD", stale_df, now=now) is True


def test_market_open_true_for_stock_with_recent_bar():
    now = datetime.now(timezone.utc)
    df = _make_df([100.0])
    df.index = pd.DatetimeIndex([now - timedelta(minutes=30)])
    assert is_market_open("AAPL", df, now=now) is True


def test_market_open_false_for_stock_with_stale_bar():
    now = datetime.now(timezone.utc)
    df = _make_df([100.0])
    df.index = pd.DatetimeIndex([now - timedelta(hours=3)])
    assert is_market_open("AAPL", df, now=now) is False


def _cfg() -> StrategyConfig:
    return StrategyConfig(fast_ma=10, slow_ma=50, rsi_period=14, rsi_buy_max=75, rsi_sell_min=35, atr_period=14)


def test_signal_status_shows_entry_active():
    last = pd.Series({"entry_signal": True, "fast_ma": 10.0, "slow_ma": 9.0, "rsi": 60.0, "close": 100.0})
    assert "ENTRY-SIGNAL" in signal_status_text(last, _cfg())


def test_signal_status_shows_indicator_breakdown_without_signal():
    last = pd.Series({"entry_signal": False, "fast_ma": 10.0, "slow_ma": 9.0, "rsi": 60.0, "close": 100.0})
    text = signal_status_text(last, _cfg())
    assert "RSI" in text
    assert "ENTRY-SIGNAL" not in text


def test_signal_status_handles_missing_indicators_without_crashing():
    last = pd.Series({"entry_signal": False, "fast_ma": float("nan"), "slow_ma": float("nan"), "rsi": float("nan")})
    text = signal_status_text(last, _cfg())
    assert isinstance(text, str)


def test_format_table_sorts_by_daily_change_descending():
    snapshots = [
        MarketSnapshot(symbol="A", price=100.0, daily_change_pct=1.0, status_text="x", market_open=True),
        MarketSnapshot(symbol="B", price=100.0, daily_change_pct=5.0, status_text="x", market_open=True),
        MarketSnapshot(symbol="C", price=100.0, daily_change_pct=-2.0, status_text="x", market_open=True),
    ]
    table = format_table(snapshots)
    assert table.index("B") < table.index("A") < table.index("C")


def test_format_table_places_unknown_change_last_without_crashing():
    snapshots = [
        MarketSnapshot(symbol="A", price=100.0, daily_change_pct=1.0, status_text="x", market_open=True),
        MarketSnapshot(symbol="B", price=None, daily_change_pct=None, status_text="warmup", market_open=False),
    ]
    table = format_table(snapshots)
    assert table.index("A") < table.index("B")
