from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

from trade_bot.config import StrategyConfig
from trade_bot.data import fetch_crypto_ohlcv, fetch_ohlcv
from trade_bot.strategy import generate_signals
from trade_bot.walkforward import warmup_bars

logger = logging.getLogger("scanner")

DEFAULT_WATCHLIST = ["AAPL", "MSFT", "GLD", "UUP", "FXE", "FXY", "SLV", "PPLT", "BTC/USD"]

# A stock/ETF symbol with no fresh bar in this long is treated as "market
# closed" rather than showing a stale price as if it were live.
STOCK_STALE_THRESHOLD = timedelta(minutes=90)


def is_crypto_symbol(symbol: str) -> bool:
    return "/" in symbol


def compute_daily_change_pct(df: pd.DataFrame) -> float | None:
    """Current close vs. the previous CALENDAR day's last close - not a
    fixed bar-count lookback, since stocks only produce a handful of bars
    per day while crypto produces many more. Falls back to the oldest bar
    in `df` if only one calendar day's worth of data is present.
    """
    if len(df) < 2:
        return None

    current_close = float(df["close"].iloc[-1])
    dates = df.index.normalize()
    last_date = dates[-1]
    prior_mask = dates < last_date

    if prior_mask.any():
        prior_close = float(df.loc[prior_mask, "close"].iloc[-1])
    else:
        prior_close = float(df["close"].iloc[0])

    if prior_close == 0:
        return None
    return (current_close / prior_close - 1) * 100


def is_market_open(symbol: str, df: pd.DataFrame, now: datetime | None = None) -> bool:
    """Crypto trades 24/7, always "open". A stock/ETF is only "open" if its
    freshest bar is recent - otherwise its price/RSI/etc. reflect the last
    close before the market shut, not live movement.
    """
    if is_crypto_symbol(symbol):
        return True
    if df.empty:
        return False
    now = now or datetime.now(timezone.utc)
    return (now - df.index[-1]) <= STOCK_STALE_THRESHOLD


def signal_status_text(last: pd.Series, cfg: StrategyConfig) -> str:
    """Short human-readable read on how close this instrument is to the
    live strategy's entry condition (see trade_bot.strategy.generate_signals).
    """
    if bool(last.get("entry_signal", False)):
        return "ENTRY-SIGNAL AKTIV"

    parts = []
    fast, slow = last.get("fast_ma"), last.get("slow_ma")
    if pd.notna(fast) and pd.notna(slow) and slow != 0:
        gap_pct = (fast / slow - 1) * 100
        parts.append(f"MA-Abstand {gap_pct:+.2f}%")

    rsi = last.get("rsi")
    if pd.notna(rsi):
        parts.append(f"RSI {rsi:.0f} (Grenze {cfg.rsi_buy_max:.0f})")

    trend = last.get("trend_ma")
    close = last.get("close")
    if pd.notna(trend) and pd.notna(close):
        parts.append("ueber Trend" if close > trend else "UNTER Trend")

    return " | ".join(parts) if parts else "warmup (noch nicht genug Bars)"


@dataclass
class MarketSnapshot:
    symbol: str
    price: float | None
    daily_change_pct: float | None
    status_text: str
    market_open: bool


def fetch_market_snapshot(
    symbol: str, cfg: StrategyConfig, stock_client, crypto_client, granularity: str = "H1",
) -> MarketSnapshot:
    max_bars = warmup_bars(cfg)
    if is_crypto_symbol(symbol):
        df = fetch_crypto_ohlcv(crypto_client, symbol, granularity, max_bars=max_bars)
    else:
        df = fetch_ohlcv(stock_client, symbol, granularity, max_bars=max_bars)

    if df.empty:
        return MarketSnapshot(symbol=symbol, price=None, daily_change_pct=None, status_text="keine Daten", market_open=False)

    signals = generate_signals(df, cfg)
    last = signals.iloc[-1]

    return MarketSnapshot(
        symbol=symbol,
        price=float(last["close"]),
        daily_change_pct=compute_daily_change_pct(df),
        status_text=signal_status_text(last, cfg),
        market_open=is_market_open(symbol, df),
    )


def format_table(snapshots: list[MarketSnapshot]) -> str:
    ranked = sorted(
        snapshots,
        key=lambda s: s.daily_change_pct if s.daily_change_pct is not None else float("-inf"),
        reverse=True,
    )

    lines = [f"{'Symbol':<10}{'Kurs':>12}{'Tag %':>10}  {'Markt':<10}  Signal"]
    for s in ranked:
        price_str = f"{s.price:.2f}" if s.price is not None else "n/a"
        change_str = f"{s.daily_change_pct:+.2f}%" if s.daily_change_pct is not None else "n/a"
        market_str = "offen" if s.market_open else "geschlossen"
        lines.append(f"{s.symbol:<10}{price_str:>12}{change_str:>10}  {market_str:<10}  {s.status_text}")

    return "\n".join(lines)


def run_scanner(
    watchlist: list[str], cfg: StrategyConfig, stock_client, crypto_client,
    granularity: str = "H1", interval_seconds: int = 60,
) -> None:
    while True:
        snapshots = []
        for symbol in watchlist:
            try:
                snapshots.append(fetch_market_snapshot(symbol, cfg, stock_client, crypto_client, granularity))
            except Exception:
                logger.exception("[%s] Fehler beim Abrufen", symbol)
                snapshots.append(
                    MarketSnapshot(symbol=symbol, price=None, daily_change_pct=None, status_text="FEHLER", market_open=False)
                )

        os.system("clear")
        print(f"Markt-Scanner - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
        print(format_table(snapshots))

        time.sleep(interval_seconds)
