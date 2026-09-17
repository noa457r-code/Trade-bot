from __future__ import annotations

import pandas as pd

from trade_bot.config import StrategyConfig


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi_values = 100 - (100 / (1 + rs))
    return rsi_values.where(avg_loss != 0, 100.0)


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Average True Range (Wilder's smoothing): measures actual recent
    volatility, so stop-loss/take-profit distances can scale with how much
    an instrument actually moves instead of a fixed percentage.
    """
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def add_indicators(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    out = df.copy()
    out["fast_ma"] = sma(out["close"], cfg.fast_ma)
    out["slow_ma"] = sma(out["close"], cfg.slow_ma)
    out["rsi"] = rsi(out["close"], cfg.rsi_period)
    out["atr"] = atr(out, cfg.atr_period)
    if cfg.trend_ma is not None:
        out["trend_ma"] = sma(out["close"], cfg.trend_ma)
    return out


def generate_signals(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Moving-average crossover strategy with an RSI confirmation filter.

    Long entry: fast MA crosses above slow MA AND RSI is below rsi_buy_max
                (avoids buying into an already-overbought spike) AND,
                if trend_ma is set, close is above trend_ma (only trade
                with the dominant long-term trend).
    Exit:       fast MA crosses below slow MA AND RSI is above rsi_sell_min
                (avoids selling into an already-oversold dip).
    """
    out = add_indicators(df, cfg)

    fast_above_slow = out["fast_ma"] > out["slow_ma"]
    crossed_up = fast_above_slow & ~fast_above_slow.shift(1, fill_value=False)
    crossed_down = ~fast_above_slow & fast_above_slow.shift(1, fill_value=False)

    out["entry_signal"] = crossed_up & (out["rsi"] < cfg.rsi_buy_max)
    out["exit_signal"] = crossed_down & (out["rsi"] > cfg.rsi_sell_min)

    valid = out["fast_ma"].notna() & out["slow_ma"].notna() & out["rsi"].notna() & out["atr"].notna()

    if cfg.trend_ma is not None:
        out["entry_signal"] &= out["close"] > out["trend_ma"]
        valid &= out["trend_ma"].notna()

    out["entry_signal"] &= valid
    out["exit_signal"] &= valid

    return out
