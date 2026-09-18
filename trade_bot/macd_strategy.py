from __future__ import annotations

import itertools
from dataclasses import dataclass

import pandas as pd

from trade_bot.strategy import atr


@dataclass
class MACDConfig:
    fast_period: int = 12   # classic MACD default (12/26/9)
    slow_period: int = 26
    signal_period: int = 9
    atr_period: int = 14


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, min_periods=period, adjust=False).mean()


def generate_signals(df: pd.DataFrame, cfg: MACDConfig) -> pd.DataFrame:
    """MACD crossover strategy, long-only (no shorts), same MA-crossover
    idea as `trade_bot.strategy` but on the MACD line vs. its signal line
    instead of two plain moving averages.

    Long entry: MACD line crosses above its signal line.
    Exit:       MACD line crosses below its signal line.
    """
    out = df.copy()
    out["atr"] = atr(out, cfg.atr_period)

    macd_line = _ema(out["close"], cfg.fast_period) - _ema(out["close"], cfg.slow_period)
    signal_line = _ema(macd_line, cfg.signal_period)
    out["macd"] = macd_line
    out["macd_signal"] = signal_line

    macd_above_signal = macd_line > signal_line
    crossed_up = macd_above_signal & ~macd_above_signal.shift(1, fill_value=False)
    crossed_down = ~macd_above_signal & macd_above_signal.shift(1, fill_value=False)

    valid = out["atr"].notna() & macd_line.notna() & signal_line.notna()
    out["entry_signal"] = crossed_up & valid
    out["exit_signal"] = crossed_down & valid

    return out


# Parameter grid for walk-forward validation, plus the same ATR stop/target
# multipliers used for the other strategies' grids so results are comparable.
PARAM_GRID: list[dict] = [
    {"fast_period": fast, "slow_period": slow, "signal_period": signal,
     "stop_loss_atr_mult": sl_mult, "take_profit_atr_mult": tp_mult}
    for (fast, slow), signal, sl_mult, tp_mult in itertools.product(
        [(8, 17), (12, 26), (19, 39)],
        [9],
        [1.5, 2.0, 3.0],
        [3.0, 4.0, 6.0],
    )
]
