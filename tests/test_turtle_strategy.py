from __future__ import annotations

import pandas as pd

from trade_bot.backtest import run_backtest
from trade_bot.config import RiskConfig
from trade_bot.turtle_strategy import TurtleConfig, generate_signals


def _make_df(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=len(closes), freq="h", tz="UTC")
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": [10.0] * len(closes)},
        index=idx,
    )


def test_entry_signal_on_breakout_above_prior_n_bar_high():
    # Flat at 10 for 3 bars, then a bar that breaks above that 3-bar high.
    df = _make_df([10, 10, 10, 15, 15, 15])
    cfg = TurtleConfig(entry_channel=3, exit_channel=2, atr_period=2)

    signals = generate_signals(df, cfg)

    assert bool(signals["entry_signal"].iloc[3]) is True


def test_no_entry_signal_without_a_breakout():
    df = _make_df([10, 10, 10, 10, 10, 10])
    cfg = TurtleConfig(entry_channel=3, exit_channel=2, atr_period=2)

    signals = generate_signals(df, cfg)

    assert not signals["entry_signal"].any()


def test_exit_signal_on_breakdown_below_prior_n_bar_low():
    df = _make_df([10, 10, 10, 15, 15, 8])
    cfg = TurtleConfig(entry_channel=3, exit_channel=2, atr_period=2)

    signals = generate_signals(df, cfg)

    assert bool(signals["exit_signal"].iloc[5]) is True


def test_no_signal_during_warmup_before_channel_is_full():
    df = _make_df([10, 10, 10, 15, 15, 8])
    cfg = TurtleConfig(entry_channel=3, exit_channel=2, atr_period=2)

    signals = generate_signals(df, cfg)

    # Fewer than `entry_channel` prior bars exist yet - can't have a valid
    # breakout signal, warmup bars must read False, not NaN/crash.
    assert not bool(signals["entry_signal"].iloc[0])
    assert not bool(signals["entry_signal"].iloc[1])
    assert not bool(signals["exit_signal"].iloc[0])


def test_atr_column_present_for_downstream_position_sizing():
    df = _make_df([10, 10, 10, 15, 15, 8])
    cfg = TurtleConfig(entry_channel=3, exit_channel=2, atr_period=2)

    signals = generate_signals(df, cfg)

    assert "atr" in signals.columns
    assert signals["atr"].iloc[3] > 0


def test_signals_are_consumable_by_run_backtest(synthetic_ohlcv):
    cfg = TurtleConfig(entry_channel=20, exit_channel=10, atr_period=14)
    risk_cfg = RiskConfig(
        initial_capital=10000, risk_per_trade=0.01, stop_loss_atr_mult=1.5,
        take_profit_atr_mult=4.0, fee_pct=0.0002,
    )

    signals = generate_signals(synthetic_ohlcv, cfg)
    result = run_backtest(signals, risk_cfg)

    assert isinstance(result.total_return_pct, float)
