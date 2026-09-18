from __future__ import annotations

from trade_bot.backtest import run_backtest
from trade_bot.config import RiskConfig
from trade_bot.macd_strategy import MACDConfig, generate_signals


def make_cfg() -> MACDConfig:
    return MACDConfig(fast_period=12, slow_period=26, signal_period=9, atr_period=14)


def test_generate_signals_produces_entries_and_exits(synthetic_ohlcv):
    cfg = make_cfg()
    signals = generate_signals(synthetic_ohlcv, cfg)

    assert "entry_signal" in signals.columns
    assert "exit_signal" in signals.columns
    # The synthetic series has a clear up-then-down trend, so the MACD
    # crossover should fire at least one entry and one exit.
    assert signals["entry_signal"].sum() >= 1
    assert signals["exit_signal"].sum() >= 1


def test_no_signals_before_indicators_are_warm(synthetic_ohlcv):
    cfg = make_cfg()
    signals = generate_signals(synthetic_ohlcv, cfg)

    warmup = cfg.slow_period + cfg.signal_period
    assert not signals["entry_signal"].iloc[:warmup].any()
    assert not signals["exit_signal"].iloc[:warmup].any()


def test_atr_column_present_for_downstream_position_sizing(synthetic_ohlcv):
    cfg = make_cfg()
    signals = generate_signals(synthetic_ohlcv, cfg)

    assert "atr" in signals.columns
    assert (signals["atr"].dropna() > 0).all()


def test_signals_are_consumable_by_run_backtest(synthetic_ohlcv):
    cfg = make_cfg()
    risk_cfg = RiskConfig(
        initial_capital=10000, risk_per_trade=0.01, stop_loss_atr_mult=1.5,
        take_profit_atr_mult=4.0, fee_pct=0.0002,
    )

    signals = generate_signals(synthetic_ohlcv, cfg)
    result = run_backtest(signals, risk_cfg)

    assert isinstance(result.total_return_pct, float)
