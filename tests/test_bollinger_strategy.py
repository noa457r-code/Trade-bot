from __future__ import annotations

from trade_bot.backtest import run_backtest
from trade_bot.bollinger_strategy import BollingerConfig, generate_signals
from trade_bot.config import RiskConfig


def make_cfg() -> BollingerConfig:
    return BollingerConfig(period=20, num_std=2.0, atr_period=14)


def test_generate_signals_produces_entries_and_exits(synthetic_ohlcv):
    cfg = make_cfg()
    signals = generate_signals(synthetic_ohlcv, cfg)

    assert "entry_signal" in signals.columns
    assert "exit_signal" in signals.columns
    # The synthetic series oscillates on top of its trend, so price should
    # cross below the lower band and back above the mean at least once.
    assert signals["entry_signal"].sum() >= 1
    assert signals["exit_signal"].sum() >= 1


def test_no_signals_before_indicators_are_warm(synthetic_ohlcv):
    cfg = make_cfg()
    signals = generate_signals(synthetic_ohlcv, cfg)

    warmup = cfg.period - 1
    assert not signals["entry_signal"].iloc[:warmup].any()
    assert not signals["exit_signal"].iloc[:warmup].any()


def test_entry_only_when_close_below_lower_band(synthetic_ohlcv):
    cfg = make_cfg()
    signals = generate_signals(synthetic_ohlcv, cfg)

    entries = signals[signals["entry_signal"]]
    assert (entries["close"] < entries["bb_lower"]).all()


def test_atr_column_present_for_downstream_position_sizing(synthetic_ohlcv):
    cfg = make_cfg()
    signals = generate_signals(synthetic_ohlcv, cfg)

    assert "atr" in signals.columns
    assert (signals["atr"].dropna() > 0).all()


def test_trend_filter_blocks_entries_below_trend_ma(synthetic_ohlcv):
    cfg_no_filter = make_cfg()
    cfg_with_filter = BollingerConfig(**{**vars(cfg_no_filter), "trend_ma": 250})

    baseline = generate_signals(synthetic_ohlcv, cfg_no_filter)
    filtered = generate_signals(synthetic_ohlcv, cfg_with_filter)

    below_trend = filtered["close"] <= filtered["trend_ma"]
    assert not (filtered["entry_signal"] & below_trend).any()
    # The filter should only ever remove entries, never add new ones.
    assert (filtered["entry_signal"] & ~baseline["entry_signal"]).sum() == 0


def test_signals_are_consumable_by_run_backtest(synthetic_ohlcv):
    cfg = make_cfg()
    risk_cfg = RiskConfig(
        initial_capital=10000, risk_per_trade=0.01, stop_loss_atr_mult=1.5,
        take_profit_atr_mult=4.0, fee_pct=0.0002,
    )

    signals = generate_signals(synthetic_ohlcv, cfg)
    result = run_backtest(signals, risk_cfg)

    assert isinstance(result.total_return_pct, float)
