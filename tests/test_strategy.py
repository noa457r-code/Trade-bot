from trade_bot.config import StrategyConfig
from trade_bot.strategy import atr, generate_signals, rsi, sma


def make_cfg() -> StrategyConfig:
    return StrategyConfig(
        fast_ma=10, slow_ma=30, rsi_period=14, rsi_buy_max=65, rsi_sell_min=35, atr_period=14
    )


def test_sma_basic():
    import pandas as pd

    s = pd.Series([1, 2, 3, 4, 5])
    result = sma(s, 2)
    assert result.iloc[-1] == 4.5
    assert result.iloc[0] != result.iloc[0]  # NaN for insufficient window


def test_rsi_bounds(synthetic_ohlcv):
    values = rsi(synthetic_ohlcv["close"], 14).dropna()
    assert (values >= 0).all()
    assert (values <= 100).all()


def test_atr_is_positive_and_reflects_volatility(synthetic_ohlcv):
    values = atr(synthetic_ohlcv, 14).dropna()
    assert (values > 0).all()

    flat = synthetic_ohlcv.copy()
    flat["high"] = flat["close"]
    flat["low"] = flat["close"]
    flat_atr = atr(flat, 14).dropna().iloc[-1]
    assert flat_atr < values.iloc[-1]


def test_generate_signals_produces_entries_and_exits(synthetic_ohlcv):
    cfg = make_cfg()
    signals = generate_signals(synthetic_ohlcv, cfg)

    assert "entry_signal" in signals.columns
    assert "exit_signal" in signals.columns
    # The synthetic series has a clear up-then-down trend, so the crossover
    # strategy should fire at least one entry.
    assert signals["entry_signal"].sum() >= 1


def test_no_signals_before_indicators_are_warm(synthetic_ohlcv):
    cfg = make_cfg()
    signals = generate_signals(synthetic_ohlcv, cfg)
    # A rolling window of size P is only valid from index P-1 onward.
    warmup = max(cfg.slow_ma, cfg.rsi_period) - 1
    assert not signals["entry_signal"].iloc[:warmup].any()
    assert not signals["exit_signal"].iloc[:warmup].any()
