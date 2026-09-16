from trade_bot.backtest import run_backtest
from trade_bot.config import RiskConfig, StrategyConfig
from trade_bot.strategy import generate_signals


def make_risk_cfg() -> RiskConfig:
    return RiskConfig(
        initial_capital=10000,
        risk_per_trade=0.01,
        stop_loss_atr_mult=2.0,
        take_profit_atr_mult=4.0,
        fee_pct=0.001,
    )


def make_strategy_cfg() -> StrategyConfig:
    return StrategyConfig(
        fast_ma=10, slow_ma=30, rsi_period=14, rsi_buy_max=65, rsi_sell_min=35, atr_period=14
    )


def test_backtest_runs_and_tracks_equity(synthetic_ohlcv):
    signals = generate_signals(synthetic_ohlcv, make_strategy_cfg())
    result = run_backtest(signals, make_risk_cfg())

    assert len(result.equity_curve) == len(synthetic_ohlcv)
    assert result.equity_curve.iloc[0] > 0


def test_backtest_never_risks_more_than_equity(synthetic_ohlcv):
    risk_cfg = make_risk_cfg()
    signals = generate_signals(synthetic_ohlcv, make_strategy_cfg())
    result = run_backtest(signals, risk_cfg)

    # Equity should never go negative even through losing trades.
    assert (result.equity_curve > 0).all()


def test_backtest_summary_is_a_string(synthetic_ohlcv):
    signals = generate_signals(synthetic_ohlcv, make_strategy_cfg())
    result = run_backtest(signals, make_risk_cfg())
    assert isinstance(result.summary(), str)
    assert "Total return" in result.summary()
