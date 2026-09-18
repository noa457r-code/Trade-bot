from __future__ import annotations

from trade_bot.config import RiskConfig, StrategyConfig
from trade_bot.strategy import generate_signals
from trade_bot.turtle_strategy import TurtleConfig
from trade_bot.turtle_strategy import generate_signals as generate_turtle_signals
from trade_bot.walkforward import _warmup_bars, run_walk_forward


def _risk_cfg() -> RiskConfig:
    return RiskConfig(
        initial_capital=10000, risk_per_trade=0.01, stop_loss_atr_mult=1.5,
        take_profit_atr_mult=4.0, fee_pct=0.0002,
    )


def test_run_walk_forward_defaults_to_sma_rsi_strategy(synthetic_ohlcv):
    strategy_cfg = StrategyConfig(fast_ma=10, slow_ma=50, rsi_period=14, rsi_buy_max=75, rsi_sell_min=35, atr_period=14)

    windows = run_walk_forward(
        synthetic_ohlcv, base_strategy=strategy_cfg, base_risk=_risk_cfg(),
        train_bars=100, test_bars=50, step_bars=50, min_trades=0,
    )

    assert all(isinstance(w.strategy_cfg, StrategyConfig) for w in windows)


def test_run_walk_forward_supports_a_custom_strategy_and_param_grid(synthetic_ohlcv):
    turtle_cfg = TurtleConfig(entry_channel=20, exit_channel=10, atr_period=14)
    turtle_grid = [
        {"entry_channel": entry, "exit_channel": exit_, "stop_loss_atr_mult": 1.5, "take_profit_atr_mult": 4.0}
        for entry, exit_ in [(10, 5), (20, 10)]
    ]

    windows = run_walk_forward(
        synthetic_ohlcv, base_strategy=turtle_cfg, base_risk=_risk_cfg(),
        train_bars=100, test_bars=50, step_bars=50, min_trades=0,
        param_grid=turtle_grid, generate_signals_fn=generate_turtle_signals,
    )

    assert all(isinstance(w.strategy_cfg, TurtleConfig) for w in windows)


def test_warmup_bars_uses_largest_int_period_field_times_three():
    strategy_cfg = StrategyConfig(
        fast_ma=10, slow_ma=50, rsi_period=14, rsi_buy_max=75, rsi_sell_min=35, atr_period=14, trend_ma=200,
    )
    assert _warmup_bars(strategy_cfg) == 200 * 3


def test_warmup_bars_ignores_none_fields():
    strategy_cfg = StrategyConfig(
        fast_ma=10, slow_ma=50, rsi_period=14, rsi_buy_max=75.0, rsi_sell_min=35.0, atr_period=14, trend_ma=None,
    )
    assert _warmup_bars(strategy_cfg) == 50 * 3


def test_warmup_bars_ignores_float_fields_even_when_yaml_parses_them_as_int():
    # YAML parses `rsi_buy_max: 75` (no decimal point) as a plain int even
    # though the field is typed float - must not be mistaken for a period.
    strategy_cfg = StrategyConfig(
        fast_ma=10, slow_ma=20, rsi_period=14, rsi_buy_max=75, rsi_sell_min=35, atr_period=14, trend_ma=None,
    )
    assert _warmup_bars(strategy_cfg) == 20 * 3


def test_warmup_bars_works_for_turtle_config():
    turtle_cfg = TurtleConfig(entry_channel=55, exit_channel=20, atr_period=14)
    assert _warmup_bars(turtle_cfg) == 55 * 3
