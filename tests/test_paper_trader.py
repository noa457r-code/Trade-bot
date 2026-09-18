from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from alpaca.common.exceptions import APIError
from alpaca.trading.enums import OrderSide, OrderType

from trade_bot.config import Config, PaperTradingConfig, RiskConfig, StrategyConfig
from trade_bot.paper_trader import PaperTrader


def make_config(max_open_positions: int = 2, kill_switch_drawdown_pct: float = 10.0) -> Config:
    risk = RiskConfig(
        initial_capital=100000.0,
        risk_per_trade=0.01,
        stop_loss_atr_mult=1.5,
        take_profit_atr_mult=6.0,
        fee_pct=0.0002,
        max_open_positions=max_open_positions,
        kill_switch_drawdown_pct=kill_switch_drawdown_pct,
    )
    strategy = StrategyConfig(
        fast_ma=10, slow_ma=50, rsi_period=14, rsi_buy_max=75, rsi_sell_min=35, atr_period=14,
    )
    return Config(
        broker="alpaca",
        environment="paper",
        instruments=["AAPL", "MSFT"],
        granularity="H1",
        strategy=strategy,
        risk=risk,
        paper_trading=PaperTradingConfig(poll_interval_seconds=60, status_update_minutes=0),
    )


def _entry_signal_frame() -> pd.DataFrame:
    return pd.DataFrame({"close": [100.0], "atr": [2.0], "entry_signal": [True], "exit_signal": [False]})


def _holding_signal_frame(price: float = 110.0, atr: float = 5.0) -> pd.DataFrame:
    return pd.DataFrame({"close": [price], "atr": [atr], "entry_signal": [False], "exit_signal": [False]})


def _make_position(avg_entry_price: str = "100.0", qty: str = "10") -> MagicMock:
    position = MagicMock()
    position.avg_entry_price = avg_entry_price
    position.qty = qty
    return position


def _make_stop_order(order_id: str = "stop-1") -> MagicMock:
    order = MagicMock()
    order.id = order_id
    order.order_type = OrderType.STOP
    return order


def _make_closing_sell_order(filled_avg_price: float) -> MagicMock:
    order = MagicMock()
    order.side = OrderSide.SELL
    order.filled_avg_price = filled_avg_price
    return order


@pytest.fixture
def trader(tmp_path):
    with patch("trade_bot.paper_trader.make_client"), patch("trade_bot.paper_trader.TradingClient"):
        t = PaperTrader(
            make_config(),
            api_key="key",
            secret_key="secret",
            safety_state_path=tmp_path / "safety_state.json",
        )
    t.trading_client = MagicMock()
    t.data_client = MagicMock()
    # No individually tracked open position for any instrument by default -
    # tests override this per-instrument where needed.
    t.trading_client.get_open_position.side_effect = APIError("no position")
    return t


def test_kill_switch_triggers_on_drawdown_and_closes_all_positions(trader):
    trader.safety_state.peak_equity = 100000.0
    trader.trading_client.get_account.return_value = MagicMock(equity="89000.0")

    with patch("trade_bot.paper_trader.send_discord_notification") as notify:
        status = trader.step()

    assert trader.safety_state.killed is True
    assert trader.trading_client.close_position.call_count == len(trader.cfg.instruments)
    assert any("KILL-SWITCH" in line for line in status)
    notify.assert_called_once()
    assert "KILL-SWITCH" in notify.call_args[0][0]


def test_kill_switch_state_persists_to_disk(trader, tmp_path):
    trader.safety_state.peak_equity = 100000.0
    trader.trading_client.get_account.return_value = MagicMock(equity="89000.0")

    with patch("trade_bot.paper_trader.send_discord_notification"):
        trader.step()

    from trade_bot.safety import SafetyState

    reloaded = SafetyState.load(trader.safety_state_path)
    assert reloaded.killed is True


def test_peak_equity_persists_to_disk_even_without_trigger(trader):
    trader.trading_client.get_account.return_value = MagicMock(equity="105000.0")
    trader.trading_client.get_all_positions.return_value = []

    with patch("trade_bot.paper_trader.generate_signals", return_value=pd.DataFrame(
        {"close": [100.0], "atr": [2.0], "entry_signal": [False], "exit_signal": [False]}
    )):
        trader.step()

    from trade_bot.safety import SafetyState

    reloaded = SafetyState.load(trader.safety_state_path)
    assert reloaded.peak_equity == 105000.0


def test_already_killed_skips_step_entirely(trader):
    trader.safety_state.killed = True
    trader.safety_state.peak_equity = 100000.0

    status = trader.step()

    trader.trading_client.get_account.assert_not_called()
    trader.trading_client.close_position.assert_not_called()
    assert status == []


def test_position_limit_blocks_new_entry_when_already_at_max(trader):
    trader.cfg.risk.max_open_positions = 1
    trader.trading_client.get_account.return_value = MagicMock(equity="100000.0")
    trader.trading_client.get_all_positions.return_value = [MagicMock()]  # already 1 open == max

    with (
        patch("trade_bot.paper_trader.generate_signals", return_value=_entry_signal_frame()),
        patch("trade_bot.paper_trader.send_discord_notification") as notify,
    ):
        status = trader.step()

    trader.trading_client.submit_order.assert_not_called()
    assert any("Limit" in line for line in status)
    assert notify.called
    assert any("Limit" in call.args[0] for call in notify.call_args_list)


def test_entries_allowed_up_to_limit_within_same_step(trader):
    trader.cfg.risk.max_open_positions = 1
    trader.trading_client.get_account.return_value = MagicMock(equity="100000.0")
    trader.trading_client.get_all_positions.return_value = []  # none open yet

    with patch("trade_bot.paper_trader.generate_signals", return_value=_entry_signal_frame()):
        trader.step()

    # Two instruments both signal entry but the limit is 1 - only the first
    # should have gotten an order in.
    assert trader.trading_client.submit_order.call_count == 1


def test_entries_allowed_for_all_instruments_under_limit(trader):
    trader.cfg.risk.max_open_positions = 2
    trader.trading_client.get_account.return_value = MagicMock(equity="100000.0")
    trader.trading_client.get_all_positions.return_value = []

    with patch("trade_bot.paper_trader.generate_signals", return_value=_entry_signal_frame()):
        trader.step()

    assert trader.trading_client.submit_order.call_count == 2


def test_moves_stop_to_breakeven_once_profit_reaches_threshold(trader):
    trader.trading_client.get_open_position.side_effect = None
    trader.trading_client.get_open_position.return_value = _make_position(avg_entry_price="100.0")
    trader.trading_client.get_orders.return_value = [_make_stop_order("stop-aapl")]

    with (
        patch("trade_bot.paper_trader.generate_signals", return_value=_holding_signal_frame(price=110.0, atr=5.0)),
        patch("trade_bot.paper_trader.send_discord_notification") as notify,
    ):
        trader.step_instrument("AAPL", open_position_count=0)

    trader.trading_client.replace_order_by_id.assert_called_once()
    order_id_arg, replace_request = trader.trading_client.replace_order_by_id.call_args[0]
    assert order_id_arg == "stop-aapl"
    assert replace_request.stop_price == 100.0
    assert notify.called


def test_does_not_move_stop_when_profit_below_threshold(trader):
    trader.trading_client.get_open_position.side_effect = None
    trader.trading_client.get_open_position.return_value = _make_position(avg_entry_price="108.0")

    with patch("trade_bot.paper_trader.generate_signals", return_value=_holding_signal_frame(price=110.0, atr=5.0)):
        trader.step_instrument("AAPL", open_position_count=0)

    trader.trading_client.replace_order_by_id.assert_not_called()


def test_does_not_repeat_breakeven_move_once_already_done(trader):
    trader.trading_client.get_open_position.side_effect = None
    trader.trading_client.get_open_position.return_value = _make_position(avg_entry_price="100.0")
    trader.trading_client.get_orders.return_value = [_make_stop_order("stop-aapl")]

    with patch("trade_bot.paper_trader.generate_signals", return_value=_holding_signal_frame(price=110.0, atr=5.0)):
        trader.step_instrument("AAPL", open_position_count=0)
        trader.step_instrument("AAPL", open_position_count=0)

    assert trader.trading_client.replace_order_by_id.call_count == 1


def test_no_crash_when_stop_leg_order_not_found(trader):
    trader.trading_client.get_open_position.side_effect = None
    trader.trading_client.get_open_position.return_value = _make_position(avg_entry_price="100.0")
    trader.trading_client.get_orders.return_value = []

    with patch("trade_bot.paper_trader.generate_signals", return_value=_holding_signal_frame(price=110.0, atr=5.0)):
        line, entered = trader.step_instrument("AAPL", open_position_count=0)

    trader.trading_client.replace_order_by_id.assert_not_called()
    assert entered is False


def test_reports_realized_profit_when_position_closes_between_steps(trader):
    trader.trading_client.get_open_position.side_effect = None
    trader.trading_client.get_open_position.return_value = _make_position(avg_entry_price="100.0", qty="10")

    # Step 1: still holding - records the open position for later close detection.
    with patch("trade_bot.paper_trader.generate_signals", return_value=_holding_signal_frame(price=105.0, atr=5.0)):
        trader.step_instrument("AAPL", open_position_count=0)

    # Step 2: position gone (take-profit/stop filled at Alpaca), closing SELL order found.
    trader.trading_client.get_open_position.side_effect = APIError("no position")
    trader.trading_client.get_orders.return_value = [_make_closing_sell_order(filled_avg_price=115.0)]

    with (
        patch("trade_bot.paper_trader.generate_signals", return_value=_holding_signal_frame(price=115.0, atr=5.0)),
        patch("trade_bot.paper_trader.send_discord_notification") as notify,
    ):
        line, entered = trader.step_instrument("AAPL", open_position_count=0)

    assert entered is False
    assert "150.00" in line  # (115 - 100) * 10 realized profit
    notify.assert_called_once()
    assert "150.00" in notify.call_args[0][0]


def test_reports_realized_loss_when_position_closes_between_steps(trader):
    trader.trading_client.get_open_position.side_effect = None
    trader.trading_client.get_open_position.return_value = _make_position(avg_entry_price="100.0", qty="10")

    with patch("trade_bot.paper_trader.generate_signals", return_value=_holding_signal_frame(price=98.0, atr=5.0)):
        trader.step_instrument("AAPL", open_position_count=0)

    trader.trading_client.get_open_position.side_effect = APIError("no position")
    trader.trading_client.get_orders.return_value = [_make_closing_sell_order(filled_avg_price=95.0)]

    with (
        patch("trade_bot.paper_trader.generate_signals", return_value=_holding_signal_frame(price=95.0, atr=5.0)),
        patch("trade_bot.paper_trader.send_discord_notification") as notify,
    ):
        line, entered = trader.step_instrument("AAPL", open_position_count=0)

    assert "-50.00" in line  # (95 - 100) * 10 realized loss
    assert "-50.00" in notify.call_args[0][0]


def test_fetch_recent_lookback_accounts_for_trend_ma(trader):
    # Regression test: _fetch_recent used to compute lookback only from
    # slow_ma/rsi_period, ignoring trend_ma entirely. With trend_ma=200 that
    # meant fewer than 200 bars were ever fetched, so trend_ma stayed NaN
    # forever and entry_signal could never fire - Jef never traded live
    # because of this.
    trader.cfg.strategy.trend_ma = 200
    with patch("trade_bot.paper_trader.fetch_ohlcv") as fetch_mock:
        fetch_mock.return_value = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        trader._fetch_recent("AAPL")

    max_bars = fetch_mock.call_args.kwargs["max_bars"]
    assert max_bars >= 200


def test_no_close_report_for_position_that_was_never_confirmed_open(trader):
    # get_open_position always raises APIError (default fixture behaviour) -
    # this instrument was never tracked as open, so a still-None position
    # must not be misreported as a "closed" trade.
    with (
        patch("trade_bot.paper_trader.generate_signals", return_value=_holding_signal_frame(price=100.0, atr=5.0)),
        patch("trade_bot.paper_trader.send_discord_notification") as notify,
    ):
        line, entered = trader.step_instrument("AAPL", open_position_count=0)

    notify.assert_not_called()
    assert line is None
    assert entered is False
