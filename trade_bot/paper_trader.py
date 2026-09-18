from __future__ import annotations

import logging
import math
import time

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.common.enums import Sort
from alpaca.trading.enums import OrderClass, OrderSide, OrderType, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    GetPortfolioHistoryRequest,
    MarketOrderRequest,
    ReplaceOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

from trade_bot.advisor import equity_based_tips, error_based_tips
from trade_bot.config import Config
from trade_bot.data import fetch_ohlcv, make_client
from trade_bot.notify import send_discord_notification
from trade_bot.risk import size_position
from trade_bot.safety import SafetyState, check_kill_switch, is_position_limit_reached
from trade_bot.strategy import generate_signals

logger = logging.getLogger("paper_trader")


class PaperTrader:
    """Trades on Alpaca's paper account using live market data.

    Entries are submitted as bracket orders (market entry + stop-loss +
    take-profit legs), so Alpaca handles exits at those levels itself. A
    strategy exit signal closes the position early via `close_position`.
    Always targets the `paper` trading endpoint - no real funds are ever
    involved.
    """

    def __init__(
        self,
        cfg: Config,
        api_key: str,
        secret_key: str,
        discord_webhook_url: str | None = None,
        safety_state_path: str = "safety_state.json",
    ):
        self.cfg = cfg
        self.data_client = make_client(api_key, secret_key)
        self.trading_client = TradingClient(api_key, secret_key, paper=True)
        self.discord_webhook_url = discord_webhook_url
        self.safety_state_path = safety_state_path
        self.safety_state = SafetyState.load(safety_state_path)
        self._error_streaks: dict[str, int] = {}
        self._breakeven_moved: dict[str, bool] = {}
        self._open_positions_seen: dict[str, tuple[float, float]] = {}

    def _fetch_recent(self, instrument: str):
        lookback_bars = max(self.cfg.strategy.slow_ma, self.cfg.strategy.rsi_period) * 3
        return fetch_ohlcv(
            self.data_client,
            instrument,
            self.cfg.granularity,
            max_bars=lookback_bars,
        )

    def _current_position(self, instrument: str):
        try:
            return self.trading_client.get_open_position(instrument)
        except APIError:
            return None

    def _find_stop_leg_order(self, instrument: str):
        orders = self.trading_client.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[instrument])
        )
        for order in orders:
            if order.order_type == OrderType.STOP:
                return order
        return None

    def _maybe_move_stop_to_breakeven(self, instrument: str, position, price: float, atr_value: float) -> None:
        """Once unrealized profit reaches `breakeven_trigger_atr_mult` ATR,
        moves the bracket order's stop-loss leg to the entry price - the
        trade can no longer close at a loss from this point on. Runs at
        most once per open position (tracked in `_breakeven_moved`,
        cleared once the position closes).
        """
        if self._breakeven_moved.get(instrument):
            return

        entry_price = float(position.avg_entry_price)
        if price - entry_price < atr_value * self.cfg.risk.breakeven_trigger_atr_mult:
            return

        stop_order = self._find_stop_leg_order(instrument)
        if stop_order is None:
            logger.warning("[%s] Breakeven-Trigger erreicht, aber kein offener Stop-Leg gefunden", instrument)
            return

        self.trading_client.replace_order_by_id(stop_order.id, ReplaceOrderRequest(stop_price=round(entry_price, 2)))
        self._breakeven_moved[instrument] = True
        logger.info("[%s] Stop auf Break-Even (%.2f) nachgezogen", instrument, entry_price)
        send_discord_notification(
            f"{instrument}: Stop auf Break-Even ({entry_price:.2f}) nachgezogen - "
            f"Trade kann nicht mehr ins Minus laufen",
            self.discord_webhook_url,
        )

    def _find_latest_closing_order(self, instrument: str):
        orders = self.trading_client.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.CLOSED, symbols=[instrument], limit=5, direction=Sort.DESC)
        )
        for order in orders:
            if order.side == OrderSide.SELL and order.filled_avg_price is not None:
                return order
        return None

    def _report_closed_trade_if_any(self, instrument: str) -> str | None:
        """Called whenever `_current_position` returns None - checks whether
        this instrument had a position we were tracking as open (recorded
        while holding, see the caller). If so, it just closed since the
        last step (signal-exit, take-profit fill, stop-loss fill, or
        break-even fill all look the same from here) - looks up the fill
        price of the closing order and reports realized P&L. Returns None
        (no false report) if this instrument was never confirmed open, e.g.
        an entry order still sitting unfilled outside market hours.
        """
        self._breakeven_moved.pop(instrument, None)
        tracked = self._open_positions_seen.pop(instrument, None)
        if tracked is None:
            return None

        entry_price, qty = tracked
        closing_order = self._find_latest_closing_order(instrument)
        if closing_order is None:
            logger.warning("[%s] Position geschlossen, aber kein Fill-Preis gefunden - P&L unbekannt", instrument)
            message = f"{instrument}: Position geschlossen, P&L unbekannt (kein Fill-Preis gefunden)"
            send_discord_notification(message, self.discord_webhook_url)
            return message

        exit_price = float(closing_order.filled_avg_price)
        pnl = (exit_price - entry_price) * qty
        logger.info("[%s] Position geschlossen @ %.2f | P&L=%+.2f", instrument, exit_price, pnl)
        message = f"{instrument}: Position geschlossen @ {exit_price:.2f} | realisierter P&L: {pnl:+.2f}"
        send_discord_notification(message, self.discord_webhook_url)
        return message

    def step_instrument(self, instrument: str, open_position_count: int) -> tuple[str | None, bool]:
        """Runs one strategy step for `instrument`, returns a one-line status
        string ONLY if there's something worth reporting in the periodic
        digest (an open position, an error, or a rejected order) -
        flat/no-signal instruments return None so the digest stays focused
        on what matters. Second value is True only when a NEW entry was
        just submitted this step (used by `step` to track the running
        open-position count against `max_open_positions` within one pass).

        `open_position_count` is how many positions are open across all
        instruments as of the start of this step, plus any entries already
        submitted earlier in the same step's loop.
        """
        df = self._fetch_recent(instrument)
        signals = generate_signals(df, self.cfg.strategy)
        last = signals.iloc[-1]
        price = float(last["close"])

        position = self._current_position(instrument)

        if position is not None:
            # Recorded on every holding step (not just once at entry) so the
            # snapshot right before a close is always the freshest one -
            # `_report_closed_trade_if_any` reads it back once this position
            # disappears, whichever way it closed.
            self._open_positions_seen[instrument] = (float(position.avg_entry_price), float(position.qty))

            if bool(last["exit_signal"]):
                self.trading_client.close_position(instrument)
                logger.info("[%s] EXIT-Signal ausgeloest @ ~%.2f - P&L folgt nach Fill", instrument, price)
                return f"{instrument}: EXIT-Signal ausgeloest @ {price:.2f}", False
            self._maybe_move_stop_to_breakeven(instrument, position, price, float(last["atr"]))
            logger.debug("[%s] Holding qty=%s @ ~%.2f", instrument, position.qty, price)
            return f"{instrument}: haelt qty={position.qty} @ {price:.2f}", False

        closed_line = self._report_closed_trade_if_any(instrument)
        if closed_line is not None:
            return closed_line, False

        if not bool(last["entry_signal"]):
            logger.debug("[%s] No signal @ %.2f", instrument, price)
            return None, False

        if is_position_limit_reached(open_position_count, self.cfg.risk.max_open_positions):
            logger.info("[%s] Entry-Signal, aber Positions-Limit erreicht - Order abgelehnt", instrument)
            send_discord_notification(
                f"{instrument}: Entry-Signal @ ~{price:.2f}, aber Positions-Limit "
                f"({self.cfg.risk.max_open_positions}) erreicht - Order abgelehnt",
                self.discord_webhook_url,
            )
            return f"{instrument}: Entry-Signal, aber Positions-Limit erreicht - abgelehnt", False

        # Equity is re-read live before every entry, so risk_per_trade is
        # sized against current account equity across all instruments
        # sharing this one paper account.
        equity = float(self.trading_client.get_account().equity)
        sizing = size_position(price, float(last["atr"]), equity, self.cfg.risk)
        quantity = math.floor(sizing.quantity)
        if quantity < 1:
            logger.debug("[%s] Signal but position size < 1 share @ %.2f | equity=%.2f", instrument, price, equity)
            return None, False

        order = MarketOrderRequest(
            symbol=instrument,
            qty=quantity,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round(sizing.take_profit_price, 2)),
            stop_loss=StopLossRequest(stop_price=round(sizing.stop_loss_price, 2)),
        )
        self.trading_client.submit_order(order)
        logger.info(
            "[%s] ENTRY @ %.2f | qty=%d | stop=%.2f | target=%.2f",
            instrument, price, quantity, sizing.stop_loss_price, sizing.take_profit_price,
        )
        send_discord_notification(
            f"ENTRY {instrument} @ ~{price:.2f} | qty={quantity} | "
            f"stop={sizing.stop_loss_price:.2f} | target={sizing.take_profit_price:.2f}",
            self.discord_webhook_url,
        )
        return f"{instrument}: ENTRY gerade ausgeloest @ {price:.2f}", True

    def _trigger_kill_switch(self, equity: float) -> list[str]:
        logger.warning(
            "KILL-SWITCH ausgeloest bei Equity %.2f (Hoch: %.2f) - schliesse alle Positionen",
            equity, self.safety_state.peak_equity,
        )
        for instrument in self.cfg.instruments:
            try:
                self.trading_client.close_position(instrument)
            except APIError:
                pass  # nothing open for this instrument, nothing to close

        message = (
            f"KILL-SWITCH ausgeloest! Konto: {equity:.2f}, Hoch: {self.safety_state.peak_equity:.2f}. "
            f"Alle offenen Positionen werden geschlossen. Bot pausiert bis manueller Reset "
            f"(`trade-bot reset-killswitch`)."
        )
        send_discord_notification(message, self.discord_webhook_url)
        return [message]

    def step(self) -> list[str]:
        """Runs a step for every instrument, returns only the noteworthy
        status lines (open positions, entries/exits, errors) - flat/no-signal
        instruments are omitted so the periodic digest stays short.

        Skips everything once the kill switch has tripped (manual reset
        required), and checks it fresh on every step before touching any
        instrument.
        """
        if self.safety_state.killed:
            logger.debug("Kill-switch aktiv - Trading pausiert")
            return []

        equity = float(self.trading_client.get_account().equity)
        triggered = check_kill_switch(equity, self.safety_state, self.cfg.risk.kill_switch_drawdown_pct)
        self.safety_state.save(self.safety_state_path)  # persist peak_equity every step, not just on trigger
        if triggered:
            return self._trigger_kill_switch(equity)

        open_position_count = len(self.trading_client.get_all_positions())

        status_lines = []
        for instrument in self.cfg.instruments:
            try:
                line, entered = self.step_instrument(instrument, open_position_count)
                if entered:
                    open_position_count += 1
                if line is not None:
                    status_lines.append(line)
                self._error_streaks[instrument] = 0
            except Exception:
                logger.exception("[%s] Error during paper trading step", instrument)
                status_lines.append(f"{instrument}: FEHLER beim Abfragen (siehe Log)")
                self._error_streaks[instrument] = self._error_streaks.get(instrument, 0) + 1
        return status_lines

    def _fetch_tips(self) -> list[str]:
        """Rule-based tips from real account state - only returns something
        when a threshold actually triggers (see trade_bot.advisor)."""
        tips = error_based_tips(self._error_streaks)
        try:
            history = self.trading_client.get_portfolio_history(
                GetPortfolioHistoryRequest(period="1M", timeframe="1D")
            )
            tips += equity_based_tips(list(history.equity), history.base_value)
        except Exception:
            logger.exception("Failed to fetch portfolio history for tips")
        return tips

    def send_status_digest(self, status_lines: list[str]) -> None:
        equity = float(self.trading_client.get_account().equity)
        body = "\n".join(status_lines) if status_lines else "keine offenen Positionen, keine Signale"
        message = f"Status-Update | Kontostand: {equity:.2f}\n{body}"

        tips = self._fetch_tips()
        if tips:
            message += "\n\nTipps:\n" + "\n".join(f"- {t}" for t in tips)

        send_discord_notification(message, self.discord_webhook_url)
        logger.info("Sent periodic status digest to Discord")

    def run_forever(self) -> None:
        logger.info(
            "Starting paper trading on %s %s (Alpaca paper account, no real funds)",
            ", ".join(self.cfg.instruments), self.cfg.granularity,
        )
        status_interval = self.cfg.paper_trading.status_update_minutes * 60
        last_status_at = time.monotonic()
        while True:
            status_lines = self.step()

            now = time.monotonic()
            if status_interval > 0 and now - last_status_at >= status_interval:
                try:
                    self.send_status_digest(status_lines)
                except Exception:
                    logger.exception("Failed to send periodic status digest")
                last_status_at = now

            time.sleep(self.cfg.paper_trading.poll_interval_seconds)
