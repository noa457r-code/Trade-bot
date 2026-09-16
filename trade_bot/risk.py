from __future__ import annotations

from dataclasses import dataclass

from trade_bot.config import RiskConfig


@dataclass
class PositionSizing:
    quantity: float
    stop_loss_price: float
    take_profit_price: float


def size_position(entry_price: float, equity: float, cfg: RiskConfig) -> PositionSizing:
    """Size a position so that hitting the stop-loss loses at most
    `risk_per_trade` of current equity.
    """
    stop_loss_price = entry_price * (1 - cfg.stop_loss_pct)
    take_profit_price = entry_price * (1 + cfg.take_profit_pct)

    risk_amount = equity * cfg.risk_per_trade
    price_risk_per_unit = entry_price - stop_loss_price
    if price_risk_per_unit <= 0:
        quantity = 0.0
    else:
        quantity = risk_amount / price_risk_per_unit

    # Never risk more notional than the available equity allows.
    max_affordable_qty = equity / entry_price
    quantity = min(quantity, max_affordable_qty)

    return PositionSizing(
        quantity=quantity,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
    )
