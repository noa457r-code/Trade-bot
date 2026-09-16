from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class StrategyConfig:
    fast_ma: int
    slow_ma: int
    rsi_period: int
    rsi_buy_max: float
    rsi_sell_min: float


@dataclass
class RiskConfig:
    initial_capital: float
    risk_per_trade: float
    stop_loss_pct: float
    take_profit_pct: float
    fee_pct: float


@dataclass
class PaperTradingConfig:
    poll_interval_seconds: int


@dataclass
class Config:
    exchange: str
    symbol: str
    timeframe: str
    strategy: StrategyConfig
    risk: RiskConfig
    paper_trading: PaperTradingConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(
            exchange=raw["exchange"],
            symbol=raw["symbol"],
            timeframe=raw["timeframe"],
            strategy=StrategyConfig(**raw["strategy"]),
            risk=RiskConfig(**raw["risk"]),
            paper_trading=PaperTradingConfig(**raw["paper_trading"]),
        )
