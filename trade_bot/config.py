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
    atr_period: int


@dataclass
class RiskConfig:
    initial_capital: float
    risk_per_trade: float
    stop_loss_atr_mult: float
    take_profit_atr_mult: float
    fee_pct: float


@dataclass
class PaperTradingConfig:
    poll_interval_seconds: int


@dataclass
class Config:
    broker: str
    environment: str
    instrument: str
    granularity: str
    strategy: StrategyConfig
    risk: RiskConfig
    paper_trading: PaperTradingConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(
            broker=raw["broker"],
            environment=raw["environment"],
            instrument=raw["instrument"],
            granularity=raw["granularity"],
            strategy=StrategyConfig(**raw["strategy"]),
            risk=RiskConfig(**raw["risk"]),
            paper_trading=PaperTradingConfig(**raw["paper_trading"]),
        )
