from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

from trade_bot.backtest import run_backtest
from trade_bot.config import Config
from trade_bot.data import fetch_ohlcv, make_client
from trade_bot.paper_trader import PaperTrader
from trade_bot.strategy import generate_signals


def _require_api_token() -> str:
    token = os.environ.get("OANDA_API_TOKEN")
    if not token:
        raise SystemExit(
            "OANDA_API_TOKEN ist nicht gesetzt. Kopiere .env.example nach .env und trage "
            "deinen (kostenlosen) OANDA-Practice-Account-Token ein."
        )
    return token


def cmd_backtest(args: argparse.Namespace) -> None:
    load_dotenv()
    cfg = Config.from_yaml(args.config)
    client = make_client(cfg.environment, _require_api_token())

    since = None
    if args.since:
        since = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    df = fetch_ohlcv(client, cfg.instrument, cfg.granularity, since=since, max_bars=args.bars)

    signals = generate_signals(df, cfg.strategy)
    result = run_backtest(signals, cfg.risk)

    print(f"Instrument: {cfg.instrument} | Granularity: {cfg.granularity} | Bars: {len(df)}")
    print(result.summary())

    if args.trades:
        for t in result.trades:
            exit_price = f"{t.exit_price:.5f}" if t.exit_price is not None else "OPEN"
            print(
                f"  {t.entry_time} entry={t.entry_price:.5f} -> "
                f"{t.exit_time} exit={exit_price} ({t.exit_reason}) pnl={t.pnl:.2f}"
            )


def cmd_paper(args: argparse.Namespace) -> None:
    load_dotenv()
    cfg = Config.from_yaml(args.config)
    trader = PaperTrader(cfg, api_token=_require_api_token())
    trader.run_forever()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(prog="trade-bot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    backtest_parser = subparsers.add_parser("backtest", help="Run a historical backtest")
    backtest_parser.add_argument("--config", default="config.yaml")
    backtest_parser.add_argument("--since", default=None, help="Datum, z.B. 2023-01-01")
    backtest_parser.add_argument("--bars", type=int, default=2000)
    backtest_parser.add_argument("--trades", action="store_true", help="Print individual trades")
    backtest_parser.set_defaults(func=cmd_backtest)

    paper_parser = subparsers.add_parser("paper", help="Run continuous paper trading (no real funds)")
    paper_parser.add_argument("--config", default="config.yaml")
    paper_parser.set_defaults(func=cmd_paper)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
