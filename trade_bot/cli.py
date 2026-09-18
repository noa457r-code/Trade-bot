from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

from trade_bot.backtest import run_backtest
from trade_bot.config import Config, RiskConfig
from trade_bot.data import fetch_ohlcv, make_client, make_crypto_client
from trade_bot.momentum_strategy import MomentumConfig, run_momentum_backtest
from trade_bot.paper_trader import PaperTrader
from trade_bot.safety import reset_kill_switch
from trade_bot.scanner import DEFAULT_WATCHLIST, run_scanner
from trade_bot.web_scanner import run_web_scanner
from trade_bot.strategy import generate_signals
from trade_bot.bollinger_strategy import PARAM_GRID as BOLLINGER_PARAM_GRID
from trade_bot.bollinger_strategy import BollingerConfig
from trade_bot.bollinger_strategy import generate_signals as generate_bollinger_signals
from trade_bot.macd_strategy import PARAM_GRID as MACD_PARAM_GRID
from trade_bot.macd_strategy import MACDConfig
from trade_bot.macd_strategy import generate_signals as generate_macd_signals
from trade_bot.turtle_strategy import PARAM_GRID as TURTLE_PARAM_GRID
from trade_bot.turtle_strategy import TurtleConfig
from trade_bot.turtle_strategy import generate_signals as generate_turtle_signals
from trade_bot.walkforward import compounded_out_of_sample_return_pct, run_walk_forward

MOMENTUM_DEFAULT_UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

# 50 liquide US-Large-Caps, 10 Sektoren x 5 - fuer breite Mehrjahres-/
# Mehrmarkt-Backtests (--instruments), nicht Teil der Live-Config.
DIVERSIFIED_50_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "ORCL", "CSCO",           # Tech
    "JPM", "BAC", "GS", "MS", "WFC",                  # Financials
    "JNJ", "UNH", "PFE", "MRK", "ABBV",               # Healthcare
    "XOM", "CVX", "COP", "SLB", "EOG",                # Energy
    "AMZN", "HD", "MCD", "NKE", "SBUX",               # Consumer Discretionary
    "PG", "KO", "PEP", "WMT", "COST",                 # Consumer Staples
    "BA", "CAT", "GE", "HON", "UPS",                  # Industrials
    "NEE", "DUK", "SO", "D", "AEP",                   # Utilities
    "LIN", "APD", "ECL", "SHW", "FCX",                # Materials
    "GOOGL", "META", "DIS", "VZ", "T",                # Communication Services
]

STRATEGIES = {
    "sma_rsi": None,  # handled separately below - uses cfg.strategy from config.yaml, not a fixed default
    "turtle": (TurtleConfig, generate_turtle_signals, TURTLE_PARAM_GRID),
    "macd": (MACDConfig, generate_macd_signals, MACD_PARAM_GRID),
    # trend_ma tested at 200 (mirroring the sma_rsi trend filter) and made
    # walk-forward results WORSE, not better (AAPL flipped from +4.77% to
    # -4.52%) - a trend filter blocks exactly the dip-buys mean-reversion
    # depends on, so it's left disabled here despite helping sma_rsi.
    "bollinger": (BollingerConfig, generate_bollinger_signals, BOLLINGER_PARAM_GRID),
}


def _format_strategy_params(strategy_cfg) -> str:
    if isinstance(strategy_cfg, TurtleConfig):
        return f"entry={strategy_cfg.entry_channel} exit={strategy_cfg.exit_channel}"
    if isinstance(strategy_cfg, MACDConfig):
        return f"macd_fast={strategy_cfg.fast_period} macd_slow={strategy_cfg.slow_period}"
    if isinstance(strategy_cfg, BollingerConfig):
        return f"period={strategy_cfg.period} std={strategy_cfg.num_std}"
    return f"fast={strategy_cfg.fast_ma} slow={strategy_cfg.slow_ma} trend={strategy_cfg.trend_ma or 'off'}"


def _require_alpaca_credentials() -> tuple[str, str]:
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise SystemExit(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY sind nicht gesetzt. Kopiere .env.example "
            "nach .env und trage deine (kostenlosen) Alpaca-Paper-Account-Keys ein."
        )
    return api_key, secret_key


def cmd_backtest(args: argparse.Namespace) -> None:
    load_dotenv()
    cfg = Config.from_yaml(args.config)
    api_key, secret_key = _require_alpaca_credentials()
    client = make_client(api_key, secret_key)

    since = None
    if args.since:
        since = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # --instruments/--granularity override config.yaml for this run only -
    # never touches the live paper-trading config.
    if args.instruments == "diversified50":
        instruments = DIVERSIFIED_50_UNIVERSE
    elif args.instruments:
        instruments = args.instruments.split(",")
    else:
        instruments = cfg.instruments
    granularity = args.granularity or cfg.granularity

    # Each instrument is backtested independently against the same starting
    # capital (not a shared portfolio simulation) - this shows per-instrument
    # edge, not combined portfolio equity.
    results = []
    for instrument in instruments:
        df = fetch_ohlcv(client, instrument, granularity, since=since, max_bars=args.bars)

        if args.strategy == "sma_rsi":
            signals = generate_signals(df, cfg.strategy)
        else:
            strategy_cls, generate_signals_fn, _ = STRATEGIES[args.strategy]
            signals = generate_signals_fn(df, strategy_cls())
        result = run_backtest(signals, cfg.risk)
        results.append((instrument, result))

        print(f"Strategy: {args.strategy} | Instrument: {instrument} | Granularity: {granularity} | Bars: {len(df)}")
        print(result.summary())

        if args.trades:
            for t in result.trades:
                exit_price = f"{t.exit_price:.5f}" if t.exit_price is not None else "OPEN"
                print(
                    f"  {t.entry_time} entry={t.entry_price:.5f} -> "
                    f"{t.exit_time} exit={exit_price} ({t.exit_reason}) pnl={t.pnl:.2f}"
                )
        print()

    if len(results) > 1:
        returns = [r.total_return_pct for _, r in results]
        drawdowns = [r.max_drawdown_pct for _, r in results]
        win_rates = [r.win_rate_pct for _, r in results]
        profitable = sum(1 for r in returns if r > 0)
        best = max(results, key=lambda pair: pair[1].total_return_pct)
        worst = min(results, key=lambda pair: pair[1].total_return_pct)
        sorted_returns = sorted(returns)
        median_return = sorted_returns[len(sorted_returns) // 2]

        print(f"=== Aggregat ueber {len(results)} Instrumente ===")
        print(f"Durchschnittliche Rendite: {sum(returns) / len(returns):.2f}% | Median: {median_return:.2f}%")
        print(f"Profitabel: {profitable}/{len(results)} ({profitable / len(results) * 100:.1f}%)")
        print(f"Durchschnittlicher Max-Drawdown: {sum(drawdowns) / len(drawdowns):.2f}%")
        print(f"Durchschnittliche Win-Rate: {sum(win_rates) / len(win_rates):.1f}%")
        print(f"Bester: {best[0]} ({best[1].total_return_pct:+.2f}%) | Schlechtester: {worst[0]} ({worst[1].total_return_pct:+.2f}%)")


def cmd_walkforward(args: argparse.Namespace) -> None:
    load_dotenv()
    cfg = Config.from_yaml(args.config)
    api_key, secret_key = _require_alpaca_credentials()
    client = make_client(api_key, secret_key)

    since = None
    if args.since:
        since = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    if args.strategy == "sma_rsi":
        base_strategy = cfg.strategy
        param_grid = None  # walkforward.run_walk_forward falls back to its own sma_rsi default grid
        generate_signals_fn = generate_signals
    else:
        strategy_cls, generate_signals_fn, param_grid = STRATEGIES[args.strategy]
        base_strategy = strategy_cls()

    for instrument in cfg.instruments:
        df = fetch_ohlcv(client, instrument, cfg.granularity, since=since, max_bars=args.bars)
        windows = run_walk_forward(
            df,
            base_strategy=base_strategy,
            base_risk=cfg.risk,
            train_bars=args.train_bars,
            test_bars=args.test_bars,
            step_bars=args.step_bars,
            param_grid=param_grid,
            generate_signals_fn=generate_signals_fn,
        )

        print(f"\n=== {args.strategy} | {instrument} | {len(windows)} walk-forward windows ===")
        if not windows:
            print("  No window produced enough trades to evaluate - widen the date range or lower --min-trades.")
            continue

        for w in windows:
            print(
                f"  train {w.train_start.date()}..{w.train_end.date()} "
                f"({_format_strategy_params(w.strategy_cfg)} "
                f"sl={w.risk_cfg.stop_loss_atr_mult}x tp={w.risk_cfg.take_profit_atr_mult}x, "
                f"in-sample {w.train_result.total_return_pct:+.2f}%) "
                f"-> test {w.test_start.date()}..{w.test_end.date()} "
                f"out-of-sample {w.test_result.total_return_pct:+.2f}% "
                f"(win {w.test_result.win_rate_pct:.1f}%, dd {w.test_result.max_drawdown_pct:.2f}%)"
            )

        compounded = compounded_out_of_sample_return_pct(windows)
        print(f"  Compounded out-of-sample return across all windows: {compounded:+.2f}%")


def cmd_momentum(args: argparse.Namespace) -> None:
    """Cross-sectional relative-momentum backtest over a universe of
    instruments (unlike `backtest`/`walkforward`, this ranks instruments
    against each other, so it can't reuse cfg.instruments from a single-
    strategy config.yaml - the universe is passed directly via --instruments).

    Uses fixed, non-optimized parameters (classic momentum lookback/rebalance
    horizons from the literature) rather than a per-window grid search, so
    unlike the other strategies this doesn't need walk-forward validation to
    stay honest - there's no in-sample fitting step that could overfit.
    """
    load_dotenv()
    api_key, secret_key = _require_alpaca_credentials()
    client = make_client(api_key, secret_key)

    since = None
    if args.since:
        since = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    instruments = args.instruments.split(",")
    price_data = {
        inst: fetch_ohlcv(client, inst, args.granularity, since=since, max_bars=args.bars)
        for inst in instruments
    }

    momentum_cfg = MomentumConfig(
        lookback_bars=args.lookback_bars, rebalance_bars=args.rebalance_bars, top_k=args.top_k,
    )
    risk_cfg = RiskConfig(
        initial_capital=10000, risk_per_trade=0.01,
        stop_loss_atr_mult=momentum_cfg.stop_loss_atr_mult, take_profit_atr_mult=0, fee_pct=0.0002,
    )

    result = run_momentum_backtest(price_data, risk_cfg, momentum_cfg)

    print(
        f"Momentum backtest | Universe: {', '.join(instruments)} | Granularity: {args.granularity} | "
        f"lookback={momentum_cfg.lookback_bars} rebalance={momentum_cfg.rebalance_bars} top_k={momentum_cfg.top_k}"
    )
    print(result.summary())
    if args.trades:
        for t in sorted(result.trades, key=lambda t: t.entry_time):
            exit_price = f"{t.exit_price:.2f}" if t.exit_price is not None else "OPEN"
            print(
                f"  {t.instrument}: {t.entry_time} entry={t.entry_price:.2f} -> "
                f"{t.exit_time} exit={exit_price} ({t.exit_reason}) pnl={t.pnl:.2f}"
            )


def cmd_scanner(args: argparse.Namespace) -> None:
    load_dotenv()
    cfg = Config.from_yaml(args.config)
    api_key, secret_key = _require_alpaca_credentials()
    stock_client = make_client(api_key, secret_key)
    crypto_client = make_crypto_client(api_key, secret_key)

    watchlist = args.instruments.split(",") if args.instruments else DEFAULT_WATCHLIST
    if args.web:
        run_web_scanner(
            watchlist, cfg.strategy, stock_client, crypto_client,
            granularity=args.granularity, interval_seconds=args.interval, port=args.port,
        )
    else:
        run_scanner(
            watchlist, cfg.strategy, stock_client, crypto_client,
            granularity=args.granularity, interval_seconds=args.interval,
        )


def cmd_paper(args: argparse.Namespace) -> None:
    load_dotenv()
    cfg = Config.from_yaml(args.config)
    api_key, secret_key = _require_alpaca_credentials()
    discord_webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    trader = PaperTrader(
        cfg, api_key=api_key, secret_key=secret_key,
        discord_webhook_url=discord_webhook_url, safety_state_path=args.state_file,
        adaptive_risk_state_path=args.adaptive_risk_state_file,
    )
    trader.run_forever()


def cmd_reset_killswitch(args: argparse.Namespace) -> None:
    state = reset_kill_switch(args.state_file)
    print(f"Kill-Switch zurueckgesetzt. Peak-Equity bleibt bei {state.peak_equity:.2f}. Bot handelt beim naechsten Step wieder normal.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(prog="trade-bot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    backtest_parser = subparsers.add_parser("backtest", help="Run a historical backtest")
    backtest_parser.add_argument("--config", default="config.yaml")
    backtest_parser.add_argument("--since", default=None, help="Datum, z.B. 2023-01-01")
    backtest_parser.add_argument("--bars", type=int, default=2000)
    backtest_parser.add_argument("--trades", action="store_true", help="Print individual trades")
    backtest_parser.add_argument(
        "--instruments", default=None,
        help="Komma-getrennte Ticker-Liste ODER 'diversified50' fuer ein 50-Instrumente-Sektorenset, "
             "ueberschreibt config.yaml nur fuer diesen Lauf (Live-Trading unberuehrt)",
    )
    backtest_parser.add_argument(
        "--granularity", default=None,
        help="Ueberschreibt config.yaml nur fuer diesen Lauf, z.B. D fuer Tages-Kerzen",
    )
    backtest_parser.add_argument(
        "--strategy", choices=list(STRATEGIES), default="sma_rsi",
        help="sma_rsi = bestehende MA-Crossover+RSI-Strategie, turtle = Donchian-Breakout "
             "(Turtle Trading System 1), macd = MACD-Crossover, bollinger = Bollinger-Band Mean-Reversion",
    )
    backtest_parser.set_defaults(func=cmd_backtest)

    momentum_parser = subparsers.add_parser(
        "momentum", help="Cross-sectional relative-momentum backtest across a universe of instruments"
    )
    momentum_parser.add_argument(
        "--instruments", default=",".join(MOMENTUM_DEFAULT_UNIVERSE),
        help="Komma-getrennte Ticker-Liste, z.B. AAPL,MSFT,GOOGL (nicht aus config.yaml - eigenes Universum)",
    )
    momentum_parser.add_argument("--granularity", default="H1")
    momentum_parser.add_argument("--since", default=None, help="Datum, z.B. 2020-01-01")
    momentum_parser.add_argument("--bars", type=int, default=6000)
    momentum_parser.add_argument("--lookback-bars", type=int, default=420, help="~3 Monate H1-Bars")
    momentum_parser.add_argument("--rebalance-bars", type=int, default=140, help="~1 Monat H1-Bars")
    momentum_parser.add_argument("--top-k", type=int, default=3)
    momentum_parser.add_argument("--trades", action="store_true", help="Print individual trades")
    momentum_parser.set_defaults(func=cmd_momentum)

    scanner_parser = subparsers.add_parser(
        "scanner", help="Terminal-Dashboard: rankt eine Watchlist live nach Tagesveraenderung + Signal-Naehe"
    )
    scanner_parser.add_argument("--config", default="config.yaml")
    scanner_parser.add_argument(
        "--instruments", default=None,
        help="Komma-getrennte Liste, Default: " + ",".join(DEFAULT_WATCHLIST),
    )
    scanner_parser.add_argument("--granularity", default="H1")
    scanner_parser.add_argument("--interval", type=int, default=60, help="Sekunden zwischen Aktualisierungen")
    scanner_parser.add_argument("--web", action="store_true", help="Web-Dashboard statt Terminal (Chart + Tabelle)")
    scanner_parser.add_argument("--port", type=int, default=8080, help="Nur mit --web: Port auf 127.0.0.1")
    scanner_parser.set_defaults(func=cmd_scanner)

    paper_parser = subparsers.add_parser("paper", help="Run continuous paper trading (no real funds)")
    paper_parser.add_argument("--config", default="config.yaml")
    paper_parser.add_argument("--state-file", default="safety_state.json")
    paper_parser.add_argument("--adaptive-risk-state-file", default="adaptive_risk_state.json")
    paper_parser.set_defaults(func=cmd_paper)

    wf_parser = subparsers.add_parser(
        "walkforward", help="Rolling walk-forward validation: refit params per window, test out-of-sample"
    )
    wf_parser.add_argument("--config", default="config.yaml")
    wf_parser.add_argument("--since", default=None, help="Datum, z.B. 2021-01-01")
    wf_parser.add_argument("--bars", type=int, default=8000)
    wf_parser.add_argument("--train-bars", type=int, default=2000, help="Bars per in-sample fitting window")
    wf_parser.add_argument("--test-bars", type=int, default=500, help="Bars per out-of-sample test window")
    wf_parser.add_argument("--step-bars", type=int, default=500, help="Bars to roll forward between windows")
    wf_parser.add_argument(
        "--strategy", choices=list(STRATEGIES), default="sma_rsi",
        help="sma_rsi = bestehende MA-Crossover+RSI-Strategie, turtle = Donchian-Breakout "
             "(Turtle Trading System 1), macd = MACD-Crossover, bollinger = Bollinger-Band Mean-Reversion",
    )
    wf_parser.set_defaults(func=cmd_walkforward)

    reset_parser = subparsers.add_parser(
        "reset-killswitch", help="Kill-Switch zuruecksetzen, Bot handelt wieder (Peak-Equity bleibt erhalten)"
    )
    reset_parser.add_argument("--state-file", default="safety_state.json")
    reset_parser.set_defaults(func=cmd_reset_killswitch)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
