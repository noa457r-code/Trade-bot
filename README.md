# Trade-bot

Ein Krypto-Trading-Bot mit Backtesting und Paper-Trading. Strategie: SMA-Crossover
(schneller/langsamer gleitender Durchschnitt) mit RSI-Filter zur Bestätigung.
Datenquelle: Binance-Marktdaten über [ccxt](https://github.com/ccxt/ccxt).

**Kein Live-Trading in dieser Version.** Backtests laufen auf historischen Daten,
Paper-Trading simuliert Trades auf Basis von Live-Kursen, ohne echte Orders zu
platzieren oder Kapital zu riskieren.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp config.example.yaml config.yaml
```

`config.yaml` enthält Symbol, Timeframe, Strategie-Parameter und Risikomanagement
(Positionsgröße pro Trade, Stop-Loss, Take-Profit, Gebühren). Werte dort anpassen.

## Backtest ausführen

```bash
python -m trade_bot.cli backtest --config config.yaml --since 2023-01-01 --bars 5000 --trades
```

Lädt historische OHLCV-Daten von Binance (öffentlich, kein API-Key nötig),
wendet die Strategie an und gibt Performance-Kennzahlen aus:
Anzahl Trades, Win-Rate, Gesamtrendite, Max Drawdown, Endkapital.

## Paper-Trading ausführen

```bash
python -m trade_bot.cli paper --config config.yaml
```

Pollt in konfigurierbarem Intervall aktuelle Kursdaten, wendet dieselbe Strategie
an und simuliert Entries/Exits inkl. Stop-Loss/Take-Profit — alles nur im
Arbeitsspeicher, keine echten Orders.

## Strategie

- **Entry:** Fast-MA kreuzt Slow-MA von unten nach oben, UND RSI liegt unter
  `rsi_buy_max` (verhindert Kauf in einen bereits überkauften Spike).
- **Exit:** Fast-MA kreuzt Slow-MA von oben nach unten, UND RSI liegt über
  `rsi_sell_min` — zusätzlich greifen Stop-Loss und Take-Profit jederzeit.
- **Positionsgröße:** so bemessen, dass ein Stop-Loss-Treffer maximal
  `risk_per_trade` des aktuellen Kapitals kostet.

## Tests

```bash
pytest
```

Tests laufen auf synthetischen Preisdaten, ohne Netzwerkzugriff.

## Wichtiger Hinweis

Dies ist ein Ausgangspunkt für Backtesting und Strategie-Entwicklung, **keine
Gewinngarantie**. Historische Performance sagt nichts über zukünftige Ergebnisse
aus. Vor jeglichem Live-Einsatz: ausführlich backtesten, mit Paper-Trading über
einen längeren Zeitraum validieren, und nur Kapital einsetzen, dessen Verlust
tragbar ist. Live-Trading (echte Orders, echtes Kapital) ist in dieser Version
bewusst noch nicht implementiert.
