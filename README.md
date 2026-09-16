# Trade-bot

Ein Aktien-Trading-Bot mit Backtesting und Paper-Trading. Strategie: SMA-Crossover
(schneller/langsamer gleitender Durchschnitt) mit RSI-Filter zur Bestätigung.
Datenquelle: [Alpaca](https://alpaca.markets/) Market-Data-API über die Library
`alpaca-py`.

**Kein Live-Trading in dieser Version.** Backtests laufen auf historischen Kursen.
Paper-Trading platziert echte Bracket-Orders (Entry + Stop-Loss + Take-Profit)
auf Alpacas Paper-Trading-Endpoint — sichtbar im Alpaca-Dashboard unter
Orders/Positions, aber mit virtuellem Kapital, ohne jedes Risiko für echtes Geld.

## Setup

1. Kostenlosen Alpaca-Account anlegen: https://alpaca.markets/
2. API-Key + Secret stehen direkt im Dashboard nach der Anmeldung (kein
   separater Freischalt-Schritt nötig).
3. Python-Umgebung einrichten:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env     # ALPACA_API_KEY + ALPACA_SECRET_KEY eintragen
cp config.example.yaml config.yaml
```

`config.yaml` enthält Instrument, Granularität (Kerzengröße), Strategie-Parameter
und Risikomanagement (Positionsgröße pro Trade, Stop-Loss, Take-Profit,
Kosten-Näherung). Werte dort anpassen.

Instrumente werden als Alpaca-Ticker angegeben, z.B. `AAPL`, `MSFT`. Gültige
Granularitäten: `M1, M5, M15, M30, H1, H4, D, W`.

Hinweis: Marktdaten für Aktien aktualisieren sich nur während der Börsenöffnungszeiten
(NYSE/NASDAQ). Außerhalb davon liefert Paper-Trading keine neuen Kerzen.

## Backtest ausführen

```bash
python -m trade_bot.cli backtest --config config.yaml --since 2023-01-01 --bars 5000 --trades
```

Lädt historische Kursdaten von Alpaca, wendet die Strategie an und gibt
Performance-Kennzahlen aus: Anzahl Trades, Win-Rate, Gesamtrendite, Max
Drawdown, Endkapital.

## Paper-Trading ausführen

```bash
python -m trade_bot.cli paper --config config.yaml
```

Pollt in konfigurierbarem Intervall aktuelle Kursdaten, wendet dieselbe Strategie
an und platziert bei einem Entry-Signal eine Bracket-Order (Market-Entry +
Stop-Loss + Take-Profit) auf dem Alpaca-Paper-Konto. Ein Exit-Signal schließt
die Position vorzeitig über `close_position`. Alles läuft gegen den
`paper`-Endpoint — virtuelles Kapital, kein echtes Geld involviert.

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
