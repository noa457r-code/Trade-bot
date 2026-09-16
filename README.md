# Trade-bot

Ein Forex-Trading-Bot mit Backtesting und Paper-Trading. Strategie: SMA-Crossover
(schneller/langsamer gleitender Durchschnitt) mit RSI-Filter zur Bestätigung.
Datenquelle: [OANDA](https://www.oanda.com/) REST-API (v20) über die Library
`oandapyV20`.

**Kein Live-Trading in dieser Version.** Backtests laufen auf historischen Kursen,
Paper-Trading simuliert Trades auf Basis von Live-Kursen, ohne echte Orders zu
platzieren oder Kapital zu riskieren.

## Setup

1. Kostenlosen OANDA-Practice-(Demo-)Account anlegen: https://www.oanda.com/demo-account/
2. API-Token erzeugen: OANDA-Account-Portal → "Manage API Access"
3. Python-Umgebung einrichten:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env     # OANDA_API_TOKEN eintragen
cp config.example.yaml config.yaml
```

`config.yaml` enthält Instrument, Granularität (Kerzengröße), Strategie-Parameter
und Risikomanagement (Positionsgröße pro Trade, Stop-Loss, Take-Profit,
Spread-Kosten-Näherung). Werte dort anpassen.

Instrumente werden in OANDA-Notation angegeben, z.B. `EUR_USD` statt `EUR/USD`.
Gültige Granularitäten: `M1, M5, M15, M30, H1, H4, D, W` (siehe
[OANDA-Doku](https://developer.oanda.com/rest-live-v20/instrument-df/#CandlestickGranularity)
für alle Codes).

## Backtest ausführen

```bash
python -m trade_bot.cli backtest --config config.yaml --since 2023-01-01 --bars 5000 --trades
```

Lädt historische Kursdaten von OANDA, wendet die Strategie an und gibt
Performance-Kennzahlen aus: Anzahl Trades, Win-Rate, Gesamtrendite, Max
Drawdown, Endkapital.

## Paper-Trading ausführen

```bash
python -m trade_bot.cli paper --config config.yaml
```

Pollt in konfigurierbarem Intervall aktuelle Kursdaten, wendet dieselbe Strategie
an und simuliert Entries/Exits inkl. Stop-Loss/Take-Profit — alles nur im
Arbeitsspeicher, keine echten Orders. Benötigt nur den API-Token (auch auf dem
Practice-Environment), kein Kapital involviert.

## Strategie

- **Entry:** Fast-MA kreuzt Slow-MA von unten nach oben, UND RSI liegt unter
  `rsi_buy_max` (verhindert Kauf in einen bereits überkauften Spike).
- **Exit:** Fast-MA kreuzt Slow-MA von oben nach unten, UND RSI liegt über
  `rsi_sell_min` — zusätzlich greifen Stop-Loss und Take-Profit jederzeit.
- **Positionsgröße:** so bemessen, dass ein Stop-Loss-Treffer maximal
  `risk_per_trade` des aktuellen Kapitals kostet.

Hinweis: Die Positionsgrößen-Berechnung geht vereinfachend davon aus, dass die
Kontowährung der Kurswährung des Paares entspricht (z.B. USD bei EUR/USD), und
berücksichtigt kein Hebel-/Margin-System eines echten Forex-Brokers. Für
Live-Trading müsste das an die jeweilige Kontowährung und Margin-Regeln des
Brokers angepasst werden.

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
