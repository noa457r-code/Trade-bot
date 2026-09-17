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

`config.yaml` enthält Instrumente, Granularität (Kerzengröße), Strategie-Parameter
und Risikomanagement (Positionsgröße pro Trade, Stop-Loss, Take-Profit,
Kosten-Näherung). Werte dort anpassen.

Instrumente werden als Liste von Alpaca-Tickern angegeben (`instruments:`, z.B.
`AAPL`, `MSFT`) — Backtest, Walk-Forward und Paper-Trading laufen dann über alle
konfigurierten Instrumente. Gültige Granularitäten: `M1, M5, M15, M30, H1, H4, D, W`.

Hinweis: Marktdaten für Aktien aktualisieren sich nur während der Börsenöffnungszeiten
(NYSE/NASDAQ). Außerhalb davon liefert Paper-Trading keine neuen Kerzen.

## Backtest ausführen

```bash
python -m trade_bot.cli backtest --config config.yaml --since 2023-01-01 --bars 5000 --trades
```

Lädt historische Kursdaten von Alpaca, wendet die Strategie an und gibt
Performance-Kennzahlen aus: Anzahl Trades, Win-Rate, Gesamtrendite, Max
Drawdown, Endkapital.

## Walk-Forward-Validierung ausführen

```bash
python -m trade_bot.cli walkforward --config config.yaml --since 2020-01-01 --bars 12000 \
    --train-bars 2000 --test-bars 500 --step-bars 500
```

Ein einfacher Backtest, der Parameter auf demselben Fenster testet, das später
auch bewertet wird, ist immer zu optimistisch (Overfitting). Walk-Forward macht
es ehrlich: Parameter werden auf einem Trainingsfenster per Grid-Search gefittet,
dann unverändert auf dem darauffolgenden, noch ungesehenen Testfenster
ausgewertet. Das Fenster rollt weiter, wiederholt sich über die gesamte
Historie, die Out-of-Sample-Ergebnisse aller Fenster werden verkettet
(compoundiert) — das zeigt, ob eine Strategie wirklich einen Vorteil hat oder
nur auf ein Fenster übergepasst war.

`strategy.trend_ma` ist bewusst NICHT Teil des Grids (siehe Kommentar in
`trade_bot/walkforward.py`) — ein Experiment zeigte, dass automatisches
Mitoptimieren des Trendfilters die Out-of-Sample-Ergebnisse verschlechtert und
instabil macht. Trendfilter-Periode wird als feste strategische Entscheidung in
`config.yaml` gesetzt, nicht pro Fenster neu gefittet.

## Paper-Trading ausführen

```bash
python -m trade_bot.cli paper --config config.yaml
```

Pollt in konfigurierbarem Intervall aktuelle Kursdaten für jedes konfigurierte
Instrument, wendet dieselbe Strategie an und platziert bei einem Entry-Signal
eine Bracket-Order (Market-Entry + Stop-Loss + Take-Profit) auf dem
Alpaca-Paper-Konto. Ein Exit-Signal schließt die Position vorzeitig über
`close_position`. Ein Fehler bei einem Instrument (z.B. Netzwerkaussetzer)
stoppt die anderen nicht. Alles läuft gegen den `paper`-Endpoint — virtuelles
Kapital, kein echtes Geld involviert.

**Dauerhaft im Hintergrund laufen lassen:** systemd-User-Service-Beispiel liegt
unter `~/.config/systemd/user/trade-bot-paper.service` (nicht Teil dieses
Repos, maschinenspezifisch) — `Restart=always` holt den Bot nach Abstürzen
automatisch zurück, `WantedBy=default.target` + `enable` startet ihn nach
Login/Reboot neu. Verwalten über `systemctl --user status/stop/restart
trade-bot-paper`.

## Strategie

- **Entry:** Fast-MA kreuzt Slow-MA von unten nach oben, UND RSI liegt unter
  `rsi_buy_max` (verhindert Kauf in einen bereits überkauften Spike), UND —
  falls `strategy.trend_ma` gesetzt ist — der Kurs liegt über diesem
  langfristigen SMA (Long nur mit dem übergeordneten Trend, nicht dagegen).
- **Exit:** Fast-MA kreuzt Slow-MA von oben nach unten, UND RSI liegt über
  `rsi_sell_min` — zusätzlich greifen Stop-Loss und Take-Profit jederzeit.
- **Positionsgröße:** so bemessen, dass ein Stop-Loss-Treffer maximal
  `risk_per_trade` des aktuellen Kapitals kostet.
- **Stop-Loss/Take-Profit:** skalieren mit dem Average True Range (ATR) statt
  einem festen Prozentsatz — passt sich der tatsächlichen Volatilität des
  Instruments an (`stop_loss_atr_mult` / `take_profit_atr_mult` in `config.yaml`).

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
