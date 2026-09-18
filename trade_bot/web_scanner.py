from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from trade_bot.scanner import MarketSnapshot, fetch_market_snapshot

logger = logging.getLogger("web_scanner")


class SnapshotStore:
    """Thread-safe holder for the latest scan - written by the background
    refresh loop, read by HTTP request threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshots: list[MarketSnapshot] = []
        self._updated_at: str | None = None

    def set(self, snapshots: list[MarketSnapshot], updated_at: str) -> None:
        with self._lock:
            self._snapshots = snapshots
            self._updated_at = updated_at

    def get_json(self) -> str:
        with self._lock:
            payload = {
                "updated_at": self._updated_at,
                "markets": [asdict(s) for s in self._snapshots],
            }
        return json.dumps(payload)


def collect_snapshots(
    watchlist: list[str], cfg, stock_client, crypto_client, granularity: str,
) -> list[MarketSnapshot]:
    """One scan pass over the whole watchlist. A single symbol's fetch error
    never takes down the others - substituted with a visible error marker.
    """
    snapshots = []
    for symbol in watchlist:
        try:
            snapshots.append(fetch_market_snapshot(symbol, cfg, stock_client, crypto_client, granularity))
        except Exception:
            logger.exception("[%s] Fehler beim Abrufen", symbol)
            snapshots.append(
                MarketSnapshot(symbol=symbol, price=None, daily_change_pct=None, status_text="FEHLER", market_open=False)
            )
    return snapshots


def _refresh_loop(store: SnapshotStore, watchlist, cfg, stock_client, crypto_client, granularity, interval_seconds):
    while True:
        snapshots = collect_snapshots(watchlist, cfg, stock_client, crypto_client, granularity)
        store.set(snapshots, datetime.now(timezone.utc).isoformat())
        time.sleep(interval_seconds)


PAGE_HTML = """<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<title>Markt-Scanner</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.5.1/chart.umd.min.js"></script>
<style>
  :root {
    color-scheme: light;
    --surface: #fcfcfb;
    --page: #f9f9f7;
    --ink: #0b0b0b;
    --ink-secondary: #52514e;
    --ink-muted: #898781;
    --grid: #e1e0d9;
    --pos: #2a78d6;
    --neg: #e34948;
    --neutral: #f0efec;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      color-scheme: dark;
      --surface: #1a1a19;
      --page: #0d0d0d;
      --ink: #ffffff;
      --ink-secondary: #c3c2b7;
      --ink-muted: #898781;
      --grid: #2c2c2a;
      --pos: #3987e5;
      --neg: #e66767;
      --neutral: #383835;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--page);
    color: var(--ink);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    padding: 24px;
  }
  h1 { font-size: 18px; margin: 0 0 4px; }
  .updated { color: var(--ink-muted); font-size: 13px; margin-bottom: 20px; }
  .card {
    background: var(--surface);
    border: 1px solid var(--grid);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 20px;
  }
  .legend { color: var(--ink-secondary); font-size: 13px; margin: 8px 0 0; }
  .legend span { display: inline-flex; align-items: center; gap: 6px; margin-right: 16px; }
  .swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--grid); }
  th { color: var(--ink-muted); font-weight: 500; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  .closed { color: var(--ink-muted); }
  .open { color: var(--pos); }
  canvas { max-height: 340px; }
</style>
</head>
<body>
  <h1>Markt-Scanner</h1>
  <div class="updated" id="updated">laedt...</div>

  <div class="card">
    <canvas id="chart"></canvas>
    <p class="legend">
      <span><span class="swatch" style="background:var(--pos)"></span>Gewinn heute</span>
      <span><span class="swatch" style="background:var(--neg)"></span>Verlust heute</span>
    </p>
  </div>

  <div class="card">
    <table id="table">
      <thead>
        <tr><th>Symbol</th><th class="num">Kurs</th><th class="num">Tag %</th><th>Markt</th><th>Signal</th></tr>
      </thead>
      <tbody></tbody>
    </table>
  </div>

<script>
const REFRESH_MS = __REFRESH_MS__;
let chart = null;
let chartFailed = false;

function color(v) {
  const style = getComputedStyle(document.documentElement);
  return v >= 0 ? style.getPropertyValue('--pos') : style.getPropertyValue('--neg');
}

async function refresh() {
  const res = await fetch('/api/snapshots');
  const data = await res.json();
  const markets = [...data.markets].sort((a, b) => (b.daily_change_pct ?? -Infinity) - (a.daily_change_pct ?? -Infinity));

  document.getElementById('updated').textContent = data.updated_at
    ? 'Aktualisiert: ' + new Date(data.updated_at).toLocaleTimeString('de-DE')
    : 'noch keine Daten';

  // Table renders unconditionally - if Chart.js failed to load from the
  // CDN (network restriction, ad-blocker), the page still shows the data
  // instead of silently rendering nothing.
  const tbody = document.querySelector('#table tbody');
  tbody.innerHTML = markets.map(m => `
    <tr>
      <td>${m.symbol}</td>
      <td class="num">${m.price !== null ? m.price.toFixed(2) : 'n/a'}</td>
      <td class="num">${m.daily_change_pct !== null ? m.daily_change_pct.toFixed(2) + '%' : 'n/a'}</td>
      <td class="${m.market_open ? 'open' : 'closed'}">${m.market_open ? 'offen' : 'geschlossen'}</td>
      <td>${m.status_text}</td>
    </tr>
  `).join('');

  if (chartFailed) return;
  try {
    const labels = markets.map(m => m.symbol);
    const values = markets.map(m => m.daily_change_pct ?? 0);
    const colors = values.map(color);

    if (!chart) {
      chart = new Chart(document.getElementById('chart'), {
        type: 'bar',
        data: { labels, datasets: [{ data: values, backgroundColor: colors, borderRadius: 4, borderSkipped: false }] },
        options: {
          plugins: { legend: { display: false }, tooltip: { callbacks: { label: (ctx) => ctx.parsed.y.toFixed(2) + '%' } } },
          scales: { y: { ticks: { callback: (v) => v + '%' } } },
        },
      });
    } else {
      chart.data.labels = labels;
      chart.data.datasets[0].data = values;
      chart.data.datasets[0].backgroundColor = colors;
      chart.update();
    }
  } catch (err) {
    chartFailed = true;
    document.querySelector('.card').insertAdjacentHTML('afterend',
      '<p style="color:var(--neg)">Chart.js konnte nicht geladen werden (Netzwerk/Blocker?) - Tabelle bleibt nutzbar.</p>');
    console.error('Chart-Rendering fehlgeschlagen:', err);
  }
}

refresh();
setInterval(refresh, REFRESH_MS);
</script>
</body>
</html>
"""


def make_handler(store: SnapshotStore, refresh_ms: int):
    page = PAGE_HTML.replace("__REFRESH_MS__", str(refresh_ms))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/api/snapshots":
                body = store.get_json().encode("utf-8")
                content_type = "application/json"
            else:
                body = page.encode("utf-8")
                content_type = "text/html; charset=utf-8"

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:  # noqa: A002 - matches BaseHTTPRequestHandler signature
            logger.debug(format, *args)

    return Handler


def run_web_scanner(
    watchlist: list[str], cfg, stock_client, crypto_client,
    granularity: str = "H1", interval_seconds: int = 60, port: int = 8080,
) -> None:
    store = SnapshotStore()
    thread = threading.Thread(
        target=_refresh_loop,
        args=(store, watchlist, cfg, stock_client, crypto_client, granularity, interval_seconds),
        daemon=True,
    )
    thread.start()

    # 127.0.0.1 only - this process holds real Alpaca credentials and must
    # never be reachable from outside the local machine.
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(store, interval_seconds * 1000))
    logger.info("Web-Scanner laeuft auf http://127.0.0.1:%d", port)
    print(f"Web-Scanner laeuft auf http://127.0.0.1:{port} (Strg+C zum Beenden)")
    server.serve_forever()
