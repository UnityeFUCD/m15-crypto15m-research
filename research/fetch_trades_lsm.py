"""Fetch the public trade tape for every market the LSM strategy traded.

WHY THIS UNBLOCKS AUDIT D
  AUDIT D reported that queue position is uncomputable, because
  /portfolio/orders/{id}/queue_position is live-only and no queue field was
  ever captured. That is true of the DIRECT measurement.

  It is not true of the QUANTITY. In a FIFO book, an order resting at price P
  fills only after everything ahead of it at P has traded. So

    queue_ahead(order) ~ volume traded at OUR price, on the side that
                         consumes our resting order, between the moment we
                         posted and the moment we filled

  and for an order that NEVER filled, the same volume over its whole resting
  life is a LOWER BOUND on what was ahead of it.

  /markets/trades is public, historical, and cursor-paginated. Combined with
  our own placement and fill timestamps it reconstructs the variable.

ASSUMPTIONS, STATED PLAINLY
  1. Strict FIFO at a price level. Kalshi does not document queue priority; if
     it is pro-rata this measure is wrong.
  2. Cancellations ahead of us are invisible. A cancel removes queue without
     trading, so realised traded-volume UNDERSTATES the queue we joined. The
     reconstruction is therefore a lower bound, and is described as such.
  3. Our own fills are excluded from the count.

Because of 1 and 2 this is a PROXY. It is reported as one and never as the
exchange's queue_position field.
"""
from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture.kalshi import KalshiClient          # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
CACHE = DATA / "trades_lsm.parquet"
N_WORKERS = 3
PAGE = 1000

O = pd.read_parquet(DATA / "orders_history.parquet")
lsm = O[O.client_order_id.astype(str).str.startswith("lsm")]
tickers = sorted(lsm.ticker.unique())
print(f"LSM markets to fetch trades for: {len(tickers)}", flush=True)

rows: list[dict] = []
done: set[str] = set()
if CACHE.exists():
    old = pd.read_parquet(CACHE)
    rows, done = old.to_dict("records"), set(old.ticker.unique())
    print(f"cache: {len(rows):,} trades over {len(done)} markets", flush=True)

todo = [t for t in tickers if t not in done]
q: queue.Queue = queue.Queue()
for t in todo:
    q.put(t)
lock = threading.Lock()
state = {"n": 0, "t0": time.time()}


def flush():
    if rows:
        pd.DataFrame(rows).drop_duplicates("trade_id").to_parquet(CACHE)


def worker():
    api = KalshiClient()
    while True:
        try:
            tk = q.get_nowait()
        except queue.Empty:
            return
        cursor, got, pages = None, [], 0
        while pages < 40:
            params = {"ticker": tk, "limit": PAGE}
            if cursor:
                params["cursor"] = cursor
            try:
                res = api.get("/markets/trades", params)
            except Exception:
                break
            if not res.ok:
                break
            body = res.body or {}
            batch = body.get("trades") or []
            got.extend(batch)
            cursor = body.get("cursor")
            pages += 1
            if not cursor or not batch:
                break
        with lock:
            for t in got:
                rows.append(dict(
                    ticker=tk, trade_id=t.get("trade_id"),
                    created_time=t.get("created_time"),
                    count=float(t.get("count_fp") or 0),
                    yes_px=float(t.get("yes_price_dollars") or 0),
                    no_px=float(t.get("no_price_dollars") or 0),
                    taker_book_side=t.get("taker_book_side"),
                    taker_outcome_side=t.get("taker_outcome_side")))
            state["n"] += 1
            if state["n"] % 25 == 0:
                el = time.time() - state["t0"]
                rate = state["n"] / max(el, 1e-9)
                print(f"  {state['n']}/{len(todo)} markets  "
                      f"{len(rows):,} trades  "
                      f"eta {q.qsize()/max(rate,1e-9)/60:.1f}min", flush=True)
                flush()


ts = [threading.Thread(target=worker, daemon=True) for _ in range(N_WORKERS)]
for t in ts:
    t.start()
for t in ts:
    t.join()
flush()
D = pd.DataFrame(rows).drop_duplicates("trade_id")
print(f"\nstored {len(D):,} trades over {D.ticker.nunique()} markets")
if len(D):
    print("median trades per market: %.0f" % D.groupby("ticker").size().median())
