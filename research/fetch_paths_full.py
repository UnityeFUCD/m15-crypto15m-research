"""Fetch the FULL price path (ml 0..15) for the complete population.

WHY THIS EXISTS
  fetch_full_population.py already requested 16 minutes of 1-minute candles
  per market and then threw away all but the single best point in [8,14]. That
  discarded exactly the data AUDIT C needs: how the favourite evolves from
  entry to settlement.

  The alternative source is ladder_paths, which does carry a 15-point path -
  but it is :00/:30 only AND it is the hot 28.6% subset that produced the
  retracted +3.99c. It cannot be used here.

  cf_index cannot substitute: it stores window_open and window_close only,
  one point per market at 900s spacing. There is no intra-window index path
  in this repo.

WHAT IS STORED
  For every market, every 1-minute candle in [0,15] minutes-to-close, as a
  JSON path of {ml, yes_bid, yes_ask, volume}. Favourite side is fixed ONCE
  at the entry observation (first valid in [8,14]) and never re-derived later
  - re-deriving it mid-path would leak the outcome into the definition of
  which side we hold.

RESUMABLE, per-thread clients, incremental flush. Same pattern as the
population fetch.
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture.kalshi import KalshiClient          # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
CACHE = DATA / "paths_full.parquet"
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
N_WORKERS = 3
FLUSH_EVERY = 500

U = pd.read_parquet(DATA / "underlying.parquet",
                    columns=["ticker", "coin", "wkey", "result", "close",
                             "series"])
U["close_utc"] = (pd.to_datetime(U.wkey, format="%y%b%d%H%M", utc=True)
                  + pd.Timedelta(hours=4))
_exp = pd.to_datetime(U.close, format="mixed", utc=True)
U = U[((_exp - U.close_utc).dt.total_seconds() / 60).round(1) == 5.0]
U = U[U.result.isin(["yes", "no"])].copy()
U["minute"] = U.close_utc.dt.minute

rows: list[dict] = []
done: set[str] = set()
if CACHE.exists():
    old = pd.read_parquet(CACHE)
    rows, done = old.to_dict("records"), set(old.ticker)

todo = U[~U.ticker.isin(done)]
print(f"population {len(U):,}   cached {len(done):,}   to fetch {len(todo):,}",
      flush=True)

q: queue.Queue = queue.Queue()
for r in todo.itertuples():
    q.put((r.ticker, r.coin, r.series, r.minute, r.result,
           int((r.close_utc - EPOCH).total_seconds())))

lock = threading.Lock()
state = {"n": 0, "err": 0, "t0": time.time()}


def flush():
    if rows:
        pd.DataFrame(rows).drop_duplicates("ticker").to_parquet(CACHE)


def worker():
    api = KalshiClient()
    while True:
        try:
            tk, coin, series, minute, result, cts = q.get_nowait()
        except queue.Empty:
            return
        body = None
        for attempt in range(3):
            try:
                res = api.get(f"/series/{series}/markets/{tk}/candlesticks",
                              {"period_interval": 1, "start_ts": cts - 16 * 60,
                               "end_ts": cts})
                if res.ok:
                    body = res.body
                break
            except Exception:
                if attempt < 2:
                    time.sleep(0.6 * (attempt + 1))
        path = []
        for k in ((body or {}).get("candlesticks") or []):
            ml = (cts - int(k.get("end_period_ts") or 0)) / 60.0
            if not (0 <= ml <= 15):
                continue
            try:
                yb = float(k["yes_bid"]["close_dollars"])
                ya = float(k["yes_ask"]["close_dollars"])
            except (KeyError, TypeError, ValueError):
                continue
            if yb <= 0 or ya >= 1 or yb >= ya:
                continue
            path.append({"ml": ml, "yb": round(yb, 4), "ya": round(ya, 4),
                         "v": float(k.get("volume_fp") or 0)})
        path.sort(key=lambda x: -x["ml"])
        # entry = FIRST valid observation in [8,14]; side frozen there
        entry = next((p for p in path if 8 <= p["ml"] <= 14), None)
        with lock:
            state["n"] += 1
            if entry is None or len(path) < 2:
                state["err"] += 1
            else:
                fy = entry["yb"] >= 0.5
                rows.append(dict(
                    ticker=tk, coin=coin, close_ts=cts, minute=minute,
                    side="yes" if fy else "no",
                    bid=entry["yb"] if fy else 1.0 - entry["ya"],
                    ask=entry["ya"] if fy else 1.0 - entry["yb"],
                    entry_ml=entry["ml"], n_pts=len(path),
                    won=1 if (("yes" if fy else "no") == result) else 0,
                    path=json.dumps(path)))
            if state["n"] % FLUSH_EVERY == 0:
                el = time.time() - state["t0"]
                rate = state["n"] / max(el, 1e-9)
                print(f"  {state['n']:,}/{len(todo):,}  kept {len(rows):,}  "
                      f"skip {state['err']:,}  {rate:.1f}/s  "
                      f"eta {q.qsize()/max(rate,1e-9)/60:.0f}min", flush=True)
                flush()


ts = [threading.Thread(target=worker, daemon=True) for _ in range(N_WORKERS)]
for t in ts:
    t.start()
for t in ts:
    t.join()
flush()
D = pd.DataFrame(rows).drop_duplicates("ticker")
print(f"\nstored {len(D):,} paths", flush=True)
if len(D):
    print("by minute:", D.minute.value_counts().sort_index().to_dict())
    print("median points per path:", D.n_pts.median())
