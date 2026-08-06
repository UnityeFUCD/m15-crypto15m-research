"""Fetch the book for the COMPLETE market population - every minute, unsampled.

WHY
  Every book dataset in this repo is sampled, and each sample turned out to
  bias a result:
    premium_history2     minute :00 only, 8 of 24 hours  -> produced a false PASS
    ladder_paths         minutes :00 and :30 only
    book_minutes_15_45   :15/:45, 2,101 randomly sampled
  Total fetched: 7,725 of 39,453 markets with a cleanly decodable close.

  The true population is balanced across minutes (:00 10,300, :15 10,338,
  :30 10,350, :45 10,346), so any residual minute or hour effect in a sample
  is an artifact of the sample, not of the market. The only way to stop
  arguing about sampling is to stop sampling.

WHAT CHANGED vs THE EARLIER FETCHES
  1. ALL minutes, not a subset.
  2. NO band filter at fetch time. Earlier scripts kept only 65-80c and threw
     the rest away, which makes the stored file itself a sample and prevents
     any test of band sensitivity. Everything with a valid quote is stored.
  3. Concurrency with PER-THREAD clients. KalshiClient holds a requests.Session
     and a _last_call rate-limiter, neither of which is shared-safe, so each
     worker builds its own and keeps its own pacing.

RESUMABLE. Re-running skips whatever is already cached. Writes incrementally
so a kill at any point loses at most one flush.
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
CACHE = DATA / "book_full.parquet"
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
for f in ("ladder_paths.parquet", "book_minutes_15_45.parquet"):
    pass  # deliberately NOT reused: those were band-filtered at fetch time

todo = U[~U.ticker.isin(done)]
print(f"population {len(U):,}   cached {len(done):,}   to fetch {len(todo):,}",
      flush=True)
print(f"minutes in population: "
      f"{U.minute.value_counts().sort_index().to_dict()}", flush=True)

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
    api = KalshiClient()            # own Session, own rate limiter
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
        best = None
        for k in ((body or {}).get("candlesticks") or []):
            ml = (cts - int(k.get("end_period_ts") or 0)) / 60.0
            if not (8 <= ml <= 14):
                continue
            try:
                yb = float(k["yes_bid"]["close_dollars"])
                ya = float(k["yes_ask"]["close_dollars"])
            except (KeyError, TypeError, ValueError):
                continue
            if yb <= 0 or ya >= 1 or yb >= ya:
                continue
            if best is None or ml > best["ml"]:
                fy = yb >= 0.5
                best = dict(ml=ml, side="yes" if fy else "no",
                            bid=yb if fy else 1.0 - ya,
                            ask=ya if fy else 1.0 - yb,
                            vol=float(k.get("volume_fp") or 0))
        with lock:
            state["n"] += 1
            if best is None:
                state["err"] += 1
            else:
                # NO band filter - store everything, filter in analysis
                rows.append(dict(ticker=tk, coin=coin, close_ts=cts,
                                 minute=minute, side=best["side"],
                                 bid=best["bid"], ask=best["ask"],
                                 vol=best["vol"], ml=best["ml"],
                                 won=1 if best["side"] == result else 0))
            if state["n"] % FLUSH_EVERY == 0:
                el = time.time() - state["t0"]
                rate = state["n"] / max(el, 1e-9)
                left = (q.qsize()) / max(rate, 1e-9) / 60
                print(f"  {state['n']:,}/{len(todo):,}  kept {len(rows):,}  "
                      f"noquote {state['err']:,}  {rate:.1f}/s  "
                      f"eta {left:.0f}min", flush=True)
                flush()


ts = [threading.Thread(target=worker, daemon=True) for _ in range(N_WORKERS)]
for t in ts:
    t.start()
for t in ts:
    t.join()

flush()
D = pd.DataFrame(rows).drop_duplicates("ticker")
print(f"\nstored {len(D):,} markets with a valid quote at 8-14 min",
      flush=True)
if len(D):
    print("by minute:", D.minute.value_counts().sort_index().to_dict())
    inband = D[(D.bid >= 0.65) & (D.bid < 0.80)]
    print(f"in-band (65-80c): {len(inband):,}")
    print("in-band by minute:",
          inband.minute.value_counts().sort_index().to_dict())
