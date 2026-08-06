"""Fill the sampling gap: fetch :15 and :45 windows with real bid AND ask.

WHY
  Every book dataset in this repo is SAMPLED:
    premium_history2  minute :00 only          (2,780 rows)
    ladder_paths      minutes :00 and :30 only (5,624 rows)

  That has two consequences that matter for HCR:
    1. the minute-00 condition cannot be properly tested - premium_history2
       contains nothing else, and ladder_paths has only one comparison minute
    2. every candidate count is a fraction of the true population, so the
       claimed 493/521 counts cannot be reconciled

  Fetching :15 and :45 closes the gap and roughly doubles the testable sample.

METHOD
  Identical to quote_ladder.py: per-minute candlesticks, take the FIRST
  eligible minute in [8,14] before the true close, record the favourite side,
  its bid and its ask. Cached and resumable.
"""
from __future__ import annotations

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture.kalshi import KalshiClient          # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
CACHE = DATA / "book_minutes_15_45.parquet"
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
api = KalshiClient()

U = pd.read_parquet(DATA / "underlying.parquet")
_exp = pd.to_datetime(U.close, format="mixed", utc=True)
U["close_utc"] = (pd.to_datetime(U.wkey, format="%y%b%d%H%M", utc=True)
                  + pd.Timedelta(hours=4))
# only markets whose close time decodes cleanly (5 min before expected_exp)
U = U[((_exp - U.close_utc).dt.total_seconds() / 60).round(1) == 5.0].copy()
U["minute"] = U.close_utc.dt.minute
TARGET = U[U.minute.isin([15, 45])].copy()
print(f"missing-minute markets available: {len(TARGET):,}")

rows, done = [], set()
if CACHE.exists():
    old = pd.read_parquet(CACHE)
    rows, done = old.to_dict("records"), set(old.ticker)
    print(f"cache: {len(rows):,} already fetched")
todo = TARGET[~TARGET.ticker.isin(done)]
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
# BUGFIX: sorting by close_utc fetched only the EARLIEST markets, which
# confounded the minute comparison with time (:15/:45 covered 18 days while
# :00/:30 covered 73). Sample RANDOMLY across the whole period instead.
if len(todo) > LIMIT:
    todo = todo.sample(LIMIT, random_state=20260806)
print(f"fetching {len(todo):,}\n")


def fetch(series, ticker, cts, tries=3):
    for a in range(tries):
        try:
            return api.get(
                f"/series/{series}/markets/{ticker}/candlesticks",
                {"period_interval": 1, "start_ts": cts - 16 * 60,
                 "end_ts": cts})
        except Exception:
            if a == tries - 1:
                return None
            time.sleep(1.0 * (a + 1))
    return None


t0, errs = time.time(), 0
for i, r in enumerate(todo.itertuples()):
    cts = int((r.close_utc - EPOCH).total_seconds())
    res = fetch(r.series, r.ticker, cts)
    if res is None or not res.ok:
        errs += 1
        continue
    best = None
    for k in ((res.body or {}).get("candlesticks") or []):
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
    if best and 0.65 <= best["bid"] < 0.80:
        rows.append(dict(ticker=r.ticker, coin=r.coin, close_ts=cts,
                         minute=r.minute, side=best["side"], bid=best["bid"],
                         ask=best["ask"], vol=best["vol"], ml=best["ml"],
                         won=1 if (best["side"] == r.result) else 0))
    if (i + 1) % 400 == 0:
        el = time.time() - t0
        print(f"  {i+1}/{len(todo)}  kept {len(rows)}  errs {errs}  "
              f"eta {el/(i+1)*(len(todo)-i-1)/60:.1f}min", flush=True)
        pd.DataFrame(rows).drop_duplicates("ticker").to_parquet(CACHE)

D = pd.DataFrame(rows).drop_duplicates("ticker")
D.to_parquet(CACHE)
print(f"\nstored {len(D):,} in-band markets at minutes :15/:45")
if len(D):
    print(D.minute.value_counts().to_string())
