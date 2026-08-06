"""Fetch 1-minute spot from Coinbase as an intra-window index PROXY.

WHY A PROXY IS NEEDED
  AUDIT C conditions on normalized distance to A0 and realized volatility
  WITHIN a window. This repo has no intra-window index data:
    cf_index      window_open / window_close only, 900s apart, 1 pt per market
    underlying    a0 and a1 endpoints
  Neither can say what the index did between open and close.

WHY COINBASE
  The CF Benchmarks Real-Time Index is computed from constituent exchanges,
  Coinbase among them. Its 1-minute spot is therefore a defensible proxy for
  intra-window index movement - not the index itself. Binance returns HTTP 451
  from this host; Kraken works but has thinner symbol coverage. Coinbase
  serves all six coins including HYPE, free and unauthenticated.

VALIDATION IS MANDATORY, NOT OPTIONAL
  A proxy that is not checked is a guess. This repo already contains ground
  truth: a0 and a1 for 41,334 windows. validate_spot_proxy.py reconstructs
  both from this data and reports the correlation and the sign-agreement rate
  on a1 >= a0. If the proxy cannot reproduce the endpoints it already knows,
  it must not be trusted for the interior, and AUDIT C fails on data grounds
  rather than being reported on a bad proxy.

Resumable and rate-limited. ~2,100 requests total, ~15 minutes.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
CACHE = DATA / "spot_1m.parquet"
COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "HYPE"]
GRAN = 60
CHUNK_MIN = 300               # Coinbase caps a response at 300 candles
SLEEP = 0.34                  # ~3 req/s, well under the public limit
START = pd.Timestamp("2026-05-24 22:00", tz="UTC")
END = pd.Timestamp("2026-08-06 00:00", tz="UTC")


def get(url, tries=4):
    for a in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "m15-research/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(2.0 * (a + 1))
                continue
            if a == tries - 1:
                return None
        except Exception:
            if a == tries - 1:
                return None
            time.sleep(0.8 * (a + 1))
    return None


rows: list[dict] = []
seen: set[tuple] = set()
if CACHE.exists():
    old = pd.read_parquet(CACHE)
    rows = old.to_dict("records")
    seen = set(zip(old.coin, old.ts))
    print(f"cache: {len(rows):,} candles")

total_chunks = len(COINS) * int(
    ((END - START).total_seconds() / 60) // CHUNK_MIN + 1)
done_chunks = 0
t0 = time.time()

for coin in COINS:
    cur = START
    while cur < END:
        nxt = min(cur + pd.Timedelta(minutes=CHUNK_MIN), END)
        done_chunks += 1
        # skip a chunk already fully cached
        want = int((nxt - cur).total_seconds() // 60)
        have = sum(1 for t in range(int(cur.timestamp()),
                                    int(nxt.timestamp()), 60)
                   if (coin, t) in seen)
        if have >= want * 0.95:
            cur = nxt
            continue
        url = (f"https://api.exchange.coinbase.com/products/{coin}-USD/"
               f"candles?granularity={GRAN}"
               f"&start={cur.strftime('%Y-%m-%dT%H:%M:%SZ')}"
               f"&end={nxt.strftime('%Y-%m-%dT%H:%M:%SZ')}")
        body = get(url)
        if isinstance(body, list):
            for c in body:
                # [ time, low, high, open, close, volume ]
                t = int(c[0])
                if (coin, t) in seen:
                    continue
                seen.add((coin, t))
                rows.append(dict(coin=coin, ts=t, low=float(c[1]),
                                 high=float(c[2]), open=float(c[3]),
                                 close=float(c[4]), vol=float(c[5])))
        time.sleep(SLEEP)
        if done_chunks % 100 == 0:
            el = time.time() - t0
            print(f"  {done_chunks}/{total_chunks} chunks  {len(rows):,} "
                  f"candles  eta {(total_chunks-done_chunks)*el/done_chunks/60:.0f}min",
                  flush=True)
            pd.DataFrame(rows).to_parquet(CACHE)
        cur = nxt

D = pd.DataFrame(rows).drop_duplicates(["coin", "ts"]).sort_values(
    ["coin", "ts"])
D.to_parquet(CACHE)
print(f"\nstored {len(D):,} 1-minute candles")
print(D.groupby("coin").size().to_string())
