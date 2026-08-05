"""Has the longshot premium stopped decaying, or is it still going?

WHY THIS IS NOW THE RIGHT QUESTION
  A day of work has ruled out the things that would make the LOSSES fixable:

    volatility     - the market prices it correctly. Edge is flat across
                     regimes (65-70c band: calm +13.68c, wild +13.03c) and
                     every true-volatility correlation is null.
    mean reversion - the apparent reversion was an overlapping-endpoint
                     artifact. With non-overlapping pairs the sign flips
                     positive in 5 of 6 coins.
    fill rate      - low fill is good news, not lost edge.
    queue depth, coin choice, rank, breadth - all rejected.

  So the losses are the correctly-priced 15-20% where a favourite loses. They
  are not a leak. What actually costs money is that the PREMIUM we are paid
  fell about 78% after Aug 1 - from ~+10.4c to ~+2.3c per contract.

  That is the number that decides whether this strategy is worth running, and
  it has only ever been measured on 10 days pre-shift and 3 days after.

WHAT THIS RECONSTRUCTS
  Candlesticks give per-minute yes_bid and yes_ask for every market. Combined
  with the settled result, that is enough to replay the strategy's entry
  exactly as the runner does it - favourite side, bid price, 8-14 minutes
  before close - on any historical day.

  Doing that from 25 May gives ~73 days of the population premium instead of
  13, which is enough to distinguish:

    a STEP DOWN  - a competitor arrived, or the exchange changed something.
                   The new level is the new reality and should be planned for.
    a DECAY      - the premium is being competed away continuously and the
                   strategy has a finite remaining life that can be estimated.
    a CYCLE      - it moves with something (volume, volatility, weekday) and
                   will come back.

  Those imply completely different decisions, and they are indistinguishable
  on 13 days.

SAMPLING AND COST
  One API call per market, so the full 41k is not practical. Sampling 8 windows
  per day per coin gives ~48 entries per day - enough for a daily estimate -
  at roughly 3,500 calls. Results are cached so this is never repeated.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from acct import api

LIVE = Path(__file__).resolve().parent
CACHE = LIVE / "premium_history2.parquet"
U = pd.read_parquet(LIVE / "underlying.parquet")
# BUG FOUND: expected_expiration_time is the window close PLUS 5 minutes, so
# using it made every "8-14 minutes before close" entry actually 3-9 minutes
# before close - deep into resolution. The wkey carries the true close on the
# ET ticker clock; all data here is May-Aug, i.e. EDT = UTC-4.
U["close"] = (pd.to_datetime(U.wkey, format="%y%b%d%H%M", utc=True)
              + pd.Timedelta(hours=4))

# 4.5% of tickers do not sit exactly 5 minutes before expected_expiration -
# their wkey does not decode cleanly. Drop them rather than guess.
_exp = pd.to_datetime(U.pop("close_meta") if "close_meta" in U else
                      pd.read_parquet(LIVE / "underlying.parquet").close,
                      format="mixed", utc=True)
U = U[((_exp - U.close).dt.total_seconds() / 60).round(1) == 5.0].copy()
print(f"kept {len(U):,} markets with a clean close time")

# sample 8 windows per coin-day, spread across the clock
U["date"] = U.close.dt.date
U["slot"] = U.close.dt.hour * 4 + U.close.dt.minute // 15
SAMPLE = (U[U.slot % 12 == 0]
          .sort_values(["coin", "close"]).reset_index(drop=True))
print(f"candidate markets to sample: {len(SAMPLE):,} "
      f"over {SAMPLE.date.nunique()} days")

done = {}
if CACHE.exists():
    old = pd.read_parquet(CACHE)
    done = set(old.ticker)
    rows = old.to_dict("records")
    print(f"cache: {len(rows):,} already fetched")
else:
    rows = []

todo = SAMPLE[~SAMPLE.ticker.isin(done)]
print(f"to fetch: {len(todo):,}\n")
def fetch(series, ticker, close_ts, tries=3):
    """The API times out intermittently; a transient failure must not lose
    the whole run, and must not be retried so hard that it competes with the
    live runner for the same connection pool."""
    for a in range(tries):
        try:
            return api(f"/series/{series}/markets/{ticker}/candlesticks",
                       {"period_interval": 1,
                        "start_ts": close_ts - 16 * 60, "end_ts": close_ts})
        except Exception:
            if a == tries - 1:
                return None, None
            time.sleep(1.5 * (a + 1))
    return None, None


t0, errs = time.time(), 0
for i, r in enumerate(todo.itertuples()):
    close_ts = int(r.close.timestamp())
    c, j = fetch(r.series, r.ticker, close_ts)
    if c != 200 or not j:
        errs += 1
        continue
    cs = j.get("candlesticks") or []
    best = None
    for k in cs:
        mins_left = (close_ts - int(k.get("end_period_ts") or 0)) / 60.0
        if not (8 <= mins_left <= 14):
            continue
        try:
            yb = float(k["yes_bid"]["close_dollars"])
            ya = float(k["yes_ask"]["close_dollars"])
        except (KeyError, TypeError, ValueError):
            continue
        if yb <= 0 or ya >= 1 or yb >= ya:
            continue
        # the runner joins the favourite's bid
        if yb >= 0.5:
            side, px = "yes", yb
        else:
            side, px = "no", 1.0 - ya
        if not (0.65 <= px < 0.80):
            continue
        vol = float(k.get("volume_fp") or 0)
        # take the EARLIEST qualifying minute, as the runner does
        if best is None or mins_left > best["mins_left"]:
            best = dict(mins_left=mins_left, side=side, px=px, vol=vol)
    if best:
        won = 1 if best["side"] == r.result else 0
        rows.append(dict(ticker=r.ticker, coin=r.coin, date=str(r.date),
                         close=r.close.isoformat(), side=best["side"],
                         px=best["px"], won=won, edge=won - best["px"],
                         vol=best["vol"], mins_left=best["mins_left"],
                         a0=r.a0, a1=r.a1, ret=r.ret))
    if (i + 1) % 250 == 0:
        el = time.time() - t0
        print(f"  {i+1:>5}/{len(todo)}  kept {len(rows):>5}  errs {errs}  "
              f"{el:.0f}s  eta {el/(i+1)*(len(todo)-i-1)/60:.1f}min",
              flush=True)
        pd.DataFrame(rows).drop_duplicates(subset=["ticker"]).to_parquet(CACHE)

D = pd.DataFrame(rows).drop_duplicates(subset=["ticker"])
D.to_parquet(CACHE)
print(f"\nreconstructed entries: {len(D):,} over {D.date.nunique()} days")
if len(D) < 300:
    sys.exit("too few to analyse yet - rerun to continue fetching")

print("\n" + "=" * 78)
print("THE PREMIUM, DAY BY DAY")
print("=" * 78)
g = D.groupby("date").agg(n=("edge", "size"), edge=("edge", "mean"),
                          win=("won", "mean"), px=("px", "mean"))
g["edge_c"] = g.edge * 100
for d, r in g.iterrows():
    bar = "#" * int(max(r.edge_c, 0) / 1.5)
    print(f"  {d}  n {int(r.n):>3}  win {r.win:.3f}  px {r.px*100:.1f}c  "
          f"{r.edge_c:>+7.2f}c  {bar}")

print("\n" + "=" * 78)
print("IS IT STILL DECAYING?")
print("=" * 78)
g = g.reset_index()
g["t"] = np.arange(len(g))
sl = np.polyfit(g.t, g.edge_c, 1)
print(f"  linear trend over {len(g)} days: {sl[0]:+.4f}c per day "
      f"(intercept {sl[1]:+.2f}c)")
half = len(g) // 2
print(f"  first half mean {g.edge_c[:half].mean():+.2f}c   "
      f"second half mean {g.edge_c[half:].mean():+.2f}c")
last14 = g.tail(14)
print(f"  last 14 days: mean {last14.edge_c.mean():+.2f}c   "
      f"trend {np.polyfit(last14.t, last14.edge_c, 1)[0]:+.4f}c/day")
RNG = np.random.default_rng(20261055)
bs = np.sort(np.array([
    np.polyfit(*(lambda i: (g.t.values[i], g.edge_c.values[i]))(
        RNG.integers(0, len(g), len(g))), 1)[0] for _ in range(6000)]))
print(f"  trend 95% CI [{bs[150]:+.4f}, {bs[5849]:+.4f}]   "
      f"P(no decay) {(bs >= 0).mean():.4f}")
D.to_parquet(CACHE)
