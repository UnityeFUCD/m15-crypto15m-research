"""Is HCR's minute-00 effect real, or is it time confounded? Day-matched test.

THE PROBLEM THIS SOLVES
  The :15/:45 book had to be fetched separately, and the two fetches do not
  cover the calendar identically:
    ladder_paths (:00/:30)      random across 73 days
    first fetch  (:15/:45)      EARLIEST 18 days   <- my sampling bug
    second fetch (:15/:45)      random across 73 days
  so the pooled :15/:45 sample is over-weighted toward May-June.

  Comparing minutes across unequal calendars measures TIME, not minute.

THE FIX
  Compare minutes only WITHIN the same day. Every day contributes a paired
  difference, so any market-wide regime shift cancels out. Days without both
  arms are dropped rather than imputed.
"""
from __future__ import annotations

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
RNG = np.random.default_rng(20260806)


def fee1(p):
    return math.ceil(0.07 * p * (1 - p) * 100) / 100.0


U = pd.read_parquet(DATA / "underlying.parquet")
U["close_ts"] = ((pd.to_datetime(U.wkey, format="%y%b%d%H%M", utc=True)
                  + pd.Timedelta(hours=4) - EPOCH).dt.total_seconds()
                 ).round().astype("int64")
Uu = (U[U.result.isin(["yes", "no"])].dropna(subset=["a0"]).query("a0>0")
      .sort_values(["coin", "close_ts"]).drop_duplicates(["coin", "close_ts"]))
parts = []
for c, g in Uu.groupby("coin"):
    g = g.sort_values("close_ts").reset_index(drop=True)
    g["r"] = np.where(g.close_ts.diff() == 900, g.a0 / g.a0.shift(1) - 1.0,
                      np.nan)
    parts.append(g)
S = pd.concat(parts, ignore_index=True)
CF = (S.groupby("close_ts").agg(r_common=("r", "mean"), nc=("r", "count"))
      .reset_index())
CF = CF[CF.nc >= 4].sort_values("close_ts").reset_index(drop=True)
CF["calm24"] = CF.r_common.abs().rolling(24).mean()

L = pd.read_parquet(DATA / "ladder_paths.parquet").drop_duplicates("ticker")
rows = []
for r in L.itertuples():
    try:
        p = json.loads(r.path)
    except Exception:
        continue
    e = sorted([k for k in p if 8 <= k["ml"] <= 14], key=lambda x: -x["ml"])
    if not e:
        continue
    e = e[0]
    yb, ya = e["bc"], e["ac"]
    if yb <= 0 or ya >= 1 or yb >= ya:
        continue
    fy = yb >= 0.5
    bid = yb if fy else 1.0 - ya
    ask = ya if fy else 1.0 - yb
    if not (0.65 <= bid < 0.80):
        continue
    ct = int((pd.to_datetime(r.ticker.split("-")[1], format="%y%b%d%H%M",
                             utc=True) + pd.Timedelta(hours=4)
              - EPOCH).total_seconds())
    rows.append(dict(ticker=r.ticker, coin=r.coin, close_ts=ct,
                     side="yes" if fy else "no", bid=bid, ask=ask,
                     won=1 if (("yes" if fy else "no") == r.result) else 0))
A = pd.DataFrame(rows)
Bx = pd.read_parquet(DATA / "book_minutes_15_45.parquet")
BOOK = (pd.concat([A, Bx[["ticker", "coin", "close_ts", "side", "bid", "ask",
                          "won"]]], ignore_index=True)
        .drop_duplicates("ticker")
        .merge(CF[["close_ts", "r_common", "calm24"]], on="close_ts",
               how="inner").dropna(subset=["calm24"]))
BOOK["utc"] = pd.to_datetime(BOOK.close_ts, unit="s", utc=True)
BOOK["minute"] = BOOK.utc.dt.minute
BOOK["day"] = BOOK.utc.dt.date
BOOK["d"] = np.where(BOOK.side == "yes", 1.0, -1.0)
BOOK["HCR"] = (BOOK.d * BOOK.r_common <= -0.0015) & (BOOK.calm24 <= 0.0030)
BOOK["maker"] = BOOK.won - BOOK.bid
BOOK["taker"] = BOOK.won - BOOK.ask - BOOK.ask.map(fee1)

print("=" * 78)
print("CALENDAR COVERAGE BY MINUTE (the confound being removed)")
print("=" * 78)
print("  %8s %7s %13s %13s %7s" % ("minute", "n", "first", "last", "days"))
for mn, g in BOOK.groupby("minute"):
    print("  %8d %7d %13s %13s %7d"
          % (mn, len(g), g.day.min(), g.day.max(), g.day.nunique()))

print("\n" + "=" * 78)
print("UNMATCHED (naive) minute comparison - HCR only")
print("=" * 78)
h = BOOK[BOOK.HCR]
print("  %8s %7s %9s %11s %11s" % ("minute", "n", "win", "maker", "taker"))
for mn, g in h.groupby("minute"):
    print("  %8d %7d %9.4f %+10.2fc %+10.2fc"
          % (mn, len(g), g.won.mean(), g.maker.mean() * 100,
             g.taker.mean() * 100))

print("\n" + "=" * 78)
print("DAY-MATCHED: minute :00 vs all other minutes, WITHIN each day")
print("=" * 78)
pairs = []
for day, g in h.groupby("day"):
    a = g[g.minute == 0]
    b = g[g.minute != 0]
    if len(a) >= 1 and len(b) >= 1:
        pairs.append((day, len(a), len(b),
                      a.taker.mean() - b.taker.mean(),
                      a.won.mean() - b.won.mean()))
if not pairs:
    print("  no days contain both arms")
else:
    P = pd.DataFrame(pairs, columns=["day", "n00", "noth", "dtaker", "dwin"])
    print("  days with both arms: %d   (mean %.1f at :00, %.1f elsewhere)"
          % (len(P), P.n00.mean(), P.noth.mean()))
    print("  mean within-day taker difference (:00 minus other): %+.2fc"
          % (P.dtaker.mean() * 100))
    print("  mean within-day win difference                    : %+.2fpp"
          % (P.dwin.mean() * 100))
    bs = np.sort(np.array([RNG.choice(P.dtaker.values, len(P), True).mean()
                           * 100 for _ in range(8000)]))
    print("  95%% CI [%+.2f, %+.2f]   P(<=0) %.4f   days favouring :00 %d/%d"
          % (bs[200], bs[7799], (bs <= 0).mean(),
             int((P.dtaker > 0).sum()), len(P)))

print("\n" + "=" * 78)
print("DAY-MATCHED: HCR vs non-HCR, WITHIN each day (the signal itself)")
print("=" * 78)
for lbl, sub in (("all minutes", BOOK),
                 ("minute :00 only", BOOK[BOOK.minute == 0]),
                 ("excluding :00", BOOK[BOOK.minute != 0])):
    pr = []
    for day, g in sub.groupby("day"):
        a = g[g.HCR]
        b = g[~g.HCR]
        if len(a) >= 1 and len(b) >= 1:
            pr.append((a.taker.mean() - b.taker.mean(),
                       a.won.mean() - b.won.mean(), len(a)))
    if len(pr) < 10:
        print("  %-18s too few paired days (%d)" % (lbl, len(pr)))
        continue
    D = pd.DataFrame(pr, columns=["dt", "dw", "n"])
    bs = np.sort(np.array([RNG.choice(D.dt.values, len(D), True).mean() * 100
                           for _ in range(8000)]))
    print("  %-18s days %3d  taker lift %+6.2fc  95%% CI [%+6.2f, %+6.2f]  "
          "P(<=0) %.4f  win lift %+5.2fpp"
          % (lbl, len(D), D.dt.mean() * 100, bs[200], bs[7799],
             (bs <= 0).mean(), D.dw.mean() * 100))
