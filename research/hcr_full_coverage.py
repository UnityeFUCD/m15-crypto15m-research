"""HCR with FULL minute coverage - the definitive test.

DATA
  ladder_paths           minutes :00 and :30, all 24 hours   (1,091 in-band)
  book_minutes_15_45     minutes :15 and :45, all 24 hours   (fetched)
  -> combined: all four minutes, all hours, real bid AND ask

  premium_history2 is DELIBERATELY EXCLUDED here. It covers only 8 of 24 hours
  at minute :00 (8.3% of daily slots) and is hour-biased, so it cannot test the
  minute condition and can distort any hour-sensitive result. It produced the
  headline +9.18c figure; this script checks that against better coverage.
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


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


# ---- common factor ----
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

# ---- book :00 / :30 ----
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
# ---- book :15 / :45 ----
Bx = pd.read_parquet(DATA / "book_minutes_15_45.parquet")
BOOK = pd.concat([A, Bx[["ticker", "coin", "close_ts", "side", "bid", "ask",
                         "won"]]], ignore_index=True).drop_duplicates("ticker")
BOOK = BOOK.merge(CF[["close_ts", "r_common", "calm24"]], on="close_ts",
                  how="inner").dropna(subset=["calm24"])
BOOK["utc"] = pd.to_datetime(BOOK.close_ts, unit="s", utc=True)
BOOK["minute"] = BOOK.utc.dt.minute
BOOK["day"] = BOOK.utc.dt.date
BOOK["d"] = np.where(BOOK.side == "yes", 1.0, -1.0)
BOOK["HCR"] = ((BOOK.d * BOOK.r_common <= -0.0015)
               & (BOOK.calm24 <= 0.0030))
BOOK["maker"] = BOOK.won - BOOK.bid
BOOK["taker"] = BOOK.won - BOOK.ask - BOOK.ask.map(fee1)

print("=" * 78)
print("COVERAGE")
print("=" * 78)
print("  markets %d   days %d   hours %d of 24"
      % (len(BOOK), BOOK.day.nunique(), BOOK.utc.dt.hour.nunique()))
print("  by minute: %s" % dict(BOOK.minute.value_counts().sort_index()))

print("\n" + "=" * 78)
print("DOES HCR WORK AT EVERY MINUTE?  (the decisive test)")
print("=" * 78)
print("  %8s %7s %9s %11s %11s" % ("minute", "n HCR", "win", "maker", "TAKER"))
for mn in (0, 15, 30, 45):
    g = BOOK[(BOOK.minute == mn) & BOOK.HCR]
    if len(g) < 15:
        print("  %8d %7d   (too few)" % (mn, len(g)))
        continue
    print("  %8d %7d %9.4f %+10.2fc %+10.2fc"
          % (mn, len(g), g.won.mean(), g.maker.mean() * 100,
             g.taker.mean() * 100))
h = BOOK[BOOK.HCR]
r = BOOK[~BOOK.HCR]
print("  %8s %7d %9.4f %+10.2fc %+10.2fc"
      % ("ALL", len(h), h.won.mean(), h.maker.mean() * 100,
         h.taker.mean() * 100))
print("  %8s %7d %9.4f %+10.2fc %+10.2fc"
      % ("non-HCR", len(r), r.won.mean(), r.maker.mean() * 100,
         r.taker.mean() * 100))

print("\n" + "=" * 78)
print("WIN RATE WITH CONFIDENCE - full coverage")
print("=" * 78)
lo, hi = wilson(int(h.won.sum()), len(h))
print("  HCR win rate      %.4f   n %d" % (h.won.mean(), len(h)))
print("  95%% Wilson CI     [%.4f, %.4f]" % (lo, hi))
gd = {k: v for k, v in h.groupby("day")}
ds = sorted(gd)
bw = np.sort(np.array([pd.concat([gd[ds[i]] for i in
                                  RNG.integers(0, len(ds), len(ds))]).won.mean()
                       for _ in range(6000)]))
print("  95%% day-clustered [%.4f, %.4f]" % (bw[150], bw[5849]))
lo2, hi2 = wilson(int(r.won.sum()), len(r))
print("  non-HCR win rate  %.4f   95%% Wilson [%.4f, %.4f]"
      % (r.won.mean(), lo2, hi2))
print("  win-rate lift     %+.2fpp" % ((h.won.mean() - r.won.mean()) * 100))

print("\n" + "=" * 78)
print("EDGE WITH CONFIDENCE - full coverage, day-clustered")
print("=" * 78)
for lbl, col in (("maker (assumes fill)", "maker"),
                 ("TAKER (real ask+fee)", "taker")):
    b = np.sort(np.array([pd.concat([gd[ds[i]] for i in
                                     RNG.integers(0, len(ds), len(ds))])[col]
                          .mean() * 100 for _ in range(6000)]))
    print("  %-24s %+6.2fc   95%% CI [%+6.2f, %+6.2f]   P(<=0) %.4f"
          % (lbl, h[col].mean() * 100, b[150], b[5849], (b <= 0).mean()))

print("\n" + "=" * 78)
print("IS THE MINUTE-00 FILTER JUSTIFIED?  (t-test, full coverage)")
print("=" * 78)
a = BOOK[(BOOK.minute == 0) & BOOK.HCR]
o = BOOK[(BOOK.minute != 0) & BOOK.HCR]
d = (a.taker.mean() - o.taker.mean()) * 100
se = np.sqrt(a.taker.var() / len(a) + o.taker.var() / len(o)) * 100
print("  minute 00  n %4d  taker %+6.2fc" % (len(a), a.taker.mean() * 100))
print("  other      n %4d  taker %+6.2fc" % (len(o), o.taker.mean() * 100))
print("  difference %+.2fc   SE %.2fc   t %.2f  -> %s"
      % (d, se, d / se,
         "JUSTIFIED" if abs(d / se) > 1.96 else "NOT justified"))

print("\n" + "=" * 78)
print("DOLLARS AT QTY 20 - does HCR beat trading everything?")
print("=" * 78)
days = BOOK.day.nunique()
print("  %-22s %7s %11s %11s %11s"
      % ("policy", "n", "maker $/day", "taker $/day", "per ct"))
for lbl, g in (("all markets", BOOK), ("HCR only", h), ("non-HCR", r)):
    print("  %-22s %7d %+11.2f %+11.2f %+10.2fc"
          % (lbl, len(g), g.maker.sum() * 20 / days,
             g.taker.sum() * 20 / days, g.taker.mean() * 100))
