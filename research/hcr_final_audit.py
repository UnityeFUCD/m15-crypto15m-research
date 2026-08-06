"""HCR final audit on the calendar-balanced full-coverage dataset.

DATA
  ladder_paths        :00 / :30, 73 days
  book_minutes_15_45  :15 / :45, 73 days (randomly sampled across the period)
  -> all four minutes, all 24 hours, real bid AND ask

  premium_history2 is EXCLUDED. It is minute-:00 only and covers 8 of 24
  hours, and it is what produced the earlier +9.18c / +10.69pp figures that
  did not survive full coverage.

THREE INDEPENDENT TESTS, all on the same balanced population:
  1. matched controls  - same coin, week, 2c price bucket, SAME SIDE
  2. permutation       - shuffle r_common within day
  3. chronological     - fixed train / valid / test boundaries

Everything is measured on TAKER edge (real ask + real fee), because that is
what the strategy would actually pay, and on day-clustered intervals.
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
SPLITS = {"train": ("2026-05-25", "2026-06-30"),
          "valid": ("2026-06-30", "2026-07-18"),
          "test": ("2026-07-18", "2026-08-07")}


def fee1(p):
    return math.ceil(0.07 * p * (1 - p) * 100) / 100.0


def build():
    U = pd.read_parquet(DATA / "underlying.parquet")
    U["close_ts"] = ((pd.to_datetime(U.wkey, format="%y%b%d%H%M", utc=True)
                      + pd.Timedelta(hours=4) - EPOCH).dt.total_seconds()
                     ).round().astype("int64")
    Uu = (U[U.result.isin(["yes", "no"])].dropna(subset=["a0"]).query("a0>0")
          .sort_values(["coin", "close_ts"])
          .drop_duplicates(["coin", "close_ts"]))
    parts = []
    for c, g in Uu.groupby("coin"):
        g = g.sort_values("close_ts").reset_index(drop=True)
        g["r"] = np.where(g.close_ts.diff() == 900,
                          g.a0 / g.a0.shift(1) - 1.0, np.nan)
        parts.append(g)
    S = pd.concat(parts, ignore_index=True)
    CF = (S.groupby("close_ts")
          .agg(r_common=("r", "mean"), nc=("r", "count")).reset_index())
    CF = CF[CF.nc >= 4].sort_values("close_ts").reset_index(drop=True)
    # calm needs 24 CONTIGUOUS windows
    CF["contig"] = CF.close_ts.diff() == 900
    CF["calm24"] = CF.r_common.abs().rolling(24).mean()
    CF.loc[CF.contig.rolling(24).min() != 1, "calm24"] = np.nan

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
    Bx = pd.read_parquet(DATA / "book_minutes_15_45.parquet")
    B = (pd.concat([pd.DataFrame(rows),
                    Bx[["ticker", "coin", "close_ts", "side", "bid", "ask",
                        "won"]]], ignore_index=True)
         .drop_duplicates("ticker")
         .merge(CF[["close_ts", "r_common", "calm24"]], on="close_ts",
                how="inner").dropna(subset=["calm24"]))
    B["utc"] = pd.to_datetime(B.close_ts, unit="s", utc=True)
    B["day"] = B.utc.dt.date
    B["week"] = B.utc.dt.isocalendar().week
    B["minute"] = B.utc.dt.minute
    B["d"] = np.where(B.side == "yes", 1.0, -1.0)
    B["HCR"] = (B.d * B.r_common <= -0.0015) & (B.calm24 <= 0.0030)
    B["taker"] = B.won - B.ask - B.ask.map(fee1)
    B["maker"] = B.won - B.bid
    B["pb"] = (B.bid * 50).round().astype(int)
    return B


B = build()
h, r = B[B.HCR], B[~B.HCR]
print("=" * 78)
print("POPULATION (calendar-balanced, all 4 minutes, 24 hours)")
print("=" * 78)
print("  markets %d   HCR %d   days %d   minutes %s"
      % (len(B), len(h), B.day.nunique(),
         sorted(B.minute.unique())))
print("  HCR      win %.4f  maker %+6.2fc  TAKER %+6.2fc"
      % (h.won.mean(), h.maker.mean() * 100, h.taker.mean() * 100))
print("  non-HCR  win %.4f  maker %+6.2fc  TAKER %+6.2fc"
      % (r.won.mean(), r.maker.mean() * 100, r.taker.mean() * 100))
print("  raw lift win %+.2fpp   taker %+.2fc"
      % ((h.won.mean() - r.won.mean()) * 100,
         (h.taker.mean() - r.taker.mean()) * 100))

print("\n" + "=" * 78)
print("TEST 1 - MATCHED CONTROLS (same coin, week, 2c price bucket, SAME side)")
print("=" * 78)
res = []
for (c, w, pb, sd), g in B.groupby(["coin", "week", "pb", "side"]):
    a, b = g[g.HCR], g[~g.HCR]
    if len(a) >= 1 and len(b) >= 2:
        res.append((len(a), a.taker.mean() - b.taker.mean(),
                    a.won.mean() - b.won.mean()))
if res:
    wts = np.array([x[0] for x in res], dtype=float)
    dt = np.array([x[1] for x in res])
    dw = np.array([x[2] for x in res])
    print("  strata %d covering %d HCR markets" % (len(res), int(wts.sum())))
    print("  weighted TAKER lift %+.2fc   win lift %+.2fpp"
          % (np.average(dt, weights=wts) * 100,
             np.average(dw, weights=wts) * 100))
    bs = np.sort(np.array([
        np.average(dt[i], weights=wts[i]) * 100
        for i in (RNG.integers(0, len(res), len(res)) for _ in range(8000))]))
    print("  95%% CI [%+.2f, %+.2f]   P(<=0) %.4f"
          % (bs[200], bs[7799], (bs <= 0).mean()))

print("\n" + "=" * 78)
print("TEST 2 - PERMUTATION (shuffle r_common within day)")
print("=" * 78)
obs = h.taker.mean() * 100
perm = []
for _ in range(3000):
    q = B.copy()
    q["rc"] = q.groupby("day").r_common.transform(
        lambda s: RNG.permutation(s.values))
    g = q[(q.d * q.rc <= -0.0015) & (q.calm24 <= 0.0030)]
    if len(g) >= 100:
        perm.append(g.taker.mean() * 100)
perm = np.array(perm)
print("  observed HCR taker %+.2fc" % obs)
print("  permuted mean %+.2fc   p95 %+.2fc   P(permuted >= observed) %.4f"
      % (perm.mean(), np.quantile(perm, .95), (perm >= obs).mean()))

print("\n" + "=" * 78)
print("TEST 3 - CHRONOLOGICAL (fixed boundaries)")
print("=" * 78)
for nm, (a, b) in SPLITS.items():
    hs = h[(h.utc >= a) & (h.utc < b)]
    rs = r[(r.utc >= a) & (r.utc < b)]
    if len(hs) < 20:
        print("  %-6s n %3d (too few)" % (nm, len(hs)))
        continue
    print("  %-6s HCR n %3d win %.4f taker %+6.2fc | non-HCR win %.4f "
          "taker %+6.2fc | lift %+6.2fc"
          % (nm, len(hs), hs.won.mean(), hs.taker.mean() * 100,
             rs.won.mean(), rs.taker.mean() * 100,
             (hs.taker.mean() - rs.taker.mean()) * 100))

print("\n" + "=" * 78)
print("TEST 4 - PER COIN (is it one coin?)")
print("=" * 78)
print("  %-6s %6s %9s %11s %11s" % ("coin", "n HCR", "win", "taker", "lift"))
for c, g in B.groupby("coin"):
    a, b = g[g.HCR], g[~g.HCR]
    if len(a) < 20:
        continue
    print("  %-6s %6d %9.4f %+10.2fc %+10.2fc"
          % (c, len(a), a.won.mean(), a.taker.mean() * 100,
             (a.taker.mean() - b.taker.mean()) * 100))

print("\n" + "=" * 78)
print("TEST 5 - DOLLARS AT q15 (does HCR beat trading everything?)")
print("=" * 78)
days = B.day.nunique()
print("  %-16s %7s %13s %13s" % ("policy", "n", "taker $/day", "per contract"))
for lbl, g in (("all markets", B), ("HCR only", h), ("non-HCR", r)):
    print("  %-16s %7d %+13.2f %+12.2fc"
          % (lbl, len(g), g.taker.sum() * 15 / days, g.taker.mean() * 100))
