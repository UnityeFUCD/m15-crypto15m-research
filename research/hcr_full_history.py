"""HCR as far back as the data allows: 73 days, with win-rate confidence.

WHAT CAN AND CANNOT BE TESTED BACK
  HCR selection    -> yes, 73 days. It needs only benchmark A0 values and the
                      book, both of which exist historically.
  cap 2            -> yes, using the runner's own series order
                      (BTC, ETH, SOL, XRP, DOGE, HYPE) as the rank proxy.
  RACE 8s/2s       -> NO. Timeout policy needs FILL LATENCY, which only exists
                      for orders we actually placed. It cannot be reconstructed
                      from historical book data at any price.

So "policy 7" is only testable in its HCR + cap2 form over the full history.
The timing layer remains a 2-day, 275-market result.
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
SERIES_ORDER = {"BTC": 0, "ETH": 1, "SOL": 2, "XRP": 3, "DOGE": 4, "HYPE": 5}


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


def tag(df):
    df = df.merge(CF[["close_ts", "r_common", "calm24"]], on="close_ts",
                  how="inner").dropna(subset=["calm24"])
    df["d"] = np.where(df.side == "yes", 1.0, -1.0)
    df["HCR"] = (df.d * df.r_common <= -0.0015) & (df.calm24 <= 0.0030)
    df["utc"] = pd.to_datetime(df.close_ts, unit="s", utc=True)
    df["day"] = df.utc.dt.date
    df["week"] = df.utc.dt.isocalendar().week
    df["ord"] = df.coin.map(SERIES_ORDER).fillna(9)
    df = df.sort_values(["close_ts", "ord"])
    df["rank"] = df.groupby("close_ts").cumcount() + 1
    return df


# ---- book 1: premium_history2 (bid only -> maker edge) ----
P = pd.read_parquet(DATA / "premium_history2.parquet")
P["close_ts"] = ((pd.to_datetime(P.close, utc=True) - EPOCH)
                 .dt.total_seconds()).round().astype("int64")
P = P.rename(columns={"px": "bid"})
P = tag(P)
P["edge"] = P.won - P.bid

# ---- book 2: ladder_paths (has real asks -> taker edge) ----
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
B = tag(pd.DataFrame(rows))
B["edge"] = B.won - B.ask - B.ask.map(fee1)      # TAKER, real cost

print("=" * 78)
print("HCR OVER THE FULL AVAILABLE HISTORY")
print("=" * 78)
for nm, df, kind in (("premium_history2", P, "maker"),
                     ("ladder_paths", B, "TAKER (real ask+fee)")):
    h = df[df.HCR]
    r = df[~df.HCR]
    print("\n  %s   %s .. %s   %d days" % (nm, df.utc.min().date(),
                                           df.utc.max().date(),
                                           df.day.nunique()))
    print("    %-10s n %5d  win %.4f  %s edge %+6.2fc"
          % ("HCR", len(h), h.won.mean(), kind, h.edge.mean() * 100))
    print("    %-10s n %5d  win %.4f  %s edge %+6.2fc"
          % ("non-HCR", len(r), r.won.mean(), kind, r.edge.mean() * 100))
    lo, hi = wilson(int(h.won.sum()), len(h))
    print("    HCR win rate 95%% Wilson CI  [%.4f, %.4f]" % (lo, hi))
    gd = {k: v for k, v in h.groupby("day")}
    ds = sorted(gd)
    bw = np.sort(np.array([pd.concat([gd[ds[i]] for i in
                                      RNG.integers(0, len(ds), len(ds))]
                                     ).won.mean() for _ in range(6000)]))
    be = np.sort(np.array([pd.concat([gd[ds[i]] for i in
                                      RNG.integers(0, len(ds), len(ds))]
                                     ).edge.mean() * 100 for _ in range(6000)]))
    print("    HCR win rate 95%% day-clustered [%.4f, %.4f]"
          % (bw[150], bw[5849]))
    print("    HCR edge     95%% day-clustered [%+.2fc, %+.2fc]  P(<=0) %.4f"
          % (be[150], be[5849], (be <= 0).mean()))

print("\n" + "=" * 78)
print("HCR + CAP 2 OVER FULL HISTORY  (RACE timing NOT testable back)")
print("=" * 78)
QTY = 20
for nm, df, kind in (("premium_history2", P, "maker"),
                     ("ladder_paths", B, "taker")):
    days = df.day.nunique()
    print("\n  %s  (%s edge, qty %d, %d days)" % (nm, kind, QTY, days))
    print("    %-26s %7s %9s %11s %11s"
          % ("policy", "n", "win", "total $", "$/day"))
    for lbl, sel in (("all markets", df),
                     ("cap2 only", df[df["rank"] <= 2]),
                     ("HCR only", df[df.HCR]),
                     ("HCR + cap2", df[df.HCR & (df["rank"] <= 2)])):
        if not len(sel):
            continue
        tot = sel.edge.sum() * QTY
        print("    %-26s %7d %9.4f %+11.2f %+11.2f"
              % (lbl, len(sel), sel.won.mean(), tot, tot / days))

print("\n" + "=" * 78)
print("STABILITY - HCR edge by week (premium_history2, maker)")
print("=" * 78)
h = P[P.HCR]
wk = h.groupby("week").agg(n=("edge", "size"), win=("won", "mean"),
                           e=("edge", "mean"))
wk["e"] = (wk.e * 100).round(2)
print(wk.to_string())
print("\n  weeks positive: %d of %d" % ((wk.e > 0).sum(), len(wk)))
print("  worst week %+.2fc   best week %+.2fc" % (wk.e.min(), wk.e.max()))
