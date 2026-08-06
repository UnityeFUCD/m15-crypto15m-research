"""Why did the in-band edge change from +3.99c to +2.54c?

Both figures were labelled "in-band 65-80c maker edge". They disagree, so at
least one is wrong or at least one is not measuring what its label says.

  project_synthesis.py  on ladder_paths  n 1,091  maker +3.99c  corrected +1.36c
  audit_unsampled.py    on book_full     n 7,504  maker +2.54c  corrected -0.17c

Three candidate explanations, tested in order of severity:

  A. DATA CONFLICT - the two files disagree about the same market. If the same
     ticker carries different bid/ask/outcome in each, one file is corrupt and
     every number built on it is void. This is the one that would be serious.

  B. POPULATION - ladder_paths holds only :00 and :30. book_full holds all
     four minutes. If :15 and :45 are worse, adding them lowers the average
     and both figures are correct measurements of different populations.

  C. SIGNAL FILTER - audit_unsampled requires nc==6 and a valid 24-window
     contiguous calm estimate, dropping markets ladder_paths keeps.

Run: python research/reconcile_edge.py
"""
from __future__ import annotations

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
P_WIN, P_LOSE = 0.843, 0.962


def fee1(p):
    return math.ceil(0.07 * p * (1 - p) * 100) / 100.0


def corr(g):
    pf = np.where(g.won == 1, P_WIN, P_LOSE)
    return (g.maker * pf).sum() / pf.sum() * 100


def stats(g, label):
    return ("  %-30s n %6d  win %.4f  maker %+6.2fc  corrected %+6.2fc  "
            "taker %+6.2fc" % (label, len(g), g.won.mean(),
                               g.maker.mean() * 100, corr(g),
                               g.taker.mean() * 100))


# ---- rebuild ladder_paths exactly as project_synthesis.py did --------------
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
    bid, ask = (yb, ya) if fy else (1.0 - ya, 1.0 - yb)
    ct = int((pd.to_datetime(r.ticker.split("-")[1], format="%y%b%d%H%M",
                             utc=True) + pd.Timedelta(hours=4)
              - EPOCH).total_seconds())
    rows.append(dict(ticker=r.ticker, coin=r.coin, close_ts=ct, bid=bid,
                     ask=ask, side="yes" if fy else "no",
                     won=1 if (("yes" if fy else "no") == r.result) else 0))
LP = pd.DataFrame(rows)
LP["minute"] = pd.to_datetime(LP.close_ts, unit="s", utc=True).dt.minute
LP["maker"] = LP.won - LP.bid
LP["taker"] = LP.won - LP.ask - LP.ask.map(fee1)
LPB = LP[(LP.bid >= 0.65) & (LP.bid < 0.80)]

BF = pd.read_parquet(DATA / "book_full.parquet")
BF["maker"] = BF.won - BF.bid
BF["taker"] = BF.won - BF.ask - BF.ask.map(fee1)
BFB = BF[(BF.bid >= 0.65) & (BF.bid < 0.80)]

print("=" * 88)
print("THE TWO FIGURES, REPRODUCED")
print("=" * 88)
print(stats(LPB, "ladder_paths in-band (old)"))
print(stats(BFB, "book_full in-band (new)"))

print("\n" + "=" * 88)
print("A. DATA CONFLICT - do the two files agree about the SAME markets?")
print("=" * 88)
m = LP.merge(BF, on="ticker", suffixes=("_lp", "_bf"))
print("  markets present in BOTH files: %d" % len(m))
if len(m):
    db = (m.bid_lp - m.bid_bf).abs()
    da = (m.ask_lp - m.ask_bf).abs()
    dw = (m.won_lp != m.won_bf)
    ds = (m.side_lp != m.side_bf)
    print("  bid  identical %.4f   max diff %.4f" % ((db < 1e-9).mean(),
                                                     db.max()))
    print("  ask  identical %.4f   max diff %.4f" % ((da < 1e-9).mean(),
                                                     da.max()))
    print("  side identical %.4f" % (~ds).mean())
    print("  won  identical %.4f" % (~dw).mean())
    verdict = ((db < 1e-9).all() and (da < 1e-9).all()
               and not ds.any() and not dw.any())
    print("  -> " + ("NO CONFLICT: the files agree exactly where they overlap"
                     if verdict else
                     "*** CONFLICT *** the files disagree - investigate before "
                     "trusting either"))

print("\n" + "=" * 88)
print("B. POPULATION - is the difference just the minutes ladder_paths lacks?")
print("=" * 88)
print("  ladder_paths minutes: %s" % LP.minute.value_counts().sort_index().to_dict())
print("  book_full    minutes: %s" % BF.minute.value_counts().sort_index().to_dict())
print()
print("  IN-BAND EDGE BY MINUTE, on the unsampled book_full:")
for mn, g in BFB.groupby("minute"):
    print(stats(g, f"    minute :{mn:02d}"))
print()
print("  APPLES TO APPLES - book_full restricted to :00/:30 only:")
print(stats(BFB[BFB.minute.isin([0, 30])], "    book_full :00/:30"))
print(stats(LPB, "    ladder_paths :00/:30"))
d = (BFB[BFB.minute.isin([0, 30])].maker.mean() - LPB.maker.mean()) * 100
print("  difference on the SAME minutes: %+.2fc" % d)

print("\n" + "=" * 88)
print("C. SIGNAL FILTER - what does requiring nc==6 and calm24 remove?")
print("=" * 88)
U = pd.read_parquet(DATA / "underlying.parquet",
                    columns=["coin", "wkey", "result", "a0"])
U["cts"] = ((pd.to_datetime(U.wkey, format="%y%b%d%H%M", utc=True)
             + pd.Timedelta(hours=4) - EPOCH).dt.total_seconds()
            ).round().astype("int64")
Uu = (U[U.result.isin(["yes", "no"])].dropna(subset=["a0"]).query("a0>0")
      .sort_values(["coin", "cts"]).drop_duplicates(["coin", "cts"]))
parts = []
for c, g in Uu.groupby("coin"):
    g = g.sort_values("cts").reset_index(drop=True)
    g["r"] = np.where(g.cts.diff() == 900, g.a0 / g.a0.shift(1) - 1.0, np.nan)
    parts.append(g)
S = pd.concat(parts, ignore_index=True)
CF = S.groupby("cts").agg(r_common=("r", "mean"), nc=("r", "count")).reset_index()
CF6 = CF[CF.nc == 6].sort_values("cts").reset_index(drop=True)
CF6["cg"] = CF6.cts.diff() == 900
CF6["calm24"] = CF6.r_common.abs().rolling(24).mean()
CF6.loc[CF6.cg.rolling(24).min() != 1, "calm24"] = np.nan
keep = set(CF6.dropna(subset=["calm24"]).cts)
print(stats(BFB, "  before signal filter"))
print(stats(BFB[BFB.close_ts.isin(keep)], "  after nc==6 + calm24"))
print("  removed %d markets (%.1f%%)"
      % (len(BFB) - BFB.close_ts.isin(keep).sum(),
         100 * (1 - BFB.close_ts.isin(keep).mean())))

print("\n" + "=" * 88)
print("VERDICT")
print("=" * 88)
lp, bf = LPB.maker.mean() * 100, BFB.maker.mean() * 100
same = BFB[BFB.minute.isin([0, 30])].maker.mean() * 100
print("  ladder_paths :00/:30 only        %+6.2fc   <- the OLD +3.99c figure" % lp)
print("  book_full    :00/:30 only        %+6.2fc" % same)
print("  book_full    all four minutes    %+6.2fc   <- the NEW figure" % bf)
print()
print("  If the middle line matches the top, the files agree and the change is")
print("  population: :15 and :45 are worse, and the old number was measured on")
print("  the better half of the day without saying so.")
