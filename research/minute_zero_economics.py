"""CORRECTION 3 - minute_zero.md economics were computed on the wrong basis.

THE ERROR
  minute_zero.md reported "+2.37c fill-corrected x 25.7 markets/day x q15 =
  +$9.11/day". Those quantities do not compose.

    corr()  = SUM(maker_i * pf_i) / SUM(pf_i)  = E[maker | FILLED]
              -> value per contract that ACTUALLY TRADES
    25.7/day = eligible/POSTED markets per day
              -> orders SUBMITTED, most but not all of which fill

  Multiplying an edge-per-filled by a count-of-posted assumes a 100% fill
  rate, which is precisely the assumption this whole project exists to
  reject. The correct quantity for a decision to post is

    EV_submitted = E[maker * 1{filled}] = mean(maker_i * pf_i)

  which equals E[maker|filled] * fill_rate, and is strictly smaller.

WHY THE CORRECTION IS LARGER THAN IT LOOKS
  AUDIT A finds :00 has the WORST fill bias of the four minutes -
  P(fill|win) 0.8167 against P(fill|lose) 1.0000, a +18.3pp gap versus the
  pooled +10.4pp. Applying a pooled fill model to :00 therefore FLATTERS it.
  This script prices :00 under its own measured fill behaviour as well.

CAVEAT CARRIED THROUGH
  The :00-specific fill rates rest on 77 live orders over 2 days, and AUDIT A
  states that cell is underpowered for its own measurement (detectable gap
  23.1pp vs observed 18.3pp). So the answer is reported as a RANGE across
  three fill models rather than a single number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
RNG = np.random.default_rng(20260806)
Q = 15

MODELS = {
    "pooled original  0.843 / 0.962": (0.843, 0.962),
    "pooled corrected 0.8848/0.9884": (0.8848, 0.9884),
    ":00-specific     0.8167/1.0000": (0.8167, 1.0000),
}

BF = pd.read_parquet(DATA / "book_full.parquet")
BF["maker"] = BF.won - BF.bid
BF["utc"] = pd.to_datetime(BF.close_ts, unit="s", utc=True)
BF["day"] = BF.utc.dt.date
B = BF[(BF.bid >= 0.65) & (BF.bid < 0.80)].copy()
Z = B[B.minute == 0]
days = B.day.nunique()
per_day = len(Z) / days

print("=" * 88)
print("MINUTE :00 ECONOMICS - CORRECTED BASIS")
print("=" * 88)
print(f"  in-band :00 markets {len(Z)}  over {days} days  = {per_day:.1f}/day")
print(f"  raw maker edge (assumes every order fills): "
      f"{Z.maker.mean()*100:+.2f}c")
print()
print("  %-32s %11s %13s %14s %12s"
      % ("fill model", "fill rate", "c/FILLED", "c/SUBMITTED", "$/day q15"))
out = {}
for name, (pw, pl) in MODELS.items():
    pf = np.where(Z.won == 1, pw, pl)
    per_filled = (Z.maker * pf).sum() / pf.sum() * 100
    ev_sub = (Z.maker * pf).mean() * 100
    fr = pf.mean()
    dpd = per_day * Q * ev_sub / 100
    out[name] = (per_filled, ev_sub, dpd)
    print("  %-32s %11.4f %+12.2fc %+13.2fc %+11.2f"
          % (name, fr, per_filled, ev_sub, dpd))

print("\n  The old figure was +$9.11/day. It used c/FILLED (+2.37c) against a")
print("  count of POSTED markets, i.e. it assumed a 100% fill rate.")

print("\n" + "=" * 88)
print("DAY-CLUSTERED INTERVAL ON EV PER SUBMITTED CONTRACT (73 days)")
print("=" * 88)
gd = {k: v for k, v in Z.groupby("day")}
ds = sorted(gd)
for name, (pw, pl) in MODELS.items():
    bs = []
    for _ in range(6000):
        s = pd.concat([gd[ds[i]] for i in RNG.integers(0, len(ds), len(ds))])
        pf = np.where(s.won == 1, pw, pl)
        bs.append((s.maker * pf).mean() * 100)
    bs = np.sort(np.array(bs))
    pt = out[name][1]
    print("  %-32s %+6.2fc  95%% CI [%+6.2f, %+6.2f]  P(<=0) %.4f"
          % (name, pt, bs[150], bs[5849], (bs <= 0).mean()))

print("\n" + "=" * 88)
print("THE SAME CORRECTION APPLIED TO EVERY MINUTE")
print("=" * 88)
pw, pl = MODELS[":00-specific     0.8167/1.0000"]
print("  (using each minute's OWN measured fill pair from AUDIT A)")
AUDIT_A = {0: (0.8167, 1.0000), 15: (0.8793, 1.0000),
           30: (0.9200, 0.9615), 45: (0.9388, 1.0000)}
print("  %-8s %7s %10s %13s %14s %12s"
      % ("minute", "n", "fillRate", "c/FILLED", "c/SUBMITTED", "$/day q15"))
for mn, g in B.groupby("minute"):
    pw, pl = AUDIT_A[mn]
    pf = np.where(g.won == 1, pw, pl)
    print("  :%02d      %7d %10.4f %+12.2fc %+13.2fc %+11.2f"
          % (mn, len(g), pf.mean(), (g.maker * pf).sum() / pf.sum() * 100,
             (g.maker * pf).mean() * 100,
             len(g) / days * Q * (g.maker * pf).mean()))

print("\n" + "=" * 88)
print("NET EFFECT OF DROPPING :30 (needs no :00 claim to be true)")
print("=" * 88)
tot = 0.0
for mn, g in B.groupby("minute"):
    pw, pl = AUDIT_A[mn]
    pf = np.where(g.won == 1, pw, pl)
    tot += len(g) / days * Q * (g.maker * pf).mean()
d30 = 0.0
for mn, g in B.groupby("minute"):
    if mn == 30:
        continue
    pw, pl = AUDIT_A[mn]
    pf = np.where(g.won == 1, pw, pl)
    d30 += len(g) / days * Q * (g.maker * pf).mean()
print("  trade all four minutes : %+7.2f $/day" % tot)
print("  drop :30 only          : %+7.2f $/day" % d30)
print("  improvement            : %+7.2f $/day" % (d30 - tot))
