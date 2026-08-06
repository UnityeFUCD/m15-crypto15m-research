"""INTERPLAY 3 - adversarial validation of the one surviving combination.

THE CANDIDATE, and why it is not a fourteenth subset search:

    commit only when  (probe unfilled at 120s)  AND  (close minute == 00)

Two components, each with independent prior support, and uncorrelated with
each other (r = -0.003):

  no-fill    +20.96pp on the win rate, n=8,201, replicated against 303 real
             exchange fills to within 0.6pp. Pooled, it is exactly priced:
             worth +15.35c at an unobtainable bid and +0.82c at the ask.
  minute :00 has now surfaced three times from different directions - a
             day-matched absolute edge (selection-corrected P=0.0020), a
             chronological split positive in train, valid and test, and now
             an ask that under-reacts to the no-fill signal.

The claim under test is narrow and mechanical: the no-fill signal is priced
ON AVERAGE but NOT at minute :00.

THIS SCRIPT TRIES TO KILL IT. Every check is one that has previously killed
something in this project:

  1  is it concentrated in a few markets, coins or weeks?
  2  does it hold with the OTHER minutes as an explicit control?
  3  does it survive day-clustering on held-out data alone?
  4  does it survive a best-of-N correction that includes the minute search?
  5  does it survive if the fill proxy is wrong in the direction that flatters?
  6  is the effect economically real after fees at a tradeable size?
  7  does it hold on the 303 REAL orders, where fills are not modelled?

A candidate that clears all seven is not proven, but it is the first thing
here to earn a prospective trial on its own evidence.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"
OUT = ROOT / "research" / "results"
RNG = np.random.default_rng(20260807)
TRAIN_END = "2026-06-30"


def fee_c(p):
    return math.ceil(0.07 * p * (1 - p) * 100) / 100.0


def dayboot(g, col="mis", n=8000):
    gd = {k: v for k, v in g.groupby("day")}
    ds = sorted(gd)
    if len(ds) < 4:
        return np.nan, np.nan, np.nan
    b = np.sort(np.array([
        pd.concat([gd[ds[i]] for i in RNG.integers(0, len(ds), len(ds))]
                  )[col].mean() * 100 for _ in range(n)]))
    return b[int(n * .025)], b[int(n * .975)], float((b <= 0).mean())


G = pd.read_parquet(DATA / "interplay_master.parquet")
G["utc"] = pd.to_datetime(G.close_ts, unit="s", utc=True)
G["day"] = G.utc.dt.date
G["minute"] = G.utc.dt.minute
G["week"] = G.utc.dt.isocalendar().week
U = G[(~G.filled) & G.ask_after.between(0.60, 0.80)].copy()
U["mis"] = U.won - U.ask_after - U.ask_after.map(fee_c)
Z = U[U.minute == 0]
OTH = U[U.minute != 0]
HO = U[U.utc >= TRAIN_END]
HOZ = HO[HO.minute == 0]

print("=" * 86)
print("ADVERSARIAL VALIDATION: commit only on (no-fill AND minute :00)")
print("=" * 86)
print(f"  full sample  n={len(Z):,} over {Z.day.nunique()} days   "
      f"taker edge {Z.mis.mean()*100:+.2f}c")
lo, hi, p = dayboot(Z)
print(f"  day-clustered 95% CI [{lo:+.2f}, {hi:+.2f}]   P(<=0) {p:.4f}")

print("\n" + "-" * 86)
print("CHECK 1 - concentration")
print("-" * 86)
s = Z.mis.sort_values()
print(f"  worst single market {s.iloc[0]*100:+.1f}c   "
      f"best {s.iloc[-1]*100:+.1f}c")
print(f"  drop the best 5 markets  -> {s.iloc[:-5].mean()*100:+.2f}c")
print(f"  drop the worst 5 markets -> {s.iloc[5:].mean()*100:+.2f}c")
print("  leave-one-coin-out:")
vals = []
for c in sorted(Z.coin.unique()):
    v = Z[Z.coin != c].mis.mean() * 100
    vals.append(v)
    print(f"    drop {c:5s} n={len(Z[Z.coin!=c]):4d}  {v:+.2f}c")
print(f"  all positive: {'YES' if min(vals) > 0 else 'NO'}")
wv = []
for w in sorted(Z.week.unique()):
    sub = Z[Z.week != w]
    if len(sub) > 50:
        wv.append(sub.mis.mean() * 100)
print(f"  leave-one-week-out: min {min(wv):+.2f}c  max {max(wv):+.2f}c  "
      f"all positive: {'YES' if min(wv) > 0 else 'NO'}")

print("\n" + "-" * 86)
print("CHECK 2 - the other minutes as an explicit control")
print("-" * 86)
print("  %-10s %7s %9s %12s %22s %9s" % ("minute", "n", "win", "taker edge",
                                         "95% CI", "P(<=0)"))
for mn, g in U.groupby("minute"):
    lo2, hi2, p2 = dayboot(g)
    print("  :%02d       %7d %9.4f %+11.2fc   [%+6.2f, %+6.2f]   %8.4f"
          % (mn, len(g), g.won.mean(), g.mis.mean() * 100, lo2, hi2, p2))
d = Z.mis.mean() - OTH.mis.mean()
print(f"\n  :00 minus the rest {d*100:+.2f}c")
paired = []
for day, g in U.groupby("day"):
    a, b = g[g.minute == 0], g[g.minute != 0]
    if len(a) and len(b):
        paired.append(a.mis.mean() - b.mis.mean())
paired = np.array(paired)
bs = np.sort(np.array([RNG.choice(paired, len(paired), True).mean() * 100
                       for _ in range(8000)]))
print(f"  day-matched within-day: {paired.mean()*100:+.2f}c   "
      f"95% CI [{bs[200]:+.2f}, {bs[7799]:+.2f}]   P(<=0) {(bs<=0).mean():.4f}")

print("\n" + "-" * 86)
print("CHECK 3 - held-out only (never used to select anything)")
print("-" * 86)
lo3, hi3, p3 = dayboot(HOZ)
print(f"  n={len(HOZ)} over {HOZ.day.nunique()} days   "
      f"{HOZ.mis.mean()*100:+.2f}c   95% CI [{lo3:+.2f}, {hi3:+.2f}]   "
      f"P(<=0) {p3:.4f}")

print("\n" + "-" * 86)
print("CHECK 4 - best-of-N including the minute search itself")
print("-" * 86)
obs = Z.mis.mean() * 100
perm = []
for _ in range(4000):
    q = U.copy()
    q["m2"] = q.groupby("day").mis.transform(
        lambda s: RNG.permutation(s.values))
    vals2 = [q[q.minute == m].m2.mean() * 100 for m in (0, 15, 30, 45)]
    perm.append(max(vals2))
perm = np.array(perm)
pp = float((perm >= obs).mean())
print(f"  observed :00 {obs:+.2f}c   permuted best-of-4 mean {perm.mean():+.2f}c"
      f"   p95 {np.quantile(perm,.95):+.2f}c")
print(f"  P(permuted best >= observed) = {pp:.4f}   -> "
      f"{'SURVIVES' if pp < 0.05 else 'does NOT survive'}")

print("\n" + "-" * 86)
print("CHECK 5 - hostile fill-proxy stress")
print("-" * 86)
print("  The proxy calls no-fill when the bid never reached our price. If it")
print("  wrongly keeps markets that WOULD have filled, it flatters the cohort.")
for drop in (0.0, 0.10, 0.25):
    if drop == 0:
        z2 = Z
    else:
        # adversarial: delete the BEST `drop` fraction, i.e. assume the proxy's
        # mistakes are exactly the winners
        k = int(len(Z) * drop)
        z2 = Z.sort_values("mis").iloc[:-k] if k else Z
    print(f"    assume proxy wrong on the best {drop:.0%}: n={len(z2):4d}  "
          f"{z2.mis.mean()*100:+.2f}c")

print("\n" + "-" * 86)
print("CHECK 6 - economics at a tradeable size")
print("-" * 86)
per_day = len(Z) / Z.day.nunique()
print(f"  {per_day:.1f} qualifying commits/day over {Z.day.nunique()} days")
for q in (5, 10, 15):
    print(f"    q{q:<3d} -> {per_day*q*Z.mis.mean():+.2f} $/day   "
          f"(held-out rate: {len(HOZ)/HOZ.day.nunique()*q*HOZ.mis.mean():+.2f})")

print("\n" + "-" * 86)
print("CHECK 7 - the 303 REAL orders, no fill proxy anywhere")
print("-" * 86)
try:
    from research.ptc_v3 import build, replay
    d = build()
    x = replay(d, 120)
    nf = x[~x.probe_filled].copy()
    nf["minute"] = pd.to_datetime(nf.close_ts, unit="s", utc=True).dt.minute
    nf = nf[nf.obs_ask.between(0.60, 0.80)]
    nf["mis"] = nf.won - nf.obs_ask - nf.obs_ask.map(fee_c)
    print(f"  real unfilled + commitable: n={len(nf)}")
    for mn, g in nf.groupby("minute"):
        print(f"    minute :{mn:02d}  n={len(g):2d}  win {g.won.mean():.4f}  "
              f"taker edge {g.mis.mean()*100:+.2f}c")
    z = nf[nf.minute == 0]
    print(f"\n  :00 on real fills: n={len(z)}  {z.mis.mean()*100:+.2f}c")
    print("  (tiny - 2 days only - reported for direction, not significance)")
except Exception as e:
    print(f"  unavailable: {e}")
