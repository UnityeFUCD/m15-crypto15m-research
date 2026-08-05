"""Fine-grained intraday edge: a smooth curve, not more bins.

WHY NOT JUST USE SMALLER BUCKETS
  The obvious way to go finer than hourly is half-hours or 15-minute slots.
  That makes the statistics WORSE, not better: 1,058 entries over 96 slots is
  ~11 entries per slot across 10 day-clusters, and 96 bins is 96 chances for
  noise to produce a winner. The hourly pass already showed the symptom - the
  best-hour permutation p was 0.0053 but the split-half correlation was only
  r=+0.2499 with a CI crossing zero. Detail that does not replicate.

  The right way to get finer resolution from the same data is to SMOOTH. A
  circular kernel over time-of-day borrows strength from neighbouring minutes,
  so the curve can be evaluated at any resolution while each point is still
  backed by hundreds of entries. It also encodes the thing we actually believe:
  that trader behaviour varies continuously across the day, not in 15-minute
  jumps.

WHAT IS RECONSTRUCTED HERE
  grid.parquet stores 'hour' in UTC and 'wkey' on Kalshi's ET ticker clock, so
  exact wall-clock time of each window is hour*60 + the wkey minute field. That
  gives 96 slots per day at full resolution, which the kernel then smooths.

THE MECHANISM THIS IS TESTING
  00-07 UTC is 20:00-03:00 ET - US retail evening into Asia's session. If the
  edge is a longshot premium paid by impatient retail buyers, that is when it
  should be largest. The hourly pass found exactly that shape (Asia +17.63c vs
  Europe +6.84c, P(gap<=0)=0.0000). This asks whether a continuous version of
  that claim holds up and REPLICATES, which the hour-level detail did not.

FOUR THINGS IT MUST SURVIVE
  1. permutation of the time labels within days
  2. split-half replication - fit on the first 5 days, score the last 5
  3. the confound check - is it really time, or is it coin mix / price / volume
     differing by time of day
  4. the post-shift regime, where the population edge is ~78% smaller
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261035)
QTY = 20
KAPPA = 12.0        # von Mises concentration ~ +/- 1.4h effective window


def load():
    D = pd.read_parquet(OUT / "grid.parquet")
    q = D[(D.px >= 0.65) & (D.px < 0.80) & (D.minute >= 8) & (D.minute <= 14)
          & (D.vol >= 2000)]
    e = (q.sort_values("minute", ascending=False)
           .groupby(["wkey", "coin"], as_index=False).first())
    e = (e.sort_values(["wkey", "minute"], ascending=[True, False])
           .groupby("wkey", as_index=False).head(3)).copy()
    e["slot"] = e.wkey.str[-2:].astype(int)
    e["tod"] = e.hour * 60 + e.slot           # minutes past midnight UTC
    return e


E = load()
days = sorted(E.day.unique())
print(f"entries {len(E):,}   days {len(days)}   distinct 15-min slots "
      f"{E.tod.nunique()} of 96")
print(f"overall edge {E.edge.mean()*100:+.2f}c\n")

theta = E.tod.to_numpy() / 1440.0 * 2 * np.pi
edge = E.edge.to_numpy()
dyc = pd.Categorical(E.day, categories=days).codes


def smooth(th, ed, grid, kappa=KAPPA):
    """von Mises kernel regression on the circle."""
    W = np.exp(kappa * np.cos(grid[:, None] - th[None, :]))
    return (W @ ed) / W.sum(axis=1) * 100


GRID = np.linspace(0, 2 * np.pi, 145)[:-1]          # every 10 minutes
curve = smooth(theta, edge, GRID)

print("=" * 78)
print("SMOOTHED INTRADAY EDGE  (von Mises kernel, ~+/-1.4h window)")
print("=" * 78)
print(f"{'UTC':>6} {'ET':>6} {'edge/ct':>9}  {'':<40}")
lo, hi = curve.min(), curve.max()
for i in range(0, 144, 6):                           # print hourly rows
    m = GRID[i] / (2 * np.pi) * 1440
    et = (int(m // 60) - 4) % 24
    n = int((curve[i] - lo) / max(hi - lo, 1e-9) * 40)
    print(f"{int(m//60):>4}:00 {et:>4}:00 {curve[i]:>+8.2f}c  "
          f"{'#'*n:<40}")
pk = GRID[int(np.argmax(curve))] / (2 * np.pi) * 1440
tr = GRID[int(np.argmin(curve))] / (2 * np.pi) * 1440
print(f"\n  peak   {int(pk//60):02d}:{int(pk%60):02d} UTC "
      f"({(int(pk//60)-4)%24:02d}:{int(pk%60):02d} ET)  {curve.max():+.2f}c")
print(f"  trough {int(tr//60):02d}:{int(tr%60):02d} UTC "
      f"({(int(tr//60)-4)%24:02d}:{int(tr%60):02d} ET)  {curve.min():+.2f}c")
print(f"  amplitude {curve.max()-curve.min():.2f}c")

print("\n" + "=" * 78)
print("TEST 1 - PERMUTATION (shuffle time labels within each day)")
print("=" * 78)
obs_amp = curve.max() - curve.min()
NP = 3000
amps = np.empty(NP)
for i in range(NP):
    sh = theta.copy()
    for d in range(len(days)):
        m = dyc == d
        sh[m] = RNG.permutation(sh[m])
    c = smooth(sh, edge, GRID)
    amps[i] = c.max() - c.min()
print(f"  observed amplitude {obs_amp:.2f}c")
print(f"  permuted amplitude exceeds it {(amps >= obs_amp).mean():.4f} of the time")
print(f"  permuted median amplitude {np.median(amps):.2f}c")

print("\n" + "=" * 78)
print("TEST 2 - SPLIT-HALF REPLICATION (the decisive one)")
print("=" * 78)
h1, h2 = days[:5], days[5:]
m1, m2 = E.day.isin(h1).to_numpy(), E.day.isin(h2).to_numpy()
c1 = smooth(theta[m1], edge[m1], GRID)
c2 = smooth(theta[m2], edge[m2], GRID)
r = np.corrcoef(c1, c2)[0, 1]
print(f"  correlation of the two half-sample curves: r = {r:+.4f}")
thr = np.median(c1)
pick = np.interp(theta[m2], GRID, c1) > thr
a, b = edge[m2][pick], edge[m2][~pick]
if len(a) > 20 and len(b) > 20:
    print(f"\n  trade only the hours the FIRST half calls good, scored on the SECOND:")
    print(f"    kept    n {len(a):>4}  edge {a.mean()*100:>+6.2f}c")
    print(f"    dropped n {len(b):>4}  edge {b.mean()*100:>+6.2f}c")
    print(f"    out-of-sample gap {(a.mean()-b.mean())*100:+.2f}c")
    d2 = E[m2].copy()
    d2["keep"] = pick
    gg = {d: x for d, x in d2.groupby("day")}
    ds = sorted(gg)
    bs = np.sort(np.array([
        (lambda ix: (np.concatenate([gg[ds[k]][gg[ds[k]].keep].edge.values for k in ix]).mean()
                     - np.concatenate([gg[ds[k]][~gg[ds[k]].keep].edge.values for k in ix]).mean()))(
            RNG.integers(0, len(ds), len(ds))) * 100
        for _ in range(4000)]))
    print(f"    95% CI [{bs[100]:+.2f}, {bs[3899]:+.2f}]   "
          f"P(gap<=0) {(bs <= 0).mean():.4f}")
    print("\n  Reversed (fit on second half, score the first):")
    thr2 = np.median(c2)
    pk2 = np.interp(theta[m1], GRID, c2) > thr2
    a2, b2 = edge[m1][pk2], edge[m1][~pk2]
    print(f"    kept    n {len(a2):>4}  edge {a2.mean()*100:>+6.2f}c")
    print(f"    dropped n {len(b2):>4}  edge {b2.mean()*100:>+6.2f}c")
    print(f"    out-of-sample gap {(a2.mean()-b2.mean())*100:+.2f}c")

print("\n" + "=" * 78)
print("TEST 3 - CONFOUND: is it time, or what trades at that time?")
print("=" * 78)
E["blk"] = np.where(E.hour < 8, "00-07 UTC", np.where(E.hour < 14, "08-13", "14-23"))
print(f"{'block':>12} {'n':>6} {'avg px':>8} {'vol':>10} {'frac_long':>10} "
      f"{'BTC share':>10} {'edge':>9}")
for k, g in E.groupby("blk"):
    print(f"{k:>12} {len(g):>6} {g.px.mean()*100:>7.1f}c {g.vol.median():>10.0f} "
          f"{g.frac_long.mean():>10.4f} {(g.coin=='BTC').mean():>10.3f} "
          f"{g.edge.mean()*100:>+8.2f}c")
print("\n  price-matched comparison (same 2c price bucket, weighted equally):")
E["pb"] = (E.px * 50).astype(int)
piv = E.pivot_table(index="pb", columns="blk", values="edge", aggfunc="mean") * 100
cnt = E.pivot_table(index="pb", columns="blk", values="edge", aggfunc="size")
ok = piv.dropna()
if len(ok) >= 3:
    w = cnt.loc[ok.index].sum(axis=1)
    adj = (ok.mul(w, axis=0).sum() / w.sum())
    print("   ", " ".join(f"{c}={adj[c]:+.2f}c" for c in adj.index))
    print(f"    price-matched gap 00-07 minus 08-13: "
          f"{adj.get('00-07 UTC', np.nan) - adj.get('08-13', np.nan):+.2f}c")

print("\n" + "=" * 78)
print("TEST 4 - DOES IT SURVIVE POST-SHIFT?")
print("=" * 78)
P = pd.read_parquet(OUT / "postshift_nofilter.parquet")
P = P[(P.vol >= 2000) & (P.px >= 0.65) & (P.px < 0.80)].copy()
if "hour" not in P.columns:
    P["hour"] = P.wkey.str[-4:-2].astype(int)
    P["hour"] = (P.hour + 4) % 24                    # ET ticker -> UTC
P["blk"] = np.where(P.hour < 8, "00-07 UTC",
                    np.where(P.hour < 14, "08-13", "14-23"))
print(f"  post-shift entries {len(P)} over {P.day.nunique()} days")
print(f"{'block':>12} {'n':>6} {'edge':>9}")
for k, g in P.groupby("blk"):
    print(f"{k:>12} {len(g):>6} {g.edge.mean()*100:>+8.2f}c")

np.save(OUT / "intraday_curve.npy", np.vstack([GRID, curve, c1, c2]))
print("\nwrote intraday_curve.npy")
