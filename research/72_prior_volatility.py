"""Does PRIOR volatility predict edge? The uncontaminated version.

WHY THE FIRST VOLATILITY RESULT CANNOT BE TRADED
  Within-window price dispersion correlates with edge at -0.4301 and the top
  quintile wins only 49.5% against 95.8% for the bottom. That looks decisive
  and it is largely an identity.

  A favourite bought at 72c that WINS travels 72 -> 100, a 28c move. One that
  LOSES travels 72 -> 0, a 72c move. So losers generate more dispersion by
  construction. "High volatility" and "we lost" are close to the same
  statement, which is why volatility measured in the first 3 minutes predicts
  nothing (corr -0.0149, CI [-0.0851,+0.0695]).

  This is the same trap as the unfilled-order finding: a real relationship that
  is an accounting consequence of the outcome rather than a cause of it.

THE UNCONTAMINATED TEST
  Volatility CLUSTERS - that is one of the most durable facts in markets. If
  the previous few windows of the same coin were violent, this one probably
  will be too. Prior-window volatility is:

    - strictly prior information, so it cannot encode this window's outcome
    - a much less noisy estimate than 3 observations inside one window, which
      is what made the 'early volatility' test underpowered rather than
      informative
    - available to the runner at decision time at zero cost

  If volatility genuinely drives edge, and volatility is persistent, then
  prior-window volatility MUST predict this window's edge. If it does not,
  one of those two things is false, and the volatility story - however good it
  looks - is not tradeable.

ALSO TESTED: is the mechanism actually about DISTANCE, not volatility?
  Settlement is A1 >= A0. What decides the favourite's survival is not
  volatility alone but the ratio of its lead to the movement still to come.
  The market's own price already encodes that ratio - a 72c favourite IS the
  market saying z ~ 0.58. So if volatility mattered beyond what price already
  says, edge would vary with volatility WITHIN a price bucket. That is the
  sharper test and it is included.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261051)

Y = pd.read_parquet(OUT / "yesno.parquet")
Y["yes_px"] = np.where(Y.maker_yes, Y.px, 1.0 - Y.px)
G = pd.read_parquet(OUT / "grid.parquet")[["wkey", "coin", "hour"]].drop_duplicates()
Y = Y.merge(G, on=["wkey", "coin"], how="left").dropna(subset=["hour"])
Y["hour"] = Y.hour.astype(int)

V = (Y.groupby(["wkey", "coin"])
       .agg(sd=("yes_px", "std"), k=("yes_px", "size"),
            hour=("hour", "first"), day=("day", "first")).reset_index())
V = V[V.k >= 4].dropna(subset=["sd"]).copy()
# wkey is YYMMMDDHHMM on the ET ticker clock, so lexical sort is chronological
V = V.sort_values(["coin", "wkey"])
for lag in (1, 2, 3):
    V[f"sd_lag{lag}"] = V.groupby("coin").sd.shift(lag)
V["sd_prior3"] = V[["sd_lag1", "sd_lag2", "sd_lag3"]].mean(axis=1)
print(f"window-coins with a volatility measure: {len(V):,}")

print("=" * 78)
print("STEP 0 - IS VOLATILITY EVEN PERSISTENT HERE?")
print("=" * 78)
P = V.dropna(subset=["sd_lag1"])
r1 = P.sd.corr(P.sd_lag1)
P3 = V.dropna(subset=["sd_prior3"])
r3 = P3.sd.corr(P3.sd_prior3)
print(f"  corr(this window sd, previous window sd)   = {r1:+.4f}  (n {len(P):,})")
print(f"  corr(this window sd, mean of prior 3)      = {r3:+.4f}  (n {len(P3):,})")
print("\n  If these are near zero, volatility does NOT cluster at the 15-minute")
print("  horizon and no prior-volatility signal can exist. That alone would")
print("  settle it.")

q = Y[(Y.px >= 0.65) & (Y.px < 0.80) & (Y.minute >= 8) & (Y.minute <= 14)
      & (Y.vol >= 2000)]
E = (q.sort_values("minute", ascending=False)
       .groupby(["wkey", "coin"], as_index=False).first())
E = E.merge(V[["wkey", "coin", "sd", "sd_lag1", "sd_prior3"]],
            on=["wkey", "coin"], how="inner")
E = (E.sort_values(["wkey", "minute"], ascending=[True, False])
       .groupby("wkey", as_index=False).head(3)).copy()
E["pxb"] = pd.cut(E.px, [0.65, 0.70, 0.75, 0.80])
EP = E.dropna(subset=["sd_prior3"]).copy()
days = sorted(EP.day.unique())
print(f"\n  entries with prior-3 volatility: {len(EP):,} over {len(days)} days")


def dayboot(df, fn, n=6000):
    gd = {d: g for d, g in df.groupby("day")}
    ds = sorted(gd)
    out = []
    for _ in range(n):
        s = pd.concat([gd[ds[i]] for i in RNG.integers(0, len(ds), len(ds))])
        v = fn(s)
        if v is not None and not np.isnan(v):
            out.append(v)
    return np.sort(np.array(out))


print("\n" + "=" * 78)
print("STEP 1 - DOES PRIOR VOLATILITY PREDICT THIS WINDOW'S EDGE?")
print("=" * 78)
EP["pb"] = pd.qcut(EP.sd_prior3, 5,
                   labels=["lowest", "low", "mid", "high", "highest"])
print(f"{'prior vol':>12} {'entries':>9} {'median prior':>14} {'avg px':>8} "
      f"{'win':>8} {'edge/ct':>9} {'$/day q20':>11}")
nd = len(days)
for k, g in EP.groupby("pb", observed=True):
    print(f"{str(k):>12} {len(g):>9} {g.sd_prior3.median()*100:>13.2f}c "
          f"{g.px.mean()*100:>7.1f}c {g.won.mean():>8.4f} "
          f"{g.edge.mean()*100:>+8.2f}c {g.edge.sum()*20/nd:>+10.2f}")
r = EP.sd_prior3.corr(EP.edge)
bs = dayboot(EP, lambda s: s.sd_prior3.corr(s.edge))
print(f"\n  corr(prior vol, edge) = {r:+.4f}   "
      f"95% CI [{bs[150]:+.4f}, {bs[5849]:+.4f}]   P(>=0) {(bs >= 0).mean():.4f}")
print(f"  for comparison, contemporaneous vol was -0.4301 (contaminated)")

print("\n" + "=" * 78)
print("STEP 2 - THE SHARPER TEST: does vol matter WITHIN a price bucket?")
print("=" * 78)
print("  A 72c favourite IS the market stating a z-score. If volatility carried")
print("  information the price does not already contain, edge would vary with")
print("  volatility at a FIXED price.\n")
print(f"{'price':>14} {'low vol edge':>14} {'high vol edge':>15} {'gap':>9} "
      f"{'n lo':>6} {'n hi':>6}")
gaps = []
for pb, g in E.groupby("pxb", observed=True):
    if len(g) < 40:
        continue
    med = g.sd.median()
    lo, hi = g[g.sd <= med], g[g.sd > med]
    if len(lo) < 12 or len(hi) < 12:
        continue
    gp = (hi.edge.mean() - lo.edge.mean()) * 100
    gaps.append(gp)
    print(f"{str(pb):>14} {lo.edge.mean()*100:>+13.2f}c "
          f"{hi.edge.mean()*100:>+14.2f}c {gp:>+8.2f}c {len(lo):>6} {len(hi):>6}")
print("\n  (contemporaneous vol, so still contaminated - shown to size the")
print("  identity, not as a signal)")
print(f"\n{'price':>14} {'low PRIOR vol':>15} {'high PRIOR vol':>16} {'gap':>9}")
for pb, g in EP.groupby("pxb", observed=True):
    if len(g) < 40:
        continue
    med = g.sd_prior3.median()
    lo, hi = g[g.sd_prior3 <= med], g[g.sd_prior3 > med]
    if len(lo) < 12 or len(hi) < 12:
        continue
    print(f"{str(pb):>14} {lo.edge.mean()*100:>+14.2f}c "
          f"{hi.edge.mean()*100:>+15.2f}c "
          f"{(hi.edge.mean()-lo.edge.mean())*100:>+8.2f}c")

print("\n" + "=" * 78)
print("STEP 3 - WOULD A PRIOR-VOL FILTER MAKE MONEY?")
print("=" * 78)
base = EP.groupby("day").edge.sum().reindex(days, fill_value=0) * 20
print(f"{'rule':>26} {'entries':>9} {'edge/ct':>9} {'$/day':>9} "
      f"{'vs keep-all':>20}")
print(f"{'keep all':>26} {len(EP):>9} {EP.edge.mean()*100:>+8.2f}c "
      f"{base.mean():>+9.2f} {'-':>20}")
for qt in (0.90, 0.80, 0.67):
    thr = EP.sd_prior3.quantile(qt)
    k = EP[EP.sd_prior3 <= thr]
    d = k.groupby("day").edge.sum().reindex(days, fill_value=0) * 20
    diff = (d - base).values
    b2 = np.sort(np.array([RNG.choice(diff, len(diff), True).mean()
                           for _ in range(8000)]))
    cut = EP[EP.sd_prior3 > thr]
    print(f"{'skip prior-vol top ' + f'{(1-qt)*100:.0f}%':>26} {len(k):>9} "
          f"{k.edge.mean()*100:>+8.2f}c {d.mean():>+9.2f} "
          f"{diff.mean():>+7.2f} [{b2[200]:+.0f},{b2[7799]:+.0f}]")
    print(f"{'':>26} discarded cohort: n {len(cut)}  "
          f"edge {cut.edge.mean()*100:+.2f}c")

print("\n" + "=" * 78)
print("VERDICT")
print("=" * 78)
if (bs >= 0).mean() < 0.05 and r < 0:
    print("  Prior volatility DOES predict edge. The mechanism is real and")
    print("  tradeable, and the discarded cohort has to be checked next.")
else:
    print("  Prior volatility does NOT predict edge, despite contemporaneous")
    print("  volatility correlating at -0.43.")
    print()
    print("  That combination has only one consistent explanation: the")
    print("  contemporaneous relationship is the payoff identity (losers travel")
    print("  further than winners), not a causal channel. Volatility does not")
    print("  forecast which favourites get overturned - it records that they")
    print("  were.")
