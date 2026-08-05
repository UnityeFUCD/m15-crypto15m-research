"""Take the LONGSHOT when the spread is wide? Verified properly.

THE CANDIDATE
  Conditioning on the spread at entry - which is visible before we order -
  splits the book cleanly:

      spread <=1c    favourite +2.32c   longshot -1.32c
      spread 1-2c    favourite -0.31c   longshot +2.31c
      spread 2-3c    favourite -2.72c   longshot +5.72c
      spread >3c     favourite +3.70c   longshot +1.25c

  As a maker we buy at the bid on either side, and the two edges sum to the
  spread, so reversing is not merely avoiding a bad trade - it converts it.
  Skipping a -2.72c entry earns zero; reversing it earns +5.72c.

WHY IT MIGHT BE REAL
  The strategy's premise is that longshots are overpriced, which is why buying
  favourites pays. That premise need not hold everywhere. In wide-spread,
  thinner markets the flow may be the other way - attention concentrated on the
  obvious side pushes the FAVOURITE to a premium instead. Different market,
  different crowd, opposite mispricing.

WHY IT MIGHT BE NOISE
  n=232 in the 2-3c cell, and eight conditions were examined, so the most
  extreme is expected to look good. The four spread buckets are also not
  independent of price - wider spreads cluster in thinner coins and quieter
  hours, both of which have their own effects.

WHAT IT HAS TO SURVIVE
  1. day-clustered intervals on each cell, not point estimates
  2. multiple-comparison correction across the buckets examined
  3. split-half - decide the rule on the first half of the days, score it on
     the second, which is what deploying would actually do
  4. the coin and hour confounds that killed the rank result
  5. dollars, against keep-all-favourites, with the fill bias applied

  Reversing a side is a STRATEGY CHANGE. Nothing here gets deployed on
  research alone regardless of how it comes out.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

LIVE = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261063)
QTY = 20
P_FILL_WIN, P_FILL_LOSE = 0.843, 0.962

D = pd.read_parquet(LIVE / "ladder_paths.parquet").drop_duplicates(
    subset=["ticker"])
rows = []
for r in D.itertuples():
    p = sorted(json.loads(r.path), key=lambda x: -x["ml"])
    e = next((k for k in p if 8 <= k["ml"] <= 14), None)
    if e is None:
        continue
    yb, ya = e["bc"], e["ac"]
    if yb <= 0 or ya >= 1 or yb >= ya:
        continue
    fav_yes = yb >= 0.5
    fav_bid = yb if fav_yes else 1.0 - ya
    lng_bid = (1.0 - ya) if fav_yes else yb
    fw = 1 if (("yes" if fav_yes else "no") == r.result) else 0
    rows.append(dict(ticker=r.ticker, coin=r.coin, date=r.date,
                     hour=int(r.ticker.split("-")[1][7:9]),
                     fav_bid=fav_bid, lng_bid=lng_bid,
                     spread_c=round((ya - yb) * 100, 2), fav_won=fw,
                     fav_edge=fw - fav_bid, lng_edge=(1 - fw) - lng_bid))
E = pd.DataFrame(rows)
E = E[(E.fav_bid >= 0.65) & (E.fav_bid < 0.80)].copy()
days = sorted(E.date.unique())
print(f"entries {len(E):,}   days {len(days)}\n")


def dayboot(df, fn, n=8000):
    gd = {d: g for d, g in df.groupby("date")}
    ds = sorted(gd)
    v = []
    for _ in range(n):
        s = pd.concat([gd[ds[i]] for i in RNG.integers(0, len(ds), len(ds))])
        x = fn(s)
        if x is not None and not (isinstance(x, float) and np.isnan(x)):
            v.append(x)
    return np.sort(np.array(v))


E["sb"] = pd.cut(E.spread_c, [0, 1.01, 2.01, 3.01, 100],
                 labels=["<=1c", "1-2c", "2-3c", ">3c"])
print("=" * 78)
print("STEP 1 - EACH CELL WITH A DAY-CLUSTERED INTERVAL")
print("=" * 78)
print(f"{'spread':>8} {'n':>6} {'fav edge':>20} {'longshot edge':>22} "
      f"{'P(long>fav)':>12}")
for k, g in E.groupby("sb", observed=True):
    if len(g) < 60:
        continue
    bf = dayboot(g, lambda s: s.fav_edge.mean() * 100)
    bl = dayboot(g, lambda s: s.lng_edge.mean() * 100)
    bd = dayboot(g, lambda s: (s.lng_edge.mean() - s.fav_edge.mean()) * 100)
    print(f"{str(k):>8} {len(g):>6} "
          f"{g.fav_edge.mean()*100:>+7.2f}c [{bf[200]:>+6.2f},{bf[7799]:>+6.2f}] "
          f"{g.lng_edge.mean()*100:>+7.2f}c [{bl[200]:>+6.2f},{bl[7799]:>+6.2f}] "
          f"{(bd > 0).mean():>12.4f}")

print("\n" + "=" * 78)
print("STEP 2 - MULTIPLE COMPARISONS: shuffle the spread labels")
print("=" * 78)
obs = 0.0
for k, g in E.groupby("sb", observed=True):
    if len(g) >= 60:
        obs = max(obs, (g.lng_edge.mean() - g.fav_edge.mean()) * 100)
perm = []
for _ in range(4000):
    sh = E.copy()
    sh["sb"] = RNG.permutation(sh.sb.values)
    m = 0.0
    for k, g in sh.groupby("sb", observed=True):
        if len(g) >= 60:
            m = max(m, (g.lng_edge.mean() - g.fav_edge.mean()) * 100)
    perm.append(m)
perm = np.array(perm)
print(f"  best observed longshot-minus-favourite gap: {obs:+.2f}c")
print(f"  shuffled labels beat it {(perm >= obs).mean():.4f} of the time")
print(f"  (this is the corrected p-value across all buckets examined)")

print("\n" + "=" * 78)
print("STEP 3 - SPLIT-HALF: decide on the first half, score on the second")
print("=" * 78)
h1, h2 = days[:len(days) // 2], days[len(days) // 2:]
A, B = E[E.date.isin(h1)], E[E.date.isin(h2)]
rule = {}
for k, g in A.groupby("sb", observed=True):
    if len(g) >= 40:
        rule[k] = "long" if g.lng_edge.mean() > g.fav_edge.mean() else "fav"
print(f"  rule learned on days 1-{len(h1)}: {rule}")
if rule:
    got = B.apply(lambda r: r.lng_edge if rule.get(r.sb, "fav") == "long"
                  else r.fav_edge, axis=1)
    base = B.fav_edge
    print(f"\n  scored on days {len(h1)+1}-{len(days)}:")
    print(f"    always favourite : {base.mean()*100:+.2f}c")
    print(f"    with reversal    : {got.mean()*100:+.2f}c")
    print(f"    gain             : {(got.mean()-base.mean())*100:+.2f}c")
    T = B.assign(d=(got - base))
    bs = dayboot(T, lambda s: s.d.mean() * 100)
    print(f"    95% CI [{bs[200]:+.2f}, {bs[7799]:+.2f}]   "
          f"P(<=0) {(bs <= 0).mean():.4f}")
    print("\n  Reversed (learn on the second half, score the first):")
    rule2 = {}
    for k, g in B.groupby("sb", observed=True):
        if len(g) >= 40:
            rule2[k] = "long" if g.lng_edge.mean() > g.fav_edge.mean() else "fav"
    got2 = A.apply(lambda r: r.lng_edge if rule2.get(r.sb, "fav") == "long"
                   else r.fav_edge, axis=1)
    print(f"    rule {rule2}")
    print(f"    always favourite {A.fav_edge.mean()*100:+.2f}c   "
          f"with reversal {got2.mean()*100:+.2f}c   "
          f"gain {(got2.mean()-A.fav_edge.mean())*100:+.2f}c")

print("\n" + "=" * 78)
print("STEP 4 - IS IT COIN OR HOUR IN DISGUISE?")
print("=" * 78)
E["resid_f"] = E.fav_edge - E.groupby("coin").fav_edge.transform("mean")
E["resid_l"] = E.lng_edge - E.groupby("coin").lng_edge.transform("mean")
print(f"{'spread':>8} {'n':>6} {'raw gap':>10} {'coin-adjusted gap':>20}")
for k, g in E.groupby("sb", observed=True):
    if len(g) < 60:
        continue
    raw = (g.lng_edge.mean() - g.fav_edge.mean()) * 100
    adj = (g.resid_l.mean() - g.resid_f.mean()) * 100
    print(f"{str(k):>8} {len(g):>6} {raw:>+9.2f}c {adj:>+19.2f}c")
print("\n  spread by coin (wide spreads cluster in thin coins):")
print(E.groupby("coin").spread_c.median().round(2).to_string())

print("\n" + "=" * 78)
print("STEP 5 - DOLLARS, WITH FILL BIAS APPLIED TO BOTH SIDES")
print("=" * 78)
def pf(won):
    return np.where(won == 1, P_FILL_WIN, P_FILL_LOSE)
E["v_fav"] = E.fav_edge * QTY * pf(E.fav_won)
E["v_rev"] = np.where(
    E.sb.isin([k for k, v in (rule or {}).items() if v == "long"]),
    E.lng_edge * QTY * pf(1 - E.fav_won), E.v_fav)
a = E.groupby("date").v_fav.sum().reindex(days, fill_value=0)
b = E.groupby("date").v_rev.sum().reindex(days, fill_value=0)
d = (b - a).values
bs = np.sort(np.array([RNG.choice(d, len(d), True).mean() for _ in range(8000)]))
print(f"  always favourite : ${a.mean():+.2f}/day")
print(f"  with reversal    : ${b.mean():+.2f}/day")
print(f"  gain ${d.mean():+.2f}/day   95% CI [{bs[200]:+.2f}, {bs[7799]:+.2f}]   "
      f"better {(b > a).sum()}/{len(days)} days   P(<=0) {(bs <= 0).mean():.4f}")
