"""Does the edge move with the hour of the day?

WHY THIS IS WORTH ASKING AND HAS NOT BEEN ASKED
  The edge is a LONGSHOT PREMIUM - payment for absorbing impatient buying of
  the cheap side. That is a behavioural quantity, not a mechanical one. It
  should depend on who is awake and trading, and crypto has strong session
  structure (Asia, Europe, US). If retail impatience concentrates in some
  sessions, the premium should too.

  What has been tested before is adjacent but different:
    - time CLUSTERING of wipeouts   -> none found
    - entry timing WITHIN a window  -> earlier is better, already maxed
  Neither of those is an hour-of-day profile of the edge itself.

THE TRAP THIS SCRIPT IS BUILT TO AVOID
  24 hours is 24 chances to find something. The best hour of 24 will always
  look impressive, and on 10 days each hour-cell holds ~100 entries across ~10
  day-clusters. A naive 'best hour' result here would be worthless.

  So the pattern has to survive three things:
    1. a PERMUTATION test - shuffle the hour labels within days and ask how
       often the best hour looks this good by chance
    2. SPLIT-HALF replication - fit the profile on the first half of the days
       and check it on the second. A real session effect replicates; noise
       does not
    3. session BLOCKS rather than single hours - fewer tests, more power, and
       a mechanism that would actually make sense

  Only if it clears all three is it worth anything, and even then it would be
  a TIME-WINDOW change, which the standing constraints forbid deploying.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261034)
QTY = 20

SESSIONS = [("Asia      00-07", range(0, 8)),
            ("Europe    08-13", range(8, 14)),
            ("US-open   14-19", range(14, 20)),
            ("US-late   20-23", range(20, 24))]


def load_pre():
    D = pd.read_parquet(OUT / "grid.parquet")
    q = D[(D.px >= 0.65) & (D.px < 0.80) & (D.minute >= 8) & (D.minute <= 14)
          & (D.vol >= 2000)]
    e = (q.sort_values("minute", ascending=False)
           .groupby(["wkey", "coin"], as_index=False).first())
    e = (e.sort_values(["wkey", "minute"], ascending=[True, False])
           .groupby("wkey", as_index=False).head(3))
    return e.copy()


E = load_pre()
days = sorted(E.day.unique())
print(f"entries {len(E):,}   days {len(days)}   hours present "
      f"{E.hour.nunique()}   overall edge {E.edge.mean()*100:+.2f}c\n")

print("=" * 78)
print("EDGE BY UTC HOUR  (pre-shift, the only regime with enough days)")
print("=" * 78)
print(f"{'hr':>3} {'n':>5} {'n/day':>7} {'avg px':>8} {'win':>7} {'edge/ct':>9} "
      f"{'$/day':>9}  {'':<24}")
prof = {}
for h in range(24):
    g = E[E.hour == h]
    if not len(g):
        continue
    ed = g.edge.mean() * 100
    prof[h] = ed
    dpd = g.edge.sum() * QTY / len(days)
    bar = ("+" * int(max(ed, 0) / 2)) or ("-" * int(min(ed, 0) / -2))
    print(f"{h:>3} {len(g):>5} {len(g)/len(days):>7.1f} {g.px.mean()*100:>7.1f}c "
          f"{g.won.mean():>7.4f} {ed:>+8.2f}c {dpd:>+9.2f}  {bar:<24}")

best = max(prof, key=prof.get)
worst = min(prof, key=prof.get)
print(f"\n  best hour  {best:02d}:00  {prof[best]:+.2f}c")
print(f"  worst hour {worst:02d}:00  {prof[worst]:+.2f}c")
print(f"  spread {prof[best]-prof[worst]:.2f}c")

print("\n" + "=" * 78)
print("TEST 1 - PERMUTATION: is the best hour better than chance produces?")
print("=" * 78)
obs_range = prof[best] - prof[worst]
obs_best = prof[best]
hrs = E.hour.to_numpy()
edg = E.edge.to_numpy()
dy = E.day.to_numpy()
nperm = 4000
mx = np.empty(nperm)
rng_ = np.empty(nperm)
for i in range(nperm):
    sh = hrs.copy()
    for d in days:                      # shuffle WITHIN day, preserving
        m = dy == d                     # both day effects and hour counts
        sh[m] = RNG.permutation(sh[m])
    means = pd.Series(edg).groupby(sh).mean() * 100
    mx[i] = means.max()
    rng_[i] = means.max() - means.min()
print(f"  observed best-hour edge {obs_best:+.2f}c   "
      f"permuted best-hour edge exceeds it {(mx >= obs_best).mean():.4f} of the time")
print(f"  observed best-worst range {obs_range:.2f}c   "
      f"permuted range exceeds it {(rng_ >= obs_range).mean():.4f} of the time")
print("\n  These are the multiple-comparison-corrected p-values. If they are")
print("  not small, the hourly variation is what random assignment produces.")

print("\n" + "=" * 78)
print("TEST 2 - SPLIT-HALF: does the profile replicate out of sample?")
print("=" * 78)
h1, h2 = days[:len(days) // 2], days[len(days) // 2:]
A = E[E.day.isin(h1)].groupby("hour").edge.mean() * 100
B = E[E.day.isin(h2)].groupby("hour").edge.mean() * 100
J = pd.concat([A.rename("first"), B.rename("second")], axis=1).dropna()
if len(J) >= 8:
    r = np.corrcoef(J["first"], J["second"])[0, 1]
    bs = np.sort(np.array([
        (lambda s: np.corrcoef(s["first"], s["second"])[0, 1])(
            J.sample(len(J), replace=True))
        for _ in range(4000)]))
    print(f"  hours in both halves: {len(J)}")
    print(f"  correlation between first-half and second-half hourly edge: "
          f"r = {r:+.4f}")
    print(f"  95% CI [{bs[100]:+.4f}, {bs[3899]:+.4f}]   "
          f"P(r<=0) {(bs <= 0).mean():.4f}")
    top = A.nlargest(6).index
    ta, tb = E[E.day.isin(h2) & E.hour.isin(top)], E[E.day.isin(h2) & ~E.hour.isin(top)]
    if len(ta) > 20 and len(tb) > 20:
        print(f"\n  hours chosen as best on the FIRST half, measured on the SECOND:")
        print(f"    chosen hours   n {len(ta):>4}  edge {ta.edge.mean()*100:>+6.2f}c")
        print(f"    the rest       n {len(tb):>4}  edge {tb.edge.mean()*100:>+6.2f}c")
        print(f"    out-of-sample gain {(ta.edge.mean()-tb.edge.mean())*100:+.2f}c")
        print("\n  This is the number that matters. Picking hours in-sample and")
        print("  measuring them out-of-sample is what deploying would actually do.")

print("\n" + "=" * 78)
print("TEST 3 - SESSION BLOCKS: fewer tests, and a mechanism that makes sense")
print("=" * 78)
print(f"{'session':>18} {'n':>6} {'n/day':>7} {'avg px':>8} {'win':>7} "
      f"{'edge/ct':>9} {'95% CI':>20}")
blocks = {}
for nm, rr in SESSIONS:
    g = E[E.hour.isin(list(rr))]
    if len(g) < 30:
        continue
    blocks[nm] = g
    gd = {d: gg for d, gg in g.groupby("day")}
    ds = sorted(gd)
    bs = np.sort(np.array([
        np.concatenate([gd[ds[i]].edge.values
                        for i in RNG.integers(0, len(ds), len(ds))]).mean() * 100
        for _ in range(4000)]))
    print(f"{nm:>18} {len(g):>6} {len(g)/len(days):>7.1f} "
          f"{g.px.mean()*100:>7.1f}c {g.won.mean():>7.4f} "
          f"{g.edge.mean()*100:>+8.2f}c "
          f"[{bs[100]:>+6.2f},{bs[3899]:>+6.2f}]")

if len(blocks) >= 2:
    print(f"\n{'pairwise':>40} {'gap':>9} {'P(gap<=0)':>11}")
    ks = list(blocks)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = blocks[ks[i]], blocks[ks[j]]
            ga = {d: gg for d, gg in a.groupby("day")}
            gb = {d: gg for d, gg in b.groupby("day")}
            ds = sorted(set(ga) & set(gb))
            if len(ds) < 4:
                continue
            bs = np.sort(np.array([
                (lambda ix: np.concatenate([ga[ds[k]].edge.values for k in ix]).mean()
                 - np.concatenate([gb[ds[k]].edge.values for k in ix]).mean())(
                    RNG.integers(0, len(ds), len(ds))) * 100
                for _ in range(4000)]))
            gap = (a.edge.mean() - b.edge.mean()) * 100
            print(f"{ks[i]+' vs '+ks[j]:>40} {gap:>+8.2f}c {(bs <= 0).mean():>11.4f}")

E.to_parquet(OUT / "hourly_profile.parquet")
pd.DataFrame({"hour": list(prof), "edge_c": list(prof.values())}).to_csv(
    OUT / "hourly_edge.csv", index=False)
print("\nwrote hourly_edge.csv for plotting")
