"""Does MAX_PER_WINDOW=3 still hold post-shift?

WHY IT NEEDS RE-CHECKING
  The cap was validated entirely pre-shift: rank-3 entries carried +16.65c
  (CI [+12.23, +21.40], p<0.0001) and beat cap 2 on 9/10 days. The mechanism
  was that a 3rd entry only exists when three coins independently clear the
  volume floor in the same window, which selects high-agreement windows.

  But post-shift the volume floor keeps 58.2% of entries instead of 31.1%. If
  three coins now clear the floor routinely rather than exceptionally, the
  selection that made rank-3 valuable is gone, and the cap is admitting
  ordinary entries rather than exceptional ones.

  That is the same failure mode that broke the dead-entry cuts, and it is
  worth checking before it costs money rather than after.

THE BAR, applied consistently now
  - compare against the correct baseline (each cap against the others, on the
    same windows)
  - the marginal entry a higher cap ADMITS must carry positive edge
  - test in dollars, day-clustered
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261023)
QTY = 15


def prep_pre():
    D = pd.read_parquet(OUT / "grid.parquet")
    q = D[(D.px >= 0.65) & (D.px < 0.80) & (D.minute >= 8) & (D.minute <= 14)
          & (D.vol >= 2000)]
    e = (q.sort_values("minute", ascending=False)
           .groupby(["wkey", "coin"], as_index=False).first())
    e = e.sort_values(["wkey", "minute"], ascending=[True, False]).copy()
    e["rank"] = e.groupby("wkey").cumcount() + 1
    return e


def prep_post():
    D = pd.read_parquet(OUT / "postshift_nofilter.parquet")
    P = D[(D.vol >= 2000) & (D.px >= 0.65) & (D.px < 0.80)].copy()
    P = P.sort_values(["wkey", "px"], ascending=[True, False])
    P["rank"] = P.groupby("wkey").cumcount() + 1
    return P


for nm, E in (("PRE-SHIFT", prep_pre()), ("POST-SHIFT", prep_post())):
    days = sorted(E.day.unique())
    print("=" * 74)
    print(f"{nm}   entries {len(E):,}   windows {E.wkey.nunique():,}   "
          f"days {len(days)}")
    print("=" * 74)
    print(f"{'rank':>6} {'n':>6} {'share of windows':>18} {'avg px':>8} "
          f"{'win':>8} {'edge/ct':>9}")
    nw = E.wkey.nunique()
    for r, g in E.groupby("rank"):
        if r > 5:
            continue
        print(f"{int(r):>6} {len(g):>6} {len(g)/nw:>18.3f} "
              f"{g.px.mean()*100:>7.1f}c {g.won.mean():>8.4f} "
              f"{g.edge.mean()*100:>+8.2f}c")
    print("\n  'share of windows' at rank 3 is the selection the cap relies on -")
    print("  if it approaches 1.0, a 3rd entry is routine, not exceptional.\n")
    print(f"{'cap':>5} {'entries':>8} {'ent/day':>9} {'edge/ct':>9} "
          f"{'$/day':>10} {'marginal edge':>15}")
    prev = None
    daily = {}
    for cap in (1, 2, 3, 4):
        K = E[E["rank"] <= cap]
        if len(K) < 30:
            continue
        d = K.groupby("day").edge.sum().reindex(days, fill_value=0) * QTY
        daily[cap] = d
        marg = E[E["rank"] == cap]
        ms = (f"{marg.edge.mean()*100:+.2f}c (n={len(marg)})"
              if len(marg) > 8 else "-")
        print(f"{cap:>5} {len(K):>8} {len(K)/len(days):>9.1f} "
              f"{K.edge.mean()*100:>+8.2f}c {d.mean():>+10.2f} {ms:>15}")
    if 2 in daily and 3 in daily:
        diff = (daily[3] - daily[2]).values
        bs = np.sort(np.array([RNG.choice(diff, len(diff), True).mean()
                               for _ in range(8000)]))
        print(f"\n  cap3 minus cap2: ${diff.mean():+.2f}/day   "
              f"95% CI [{bs[200]:+.2f}, {bs[7799]:+.2f}]   "
              f"better {(daily[3] > daily[2]).sum()}/{len(days)} days")
        m3 = E[E["rank"] == 3]
        if len(m3) > 8:
            g = {x: gg for x, gg in m3.groupby("day")}
            ds = sorted(g)
            if len(ds) >= 2:
                b2 = np.sort(np.array([
                    np.concatenate([g[ds[i]].edge.values
                                    for i in RNG.integers(0, len(ds), len(ds))]).mean() * 100
                    for _ in range(6000)]))
                print(f"  the rank-3 cohort the cap ADMITS: edge "
                      f"{m3.edge.mean()*100:+.2f}c  "
                      f"95% CI [{b2[150]:+.2f}, {b2[5849]:+.2f}]  "
                      f"P(<=0) {(b2 <= 0).mean():.4f}")
    if 3 in daily and 4 in daily:
        diff = (daily[4] - daily[3]).values
        print(f"  cap4 minus cap3: ${diff.mean():+.2f}/day   "
              f"better {(daily[4] > daily[3]).sum()}/{len(days)} days")
    print()
