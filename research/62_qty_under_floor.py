"""Is qty 15 still right now that the stop is a $211 FLOOR, not 20% of peak?

WHY THE OLD ANSWER NO LONGER APPLIES
  The qty 15 recommendation came from a simulation with the runner's stop at
  20% of a persistent peak. Under that rule a single bad window at qty 30 was
  17.9% of bank - nearly the whole stop budget - so P(stopped) was 16.0% at
  qty 30 against 1.1% at qty 15, and the 5th percentile outcome collapsed to
  $353 at qty 30.

  The stop is now an ABSOLUTE FLOOR of $211. At today's equity that is $181 of
  room rather than $78. The whole reason qty 30 was dangerous - that a couple
  of bad windows exhausted the budget - is much weaker.

  So the earlier conclusion was correct for the rule that existed then, and has
  to be re-derived for the rule that exists now. Leaving qty 15 in place
  because it was right yesterday would be exactly the mistake that the stale
  dead-entry cuts and the stale sizing model already made today.

CONSTRAINT
  The user's standing instruction caps size at 30. Nothing above that is
  considered.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261025)
D = pd.read_parquet(OUT / "grid.parquet")
EQUITY = 392.49
FLOOR = 211.0
NPATH, NDAY = 20000, 30
days = sorted(D.day.unique())

q = D[(D.px >= 0.65) & (D.px < 0.80) & (D.minute >= 8) & (D.minute <= 14)
      & (D.vol >= 2000)]
E = (q.sort_values("minute", ascending=False)
       .groupby(["wkey", "coin"], as_index=False).first())
E = (E.sort_values(["wkey", "minute"], ascending=[True, False])
       .groupby("wkey", as_index=False).head(3)).copy()
E["ec"] = np.where(E.won == 1, (1 - E.px), -E.px)
WE = E.groupby("wkey").ec.sum().to_numpy()
K = max(int(round(len(WE) / len(days))), 1)
print(f"window pool {len(WE)}   {K} windows/day   "
      f"mean {WE.mean()*100:+.2f}c/contract")
print(f"equity ${EQUITY:.2f}   FLOOR ${FLOOR:.0f}   room ${EQUITY-FLOOR:.2f}\n")


def sim(qty, shift=0.0, floor=FLOOR):
    pool = (WE + shift / 100.0) * qty
    idx = RNG.integers(0, len(pool), size=(NPATH, NDAY * K))
    draws = pool[idx]
    eq = np.full(NPATH, EQUITY)
    dead = np.zeros(NPATH, dtype=bool)
    for t in range(draws.shape[1]):
        live = ~dead
        eq[live] += draws[live, t]
        dead |= (eq < floor)
    return eq, dead


print("=" * 78)
print("30-DAY OUTCOME UNDER THE $211 FLOOR (measured edge)")
print("=" * 78)
print(f"{'qty':>5} {'mean':>10} {'median':>10} {'5th pct':>10} {'95th':>10} "
      f"{'P(floored)':>11} {'worst window':>13}")
for qty in (10, 15, 20, 25, 30):
    eq, dead = sim(qty)
    print(f"{qty:>5} {eq.mean():>10,.0f} {np.median(eq):>10,.0f} "
          f"{np.percentile(eq, 5):>10,.0f} {np.percentile(eq, 95):>10,.0f} "
          f"{dead.mean():>11.4f} {WE.min()*qty:>+13.2f}")

print("\n" + "=" * 78)
print("UNDER A DEGRADED EDGE - the regime we are actually in")
print("=" * 78)
print("  post-shift population edge is ~+2.8 to +6.1c against pre-shift +11.3c")
print("  so the pool is shifted down to match\n")
for shift, lbl in ((-5.0, "edge -5c (post-shift-ish)"),
                   (-9.0, "edge -9c (near zero)")):
    print(f"  {lbl}")
    print(f"{'qty':>5} {'mean':>10} {'median':>10} {'5th pct':>10} "
          f"{'P(floored)':>11}")
    for qty in (10, 15, 20, 25, 30):
        eq, dead = sim(qty, shift=shift)
        print(f"{qty:>5} {eq.mean():>10,.0f} {np.median(eq):>10,.0f} "
              f"{np.percentile(eq, 5):>10,.0f} {dead.mean():>11.4f}")
    print()

print("=" * 78)
print("HOW MANY CONSECUTIVE WIPEOUTS DOES THE FLOOR ABSORB?")
print("=" * 78)
worst = WE.min()
print(f"  worst observed window: {worst*100:+.2f}c/contract")
print(f"{'qty':>5} {'worst window $':>16} {'% of equity':>13} "
      f"{'wipeouts to floor':>19}")
for qty in (10, 15, 20, 25, 30):
    w = worst * qty
    print(f"{qty:>5} {w:>+16.2f} {abs(w)/EQUITY*100:>12.1f}% "
          f"{int((EQUITY-FLOOR)/abs(w)):>19}")
print("\n  Under the old 20%-of-peak stop, qty 30 had roughly ONE bad window of")
print("  budget. The floor changes that materially.")

print("\n" + "=" * 78)
print("THE HONEST CAVEAT")
print("=" * 78)
print("  This simulation resamples PRE-SHIFT windows. The post-shift edge is")
print("  lower and its window-level variance is not yet measurable on 3 days,")
print("  so the degraded-edge rows above are the relevant ones, and they are")
print("  a shift applied to pre-shift variance - not measured post-shift risk.")
