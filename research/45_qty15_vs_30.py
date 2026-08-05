"""Is quantity 15 more profitable than quantity 30?

WHY THE NAIVE ANSWER IS WRONG IN BOTH DIRECTIONS
  "Half the size makes half the money" is true only for a strategy that never
  stops. This one has hard stops, so size changes the PROBABILITY OF BEING
  STOPPED, and a stop ends the compounding. Halving size does not halve the
  outcome; it changes the shape of the distribution.

  Equally, "smaller is always safer therefore better" ignores that the edge is
  positive - staying small forgoes real money.

  The decisive quantity is terminal wealth over a horizon WITH the stops
  active, not P&L per window.

WHAT MAKES THIS TESTABLE NOW
  We know the per-contract edge is scale-free at our size: a 30-lot is ~1.2%
  of the longshot flow in an entered window, so 15 and 30 both fill from the
  same distribution. Fill rate could differ slightly (a smaller order sits
  less deep), which would FAVOUR 15 - so ignoring it is conservative against
  the size we currently run.

  The live drawdown controls are simulated exactly: runner stop at 20% of a
  persistent peak, watchdog at 22%, and the stop ends trading for the path.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260915)
D = pd.read_parquet(OUT / "grid.parquet")
BANK = 411.23
NPATH, NDAY = 20000, 30
STOP_FRAC = 0.20
days = sorted(D.day.unique())

q = D[(D.px >= 0.65) & (D.px < 0.85) & (D.minute >= 8) & (D.minute <= 14)
      & (D.vol >= 2000)]
E = (q.sort_values("minute", ascending=False)
       .groupby(["wkey", "coin"], as_index=False).first())
E = (E.sort_values(["wkey", "minute"], ascending=[True, False])
       .groupby("wkey", as_index=False).head(3)).copy()
# per-window edge in CONTRACT units, so it scales exactly with qty
E["ec"] = np.where(E.won == 1, (1 - E.px), -E.px)
WE = E.groupby("wkey").ec.sum().to_numpy()      # window edge per contract
nwin_day = E.wkey.nunique() / len(days)
K = max(int(round(nwin_day)), 1)
print(f"window pool {len(WE)}   windows/day {nwin_day:.1f}   "
      f"mean window edge {WE.mean()*100:+.2f}c/contract")
print(f"bank ${BANK:.2f}   stop at {STOP_FRAC:.0%} of peak   "
      f"{NPATH:,} paths x {NDAY} days\n")


def sim(qty, shift=0.0, stop=True):
    pool = (WE + shift / 100.0) * qty          # dollars per window at this qty
    idx = RNG.integers(0, len(pool), size=(NPATH, NDAY * K))
    draws = pool[idx]
    eq = np.full(NPATH, BANK)
    peak = np.full(NPATH, BANK)
    dead = np.zeros(NPATH, dtype=bool)
    for t in range(draws.shape[1]):
        live = ~dead
        eq[live] += draws[live, t]
        peak = np.maximum(peak, eq)
        if stop:
            dead |= (eq < (1 - STOP_FRAC) * peak)
    return eq, dead


print("=== TERMINAL WEALTH AFTER 30 DAYS, WITH THE LIVE STOPS ACTIVE ===")
print(f"{'qty':>5} {'mean':>10} {'median':>10} {'5th pct':>10} {'95th':>10} "
      f"{'P(stopped)':>11} {'P(ruin)':>9}")
res = {}
for qty in (5, 10, 15, 20, 30):
    eq, dead = sim(qty)
    res[qty] = eq
    print(f"{qty:>5} {eq.mean():>10,.0f} {np.median(eq):>10,.0f} "
          f"{np.percentile(eq, 5):>10,.0f} {np.percentile(eq, 95):>10,.0f} "
          f"{dead.mean():>11.4f} {(eq < 0.5*BANK).mean():>9.4f}")

print("\n=== IS 15 'MORE PROFITABLE' THAN 30? ===")
a, b = res[15], res[30]
print(f"  qty 15 mean ${a.mean():,.0f}   median ${np.median(a):,.0f}   "
      f"5th ${np.percentile(a,5):,.0f}")
print(f"  qty 30 mean ${b.mean():,.0f}   median ${np.median(b):,.0f}   "
      f"5th ${np.percentile(b,5):,.0f}")
print(f"  30 beats 15 on the MEAN by ${b.mean()-a.mean():,.0f}")
print(f"  15 beats 30 at the 5th PERCENTILE by "
      f"${np.percentile(a,5)-np.percentile(b,5):,.0f}")
print(f"  P(qty30 path ends below qty15 median): "
      f"{(b < np.median(a)).mean():.4f}")

print("\n=== NOW UNDER A DEGRADED EDGE (the regime we may be in) ===")
print(f"{'qty':>5} {'shift':>8} {'mean':>10} {'median':>10} {'5th pct':>10} "
      f"{'P(stopped)':>11}")
for shift in (0.0, -4.0, -8.0):
    for qty in (15, 30):
        eq, dead = sim(qty, shift=shift)
        print(f"{qty:>5} {shift:>+8.1f}c {eq.mean():>10,.0f} "
              f"{np.median(eq):>10,.0f} {np.percentile(eq, 5):>10,.0f} "
              f"{dead.mean():>11.4f}")
    print()

print("=== THE SINGLE-WINDOW EXPOSURE THAT ACTUALLY TRIGGERED THE HALT ===")
worst = WE.min()
print(f"{'qty':>5} {'worst window $':>16} {'% of ${:.0f} bank'.format(BANK):>18}")
for qty in (5, 10, 15, 20, 30):
    print(f"{qty:>5} {worst*qty:>+16.2f} {abs(worst*qty)/BANK*100:>17.1f}%")
print("\n  The 2026-08-05 halt came from a -$73.50 window at qty 30. The same")
print("  window at qty 15 costs half that, and the stop would not have fired.")

print("\n=== HOW LONG TO REACH A TARGET, ACCOUNTING FOR STOPS ===")
for target in (600, 800):
    print(f"  P(equity >= ${target} within 30 days):")
    for qty in (10, 15, 20, 30):
        eq, dead = sim(qty)
        print(f"    qty {qty:>2}: {(eq >= target).mean():.4f}")
