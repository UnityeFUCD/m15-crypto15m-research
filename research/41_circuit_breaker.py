"""Is a bankroll circuit breaker worth having, and at what level?

WHAT ALREADY EXISTS
  MAX_DD_FRAC = 0.30 - a PRE-SUBMIT envelope requiring
      cash - proposed_cost  >=  0.70 x peak_equity
  So a 50% breaker would be LOOSER than what is already enforced. The real
  question is not "should we add one" but "is 30% the right level".

THE THEORY, WHICH THE SIMULATION MUST RESPECT
  With a POSITIVE edge and SERIALLY INDEPENDENT outcomes - both measured: the
  edge is positive and window P&L autocorrelation is null at every lag - a stop
  loss strictly REDUCES expected terminal wealth. It cannot improve returns; it
  can only cap the tail. So a breaker is insurance, and the question is the
  premium: how much expected growth does each level cost, and how much ruin
  does it prevent?

  A breaker earns its keep only in the scenario the historical data cannot
  price: that the edge has silently turned negative. So the simulation runs
  BOTH regimes - the measured one, and a dead-edge one.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260908)
D = pd.read_parquet(OUT / "grid.parquet")
QTY, BANK = 30, 447.31
NPATH, NWIN = 20000, 96 * 14          # 14 days of windows

q = D[(D.px >= 0.65) & (D.px < 0.85) & (D.minute >= 8) & (D.minute <= 14)
      & (D.vol >= 2000)]
E = (q.sort_values("minute", ascending=False)
       .groupby(["wkey", "coin"], as_index=False).first())
E = (E.sort_values(["wkey", "minute"], ascending=[True, False])
       .groupby("wkey", as_index=False).head(3)).copy()
E["pnl"] = np.where(E.won == 1, (1 - E.px) * QTY, -E.px * QTY)
WP = E.groupby("wkey").pnl.sum().to_numpy()      # whole-window P&L
print(f"window P&L pool: n {len(WP)}  mean ${WP.mean():+.2f}  "
      f"sd ${WP.std():.2f}  min ${WP.min():+.2f}  max ${WP.max():+.2f}")


def run(levels, shift=0.0):
    """levels: fraction of PEAK equity at which trading stops permanently.
    shift: cents/contract added to every outcome, to model a degraded edge."""
    pool = WP + shift / 100.0 * QTY * 1.8      # ~1.8 positions per window
    idx = RNG.integers(0, len(pool), size=(NPATH, NWIN))
    draws = pool[idx]
    out = {}
    for lv in levels:
        eq = np.full(NPATH, BANK)
        peak = np.full(NPATH, BANK)
        dead = np.zeros(NPATH, dtype=bool)
        for t in range(NWIN):
            live = ~dead
            eq[live] += draws[live, t]
            peak = np.maximum(peak, eq)
            if lv is not None:
                dead |= eq < lv * peak
        out[lv] = dict(final=eq, ruin=(eq < 0.5 * BANK).mean(),
                       stopped=dead.mean())
    return out


print(f"\nbank ${BANK:.2f}   {NPATH:,} paths x {NWIN} windows (~14 days)")
for lbl, shift in (("MEASURED EDGE (what we observe today)", 0.0),
                   ("DEGRADED: edge cut to zero", -9.95),
                   ("DEAD: edge turned -5c/contract", -14.95)):
    print(f"\n=== {lbl} ===")
    res = run([None, 0.70, 0.60, 0.50, 0.40], shift=shift)
    print(f"{'breaker':>10} {'median final':>14} {'mean final':>12} "
          f"{'5th pct':>10} {'P(ruin)':>9} {'P(stopped)':>11}")
    for lv, r in res.items():
        nm = "none" if lv is None else f"{lv:.0%} of peak"
        print(f"{nm:>10} {np.median(r['final']):>14,.0f} "
              f"{r['final'].mean():>12,.0f} "
              f"{np.percentile(r['final'], 5):>10,.0f} "
              f"{r['ruin']:>9.4f} {r['stopped']:>11.4f}")

print("\n=== WHAT THE CURRENT 30% ENVELOPE COSTS AND BUYS ===")
a = run([None, 0.70], shift=0.0)
cost = a[None]["final"].mean() - a[0.70]["final"].mean()
print(f"  at the measured edge: costs ${cost:,.0f} of expected terminal wealth")
print(f"    P(ruin) {a[None]['ruin']:.4f} -> {a[0.70]['ruin']:.4f}")
b = run([None, 0.70], shift=-14.95)
cost2 = b[None]["final"].mean() - b[0.70]["final"].mean()
print(f"  if the edge were DEAD (-5c): saves ${-cost2:,.0f}")
print(f"    P(ruin) {b[None]['ruin']:.4f} -> {b[0.70]['ruin']:.4f}")
print("\n  That asymmetry IS the case for the breaker: it costs a little when")
print("  we are right and saves a lot when we are wrong. A 50% breaker would")
print("  be LOOSER than the 30% envelope already deployed, so it would add")
print("  nothing - the existing control already binds first.")
