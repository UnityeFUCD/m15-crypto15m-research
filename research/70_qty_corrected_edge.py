"""Is qty 20 still right once the edge is corrected for maker fill bias?

WHY THIS HAS TO BE RE-DERIVED
  The qty decision (15, then 20) came from simulations driven by the measured
  window-level edge distribution taken from grid.parquet. Those numbers assume
  every posted order fills.

  They do not. Measured on 120 live orders, we fill 96.2% of losers and 84.3%
  of winners, because a losing favourite must fall back through our resting bid
  while a winning one runs away. Correcting for that cuts realised edge by
  27.6% pre-shift and 58.9% post-shift.

  Position sizing is the one decision most sensitive to the edge estimate. A
  strategy sized for +11.76c/contract that actually earns +9.87c has a
  different optimal size, a different ruin probability, and a different
  drawdown profile - and we are currently $79.40 off peak, so this is not
  academic.

WHAT CHANGES AND WHAT DOES NOT
  The fill bias does NOT scale the distribution uniformly. It drops winners
  more often than losers, so it removes mass from the RIGHT tail specifically.
  Mean falls and the loss tail becomes relatively heavier - which is worse for
  sizing than a simple proportional haircut would suggest.

  Modelling it as a scalar edge reduction would understate the risk. So the
  simulation resamples entries and applies the fill model per entry, letting
  the distribution reshape itself.

CONSTRAINTS
  Hard cap of 30. Floor of $211 - the absolute stop, replacing the old
  20%-of-peak rule at the user's explicit instruction.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261042)

EQUITY = 378.47
FLOOR = 211.0
P_FILL_LOSE = 0.962
P_FILL_WIN = 0.843
NPATH, NDAY = 12000, 30

D = pd.read_parquet(OUT / "grid.parquet")
q = D[(D.px >= 0.65) & (D.px < 0.80) & (D.minute >= 8) & (D.minute <= 14)
      & (D.vol >= 2000)]
E = (q.sort_values("minute", ascending=False)
       .groupby(["wkey", "coin"], as_index=False).first())
E = (E.sort_values(["wkey", "minute"], ascending=[True, False])
       .groupby("wkey", as_index=False).head(3)).copy()
E["ec"] = np.where(E.won == 1, 1 - E.px, -E.px)
days = sorted(E.day.unique())

# window-level vectors, keeping entries separate so fill can be applied per entry
grp = [g for _, g in E.groupby("wkey")]
WIN_EC = [g.ec.to_numpy() for g in grp]
WIN_WON = [g.won.to_numpy().astype(bool) for g in grp]
K = max(int(round(len(grp) / len(days))), 1)
print(f"windows {len(grp)}   {K} windows/day   entries {len(E)}")
print(f"equity ${EQUITY:.2f}   floor ${FLOOR:.0f}   room ${EQUITY-FLOOR:.2f}\n")

flat = np.concatenate(WIN_EC)
flat_won = np.concatenate(WIN_WON)
pf = np.where(flat_won, P_FILL_WIN, P_FILL_LOSE)
print("=" * 78)
print("WHAT THE FILL BIAS DOES TO THE EDGE DISTRIBUTION")
print("=" * 78)
print(f"{'':>24} {'mean':>10} {'sd':>9} {'5th pct':>10} {'P(loss)':>9}")
print(f"{'assumed (100% fill)':>24} {flat.mean()*100:>+9.2f}c "
      f"{flat.std()*100:>8.2f}c {np.percentile(flat,5)*100:>+9.2f}c "
      f"{(flat<0).mean():>9.4f}")
eff = flat * pf                       # expected contribution per posted entry
print(f"{'corrected (real fill)':>24} {eff.mean()*100:>+9.2f}c "
      f"{eff.std()*100:>8.2f}c {np.percentile(eff,5)*100:>+9.2f}c "
      f"{(flat<0).mean():>9.4f}")
print(f"\n  mean edge falls {(1-eff.mean()/flat.mean())*100:.1f}%")
print("  note the 5th percentile barely moves - the bias removes right-tail")
print("  mass, not left-tail mass. Losses are unaffected; wins are rarer.")


# pad windows to a fixed width so the whole simulation vectorises. Looping over
# unique windows per timestep was quadratic and had to be killed mid-run.
WMAX = max(len(v) for v in WIN_EC)
PAD_EC = np.zeros((len(grp), WMAX))
PAD_PF = np.zeros((len(grp), WMAX))
PAD_OK = np.zeros((len(grp), WMAX), dtype=bool)
for i, (ec, wn) in enumerate(zip(WIN_EC, WIN_WON)):
    PAD_EC[i, :len(ec)] = ec
    PAD_PF[i, :len(ec)] = np.where(wn, P_FILL_WIN, P_FILL_LOSE)
    PAD_OK[i, :len(ec)] = True
print(f"padded window matrix {PAD_EC.shape} (max {WMAX} entries/window)\n")


def sim(qty, corrected, shift=0.0):
    """Resample windows; apply the per-entry fill model when corrected."""
    eq = np.full(NPATH, EQUITY)
    dead = np.zeros(NPATH, dtype=bool)
    for _ in range(NDAY * K):
        if dead.all():
            break
        sel = RNG.integers(0, len(grp), size=NPATH)
        ec = (PAD_EC[sel] + shift / 100.0) * PAD_OK[sel]
        if corrected:
            hit = (RNG.random((NPATH, WMAX)) < PAD_PF[sel]) & PAD_OK[sel]
            draw = (hit * ec).sum(axis=1) * qty
        else:
            draw = ec.sum(axis=1) * qty
        eq = np.where(dead, eq, eq + draw)
        dead |= eq < FLOOR
    return eq, dead


print("\n" + "=" * 78)
print("30-DAY OUTCOME - ASSUMED vs CORRECTED EDGE")
print("=" * 78)
for lbl, corrected in (("ASSUMED 100% fill", False), ("CORRECTED fill bias", True)):
    print(f"\n  {lbl}")
    print(f"{'qty':>5} {'mean':>10} {'median':>10} {'5th pct':>10} "
          f"{'P(floored)':>12} {'worst window $':>16}")
    for qty in (10, 15, 20, 25, 30):
        eq, dead = sim(qty, corrected)
        worst = min(v.sum() for v in WIN_EC) * qty
        print(f"{qty:>5} {eq.mean():>10,.0f} {np.median(eq):>10,.0f} "
              f"{np.percentile(eq,5):>10,.0f} {dead.mean():>12.4f} "
              f"{worst:>+16.2f}")

print("\n" + "=" * 78)
print("AND UNDER THE POST-SHIFT EDGE, WHICH IS WHERE WE ACTUALLY ARE")
print("=" * 78)
print("  post-shift population edge is ~+5.1c against pre-shift +11.8c, so the")
print("  pool is shifted down to match, THEN the fill bias is applied\n")
for shift, lbl in ((-6.0, "post-shift-like (-6c)"), (-9.0, "near zero (-9c)")):
    print(f"  {lbl}")
    print(f"{'qty':>5} {'mean':>10} {'median':>10} {'5th pct':>10} "
          f"{'P(floored)':>12}")
    for qty in (10, 15, 20, 25, 30):
        eq, dead = sim(qty, True, shift=shift)
        print(f"{qty:>5} {eq.mean():>10,.0f} {np.median(eq):>10,.0f} "
              f"{np.percentile(eq,5):>10,.0f} {dead.mean():>12.4f}")
    print()

print("=" * 78)
print("DRAWDOWN CAPACITY AT TODAY'S EQUITY")
print("=" * 78)
worst = min(v.sum() for v in WIN_EC)
print(f"  worst observed window {worst*100:+.2f}c/contract")
print(f"{'qty':>5} {'worst window $':>16} {'% of equity':>13} "
      f"{'such windows to floor':>23}")
for qty in (10, 15, 20, 25, 30):
    w = worst * qty
    print(f"{qty:>5} {w:>+16.2f} {abs(w)/EQUITY*100:>12.1f}% "
          f"{int((EQUITY-FLOOR)/abs(w)):>23}")

print("\n" + "=" * 78)
print("READING IT")
print("=" * 78)
print("  The question is not which qty has the highest mean - higher size")
print("  always does while the edge is positive. It is whether qty 20 still")
print("  leaves acceptable floor risk once the edge is corrected downward,")
print("  and whether the corrected numbers change the ranking at all.")
