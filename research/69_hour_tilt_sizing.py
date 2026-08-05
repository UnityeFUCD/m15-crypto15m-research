"""Don't FILTER on the hour - TILT SIZE by it. Keep every entry, move contracts.

WHY FILTERING FAILED AND TILTING MIGHT NOT
  Every hours filter lost money: -$34 to -$96/day walk-forward. The reason is
  simple and worth stating plainly - the worst block (08-13 UTC) still earns
  +6.84c/contract. Discarding profitable entries cannot help. Edge per contract
  rose to +17.53c and dollars fell to $150/day, because entries per day
  collapsed from 105.8 to 45.1.

  Tilting has no such defect. It keeps every entry and only moves contracts
  from lower-edge hours to higher-edge ones, holding the TOTAL contract count
  fixed. Nothing profitable is thrown away, and gross exposure is unchanged.

  Rough arithmetic on the pre-shift book: 293 entries at +17.63c in 00-07 and
  765 at +9.51c elsewhere. Holding 21,160 total contracts fixed, moving from
  flat qty 20 to 26/17.7 gives about +$14/day. That is worth testing properly.

THE REASON TO BE SUSPICIOUS
  An edge-weighted sizing model was deployed earlier today and REVERSED, because
  it was fitted pre-shift and stopped ranking once the distributions moved. This
  is another sizing model fitted on pre-shift data. The post-shift coin-adjusted
  time gap is only +2.68c with P(gap<=0)=0.3700 - not significant on 3 days.

  So the prior here is bad, and the test has to be built to fail:
    - walk-forward only, never fitting on the day being scored
    - contract-neutral by construction, verified numerically
    - dollars, day-clustered
    - and reported for BOTH regimes, with the post-shift result given the
      weight its sample size deserves

CONSTRAINTS RESPECTED
  Max size 30 is a hard cap. Tilts are clipped so no hour ever exceeds it, and
  the mean stays at the deployed qty of 20.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261037)
QTY, QMAX, QMIN = 20, 30, 10
KAPPA = 12.0
GRID = np.linspace(0, 2 * np.pi, 145)[:-1]


def smooth(th, ed, grid=GRID, kappa=KAPPA):
    W = np.exp(kappa * np.cos(grid[:, None] - th[None, :]))
    return (W @ ed) / W.sum(axis=1) * 100


def load_pre():
    D = pd.read_parquet(OUT / "grid.parquet")
    q = D[(D.px >= 0.65) & (D.px < 0.80) & (D.minute >= 8) & (D.minute <= 14)
          & (D.vol >= 2000)]
    e = (q.sort_values("minute", ascending=False)
           .groupby(["wkey", "coin"], as_index=False).first())
    e = (e.sort_values(["wkey", "minute"], ascending=[True, False])
           .groupby("wkey", as_index=False).head(3)).copy()
    e["tod"] = e.hour * 60 + e.wkey.str[-2:].astype(int)
    return e


def load_post():
    P = pd.read_parquet(OUT / "postshift_nofilter.parquet")
    P = P[(P.vol >= 2000) & (P.px >= 0.65) & (P.px < 0.80)].copy()
    P["hour"] = (P.wkey.str[-4:-2].astype(int) + 4) % 24
    P["tod"] = P.hour * 60 + P.wkey.str[-2:].astype(int)
    return (P.sort_values(["wkey", "px"], ascending=[True, False])
             .groupby("wkey", as_index=False).head(3)).copy()


def tilt_qty(curve, tod, strength):
    """Map the fitted hourly curve to a contract count, mean-preserving."""
    v = np.interp(tod / 1440 * 2 * np.pi, GRID, curve)
    z = (v - v.mean()) / (v.std() + 1e-9)
    q = QTY * (1.0 + strength * z)
    return np.clip(q, QMIN, QMAX)


for lbl, E in (("PRE-SHIFT", load_pre()), ("POST-SHIFT", load_post())):
    days = sorted(E.day.unique())
    print("=" * 78)
    print(f"{lbl}   entries {len(E):,}   days {len(days)}")
    print("=" * 78)
    if len(days) < 4:
        print("  too few days for walk-forward\n")
        continue
    flat = E.groupby("day").edge.sum().reindex(days, fill_value=0) * QTY
    print(f"{'rule':>28} {'$/day':>10} {'contracts/day':>15} "
          f"{'vs flat qty20':>22}")
    print(f"{'flat qty 20':>28} {flat.mean():>10.2f} "
          f"{len(E)*QTY/len(days):>15.0f} {'-':>22}")
    for strength in (0.15, 0.25, 0.35, 0.50):
        pnl, ct = [], []
        for i in range(2, len(days)):
            tr = E[E.day.isin(days[:i])]
            te = E[E.day == days[i]]
            if len(tr) < 60 or not len(te):
                continue
            c = smooth(tr.tod.to_numpy() / 1440 * 2 * np.pi, tr.edge.to_numpy())
            q = tilt_qty(c, te.tod.to_numpy(), strength)
            pnl.append(float((te.edge.to_numpy() * q).sum()))
            ct.append(float(q.sum()))
        if not pnl:
            continue
        base = flat.reindex(days[2:]).values[:len(pnl)]
        bct = np.array([len(E[E.day == d]) * QTY for d in days[2:]][:len(pnl)])
        d = np.array(pnl) - base
        bs = np.sort(np.array([RNG.choice(d, len(d), True).mean()
                               for _ in range(8000)]))
        drift = (np.mean(ct) / np.mean(bct) - 1) * 100
        print(f"{'tilt strength ' + f'{strength:.2f}':>28} "
              f"{np.mean(pnl):>10.2f} {np.mean(ct):>15.0f} "
              f"{d.mean():>+8.2f} [{bs[200]:+.0f},{bs[7799]:+.0f}] "
              f"P<=0 {(bs <= 0).mean():.3f}")
        if abs(drift) > 3:
            print(f"{'':>28}  WARNING contract drift {drift:+.1f}% - not neutral")
    print()

print("=" * 78)
print("CONTRACT NEUTRALITY CHECK (the failure mode of the last sizing model)")
print("=" * 78)
E = load_pre()
days = sorted(E.day.unique())
c = smooth(E.tod.to_numpy() / 1440 * 2 * np.pi, E.edge.to_numpy())
for strength in (0.15, 0.25, 0.35, 0.50):
    q = tilt_qty(c, E.tod.to_numpy(), strength)
    print(f"  strength {strength:.2f}: mean qty {q.mean():>6.2f}  "
          f"min {q.min():>5.1f}  max {q.max():>5.1f}  "
          f"clipped {(q >= QMAX - 1e-9).sum() + (q <= QMIN + 1e-9).sum():>4} "
          f"of {len(q)}")
print("\n  The previous model failed partly because clipping broke neutrality -")
print("  mean qty drifted once the distribution moved. If 'clipped' is large,")
print("  the same thing is happening here.")

print("\n" + "=" * 78)
print("WHAT WOULD MAKE THIS DEPLOYABLE, AND WHY IT IS NOT YET")
print("=" * 78)
print("  The pre-shift evidence for the time effect is strong and survives the")
print("  coin-mix confound (+8.19c, CI [+5.38,+11.35]). The post-shift evidence")
print("  is +2.68c with P(gap<=0)=0.37 on 3 days - directionally the same, far")
print("  from significant.")
print()
print("  A tilt fitted pre-shift, on XRP/SOL/DOGE, applied to a live book that")
print("  is now ETH/XRP/HYPE/DOGE/SOL - where DOGE and SOL carry NEGATIVE edge")
print("  - is the same setup that produced two reversals earlier today.")
print()
print("  What settles it: post-shift days. The effect size needs ~10 days at")
print("  the current entry rate to separate +2.68c from zero.")
