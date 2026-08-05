"""Recalibrate the deployed thresholds on POST-SHIFT data.

WHY THIS MATTERS MORE THAN ANY NEW SIGNAL
  Three live parameters were all fitted on the pre-shift regime:

      MIN_VOLUME  2000     chosen by walk-forward on Jul 23 - Aug 1
      OTHERS_MAX  0.71     the cross-coin cut of the dead-entry filter
      FLONG_MAX   0.5046   the flow cut of the same filter

  The market level-shifted after Aug 1 - the population surplus fell from
  +10-14pp to +4-5pp, and daily qualifying entries roughly doubled. A cut point
  chosen on one distribution can sit in completely the wrong place on another:
  if volumes are systematically higher now, a 2,000 floor that once selected
  the top half might now select almost everything and do nothing at all.

  So this is not a search for a new edge. It is checking whether the controls
  already deployed still point where they were aimed. That is cheaper to fix
  and more likely to be wrong than anything undiscovered.

WHAT IS CHECKED
  1. have the underlying DISTRIBUTIONS moved? (if not, the cuts are still fine)
  2. does the volume floor still separate edge, and where is the best cut now?
  3. does the dead-entry conjunction still identify worthless entries?
  4. would recalibrating actually pay, in dollars, day-clustered?
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261017)
QTY = 15

PRE = pd.read_parquet(OUT / "grid.parquet")
q = PRE[(PRE.px >= 0.65) & (PRE.px < 0.85) & (PRE.minute >= 8)
        & (PRE.minute <= 14) & (PRE.vol >= 2000)]
E_pre = (q.sort_values("minute", ascending=False)
           .groupby(["wkey", "coin"], as_index=False).first())
# unfiltered pre-shift, for distribution comparison
qa = PRE[(PRE.px >= 0.65) & (PRE.px < 0.85) & (PRE.minute >= 8)
         & (PRE.minute <= 14)]
A_pre = (qa.sort_values("minute", ascending=False)
           .groupby(["wkey", "coin"], as_index=False).first())
POST = pd.read_parquet(OUT / "postshift_windows.parquet")

print("=" * 76)
print("1. HAVE THE DISTRIBUTIONS MOVED? (if not, the cuts are still aimed right)")
print("=" * 76)
print(f"{'quantity':>12} {'pct':>6} {'PRE-shift':>12} {'POST-shift':>12} "
      f"{'shift':>10}")
for col, lbl in (("vol", "volume"), ("frac_long", "longshot share"),
                 ("px", "entry price")):
    for p in (10, 25, 50, 75, 90):
        a = np.percentile(A_pre[col].dropna(), p)
        b = np.percentile(POST[col].dropna(), p)
        print(f"{lbl:>12} {p:>5}th {a:>12.4f} {b:>12.4f} "
              f"{(b/a-1)*100 if a else float('nan'):>+9.1f}%")
    print()

print("=" * 76)
print("2. DOES THE VOLUME FLOOR STILL SEPARATE EDGE? (post-shift)")
print("=" * 76)
print(f"{'floor':>8} {'n':>6} {'share kept':>11} {'win':>8} {'edge/ct':>9} "
      f"{'total edge':>11}")
best = None
for v in (0, 500, 1000, 1500, 2000, 3000, 4000, 6000, 8000):
    g = POST[POST.vol >= v]
    if len(g) < 40:
        continue
    tot = g.edge.sum() * 100
    print(f"{v:>8} {len(g):>6} {len(g)/len(POST):>11.3f} {g.won.mean():>8.4f} "
          f"{g.edge.mean()*100:>+9.2f} {tot:>+11.1f}")
    if best is None or tot > best[1]:
        best = (v, tot)
print(f"\n  total-edge optimum post-shift: floor {best[0]}  "
      f"(deployed is 2000)")
print("  'total edge' is what matters - a higher floor raises edge/contract")
print("  but discards entries, and we are opportunity-constrained.")

print("\n" + "=" * 76)
print("3. DOES THE DEAD-ENTRY CONJUNCTION STILL FIND WORTHLESS ENTRIES?")
print("=" * 76)
P = POST[POST.vol >= 2000].copy()
GX = P.groupby("wkey").agg(s_px=("px", "sum"), n=("px", "size")).reset_index()
P = P.merge(GX, on="wkey", how="left")
P["others_px"] = np.where(P.n >= 3, (P.s_px - P.px) / (P.n - 1).clip(lower=1),
                          np.nan)
P = P.dropna(subset=["others_px"])
print(f"  post-shift entries with a cross-section: {len(P)}")
dead = P[(P.others_px <= 0.71) & (P.frac_long <= 0.5046)]
rest = P[~((P.others_px <= 0.71) & (P.frac_long <= 0.5046))]
print(f"\n  DEPLOYED cuts (others_px<=0.71 AND frac_long<=0.5046):")
print(f"    flagged dead : n {len(dead):>4}  win {dead.won.mean() if len(dead) else float('nan'):.4f}  "
      f"edge {dead.edge.mean()*100 if len(dead) else float('nan'):+.2f}c")
print(f"    kept         : n {len(rest):>4}  win {rest.won.mean():.4f}  "
      f"edge {rest.edge.mean()*100:+.2f}c")
if len(dead) > 15:
    g = {d: gg for d, gg in dead.groupby("day")}
    ds = sorted(g)
    bs = np.sort(np.array([
        np.concatenate([g[ds[i]].edge.values
                        for i in RNG.integers(0, len(ds), len(ds))]).mean() * 100
        for _ in range(6000)]))
    print(f"    dead-cohort edge 95% CI [{bs[150]:+.2f}, {bs[5849]:+.2f}]")
    print(f"    -> the filter only earns its place if this is <= 0")

print("\n  SWEEPING THE CUTS on post-shift data:")
print(f"{'others<=':>10} {'flong<=':>9} {'n dead':>8} {'dead edge':>11} "
      f"{'kept edge':>11} {'kept total':>11}")
rows = []
for o in (0.66, 0.69, 0.71, 0.74, 0.77):
    for f in (0.40, 0.45, 0.5046, 0.55, 0.60):
        d = P[(P.others_px <= o) & (P.frac_long <= f)]
        k = P[~((P.others_px <= o) & (P.frac_long <= f))]
        if len(d) < 8 or len(k) < 60:
            continue
        rows.append((o, f, len(d), d.edge.mean() * 100, k.edge.mean() * 100,
                     k.edge.sum() * 100))
        print(f"{o:>10.2f} {f:>9.4f} {len(d):>8} {d.edge.mean()*100:>+11.2f} "
              f"{k.edge.mean()*100:>+11.2f} {k.edge.sum()*100:>+11.1f}")
if rows:
    bestk = max(rows, key=lambda r: r[5])
    print(f"\n  best kept-total post-shift: others<={bestk[0]:.2f} "
          f"flong<={bestk[1]:.4f}  (deployed 0.71 / 0.5046)")

print("\n" + "=" * 76)
print("4. WOULD RECALIBRATING PAY? dollars, day-clustered")
print("=" * 76)
days = sorted(P.day.unique())
def dollars(o, f):
    k = P[~((P.others_px <= o) & (P.frac_long <= f))]
    return k.groupby("day").edge.sum().reindex(days, fill_value=0) * QTY
base = dollars(0.71, 0.5046)
print(f"{'candidate':>22} {'$/day':>9} {'vs deployed':>12} {'days won':>9}")
print(f"{'DEPLOYED 0.71/0.5046':>22} {base.mean():>+9.2f} {0.0:>+12.2f} "
      f"{'-':>9}")
for o, f in ((0.66, 0.5046), (0.74, 0.5046), (0.71, 0.45), (0.71, 0.55),
             (0.66, 0.45), (0.74, 0.55), (1.01, 1.01)):
    c = dollars(o, f)
    d = (c - base).values
    lbl = "NO FILTER" if o > 1 else f"{o:.2f} / {f:.4f}"
    print(f"{lbl:>22} {c.mean():>+9.2f} {d.mean():>+12.2f} "
          f"{(c > base).sum():>6}/{len(days)}")
print(f"\n  only {len(days)} post-shift days - treat every difference here as")
print("  indicative. The point is whether the deployed cut is CLEARLY wrong,")
print("  not to chase a new optimum on three days of data.")
