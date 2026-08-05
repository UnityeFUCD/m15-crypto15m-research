"""THE UNTESTED HYPOTHESIS: is the whole complex on a knife edge at once?

WHY EVERY PREVIOUS TEST WAS LOOKING IN THE WRONG PLACE
  Fourteen features have failed to predict a wipeout: price, volume, longshot
  share, cross-sectional price, dispersion, drift, slope, curvature, path
  volatility, range, volume growth, longshot trend, entry minute, time
  autocorrelation.

  Every one of them is a KALSHI quantity - a price or a flow. But a wipeout is
  not a Kalshi event. It is the underlying index crossing its strike in every
  coin at once. What governs that is how far each index sits from its own
  strike, measured against how far it could plausibly travel in the time left:

      z = (index_now - strike) / (sigma * sqrt(seconds_remaining))

  The Kalshi price is a NOISY, SATURATING function of z. Two windows both
  quoted at 73c can be completely different underneath: one with a large margin
  and high volatility, one with a razor-thin margin and low volatility. Price
  cannot tell them apart. z can.

THE SHARP PREDICTION
  A wipeout needs EVERY coin to flip. If every coin is sitting on a thin margin
  simultaneously, one modest market-wide move flips all of them. If margins are
  wide, the same move does nothing. So the cross-sectional MINIMUM (or mean) of
  z should separate wipeout windows - and price, which averages over volatility
  and time, should not.

  This also explains why alignment mattered (19 of 20 wipeouts had every
  favourite pointing the same way, 5.65% vs 1.54%): alignment is necessary but
  not sufficient. Aligned AND thin is the dangerous state.

WHAT WOULD MAKE THIS A REAL FINDING
  It must beat price (which is already known), survive day-clustered
  permutation, hold out of sample, and produce a rule that gains dollars rather
  than merely separating rates.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260921)

M = pd.read_parquet(OUT / "mechanism.parquet")
M["wkey"] = M.win.str.split("-").str[1]
# z at the moment we would enter: the first trade inside the live entry window
M = M[(M.secs >= 480) & (M.secs <= 840)].copy()
Z = (M.sort_values("secs", ascending=False)
       .groupby(["wkey", "coin"], as_index=False).first()
       [["wkey", "coin", "z", "px", "won", "day", "secs"]])
print(f"z observations at entry: {len(Z):,}  windows {Z.wkey.nunique():,}  "
      f"coins {sorted(Z.coin.unique())}")

# restrict to the entries the live policy would actually take
G = pd.read_parquet(OUT / "grid.parquet")
q = G[(G.px >= 0.65) & (G.px < 0.85) & (G.minute >= 8) & (G.minute <= 14)
      & (G.vol >= 2000)]
E = (q.sort_values("minute", ascending=False)
       .groupby(["wkey", "coin"], as_index=False).first())
E = (E.sort_values(["wkey", "minute"], ascending=[True, False])
       .groupby("wkey", as_index=False).head(3))[["wkey", "coin", "px", "won", "day"]]
J = E.merge(Z[["wkey", "coin", "z"]], on=["wkey", "coin"], how="inner")
print(f"entries with a z value: {len(J):,} of {len(E):,}\n")

# cross-sectional z state per window, over ALL coins quoting - not just ours,
# because the market-wide move is what matters
ALLZ = Z.groupby("wkey").agg(z_min=("z", "min"), z_mean=("z", "mean"),
                             z_max=("z", "max"), z_sd=("z", "std"),
                             n_coins=("z", "size")).reset_index()
W = J.groupby("wkey").agg(n=("won", "size"),
                          losses=("won", lambda s: int((s == 0).sum())),
                          px=("px", "mean"), our_zmin=("z", "min"),
                          our_zmean=("z", "mean"),
                          day=("day", "first")).reset_index()
W = W.merge(ALLZ, on="wkey", how="left")
Wm = W[(W.n >= 2) & W.z_min.notna()].copy()
Wm["wipe"] = (Wm.losses == Wm.n).astype(int)
print(f"windows with 2+ positions and cross-sectional z: {len(Wm)}   "
      f"wipeouts {Wm.wipe.sum()} ({Wm.wipe.mean():.4f})\n")


def dayperm(df, col, n=6000):
    """Day-clustered permutation of the wipeout label."""
    med = df[col].median()
    a = df[df[col] <= med].wipe.mean()
    b = df[df[col] > med].wipe.mean()
    real = b - a
    null = []
    for _ in range(n):
        S = df.copy()
        S["wipe"] = S.groupby("day").wipe.transform(
            lambda s: RNG.permutation(s.values))
        null.append(S[S[col] > med].wipe.mean() - S[S[col] <= med].wipe.mean())
    null = np.array(null)
    return a, b, real, (np.abs(null) >= abs(real)).mean()


print("=== DOES CROSS-SECTIONAL z SEPARATE WIPEOUTS? ===")
print("  (price is included as the benchmark it must beat)\n")
print(f"{'feature':>12} {'low half':>10} {'high half':>10} {'diff':>9} "
      f"{'ratio':>8} {'p (day-clustered)':>19}")
cands = ["z_min", "z_mean", "z_max", "z_sd", "our_zmin", "our_zmean", "px"]
hits = []
for c in cands:
    d = Wm.dropna(subset=[c])
    if len(d) < 60:
        continue
    a, b, real, p = dayperm(d, c)
    ratio = b / a if a > 0 else np.nan
    star = "  <-- " if p < 0.05 else ""
    print(f"{c:>12} {a:>10.4f} {b:>10.4f} {real:>+9.4f} {ratio:>8.2f} "
          f"{p:>19.4f}{star}")
    if p < 0.05:
        hits.append(c)

print("\n=== WIPEOUT RATE BY CROSS-SECTIONAL MINIMUM z (the knife-edge test) ===")
Wm["zq"] = pd.qcut(Wm.z_min.rank(method="first"), 5, labels=False,
                   duplicates="drop")
print(f"{'quintile':>9} {'n':>6} {'mean z_min':>11} {'mean px':>9} "
      f"{'wipeout rate':>13} {'window P&L':>12}")
Wm2 = Wm.merge(
    J.assign(pnl=np.where(J.won == 1, (1 - J.px) * 15, -J.px * 15))
     .groupby("wkey").pnl.sum().rename("pnl"), on="wkey", how="left")
for qq, g in Wm2.dropna(subset=["zq"]).groupby("zq"):
    print(f"{int(qq):>9} {len(g):>6} {g.z_min.mean():>11.3f} "
          f"{g.px.mean()*100:>8.1f}c {g.wipe.mean():>13.4f} "
          f"{g.pnl.mean():>+12.2f}")

print("\n=== THE DECISIVE CHECK: does z beat PRICE at the same job? ===")
print("  If z only works because it correlates with price, it adds nothing.")
print("  Split on z INSIDE price terciles:\n")
Wm["pt"] = pd.qcut(Wm.px.rank(method="first"), 3, labels=False, duplicates="drop")
print(f"{'price tercile':>14} {'mean px':>9} {'low z wipe':>12} "
      f"{'high z wipe':>13} {'n low':>7} {'n high':>7}")
for pt, g in Wm.dropna(subset=["pt"]).groupby("pt"):
    med = g.z_min.median()
    lo, hi = g[g.z_min <= med], g[g.z_min > med]
    if len(lo) < 15 or len(hi) < 15:
        continue
    print(f"{int(pt):>14} {g.px.mean()*100:>8.1f}c {lo.wipe.mean():>12.4f} "
          f"{hi.wipe.mean():>13.4f} {len(lo):>7} {len(hi):>7}")
print("\n  If the gap survives INSIDE each price tercile, z carries information")
print("  price does not - which is the whole claim.")

if hits:
    print(f"\n=== OUT OF SAMPLE: {hits} ===")
    days = sorted(Wm.day.unique())
    cut = days[len(days) // 2]
    for c in hits:
        tr, te = Wm[Wm.day < cut], Wm[Wm.day >= cut]
        thr = tr[c].median()
        lo, hi = te[te[c] <= thr], te[te[c] > thr]
        if len(lo) < 10 or len(hi) < 10:
            print(f"  {c:>10}: too few out of sample")
            continue
        print(f"  {c:>10}: OOS wipeout  low {lo.wipe.mean():.4f} ({lo.wipe.sum()}/{len(lo)})"
              f"   high {hi.wipe.mean():.4f} ({hi.wipe.sum()}/{len(hi)})")
else:
    print("\n=== CROSS-SECTIONAL z DOES NOT SEPARATE WIPEOUTS EITHER ===")
