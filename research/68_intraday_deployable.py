"""Is the intraday edge pattern a TIME effect, or a coin effect wearing a clock?
And if it is real, does an hours filter make money walk-forward, in dollars?

THE TWO WAYS THIS DIES
  1. COIN CONFOUND. Post-shift edge by coin in the live band is wildly uneven:
     ETH +12.97c, XRP +5.42c, HYPE +4.53c, SOL -1.83c, DOGE -2.32c. If the
     high-edge coins simply trade more in the high-edge hours, then the
     'intraday pattern' is the coin ranking re-expressed, and filtering by hour
     is a worse way of doing what filtering by coin would do directly.

     This must be tested WITHIN coin. If the time shape survives inside each
     coin separately, it is a time effect.

  2. DOLLARS. Cutting hours cuts entries. The 08-13 UTC block is 32% of the
     pre-shift book. A rule that raises edge-per-contract while removing a
     third of the volume can easily lower total dollars. The hourly pass
     reported cents; only dollars decide.

AND THE DATA CAVEAT THAT APPLIES TO BOTH
  grid.parquet is effectively XRP/SOL/DOGE - BTC has 347 rows and ZERO
  in-band entries, ETH only 12. Two of those three coins (SOL, DOGE) now carry
  NEGATIVE post-shift edge. So the pre-shift curve is fitted largely on coins
  that no longer pay. The post-shift set has the right coin mix but only 3
  days. Neither is clean; both are reported.

WALK-FORWARD, because that is what deploying would actually do
  For each day, fit the smooth curve on PRIOR days only, choose the hours it
  calls good, and score that day. No day ever informs its own decision.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261036)
QTY = 20
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
    et = P.wkey.str[-4:-2].astype(int)
    P["hour"] = (et + 4) % 24
    P["tod"] = P.hour * 60 + P.wkey.str[-2:].astype(int)
    P = (P.sort_values(["wkey", "px"], ascending=[True, False])
           .groupby("wkey", as_index=False).head(3)).copy()
    return P


PRE, POST = load_pre(), load_post()

print("=" * 78)
print("TEST A - IS THE TIME SHAPE PRESENT *WITHIN* EACH COIN?")
print("=" * 78)
for lbl, E in (("PRE-SHIFT", PRE), ("POST-SHIFT", POST)):
    print(f"\n  {lbl}")
    print(f"{'coin':>7} {'n':>6} {'00-07':>9} {'08-13':>9} {'14-23':>9} "
          f"{'00-07 minus rest':>18}")
    blk = np.where(E.hour < 8, 0, np.where(E.hour < 14, 1, 2))
    E = E.assign(blk=blk)
    gaps = []
    for c, g in E.groupby("coin"):
        if len(g) < 40:
            continue
        m = [g[g.blk == b].edge.mean() * 100 if (g.blk == b).sum() >= 8
             else np.nan for b in (0, 1, 2)]
        a = g[g.blk == 0].edge.mean() * 100
        r = g[g.blk != 0].edge.mean() * 100
        if (g.blk == 0).sum() >= 8 and (g.blk != 0).sum() >= 8:
            gaps.append(a - r)
        print(f"{c:>7} {len(g):>6} " + " ".join(
            f"{x:>+8.2f}c" if not np.isnan(x) else f"{'-':>9}" for x in m)
            + f" {a-r:>+17.2f}c")
    if len(gaps) >= 3:
        print(f"\n    coins with a POSITIVE 00-07 gap: "
              f"{sum(1 for x in gaps if x > 0)}/{len(gaps)}   "
              f"mean gap {np.mean(gaps):+.2f}c")
        print("    If the effect were pure coin mix, the within-coin gaps would")
        print("    scatter around zero. Consistent sign across coins is the test.")

print("\n" + "=" * 78)
print("TEST B - COIN-ADJUSTED: remove each coin's own mean, then re-test time")
print("=" * 78)
for lbl, E in (("PRE-SHIFT", PRE), ("POST-SHIFT", POST)):
    E = E.copy()
    E["resid"] = E.edge - E.groupby("coin").edge.transform("mean")
    blk = np.where(E.hour < 8, "00-07", np.where(E.hour < 14, "08-13", "14-23"))
    E["blk"] = blk
    print(f"\n  {lbl}   (residual edge after removing coin means)")
    for k, g in E.groupby("blk"):
        print(f"{k:>10} n {len(g):>5}  raw {g.edge.mean()*100:>+7.2f}c   "
              f"coin-adjusted {g.resid.mean()*100:>+7.2f}c")
    a = E[E.blk == "00-07"]
    r = E[E.blk != "00-07"]
    gg = {d: x for d, x in E.groupby("day")}
    ds = sorted(gg)
    if len(ds) >= 3:
        bs = np.sort(np.array([
            (lambda ix: (np.concatenate([gg[ds[k]][gg[ds[k]].blk == "00-07"].resid.values for k in ix]).mean()
                         - np.concatenate([gg[ds[k]][gg[ds[k]].blk != "00-07"].resid.values for k in ix]).mean()))(
                RNG.integers(0, len(ds), len(ds))) * 100
            for _ in range(4000)]))
        print(f"    coin-adjusted 00-07 gap "
              f"{(a.resid.mean()-r.resid.mean())*100:+.2f}c   "
              f"95% CI [{bs[100]:+.2f}, {bs[3899]:+.2f}]   "
              f"P(gap<=0) {(bs <= 0).mean():.4f}")

print("\n" + "=" * 78)
print("TEST C - WALK-FORWARD IN DOLLARS (fit on prior days, score the next)")
print("=" * 78)
for lbl, E in (("PRE-SHIFT", PRE), ("POST-SHIFT", POST)):
    days = sorted(E.day.unique())
    if len(days) < 4:
        print(f"\n  {lbl}: only {len(days)} days - walk-forward not meaningful")
        continue
    print(f"\n  {lbl}   {len(days)} days")
    print(f"{'rule':>26} {'$/day':>10} {'ent/day':>9} {'edge/ct':>9} "
          f"{'vs all-hours':>13}")
    allh = E.groupby("day").edge.sum().reindex(days, fill_value=0) * QTY
    print(f"{'trade ALL hours':>26} {allh.mean():>10.2f} "
          f"{len(E)/len(days):>9.1f} {E.edge.mean()*100:>+8.2f}c "
          f"{'-':>13}")
    for qtl, nm in ((0.50, "walk-fwd: top 50% hours"),
                    (0.33, "walk-fwd: top 67% hours"),
                    (0.25, "walk-fwd: top 75% hours")):
        pnl, ent, edg = [], [], []
        for i in range(2, len(days)):
            tr = E[E.day.isin(days[:i])]
            te = E[E.day == days[i]]
            if len(tr) < 60 or not len(te):
                continue
            c = smooth(tr.tod.to_numpy() / 1440 * 2 * np.pi, tr.edge.to_numpy())
            thr = np.quantile(c, qtl)
            keep = te[np.interp(te.tod / 1440 * 2 * np.pi, GRID, c) > thr]
            pnl.append(keep.edge.sum() * QTY)
            ent.append(len(keep))
            edg.append(keep.edge.mean() * 100 if len(keep) else np.nan)
        if not pnl:
            continue
        base = allh.reindex(days[2:]).values[:len(pnl)]
        d = np.array(pnl) - base
        bs = np.sort(np.array([RNG.choice(d, len(d), True).mean()
                               for _ in range(6000)]))
        print(f"{nm:>26} {np.mean(pnl):>10.2f} {np.mean(ent):>9.1f} "
              f"{np.nanmean(edg):>+8.2f}c "
              f"{d.mean():>+7.2f} [{bs[150]:+.0f},{bs[5849]:+.0f}]")
    print(f"\n    fixed block rule (no fitting): drop 08-13 UTC")
    k = E[~((E.hour >= 8) & (E.hour < 14))]
    kd = k.groupby("day").edge.sum().reindex(days, fill_value=0) * QTY
    d = (kd - allh).values
    bs = np.sort(np.array([RNG.choice(d, len(d), True).mean()
                           for _ in range(6000)]))
    print(f"{'drop 08-13 UTC':>26} {kd.mean():>10.2f} "
          f"{len(k)/len(days):>9.1f} {k.edge.mean()*100:>+8.2f}c "
          f"{d.mean():>+7.2f} [{bs[150]:+.0f},{bs[5849]:+.0f}]")

print("\n" + "=" * 78)
print("READING IT")
print("=" * 78)
print("  An hours filter can only help if it raises DOLLARS, not cents. It")
print("  removes entries, and at 3 per window with a gross cap we are not")
print("  always capital-bound, so lost entries are lost profit rather than")
print("  capital freed for better ones.")
print()
print("  Trading hours is NOT on the do-not-change list - that list covers")
print("  size, stops, the 65-85c band and the 8-14 minute window. But nothing")
print("  gets deployed on a pre-shift fit alone, given the coin mismatch.")
