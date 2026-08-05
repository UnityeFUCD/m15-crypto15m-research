"""Does longshot SHARE add edge beyond the volume floor? - properly powered.

WHY THIS REDO
  25_beyond_volume.py concluded "nothing adds to the volume filter" from 31
  entries. That is not a null result, it is an absent test: the adverse-
  selection extraction keeps only the first in-band trade per window, so almost
  nothing survived `vol >= 2000` AND the 65-85c band. Reporting that as a
  finding would have been giving up with no evidence either way.

  grid.parquet carries both `vol` and `frac_long` at one row per minute per
  window, which leaves 2,495 rows inside the live filter for DOGE/SOL/XRP -
  enough to actually answer it.

THE QUESTION
  The volume floor is deployed and works. Longshot share is highly correlated
  with activity (rho 0.15 with raw volume, but 0.997 with recent longshot flow)
  so it may be the same signal. If it adds edge AT FIXED VOLUME it is a second
  filter worth having; if not, the floor already has it.

  Entries are selected exactly as the runner selects them: first qualifying
  minute per (window, coin), capped per window, window-equal-weighted.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260813)
D = pd.read_parquet(OUT / "grid.parquet")
QTY = 30
days = sorted(D.day.unique())


def entries(vmin, cap=3, band=(0.65, 0.85), tw=(8, 14)):
    q = D[(D.px >= band[0]) & (D.px < band[1]) & (D.minute >= tw[0])
          & (D.minute <= tw[1]) & (D.vol >= vmin)]
    e = (q.sort_values("minute", ascending=False)
           .groupby(["wkey", "coin"], as_index=False).first())
    return (e.sort_values(["wkey", "minute"], ascending=[True, False])
              .groupby("wkey", as_index=False).head(cap))


E = entries(2000)
E["pb"] = (E.px * 20).astype(int)
print(f"INSIDE THE LIVE FILTER: {len(E):,} entries, {E.wkey.nunique():,} windows, "
      f"{E.day.nunique()} days")
print(f"  win {E.won.mean():.4f}   edge {E.edge.mean()*100:+.2f}c   "
      f"coins {sorted(E.coin.unique())}\n")


def dayboot(df, mask_col, n=5000):
    ds = sorted(df.day.unique())
    g = {d: gg for d, gg in df.groupby("day")}
    out = []
    for _ in range(n):
        s = pd.concat([g[ds[k]] for k in RNG.choice(len(ds), len(ds), True)])
        a, b = s[~s[mask_col]], s[s[mask_col]]
        if len(a) > 30 and len(b) > 30:
            out.append((b.edge.mean() - a.edge.mean()) * 100)
    return np.sort(np.array(out))


print("=== LONGSHOT SHARE, INSIDE THE LIVE FILTER ===")
E["hi"] = E.groupby(["coin", "pb"]).frac_long.transform(lambda s: s > s.median())
a, b = E[~E.hi], E[E.hi]
bs = dayboot(E, "hi")
lo, hi = bs[int(.025 * len(bs))], bs[int(.975 * len(bs))]
p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
print(f"  low  frac_long: {len(a):>5,} entries  win {a.won.mean():.4f}  "
      f"edge {a.edge.mean()*100:+.2f}c")
print(f"  high frac_long: {len(b):>5,} entries  win {b.won.mean():.4f}  "
      f"edge {b.edge.mean()*100:+.2f}c")
print(f"  DIFF {(b.edge.mean()-a.edge.mean())*100:+.2f}c   "
      f"95% CI [{lo:+.2f}, {hi:+.2f}]   p {p:.4f}")

print("\n=== QUARTILES OF LONGSHOT SHARE, inside the live filter ===")
E["fq"] = E.groupby(["coin", "pb"]).frac_long.transform(
    lambda s: pd.qcut(s.rank(method="first"), 4, labels=False, duplicates="drop"))
print(f"{'q':>3} {'n':>6} {'frac_long':>10} {'avg px':>8} {'win':>8} "
      f"{'implied':>9} {'excess':>8} {'edge':>9} {'$/day @30':>10}")
for q, g in E.dropna(subset=["fq"]).groupby("fq"):
    print(f"{int(q):>3} {len(g):>6,} {g.frac_long.mean():>10.4f} "
          f"{g.px.mean()*100:>7.1f}c {g.won.mean():>8.4f} {g.px.mean():>9.4f} "
          f"{(g.won.mean()-g.px.mean())*100:>+7.2f}c {g.edge.mean()*100:>+9.2f} "
          f"{g.edge.sum()*QTY/len(days):>+10.2f}")

print("\n=== HORSE RACE at fixed volume (3x3, inside the live filter) ===")
E["vq"] = E.groupby(["coin", "pb"]).vol.transform(
    lambda s: pd.qcut(s.rank(method="first"), 3, labels=False, duplicates="drop"))
E["lq"] = E.groupby(["coin", "pb"]).frac_long.transform(
    lambda s: pd.qcut(s.rank(method="first"), 3, labels=False, duplicates="drop"))
print(f"{'':>20}" + "".join(f"{'vol q' + str(i):>14}" for i in range(3)))
for lq in range(3):
    row = ""
    for vq in range(3):
        g = E[(E.lq == lq) & (E.vq == vq)]
        row += (f"{g.edge.mean()*100:>+8.2f}({len(g):>3})" if len(g) > 30
                else f"{'-':>14}")
    print(f"  frac_long q{lq}       {row}")
print("\n  DOWN a column = effect of longshot share at fixed volume.")

print("\n=== WALK-FORWARD: would filtering on longshot share have paid? ===")
CUT = days[len(days) // 2]
thr = E[E.day < CUT].groupby(["coin", "pb"]).frac_long.median()
te = E[E.day >= CUT].copy()
te["thr"] = [thr.get(k, np.nan) for k in zip(te.coin, te.pb)]
te = te.dropna(subset=["thr"])
te["keep"] = te.frac_long > te.thr
k, allr = te[te.keep], te
nd = te.day.nunique()
print(f"  out-of-sample days {nd}")
print(f"  ALL entries          : {len(allr):>5,}  edge {allr.edge.mean()*100:+.2f}c"
      f"   ${allr.edge.sum()*QTY/nd:+.2f}/day")
print(f"  high frac_long only  : {len(k):>5,}  edge {k.edge.mean()*100:+.2f}c"
      f"   ${k.edge.sum()*QTY/nd:+.2f}/day")
print(f"  -> filtering raises edge/contract by "
      f"{(k.edge.mean()-allr.edge.mean())*100:+.2f}c but trades "
      f"{(1-len(k)/len(allr))*100:.0f}% fewer, for "
      f"${(k.edge.sum()-allr.edge.sum())*QTY/nd:+.2f}/day")

print("\n=== AND THE DRAWDOWN QUESTION THE USER ACTUALLY ASKED ===")
for lbl, g in (("all entries", allr), ("high frac_long", k)):
    per = g.groupby("day").edge.sum() * QTY
    loss = g[g.won == 0]
    print(f"  {lbl:>16}: loss rate {1-g.won.mean():.4f}   "
          f"mean loss ${loss.edge.mean()*QTY:+.2f}   "
          f"worst day ${per.min():+.2f}   sd ${per.std():.2f}")
