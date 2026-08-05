"""Is the cap-2 -> cap-3 gain real, or one or two lucky days?

A 20% jump in $/day from a single extra entry per window is exactly the shape
of a result that turns out to be two outlier windows, so it gets checked day by
day and by day-clustered bootstrap before it goes anywhere near the runner.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260809)
D = pd.read_parquet(OUT / "grid.parquet")
QTY, MAX_GROSS = 30, 110.0
days = sorted(D.day.unique())


def take(plo, phi, mlo, mhi, vmin, cap):
    q = D[(D.px >= plo) & (D.px < phi) & (D.minute >= mlo) & (D.minute <= mhi)
          & (D.vol >= vmin)]
    e = (q.sort_values("minute", ascending=False)
           .groupby(["wkey", "coin"], as_index=False).first())
    e = (e.sort_values(["wkey", "minute"], ascending=[True, False])
           .groupby("wkey", as_index=False).head(cap))
    keep = []
    for _, g in e.groupby("wkey"):
        gross, rank = 0.0, 0
        for _, r in g.iterrows():
            c = QTY * r.px
            if gross + c > MAX_GROSS:
                continue
            gross += c
            rank += 1
            keep.append((r.day, r.wkey, r.coin, r.px, r.edge, r.won, rank))
    return pd.DataFrame(keep, columns=["day", "wkey", "coin", "px", "edge",
                                       "won", "rank"])


T = take(0.65, 0.85, 8, 14, 2000, 3)          # rank identifies the Nth entry
print("=== EDGE BY ENTRY RANK WITHIN A WINDOW (vmin 2000, 65-85c, 8-14) ===")
print(f"{'rank':>6} {'n':>6} {'ent/day':>8} {'avg px':>8} {'win':>7} {'edge c':>8} "
      f"{'$/day @30':>10}")
for r, g in T.groupby("rank"):
    print(f"{int(r):>6} {len(g):>6,} {len(g)/len(days):>8.1f} "
          f"{g.px.mean()*100:>7.1f}c {g.won.mean():>7.4f} "
          f"{g.edge.mean()*100:>+8.2f} {g.edge.sum()*QTY/len(days):>+10.2f}")
print("  rank 3 is exactly the entry that cap=3 adds and cap=2 refuses.")

print("\n=== DAY BY DAY ===")
print(f"{'day':>12} {'cap2 $':>9} {'cap3 $':>9} {'diff':>9} {'rank3 n':>8} "
      f"{'rank3 edge':>11}")
d2, d3 = [], []
for d in days:
    g = T[T.day == d]
    s3 = g.edge.sum() * QTY
    s2 = g[g["rank"] <= 2].edge.sum() * QTY
    m = g[g["rank"] == 3]
    d2.append(s2)
    d3.append(s3)
    print(f"{d:>12} {s2:>+9.2f} {s3:>+9.2f} {s3-s2:>+9.2f} {len(m):>8} "
          f"{(m.edge.mean()*100 if len(m) else 0):>+11.2f}")
d2, d3 = np.array(d2), np.array(d3)
print(f"\n  cap3 beat cap2 on {(d3 > d2).sum()}/{len(days)} days")
print(f"  mean ${d3.mean():+.2f} vs ${d2.mean():+.2f}/day  "
      f"(+${d3.mean()-d2.mean():.2f})")

r3 = T[T["rank"] == 3]
g = {d: gg.edge.to_numpy() for d, gg in r3.groupby("day")}
ds = list(g)
bs = np.sort(np.array([np.concatenate(
    [g[ds[k]] for k in RNG.choice(len(ds), len(ds), True)]).mean()
    for _ in range(6000)]))
lo, hi = bs[int(.025 * len(bs))] * 100, bs[int(.975 * len(bs))] * 100
print(f"\n  rank-3 edge {r3.edge.mean()*100:+.2f}c  "
      f"day-clustered 95% CI [{lo:+.2f}, {hi:+.2f}]  "
      f"p(<=0) {(bs <= 0).mean():.4f}")
print(f"  rank-3 entries come from {r3.wkey.nunique():,} distinct windows on "
      f"{r3.day.nunique()} days")
top = r3.reindex(r3.edge.abs().sort_values(ascending=False).index).head(3)
print(f"  3 largest single contributions: "
      f"{[f'{x.coin} {x.edge*100:+.0f}c' for _, x in top.iterrows()]}")
print(f"  edge excluding those 3: "
      f"{r3.drop(top.index).edge.mean()*100:+.2f}c  "
      f"(if this collapses, the gain was outliers)")
