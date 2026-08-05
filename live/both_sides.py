"""Three questions: trade the OTHER side, cap correlated coins, and do bands move?

1. THE OPPOSITE TRADE
   As a maker we buy at the BID on whichever side we choose. If the favourite's
   bid is 0.70 and its ask 0.72, the longshot's bid is 0.28. Then

       favourite edge = P(fav wins) - 0.70
       longshot  edge = (1 - P(fav wins)) - 0.28 = 0.72 - P(fav wins)

   and the two SUM TO THE SPREAD. Whenever the favourite side is only mildly
   negative, the longshot side is positive by construction, because a maker
   earns the spread from either direction.

   The whole strategy rests on longshots being OVERPRICED - that is the premium
   we collect. So on average the favourite side must win. But "on average" is
   not "always", and if there is an identifiable condition where the favourite
   is the overpriced one, we should be taking the other side there rather than
   merely skipping it. Skipping earns zero; reversing earns the premium.

2. CORRELATION-AWARE POSITION LIMITS
   Coins inside one window correlate 0.768, so three positions is close to one
   bet at triple size - effective independent bets are 1.24, not 3. Capping at
   one coin per window would make each window a genuine single bet. Live data
   hinted cap 1 beat cap 3 ($203 vs $129) but the interval was far too wide.
   73 days settles it.

3. DO THE BANDS MOVE?
   The 65-80c band is fixed. If the profitable region drifts - 70-75c one week,
   65-70c the next - then a fixed band is sometimes pointed at the wrong place,
   and that is invisible in a pooled average.

WHAT WOULD MAKE ANY OF THESE REAL
   Same bar as everything else: discards verifiably negative, dollars not
   cents, day-clustered. And for the reversal specifically, the condition has
   to be knowable BEFORE the entry - not defined by the outcome.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

LIVE = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261062)
QTY = 20

D = pd.read_parquet(LIVE / "ladder_paths.parquet").drop_duplicates(
    subset=["ticker"])
rows = []
for r in D.itertuples():
    p = sorted(json.loads(r.path), key=lambda x: -x["ml"])
    entry = next((k for k in p if 8 <= k["ml"] <= 14), None)
    if entry is None:
        continue
    yb, ya = entry["bc"], entry["ac"]
    if yb <= 0 or ya >= 1 or yb >= ya:
        continue
    fav_yes = yb >= 0.5
    fav_bid = yb if fav_yes else 1.0 - ya          # buy favourite at its bid
    lng_bid = (1.0 - ya) if fav_yes else yb        # buy longshot at its bid
    fav_won = 1 if (("yes" if fav_yes else "no") == r.result) else 0
    rows.append(dict(ticker=r.ticker, coin=r.coin, date=r.date,
                     wkey=r.ticker.split("-")[1], ml=entry["ml"],
                     fav_bid=fav_bid, lng_bid=lng_bid,
                     spread=(ya - yb),
                     fav_won=fav_won,
                     fav_edge=fav_won - fav_bid,
                     lng_edge=(1 - fav_won) - lng_bid))
E = pd.DataFrame(rows)
E = E[(E.fav_bid >= 0.60) & (E.fav_bid < 0.90)].copy()
days = sorted(E.date.unique())
print(f"entries {len(E):,}   days {len(days)}   windows {E.wkey.nunique():,}\n")


def dayboot(df, fn, n=6000):
    gd = {d: g for d, g in df.groupby("date")}
    ds = sorted(gd)
    v = []
    for _ in range(n):
        s = pd.concat([gd[ds[i]] for i in RNG.integers(0, len(ds), len(ds))])
        x = fn(s)
        if x is not None and not (isinstance(x, float) and np.isnan(x)):
            v.append(x)
    return np.sort(np.array(v))


print("=" * 78)
print("1. FAVOURITE vs LONGSHOT, SAME MARKETS, BOTH AS MAKER")
print("=" * 78)
B = E[(E.fav_bid >= 0.65) & (E.fav_bid < 0.80)]
for lbl, col, px in (("buy FAVOURITE at its bid", "fav_edge", "fav_bid"),
                     ("buy LONGSHOT  at its bid", "lng_edge", "lng_bid")):
    bs = dayboot(B, lambda s, c=col: s[c].mean() * 100)
    print(f"  {lbl}: n {len(B)}  avg px {B[px].mean()*100:>5.1f}c  "
          f"edge {B[col].mean()*100:>+6.2f}c  "
          f"95% CI [{bs[150]:+.2f}, {bs[5849]:+.2f}]")
print(f"\n  they sum to the spread: {(B.fav_edge + B.lng_edge).mean()*100:+.2f}c "
      f"vs measured spread {B.spread.mean()*100:.2f}c")

print("\n" + "=" * 78)
print("   WHERE IS THE FAVOURITE SIDE THE WRONG SIDE?")
print("=" * 78)
print(f"{'condition':>22} {'n':>7} {'fav edge':>10} {'longshot edge':>15} "
      f"{'better side':>13}")
E["pxb"] = pd.cut(E.fav_bid, [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
for k, g in E.groupby("pxb", observed=True):
    if len(g) < 80:
        continue
    f, l = g.fav_edge.mean() * 100, g.lng_edge.mean() * 100
    print(f"{str(k):>22} {len(g):>7} {f:>+9.2f}c {l:>+14.2f}c "
          f"{('FAVOURITE' if f > l else 'LONGSHOT'):>13}")
print()
E["sb"] = pd.cut(E.spread * 100, [0, 1.01, 2.01, 3.01, 20])
for k, g in E.groupby("sb", observed=True):
    if len(g) < 80:
        continue
    f, l = g.fav_edge.mean() * 100, g.lng_edge.mean() * 100
    print(f"{'spread ' + str(k) + 'c':>22} {len(g):>7} {f:>+9.2f}c "
          f"{l:>+14.2f}c {('FAVOURITE' if f > l else 'LONGSHOT'):>13}")

print("\n" + "=" * 78)
print("2. ONE COIN PER WINDOW vs THREE (coins correlate 0.768)")
print("=" * 78)
W = B.sort_values(["wkey", "fav_bid"], ascending=[True, False])
print(f"{'rule':>26} {'entries':>9} {'edge/ct':>9} {'$/day':>9} "
      f"{'vs cap 3':>22}")
cap3 = W.groupby("wkey", as_index=False).head(3)
d3 = cap3.groupby("date").fav_edge.sum().reindex(days, fill_value=0) * QTY
print(f"{'cap 3 (deployed)':>26} {len(cap3):>9} "
      f"{cap3.fav_edge.mean()*100:>+8.2f}c {d3.mean():>+9.2f} {'-':>22}")
for cap in (1, 2):
    k = W.groupby("wkey", as_index=False).head(cap)
    d = k.groupby("date").fav_edge.sum().reindex(days, fill_value=0) * QTY
    diff = (d - d3).values
    bs = np.sort(np.array([RNG.choice(diff, len(diff), True).mean()
                           for _ in range(8000)]))
    print(f"{'cap ' + str(cap):>26} {len(k):>9} {k.fav_edge.mean()*100:>+8.2f}c "
          f"{d.mean():>+9.2f} {diff.mean():>+7.2f} "
          f"[{bs[200]:+.0f},{bs[7799]:+.0f}] P<=0 {(bs <= 0).mean():.3f}")
mar = W.groupby("wkey", as_index=False).head(3).groupby("wkey").tail(2)
print(f"\n  the entries cap 3 admits beyond the first: n {len(mar)}  "
      f"edge {mar.fav_edge.mean()*100:+.2f}c")
bs = dayboot(mar, lambda s: s.fav_edge.mean() * 100)
print(f"  95% CI [{bs[150]:+.2f}, {bs[5849]:+.2f}]   "
      f"P(marginal entries NEGATIVE) {(bs < 0).mean():.4f}")

print("\n" + "=" * 78)
print("3. DOES THE PROFITABLE BAND MOVE OVER TIME?")
print("=" * 78)
E["wk"] = pd.to_datetime(E.date).dt.isocalendar().week
print(f"{'week':>6} {'n':>6} " + "".join(f"{b:>11}" for b in
      ("60-65", "65-70", "70-75", "75-80", "80-85")) + f" {'best':>8}")
for w, g in E.groupby("wk"):
    if len(g) < 60:
        continue
    cells, best, bv = [], None, -99
    for lo, hi in ((0.60, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 0.80),
                   (0.80, 0.85)):
        s = g[(g.fav_bid >= lo) & (g.fav_bid < hi)]
        if len(s) < 12:
            cells.append(f"{'-':>11}")
            continue
        v = s.fav_edge.mean() * 100
        cells.append(f"{v:>+10.2f}c")
        if v > bv:
            bv, best = v, f"{lo*100:.0f}-{hi*100:.0f}"
    print(f"{int(w):>6} {len(g):>6} " + "".join(cells) + f" {best or '-':>8}")
print("\n  If 'best' jumps around week to week, the optimum is not stable and a")
print("  fixed band is sometimes aimed at the wrong place. If it is usually the")
print("  same slice, the band is fine and the variation is noise.")
