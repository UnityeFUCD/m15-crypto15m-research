"""Where inside the band do we actually make money? 73 days, real book.

THE QUESTION
  Fill bias is structural and cannot be priced away - fill|LOSE is 1.0000 at
  every posting price, so we always take the losers. Realised edge is therefore

      P(win) * fill|WIN  minus  price,  all over  P(fill)

  which for a 0.75 population win rate at 69.7c comes to +2.3c. That matches
  live. To do better we need a HIGHER WIN RATE AT THE SAME PRICE, or the SAME
  WIN RATE AT A LOWER PRICE.

  Both are choices about WHERE INSIDE the existing band to concentrate - which
  price, which minute, which coin. Nothing here widens the 65-80c band or the
  8-14 minute window; it only asks which part of them is worth having.

WHY THIS HAS TO BE REDONE
  Every previous answer to this came from grid.parquet, which the exchange's
  own data has now contradicted twice in the same direction:

      population edge   grid +11.76c   reconstruction +3.5c
      00-07 hour effect grid  +8.12c   reconstruction +0.09c

  It overstates effects roughly threefold. Any band or timing optimum derived
  from it is suspect, including the ones currently deployed.

  The stored paths carry the full per-minute book for ~4,000 markets over 73
  days, so the entry can be re-simulated at ANY minute and ANY price and scored
  against the real settlement. That is the surface this builds.

THE BAR
  A concentration rule discards entries, so what it discards must be verifiably
  NEGATIVE, and it must win in DOLLARS after the lost volume - the trap that
  killed the hours filter, the cap changes and the coin drops.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

LIVE = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261061)
QTY = 20

D = pd.read_parquet(LIVE / "ladder_paths.parquet").drop_duplicates(
    subset=["ticker"])
print(f"paths: {len(D):,}   days {D.date.nunique()}")

rows = []
for r in D.itertuples():
    p = sorted(json.loads(r.path), key=lambda x: -x["ml"])
    for k in p:
        ml = k["ml"]
        if not (6 <= ml <= 14):
            continue
        yb, ya = k["bc"], k["ac"]
        if yb <= 0 or ya >= 1 or yb >= ya:
            continue
        side, bid = ("yes", yb) if yb >= 0.5 else ("no", 1.0 - ya)
        if not (0.60 <= bid < 0.90):
            continue
        rows.append(dict(ticker=r.ticker, coin=r.coin, date=r.date,
                         ml=round(ml), side=side, px=bid,
                         won=1 if side == r.result else 0,
                         edge=(1 if side == r.result else 0) - bid))
E = pd.DataFrame(rows)
print(f"observations: {len(E):,}\n")


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
print("EDGE BY ENTRY PRICE  (all minutes 8-14, the deployed window)")
print("=" * 78)
B = E[(E.ml >= 8) & (E.ml <= 14)]
B = B.sort_values(["ticker", "ml"], ascending=[True, False]).groupby(
    "ticker", as_index=False).first()
print(f"{'price':>12} {'n':>7} {'win':>8} {'edge/ct':>9} {'95% CI':>20}")
for lo, hi in ((0.60, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 0.80),
               (0.80, 0.85), (0.85, 0.90)):
    g = B[(B.px >= lo) & (B.px < hi)]
    if len(g) < 60:
        continue
    bs = dayboot(g, lambda s: s.edge.mean() * 100)
    print(f"{f'{lo*100:.0f}-{hi*100:.0f}c':>12} {len(g):>7} {g.won.mean():>8.4f} "
          f"{g.edge.mean()*100:>+8.2f}c "
          f"[{bs[150]:>+7.2f},{bs[5849]:>+7.2f}]")
print("\n  The deployed band is 65-80c. If the 60-65c slice is strongly")
print("  positive that is a widening, which is off the table. What matters")
print("  here is whether any slice INSIDE 65-80 is negative.")

print("\n" + "=" * 78)
print("EDGE BY MINUTES LEFT  (inside the deployed 65-80c band)")
print("=" * 78)
M = E[(E.px >= 0.65) & (E.px < 0.80)]
print(f"{'min left':>10} {'n':>7} {'avg px':>8} {'win':>8} {'edge/ct':>9} "
      f"{'95% CI':>20}")
for m in range(14, 5, -1):
    g = M[M.ml == m]
    if len(g) < 60:
        continue
    bs = dayboot(g, lambda s: s.edge.mean() * 100)
    print(f"{m:>10} {len(g):>7} {g.px.mean()*100:>7.1f}c {g.won.mean():>8.4f} "
          f"{g.edge.mean()*100:>+8.2f}c [{bs[150]:>+7.2f},{bs[5849]:>+7.2f}]")
print("\n  The runner enters at 8-14. Minutes 6-7 are shown only to see the")
print("  shape - using them would widen the window, which is off the table.")

print("\n" + "=" * 78)
print("EDGE BY COIN  (73 days - the 3-day answer said drop nothing)")
print("=" * 78)
print(f"{'coin':>7} {'n':>7} {'avg px':>8} {'win':>8} {'edge/ct':>9} "
      f"{'95% CI':>20} {'P(<0)':>8}")
for c, g in B.groupby("coin"):
    if len(g) < 60:
        continue
    bs = dayboot(g, lambda s: s.edge.mean() * 100)
    print(f"{c:>7} {len(g):>7} {g.px.mean()*100:>7.1f}c {g.won.mean():>8.4f} "
          f"{g.edge.mean()*100:>+8.2f}c [{bs[150]:>+7.2f},{bs[5849]:>+7.2f}] "
          f"{(bs < 0).mean():>8.4f}")

print("\n" + "=" * 78)
print("THE SURFACE - price x minute, inside the deployed box")
print("=" * 78)
S = E[(E.px >= 0.65) & (E.px < 0.80) & (E.ml >= 8) & (E.ml <= 14)].copy()
S["pb"] = pd.cut(S.px, [0.65, 0.70, 0.75, 0.80])
piv = S.pivot_table(index="ml", columns="pb", values="edge",
                    aggfunc="mean", observed=True) * 100
cnt = S.pivot_table(index="ml", columns="pb", values="edge",
                    aggfunc="size", observed=True)
print("  edge/contract:")
print(piv.round(2).to_string())
print("\n  n:")
print(cnt.to_string())

print("\n" + "=" * 78)
print("DOES CONCENTRATING PAY, IN DOLLARS?")
print("=" * 78)
days = sorted(B.date.unique())
base = B.groupby("date").edge.sum().reindex(days, fill_value=0) * QTY
print(f"{'rule':>28} {'entries':>9} {'edge/ct':>9} {'$/day':>9} "
      f"{'vs keep-all':>22}")
print(f"{'keep all 65-80c':>28} {len(B):>9} {B.edge.mean()*100:>+8.2f}c "
      f"{base.mean():>+9.2f} {'-':>22}")
for lbl, sel in (("only 65-70c", (B.px < 0.70)),
                 ("only 70-80c", (B.px >= 0.70)),
                 ("only 65-75c", (B.px < 0.75)),
                 ("only 75-80c", (B.px >= 0.75))):
    k = B[sel]
    if len(k) < 100:
        continue
    d = k.groupby("date").edge.sum().reindex(days, fill_value=0) * QTY
    diff = (d - base).values
    bs = np.sort(np.array([RNG.choice(diff, len(diff), True).mean()
                           for _ in range(8000)]))
    cut = B[~sel]
    print(f"{lbl:>28} {len(k):>9} {k.edge.mean()*100:>+8.2f}c "
          f"{d.mean():>+9.2f} {diff.mean():>+7.2f} "
          f"[{bs[200]:+.0f},{bs[7799]:+.0f}] P<=0 {(bs <= 0).mean():.3f}")
    print(f"{'':>28} discards: n {len(cut)} edge {cut.edge.mean()*100:+.2f}c")
