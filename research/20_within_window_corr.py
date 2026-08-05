"""Do the coins inside one window lose TOGETHER?

WHY THIS DECIDES THE GROSS CAP
  Live, positions from the previous window have not settled when the next
  window opens, so raising the per-window cap to 3 does nothing unless
  MAX_GROSS rises too. Raising MAX_GROSS concentrates more capital into
  simultaneously-open positions.

  That is only safe if those positions fail independently. If the five coins
  share one market-wide move - and they plainly might, they are all crypto -
  then N open positions is closer to ONE bet of size N, and the tail is far
  fatter than the per-entry win rate suggests.

  This measures the joint loss distribution directly rather than assuming it,
  and prices the worst case that actually matters: how much of the account can
  a single bad window take.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
D = pd.read_parquet(OUT / "grid.parquet")
QTY = 30


def take(vmin, cap=5):
    q = D[(D.px >= 0.65) & (D.px < 0.85) & (D.minute >= 8) & (D.minute <= 14)
          & (D.vol >= vmin)]
    e = (q.sort_values("minute", ascending=False)
           .groupby(["wkey", "coin"], as_index=False).first())
    return (e.sort_values(["wkey", "minute"], ascending=[True, False])
              .groupby("wkey", as_index=False).head(cap))


for vmin in (500, 2000):
    T = take(vmin)
    W = T.groupby("wkey").agg(n=("won", "size"), k=("won", "sum"),
                              pnl=("edge", "sum"))
    print(f"=== vmin {vmin}:  {len(W):,} windows, {len(T):,} entries, "
          f"per-entry win {T.won.mean():.4f} ===")
    print(f"{'entries':>8} {'windows':>8}  " +
          "".join(f"{'losses=' + str(i):>11}" for i in range(6)))
    for n, g in W.groupby("n"):
        if n < 2 or len(g) < 15:
            continue
        row = ""
        for L in range(6):
            if L > n:
                row += f"{'':>11}"
                continue
            obs = ((g.n - g.k) == L).mean()
            row += f"{obs:>11.3f}"
        print(f"{int(n):>8} {len(g):>8}  {row}")
    # what independence would predict for the all-lose corner
    p = 1 - T.won.mean()
    print(f"{'':>8} {'INDEP':>8}  " +
          "".join(f"{'':>11}" for _ in range(0)) )
    for n, g in W.groupby("n"):
        if n < 2 or len(g) < 15:
            continue
        obs_all = ((g.k == 0)).mean()
        ind_all = p ** n
        ratio = obs_all / ind_all if ind_all > 0 else float("nan")
        print(f"   n={int(n)}: P(all {int(n)} lose) observed {obs_all:.4f}  "
              f"independent {ind_all:.4f}  ratio {ratio:>5.1f}x  "
              f"({int(obs_all*len(g))} of {len(g)} windows)")
    print()

print("=== WHAT A SINGLE BAD WINDOW COSTS, at qty 30 ===")
T = take(2000)
W = T.groupby("wkey").agg(n=("won", "size"), pnl=("edge", "sum"),
                          cost=("px", "sum"))
W["dollars"] = W.pnl * QTY
W["exposure"] = W.cost * QTY
print(f"  windows {len(W):,}   worst single window ${W.dollars.min():+.2f}")
for q in (0.1, 1, 5, 25, 50):
    print(f"  {q:>4.1f}th pct window P&L  ${np.percentile(W.dollars, q):+8.2f}")
print(f"\n  worst 5 windows: "
      f"{[f'{x:+.0f}' for x in W.dollars.nsmallest(5)]}")

print("\n=== SIMULTANEOUS EXPOSURE: what if TWO windows overlap ===")
print("  live, the previous window has not settled, so two windows are open at")
print("  once. Pairing consecutive windows to price that:")
ws = sorted(W.index)
pair = [W.loc[a, "dollars"] + W.loc[b, "dollars"]
        for a, b in zip(ws, ws[1:])]
pair = np.array(pair)
print(f"  worst consecutive-window pair  ${pair.min():+.2f}")
for q in (0.1, 1, 5):
    print(f"  {q:>4.1f}th pct pair P&L        ${np.percentile(pair, q):+8.2f}")
eq = 338.81
print(f"\n  against equity ${eq:.2f}:")
print(f"    worst observed pair = {abs(pair.min())/eq*100:.1f}% of the account")
print(f"    1st-pct pair        = {abs(np.percentile(pair,1))/eq*100:.1f}%")
