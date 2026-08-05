"""What the 80-85c band REALLY does to the book.

THE USER'S INSIGHT, WHICH IS THE RIGHT FRAME
  A binary bought at price p breaks even at a win rate of exactly p. So each
  band carries its own hurdle:

      65c fill -> must win 65% of the time just to break even
      82c fill -> must win 82%

  A band can post a high win rate and still be the worst thing in the book, if
  the hurdle it must clear is higher still. Reporting raw win rates hides this
  completely: 0.90 looks better than 0.75 until you notice one needed 0.88 and
  the other needed 0.68.

  This computes SURPLUS = realised win rate - required win rate for every band,
  the margin for error each one has, and what the blended book looks like with
  and without the expensive tail.

ALSO ANSWERED
  How much statistical room each band has. A band with 7 live positions has a
  standard error near 17 percentage points, so a 10-point deficit there means
  almost nothing - while the same deficit on 154 positions would be decisive.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260917)
D = pd.read_parquet(OUT / "grid.parquet")
QTY = 15

q = D[(D.minute >= 8) & (D.minute <= 14) & (D.vol >= 2000)
      & (D.px >= 0.55) & (D.px < 0.95)]
E = (q.sort_values("minute", ascending=False)
       .groupby(["wkey", "coin"], as_index=False).first()).copy()
E["bucket"] = pd.cut(E.px, [.55, .65, .70, .75, .80, .85, .95],
                     labels=["55-65", "65-70", "70-75", "75-80", "80-85", "85-95"])

print("=" * 78)
print("REQUIRED vs REALISED WIN RATE - the hurdle each band must clear")
print("=" * 78)
print(f"{'band':>8} {'n':>6} {'avg px':>8} {'REQUIRED':>10} {'REALISED':>10} "
      f"{'SURPLUS':>9} {'SE':>7} {'surplus/SE':>11} {'edge/ct':>9}")
rows = []
for b, g in E.groupby("bucket", observed=True):
    req = g.px.mean()
    real = g.won.mean()
    se = np.sqrt(real * (1 - real) / len(g))
    rows.append(dict(band=str(b), n=len(g), px=req, req=req, real=real,
                     surplus=real - req, se=se, edge=g.edge.mean()))
    print(f"{str(b):>8} {len(g):>6} {req*100:>7.1f}c {req*100:>9.1f}% "
          f"{real*100:>9.1f}% {(real-req)*100:>+8.1f}pp {se*100:>6.1f}pp "
          f"{(real-req)/se:>11.2f} {g.edge.mean()*100:>+9.2f}c")
R = pd.DataFrame(rows)
print("\n  SURPLUS is the whole story: how far above its own break-even the")
print("  band actually wins. 'surplus/SE' is how many standard errors that is")
print("  - under about 2, the band is not established either way.")

print("\n" + "=" * 78)
print("MARGIN FOR ERROR - how much can the win rate fall before it loses money")
print("=" * 78)
print(f"{'band':>8} {'surplus':>9} {'capital/contract':>17} "
      f"{'$ per 1pp of win rate':>23} {'return on capital':>18}")
for _, r in R.iterrows():
    print(f"{r.band:>8} {r.surplus*100:>+8.1f}pp {r.px*QTY:>16.2f} "
          f"{0.01*QTY:>22.2f} {r.edge/r.px*100:>17.2f}%")
print("\n  Every band earns the same $ per point of win rate at a given size,")
print("  but the expensive ones TIE UP MORE CAPITAL to do it. Return on")
print("  capital is the column that matters when capital is finite.")

print("\n" + "=" * 78)
print("BOOK COMPOSITION - what each band contributes and costs")
print("=" * 78)
E["pnl"] = np.where(E.won == 1, (1 - E.px) * QTY, -E.px * QTY)
E["cap"] = E.px * QTY
tot_pnl, tot_cap = E.pnl.sum(), E.cap.sum()
print(f"{'band':>8} {'n':>6} {'% of entries':>13} {'% of capital':>13} "
      f"{'% of P&L':>10} {'P&L per $ capital':>19}")
for b, g in E.groupby("bucket", observed=True):
    print(f"{str(b):>8} {len(g):>6} {len(g)/len(E)*100:>12.1f}% "
          f"{g.cap.sum()/tot_cap*100:>12.1f}% {g.pnl.sum()/tot_pnl*100:>9.1f}% "
          f"{g.pnl.sum()/g.cap.sum():>18.4f}")
print("\n  A band that takes more capital than it returns P&L is diluting the")
print("  book even when its own P&L is positive.")

print("\n" + "=" * 78)
print("BLENDED BOOK, WITH AND WITHOUT THE EXPENSIVE TAIL")
print("=" * 78)
base = E[(E.px >= 0.65) & (E.px < 0.85)]
for lbl, sub in (("65-85c (current live band)", base),
                 ("65-80c (drop 80-85)", E[(E.px >= 0.65) & (E.px < 0.80)]),
                 ("65-75c (drop 75+)", E[(E.px >= 0.65) & (E.px < 0.75)])):
    req = sub.px.mean()
    real = sub.won.mean()
    print(f"  {lbl:<28} n {len(sub):>4}  required {req*100:>5.1f}%  "
          f"realised {real*100:>5.1f}%  surplus {(real-req)*100:>+5.1f}pp  "
          f"${sub.pnl.sum()/len(sub):>+6.2f}/entry  "
          f"ret/$cap {sub.pnl.sum()/sub.cap.sum()*100:>+5.2f}%")

print("\n" + "=" * 78)
print("WHAT DOES THE LIVE 80-85c DEFICIT ACTUALLY TELL US?")
print("=" * 78)
print("  live 80-85c: 7 positions, realised 71.4%, required ~82%")
n_live, p_live = 7, 0.714
se_live = np.sqrt(p_live * (1 - p_live) / n_live)
print(f"  standard error on 7 positions: {se_live*100:.1f}pp")
print(f"  deficit -10.6pp is {abs(-0.106)/se_live:.2f} standard errors")
bt = E[(E.px >= 0.80) & (E.px < 0.85)]
print(f"\n  backtest 80-85c: {len(bt)} positions, realised {bt.won.mean()*100:.1f}%, "
      f"required {bt.px.mean()*100:.1f}%")
print(f"  backtest surplus {(bt.won.mean()-bt.px.mean())*100:+.1f}pp on "
      f"{len(bt)} observations (SE "
      f"{np.sqrt(bt.won.mean()*(1-bt.won.mean())/len(bt))*100:.1f}pp)")
# how likely is the live result if the backtest rate is true?
from math import comb
p_bt = bt.won.mean()
pk = sum(comb(7, k) * p_bt**k * (1-p_bt)**(7-k) for k in range(0, 6))
print(f"\n  If the backtest win rate {p_bt:.3f} were true, P(<=5 wins in 7) "
      f"= {pk:.4f}")
print("  So the live result is unusual but far from impossible. 7 positions")
print("  cannot overturn 154. The honest verdict is UNRESOLVED, and the band")
print("  needs roughly 60-100 live positions to decide.")
