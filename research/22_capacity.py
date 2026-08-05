"""How large can this strategy get before it eats its own edge?

WHY THIS IS THE BINDING QUESTION
  Capital is NOT the constraint. At qty 30 the account holds ~$110 gross at a
  time and turns it over every 15 minutes, so a few hundred dollars funds the
  whole thing. What limits the strategy is FILL CAPACITY: the edge is payment
  for absorbing impatient longshot buying, so our size cannot exceed the flow
  that is actually arriving. Past that point we either do not get filled (the
  edge is unrealised) or we move the price (the edge is competed away).

  So capacity is set by the longshot-buying volume per window, not by bankroll.

MEASURED HERE
  - longshot volume actually available in the windows the strategy enters
  - our current 30-lot as a share of it
  - the size at which we would become a material fraction of that flow
  - what each size implies for annual P&L, if edge per contract held constant
    (it will not - that is the point)
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
D = pd.read_parquet(OUT / "grid.parquet")
QTY = 30
days = sorted(D.day.unique())

q = D[(D.px >= 0.65) & (D.px < 0.85) & (D.minute >= 8) & (D.minute <= 14)
      & (D.vol >= 2000)]
e = (q.sort_values("minute", ascending=False)
       .groupby(["wkey", "coin"], as_index=False).first())
e = (e.sort_values(["wkey", "minute"], ascending=[True, False])
       .groupby("wkey", as_index=False).head(3))
# vol is cumulative traded contracts at entry; frac_long is the share of that
# volume where the TAKER bought the cheap side - the flow we are paid to absorb
e["long_vol"] = e.vol * e.frac_long
print(f"entries {len(e):,} over {len(days)} days "
      f"({len(e)/len(days):.1f}/day), 5 coins\n")

print("=== FLOW AVAILABLE IN THE WINDOWS WE ENTER (contracts, by entry time) ===")
print(f"{'percentile':>12} {'total vol':>11} {'longshot vol':>13} "
      f"{'our 30 as % of longshot':>25}")
for p in (10, 25, 50, 75, 90):
    tv = np.percentile(e.vol, p)
    lv = np.percentile(e.long_vol, p)
    print(f"{p:>11}th {tv:>11,.0f} {lv:>13,.0f} {QTY/max(lv,1)*100:>24.1f}%")
print(f"\n  median longshot volume per entered window: {e.long_vol.median():,.0f}")
print(f"  our 30-lot is {QTY/e.long_vol.median()*100:.2f}% of the median")

print("\n=== BUT THAT IS CUMULATIVE FLOW; WE ONLY SEE FLOW AFTER WE POST ===")
print("  We rest for a few minutes inside the 8-14 minute band, so the")
print("  relevant pool is the flow arriving in THAT slice, not the whole")
print("  window. Approximating it by the volume accruing between minute 14")
print("  and minute 8:")
# Select the windows on the ENTRY condition, then read both minutes from the
# unfiltered data. Applying `vol >= 2000` at both minutes would drop minute 14
# outright, since cumulative volume is lower earlier in the window.
keys = set(zip(e.wkey, e.coin))
sub = D[D.minute.isin([8, 14])].copy()
sub = sub[[k in keys for k in zip(sub.wkey, sub.coin)]]
piv = sub.pivot_table(index=["wkey", "coin"], columns="minute", values="vol")
piv = piv.dropna()
piv["slice"] = piv[8] - piv[14]
sl = piv["slice"].clip(lower=0)
print(f"\n  windows measured {len(sl):,}")
for p in (10, 25, 50, 75):
    print(f"    {p:>2}th pct volume arriving in the 8-14 min slice: "
          f"{np.percentile(sl, p):>8,.0f} contracts")
med_slice = np.median(sl)
print(f"\n  at the median, {med_slice:,.0f} contracts trade while we rest.")
print(f"  our 30 is {QTY/max(med_slice,1)*100:.1f}% of that. Roughly half of")
print(f"  that flow is longshot-buying, so we are ~{QTY/max(med_slice*0.5,1)*100:.1f}% "
      f"of the flow we are paid to absorb.")

print("\n=== SIZE LADDER: P&L if edge held constant (IT WILL NOT) ===")
print(f"{'qty':>6} {'% of tradeable slice':>21} {'$/day':>10} {'$/year':>12} "
      f"{'verdict':>28}")
edge_c = 11.29           # backtest edge at this configuration
ent_day = len(e) / len(days)
for qty in (30, 60, 100, 200, 500, 1000):
    share = qty / max(med_slice * 0.5, 1) * 100
    day = ent_day * qty * edge_c / 100
    if share < 5:
        v = "plausible, flow absorbs it"
    elif share < 15:
        v = "fill rate will degrade"
    elif share < 40:
        v = "we ARE the flow; edge decays"
    else:
        v = "not executable"
    print(f"{qty:>6} {share:>20.1f}% {day:>10,.0f} {day*365:>12,.0f} {v:>28}")

print("\n=== WHAT $100k/YEAR REQUIRES ===")
need_day = 100_000 / 365
print(f"  $100,000/year = ${need_day:.0f}/day")
qty_needed = need_day / (ent_day * edge_c / 100)
print(f"  at {ent_day:.0f} entries/day and +{edge_c:.2f}c/contract that needs "
      f"qty {qty_needed:.0f}")
print(f"  = {qty_needed/max(med_slice*0.5,1)*100:.1f}% of the longshot flow "
      f"arriving while we rest")
print(f"  capital to fund it: ~${qty_needed*0.73*3:,.0f} gross at any moment")
print("\n  AT THE LIVE EDGE (+5.60c/contract, not the backtest's +11.29c):")
qn2 = need_day / (ent_day * 5.60 / 100)
print(f"  needs qty {qn2:.0f} = "
      f"{qn2/max(med_slice*0.5,1)*100:.1f}% of the flow")
