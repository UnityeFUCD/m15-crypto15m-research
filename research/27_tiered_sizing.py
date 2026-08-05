"""Use longshot share to SIZE, not to filter.

THE TRADEOFF FILTERING CREATES
  High longshot share carries +15.81c against +12.56c out of sample, and cuts
  the loss rate from 13.9% to 10.7%. But filtering on it throws away 36% of
  entries and costs $76/day. Capital is not the binding constraint here - the
  gross cap is rarely reached - so declining an entry buys nothing back.

THE ALTERNATIVE
  Keep every entry, but size it by its longshot share. That captures the edge
  difference without surrendering the volume of trades. Position size stays
  capped at 30 - the top tier is NOT raised, the weak tier is lowered - so this
  cannot increase maximum exposure.

SCORED ON WHAT WAS ASKED FOR
  edge per contract, loss rate, daily sd, worst day, drawdown, and $/day, all
  walk-forward: tiers are set on the first half of days and applied to the
  second.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260814)
D = pd.read_parquet(OUT / "grid.parquet")
days = sorted(D.day.unique())
CUT = days[len(days) // 2]

q = D[(D.px >= 0.65) & (D.px < 0.85) & (D.minute >= 8) & (D.minute <= 14)
      & (D.vol >= 2000)]
E = (q.sort_values("minute", ascending=False)
       .groupby(["wkey", "coin"], as_index=False).first())
E = (E.sort_values(["wkey", "minute"], ascending=[True, False])
       .groupby("wkey", as_index=False).head(3)).copy()
E["pb"] = (E.px * 20).astype(int)

# tier thresholds learned on PAST days only
thr = E[E.day < CUT].groupby(["coin", "pb"]).frac_long.median()
te = E[E.day >= CUT].copy()
te["thr"] = [thr.get(k, np.nan) for k in zip(te.coin, te.pb)]
te = te.dropna(subset=["thr"]).copy()
te["hi"] = te.frac_long > te.thr
nd = te.day.nunique()
print(f"out-of-sample: {len(te):,} entries over {nd} days "
      f"({len(te)/nd:.0f}/day)\n")

MAX_GROSS = 110.0


def sim(qty_hi, qty_lo, label):
    t = te.copy()
    t["qty"] = np.where(t.hi, qty_hi, qty_lo)
    keep = []
    for _, g in t.groupby("wkey"):
        gross = 0.0
        for r in g.itertuples():
            c = r.qty * r.px
            if gross + c > MAX_GROSS or r.qty <= 0:
                continue
            gross += c
            keep.append((r.day, r.edge * r.qty, r.qty, r.won, r.px * r.qty))
    k = pd.DataFrame(keep, columns=["day", "pnl", "qty", "won", "cost"])
    per = k.groupby("day").pnl.sum().reindex(sorted(te.day.unique()),
                                             fill_value=0.0)
    eq = np.concatenate([[0.0], np.cumsum(per.values)])
    dd = (np.maximum.accumulate(eq) - eq).max()
    losses = k[k.won == 0]
    return dict(label=label, n=len(k), contracts=k.qty.sum(),
                edge=k.pnl.sum() / max(k.qty.sum(), 1) * 100,
                day=per.mean(), sd=per.std(ddof=1), worst=per.min(),
                dd=dd, lossrate=1 - k.won.mean(),
                meanloss=losses.pnl.mean() if len(losses) else 0.0,
                maxexp=k.cost.max())


rows = [
    sim(30, 30, "CURRENT  flat 30"),
    sim(30, 0,  "filter   30 / drop low"),
    sim(30, 20, "tiered   30 / 20"),
    sim(30, 15, "tiered   30 / 15"),
    sim(30, 10, "tiered   30 / 10"),
    sim(25, 15, "tiered   25 / 15"),
]
print(f"{'configuration':>24} {'n':>5} {'contracts':>10} {'edge c':>8} "
      f"{'$/day':>9} {'sd':>8} {'worst day':>10} {'maxDD':>8} {'loss rate':>10}")
for r in rows:
    print(f"{r['label']:>24} {r['n']:>5} {r['contracts']:>10,.0f} "
          f"{r['edge']:>+8.2f} {r['day']:>+9.2f} {r['sd']:>8.2f} "
          f"{r['worst']:>+10.2f} {r['dd']:>8.2f} {r['lossrate']:>10.4f}")

base = rows[0]
print(f"\n=== VERSUS THE CURRENT FLAT-30 CONFIG ===")
for r in rows[1:]:
    print(f"  {r['label']:>24}  $/day {r['day']-base['day']:>+7.2f}  "
          f"edge {r['edge']-base['edge']:>+5.2f}c  "
          f"sd {r['sd']-base['sd']:>+7.2f}  "
          f"loss rate {r['lossrate']-base['lossrate']:>+7.4f}  "
          f"max exposure ${r['maxexp']:.2f}")

print("\n=== IS THE SIGNAL EVEN AVAILABLE LIVE? ===")
print("  frac_long = cumulative longshot-buying volume / cumulative volume, at")
print("  the entry instant. The runner currently reads `volume_fp` off the")
print("  market object, which gives the DENOMINATOR only. The numerator needs")
print("  the trade feed (GET /markets/trades) per candidate market, so this")
print("  costs roughly one extra API call per candidate per scan.")
print("  Measured API latency this session: ~13ms for a markets call, against")
print("  a 5s cycle and a 20.4s median entry lateness, so the added latency is")
print("  not material. Rate limits are the real question, not speed.")

print("\n=== WHAT IT DOES NOT FIX ===")
lo = te[~te.hi]
print(f"  low-share entries still carry a POSITIVE edge "
      f"({lo.edge.mean()*100:+.2f}c on {len(lo)} entries), so they are not")
print("  losers to be eliminated - they are simply worth less per dollar of")
print("  risk. That is why sizing beats filtering here.")
