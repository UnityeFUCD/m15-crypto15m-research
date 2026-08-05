"""Under a BINDING capital cap, which volume floor is right?

THE SHIFT
  The live runner is gross-capped, not opportunity-capped: 85 gross_cap blocks
  against 50 volume-filter skips in four minutes. When capital is the binding
  constraint the objective is not $/day from unlimited entries, it is edge per
  dollar deployed - you hold about the same number of positions either way, so
  you want them to be the best ones available.

  That flips the volume floor from a throughput cost into a free upgrade, but
  only up to the point where the floor is so high that capital sits idle.
  This finds that point.

MODEL FIDELITY
  Positions are held to settlement, and capital is released at the window close
  plus a settlement lag - which is why two windows overlap live and why the
  earlier scripts (which settled instantly at the boundary) overstated
  throughput. The $110 gross cap is enforced continuously, not per window.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
D = pd.read_parquet(OUT / "grid.parquet")
QTY, GROSS = 30, 110.0
LAG_S = 45.0            # observed: settlement credit lands ~30-60s after close
days = sorted(D.day.unique())

# absolute entry/close time for every candidate, so the ledger is continuous
W = D.copy()
W["close"] = pd.to_datetime(W.wkey, format="%y%b%d%H%M", utc=True, errors="coerce")
W = W.dropna(subset=["close"])
W["t_entry"] = W.close - pd.to_timedelta(W.secs, unit="s")
W["t_free"] = W.close + pd.to_timedelta(LAG_S, unit="s")


def run(vmin, cap, band=(0.65, 0.85), tw=(8, 14), gross_cap=GROSS):
    q = W[(W.px >= band[0]) & (W.px < band[1]) & (W.minute >= tw[0])
          & (W.minute <= tw[1]) & (W.vol >= vmin)]
    if len(q) < 50:
        return None
    e = (q.sort_values("minute", ascending=False)
           .groupby(["wkey", "coin"], as_index=False).first())
    e = (e.sort_values(["wkey", "minute"], ascending=[True, False])
           .groupby("wkey", as_index=False).head(cap))
    e = e.sort_values("t_entry")
    open_pos = []                      # (t_free, cost)
    taken, blocked = [], 0
    for r in e.itertuples():
        open_pos = [p for p in open_pos if p[0] > r.t_entry]
        g = sum(c for _, c in open_pos)
        cost = QTY * r.px
        if g + cost > gross_cap:
            blocked += 1
            continue
        open_pos.append((r.t_free, cost))
        taken.append((r.day, r.edge * QTY, cost))
    if len(taken) < 30:
        return None
    t = pd.DataFrame(taken, columns=["day", "pnl", "cost"])
    per = t.groupby("day").pnl.sum().reindex(days, fill_value=0.0).values
    return dict(vmin=vmin, cap=cap, n=len(t), ent_day=len(t) / len(days),
                blocked=blocked, block_rate=blocked / max(len(e), 1),
                edge_c=t.pnl.sum() / (len(t) * QTY) * 100,
                roc=t.pnl.sum() / t.cost.sum() * 100,
                day=per.mean(), sd=per.std(ddof=1), worst=per.min(),
                neg=(per < 0).sum())


print(f"gross cap ${GROSS:.0f}   qty {QTY}   settlement lag {LAG_S:.0f}s   "
      f"{len(days)} days\n")
print("=== VOLUME FLOOR SWEEP, capital-constrained (65-85c, 8-14 min) ===")
print(f"{'vmin':>6} {'cap':>4} {'ent/day':>8} {'blocked':>8} {'edge c':>8} "
      f"{'ret/$ dep':>10} {'$/day':>9} {'sd':>8} {'worst day':>10} {'neg':>4}")
best, res = None, []
for cap in (2, 3):
    for vmin in (0, 500, 1000, 2000, 3000, 4000, 6000):
        r = run(vmin, cap)
        if r is None:
            continue
        res.append(r)
        print(f"{vmin:>6} {cap:>4} {r['ent_day']:>8.1f} {r['block_rate']:>8.3f} "
              f"{r['edge_c']:>+8.2f} {r['roc']:>+10.2f}% {r['day']:>+9.2f} "
              f"{r['sd']:>8.2f} {r['worst']:>+10.2f} {r['neg']:>4}")
    print()

R = pd.DataFrame(res)
b = R.loc[R.day.idxmax()]
cur = R[(R.vmin == 2000) & (R.cap == 3)].iloc[0]
old = R[(R.vmin == 500) & (R.cap == 2)].iloc[0]
print("=== COMPARISON AT THE REAL CONSTRAINT ===")
for lbl, r in (("was (v500/cap2)  ", old), ("now (v2000/cap3) ", cur),
               ("best in sweep    ", b)):
    print(f"  {lbl} vmin {int(r.vmin):>5} cap {int(r.cap)}  "
          f"${r.day:>+7.2f}/day  edge {r.edge_c:>+6.2f}c  "
          f"ret/dollar {r.roc:>+5.2f}%  worst day ${r.worst:>+7.2f}")

print("\n=== DOES A BIGGER GROSS CAP ACTUALLY PAY, given the correlation? ===")
print("  P(all 3 lose) is 6.2x independence, so more simultaneous positions")
print("  buy return at a worse-than-linear risk. Pricing both sides:\n")
print(f"{'gross':>7} {'ent/day':>8} {'$/day':>9} {'worst day':>10} "
       f"{'worst/equity':>13}")
EQ = 338.81
for g in (110, 150, 200, 260):
    r = run(2000, 3, gross_cap=g)
    if r:
        print(f"{g:>7} {r['ent_day']:>8.1f} {r['day']:>+9.2f} "
              f"{r['worst']:>+10.2f} {abs(r['worst'])/EQ*100:>12.1f}%")
print("\n  worst day here is the worst of only 10 observed days, so it is a")
print("  floor on the true tail, not an estimate of it.")
