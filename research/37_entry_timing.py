"""Does entering LATER in the window raise throughput enough to beat the edge cost?

THE LEVER
  Live, the median entry is at 13.75 minutes remaining - the runner takes the
  first qualifying moment. Kalshi reserves the capital from that instant until
  settlement, so each entry consumes ~13.75 minutes of capital-lock. Waiting
  until 10 minutes remaining would cut that to ~10, a 27% reduction, which
  under a BINDING capital constraint converts directly into more entries.

  177 orders are refused for every 36 taken, so the constraint is emphatically
  binding. Throughput is therefore a first-class objective, not a detail.

THE COST SIDE
  Edge is not constant across the window. Entering later means the favourite
  has had longer to establish itself (win rate usually rises) but the price has
  usually risen too (so edge per contract may fall). The net is an empirical
  question.

WHAT THIS SIMULATES
  A chronological capital ledger where a position locks MAX_GROSS-limited funds
  from entry until the window closes plus a settlement lag - which is what the
  live runner actually experiences and what the earlier backtests missed by
  settling instantly at window boundaries. Policies differ ONLY in when they
  are willing to enter.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260829)
D = pd.read_parquet(OUT / "grid.parquet")
QTY, LAG = 30, 45.0
days = sorted(D.day.unique())

W = D.copy()
W["close"] = pd.to_datetime(W.wkey, format="%y%b%d%H%M", utc=True, errors="coerce")
W = W.dropna(subset=["close"]).copy()
W["t_entry"] = W.close - pd.to_timedelta(W.secs, unit="s")
W["t_free"] = W.close + pd.to_timedelta(LAG, unit="s")

print("=== EDGE BY MINUTES REMAINING AT ENTRY (65-85c, vol>=2000) ===")
q = W[(W.px >= 0.65) & (W.px < 0.85) & (W.vol >= 2000) & (W.minute >= 5)
      & (W.minute <= 14)]
print(f"{'min left':>9} {'n':>6} {'avg px':>8} {'win':>8} {'edge/ct':>9} "
      f"{'edge per capital-min':>22}")
for m, g in q.groupby("minute"):
    if len(g) < 40:
        continue
    lock = m + LAG / 60.0
    print(f"{int(m):>9} {len(g):>6,} {g.px.mean()*100:>7.1f}c {g.won.mean():>8.4f} "
          f"{g.edge.mean()*100:>+9.2f} "
          f"{g.edge.mean()*100/(g.px.mean()*lock):>+22.4f}")
print("\n  last column = cents of edge per dollar-minute of capital locked,")
print("  which is the objective when capital is the binding constraint.")


def sim(min_lo, min_hi, gross_cap=110.0, cap=3, vmin=2000):
    """Chronological ledger. Policy differs only in the entry window."""
    s = W[(W.px >= 0.65) & (W.px < 0.85) & (W.vol >= vmin)
          & (W.minute >= min_lo) & (W.minute <= min_hi)]
    if len(s) < 50:
        return None
    e = (s.sort_values("minute", ascending=False)
           .groupby(["wkey", "coin"], as_index=False).first())
    e = (e.sort_values(["wkey", "minute"], ascending=[True, False])
           .groupby("wkey", as_index=False).head(cap))
    e = e.sort_values("t_entry")
    openp, taken, blocked = [], [], 0
    for r in e.itertuples():
        openp = [p for p in openp if p[0] > r.t_entry]
        g = sum(c for _, c in openp)
        cost = QTY * r.px
        if g + cost > gross_cap:
            blocked += 1
            continue
        openp.append((r.t_free, cost))
        taken.append((r.day, r.edge * QTY, cost, r.minute))
    if len(taken) < 30:
        return None
    t = pd.DataFrame(taken, columns=["day", "pnl", "cost", "minute"])
    per = t.groupby("day").pnl.sum().reindex(days, fill_value=0.0)
    lock = (t.minute + LAG / 60.0)
    return dict(n=len(t), ent_day=len(t) / len(days), blocked=blocked,
                block_rate=blocked / max(len(e), 1),
                edge=t.pnl.sum() / (len(t) * QTY) * 100,
                day=per.mean(), sd=per.std(ddof=1),
                mean_left=t.minute.mean(),
                capmin=(t.cost * lock).sum(),
                per_capmin=t.pnl.sum() / (t.cost * lock).sum())


print("\n=== POLICY COMPARISON under a $110 chronological gross cap ===")
print(f"{'entry window':>14} {'ent/day':>8} {'blocked':>8} {'avg left':>9} "
      f"{'edge/ct':>8} {'$/day':>9} {'$ per capital-min':>19}")
POL = [(8, 14, "8-14 (current)"), (8, 12, "8-12"), (8, 11, "8-11"),
       (8, 10, "8-10"), (8, 9, "8-9"), (5, 11, "5-11"), (5, 9, "5-9"),
       (11, 14, "11-14")]
res = {}
for lo, hi, lbl in POL:
    r = sim(lo, hi)
    if not r:
        continue
    res[lbl] = r
    print(f"{lbl:>14} {r['ent_day']:>8.1f} {r['block_rate']:>8.3f} "
          f"{r['mean_left']:>9.2f} {r['edge']:>+8.2f} {r['day']:>+9.2f} "
          f"{r['per_capmin']:>19.5f}")

b = res.get("8-14 (current)")
print(f"\n=== VERSUS CURRENT ===")
for lbl, r in res.items():
    if lbl.startswith("8-14"):
        continue
    print(f"  {lbl:>14}  $/day {r['day']-b['day']:>+8.2f} "
          f"({(r['day']/b['day']-1)*100:>+6.1f}%)   "
          f"entries {r['ent_day']-b['ent_day']:>+6.1f}/day   "
          f"capital-min {(r['capmin']/b['capmin']-1)*100:>+6.1f}%")

print("\n=== TIGHTER CAPITAL: does the ranking change when it binds harder? ===")
print("   (live blocks 177 per 36 taken, so the effective cap is tighter than")
print("    this backtest's, which settles positions promptly)\n")
print(f"{'gross cap':>10} " + "".join(f"{l:>16}" for l in
                                      ("8-14 (current)", "8-11", "8-10")))
for gc in (110, 70, 45, 30):
    row = ""
    for lo, hi, lbl in ((8, 14, "a"), (8, 11, "b"), (8, 10, "c")):
        r = sim(lo, hi, gross_cap=gc)
        row += f"{r['day']:>+16.2f}" if r else f"{'-':>16}"
    print(f"{gc:>10} {row}")
print("\n  If later entry wins only at a tight cap, that is the regime we are")
print("  actually in, and the backtest's loose cap understates the gain.")
