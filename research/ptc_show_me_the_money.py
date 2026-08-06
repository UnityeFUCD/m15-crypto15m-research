"""The PTC backtest in dollars, trade by trade, on the real 2 days.

This is the same replay as ptc_v3.py - real orders, real fill timestamps, real
prices, real settlements, all 303 LSM orders - but reported as money instead
of p-values.

Question it answers: if PTC had been running on Aug 4-5 instead of the
existing strategy, what would the account have done?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.ptc_v3 import build, replay, control            # noqa: E402

d = build()
Q = 15

print("=" * 84)
print("THE BACKTEST, IN DOLLARS - Aug 4-5 2026, all 303 real orders, q15")
print("=" * 84)

arms = {"CONTROL (what you actually ran)": control(d, Q),
        "PROBE_ONLY_60 (probe, never commit)": replay(d, 60, commit=False),
        "PTC_60": replay(d, 60, qty=Q),
        "PTC_120": replay(d, 120, qty=Q)}

print("\n  %-38s %10s %10s %10s" % ("arm", "Aug 4", "Aug 5", "TOTAL"))
for k, v in arms.items():
    dd = v.groupby("day").pnl.sum()
    days = sorted(dd.index)
    print("  %-38s %+10.2f %+10.2f %+10.2f"
          % (k, dd.get(days[0], 0.0), dd.get(days[1], 0.0), v.pnl.sum()))

print("\n" + "=" * 84)
print("EVERY PTC_120 TRADE, START TO FINISH")
print("=" * 84)
x = replay(d, 120, qty=Q)
print("\n  PROBES (q1 each, the diagnostic layer):")
pf = x[x.probe_filled]
print("    filled     %3d probes   P&L %+8.2f   win rate %.4f"
      % (len(pf), pf.probe_pnl.sum(), pf.won.mean()))
nf = x[~x.probe_filled]
print("    NOT filled %3d probes   P&L %+8.2f   win rate %.4f  <- the signal"
      % (len(nf), 0.0, nf.won.mean()))

print("\n  COMMITMENTS (q%d IOC, only from the no-fill branch):" % Q)
sel = x[x.selected].copy()
sel["pnl_commit"] = sel.ioc_qty * (sel.won - sel.exec_price) - sel.ioc_fee
print("    %-28s %8s %6s %8s %10s"
      % ("market", "ask", "won", "fee", "P&L"))
for r in sel.sort_values("close_dt").itertuples():
    print("    %-28s %8.4f %6d %8.4f %+10.2f"
          % (r.ticker, r.exec_price, int(r.won), r.ioc_fee, r.pnl_commit))
print("    %-28s %8s %6s %8.4f %+10.2f"
      % ("TOTAL", "", "", sel.ioc_fee.sum(), sel.pnl_commit.sum()))

print("\n  BOTTOM LINE")
print("    probe P&L      %+8.2f" % x.probe_pnl.sum())
print("    commit P&L     %+8.2f" % sel.pnl_commit.sum())
print("    PTC_120 total  %+8.2f" % x.pnl.sum())
print("    what you ran   %+8.2f  (standardized to q%d)"
      % (control(d, Q).pnl.sum(), Q))
print("    difference     %+8.2f" % (x.pnl.sum() - control(d, Q).pnl.sum()))

print("\n" + "=" * 84)
print("WHY 6 TRADES CANNOT SETTLE IT - the same backtest, one input changed")
print("=" * 84)
print("\n  Each row re-runs the identical backtest with ONE thing perturbed.")
print("  If the answer were solid these would cluster. They do not.\n")
print("  %-46s %12s" % ("variation", "PTC_120 total"))
rows = [("as run", replay(d, 120, qty=Q).pnl.sum())]
rows.append(("cancel 1s later (60s->61s equivalent)",
             replay(d, 120, qty=Q, latency=1.0).pnl.sum()))
rows.append(("cancel 3s later", replay(d, 120, qty=Q, latency=3.0).pnl.sum()))
rows.append(("pay 1c more on every commit",
             replay(d, 120, qty=Q, slippage_c=1.0).pnl.sum()))
rows.append(("pay 2c more on every commit",
             replay(d, 120, qty=Q, slippage_c=2.0).pnl.sum()))
rows.append(("IOC only half-fills",
             replay(d, 120, qty=Q, fill_fraction=0.5).pnl.sum()))
rows.append(("wait 60s instead of 120s", replay(d, 60, qty=Q).pnl.sum()))
rows.append(("ask ceiling 78c instead of 80c",
             replay(d, 120, qty=Q, ceiling=0.78).pnl.sum()))
rows.append(("ask ceiling 82c instead of 80c",
             replay(d, 120, qty=Q, ceiling=0.82).pnl.sum()))
for nm, v in rows:
    print("  %-46s %+12.2f" % (nm, v))

lo = min(v for _, v in rows)
hi = max(v for _, v in rows)
print("\n  range across those variations: %+.2f to %+.2f" % (lo, hi))
print("  Same 2 days, same real fills. Small, defensible changes to the rules")
print("  move the answer by $%.2f. That is the problem - not that the backtest"
      % (hi - lo))
print("  wasn't run, but that 6 trades cannot hold an answer still.")

print("\n" + "=" * 84)
print("WHAT THE 2 DAYS DO ESTABLISH (this part IS solid)")
print("=" * 84)
for wait in (60, 120):
    y = replay(d, wait, qty=Q)
    f, n = y[y.probe_filled], y[~y.probe_filled]
    print("\n  cancel at %ds:" % wait)
    print("    probe FILLED     n=%3d   win rate %.4f" % (len(f), f.won.mean()))
    print("    probe NOT filled n=%3d   win rate %.4f" % (len(n), n.won.mean()))
    print("    gap %+.1f percentage points"
          % ((n.won.mean() - f.won.mean()) * 100))
print("\n  303 orders is enough to see this. 6 commitments is not enough to")
print("  price it. Those are different sample sizes for different questions.")
