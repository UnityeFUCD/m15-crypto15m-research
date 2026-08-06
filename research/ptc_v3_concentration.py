"""Why the PTC sensitivity tables are incoherent, and how big a trial must be.

Part 4 asks for a coherence check: "If additional latency improves the result
materially, flag that as a noise or selection warning." It does, badly:

  PTC_60 cancel latency  0.0-1.0s -> +$2.87    2.0-3.0s -> +$13.27
  PTC_60 slippage        0c +2.87  1c +0.85  2c -6.17  3c +7.18  5c -0.95
  PTC_120 slippage       0c +8.88  1c +11.07 2c +7.51  3c +4.09  5c +8.52

A cost that makes money is not a robustness property, it is a selection
artifact. Two mechanisms produce it and this script measures both:

  1  LATENCY PIVOT. A longer cancel race reclassifies a probe from no-fill to
     filled, which deletes a COMMITMENT. If that commitment was a loser, the
     handicap "helps". With single-digit commit counts one order flips it.

  2  CEILING DESELECTION. Slippage is added to exec_price BEFORE the 60-80c
     eligibility test, so more cost pushes candidates out of the band and
     silently removes them. Higher cost then buys fewer, different trades -
     not the same trades at worse prices.

It then computes the sample size the prospective trial needs, from the
observed dispersion rather than an assumed one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.ptc_v3 import build, replay, control, PRIMARY_QTY  # noqa: E402

data = build()
print("=" * 84)
print("1. CONCENTRATION - how few orders carry each result?")
print("=" * 84)
for wait, nm in ((60, "PTC_60"), (120, "PTC_120")):
    x = replay(data, wait)
    sel = x[x.selected].copy()
    sel["contrib"] = sel.ioc_qty * (sel.won - sel.exec_price) - sel.ioc_fee
    sel = sel.sort_values("contrib")
    tot = float(x.pnl.sum())
    ioc_tot = float(sel.contrib.sum())
    print(f"\n  {nm}: total ${tot:+.2f}, of which the IOC branch is ${ioc_tot:+.2f} "
          f"across {len(sel)} commits")
    print("    %-26s %8s %8s %10s" % ("ticker", "ask", "won", "P&L"))
    for r in sel.itertuples():
        print("    %-26s %8.4f %8d %+10.2f" % (r.ticker, r.exec_price, r.won,
                                               r.contrib))
    if len(sel):
        best = sel.contrib.max()
        worst = sel.contrib.min()
        print(f"    single best commit  {best:+.2f} = "
              f"{100*best/ioc_tot if ioc_tot else float('nan'):.0f}% of the IOC branch")
        print(f"    single worst commit {worst:+.2f}")
        print(f"    drop the best commit -> IOC branch becomes "
              f"${ioc_tot - best:+.2f}")

print("\n" + "=" * 84)
print("2. THE LATENCY PIVOT - which orders flip, and what do they cost?")
print("=" * 84)
base = replay(data, 60, latency=0.0)
lat2 = replay(data, 60, latency=2.0)
flip = data.index[(base.selected.values) & (~lat2.selected.values)]
print(f"  commitments present at 0.0s latency but absent at 2.0s: {len(flip)}")
for i in flip:
    r = base.loc[i]
    print(f"    {r.ticker:26s} ask {r.exec_price:.4f} won {int(r.won)} "
          f"P&L {r.ioc_qty*(r.won-r.exec_price)-r.ioc_fee:+.2f}  "
          f"first_fill {r.first_fill_seconds:.2f}s")
print("\n  Those orders filled between 1.0s and 2.0s. A longer cancel race")
print("  catches them as probe fills, which removes the commitment. The gain")
print("  is the removal of losing commitments, not a better execution.")

print("\n" + "=" * 84)
print("3. CEILING DESELECTION - slippage removes trades instead of costing them")
print("=" * 84)
print("  %-10s %9s %14s %16s" % ("slippage", "commits", "total P&L",
                                 "P&L if SAME set"))
b = replay(data, 60, slippage_c=0.0)
base_set = set(data.loc[b.selected.values, "ticker"])
for sl in (0.0, 1.0, 2.0, 3.0, 5.0):
    s = replay(data, 60, slippage_c=sl)
    # counterfactual: keep the ORIGINAL selection, just pay more
    keep = data.ticker.isin(base_set).values
    same = float(s.probe_pnl.sum()
                 + (PRIMARY_QTY * (s.won - s.exec_price) - s.ioc_fee)[keep].sum())
    print("  %-10.2f %9d %14.4f %16.4f"
          % (sl, int(s.selected.sum()), float(s.pnl.sum()), same))
print("\n  The right-hand column is monotone: paying more on the SAME trades")
print("  always costs money. The left-hand column is not, because the trade")
print("  SET changes. Only the right-hand column is a cost sensitivity.")

print("\n" + "=" * 84)
print("4. REQUIRED SAMPLE SIZE FOR THE PROSPECTIVE TRIAL")
print("=" * 84)
Z80 = 2.802
arms = {"CONTROL": control(data, PRIMARY_QTY),
        "PROBE_ONLY_60": replay(data, 60, commit=False),
        "PTC_60": replay(data, 60), "PTC_120": replay(data, 120)}
w = {k: v.groupby("close_dt").pnl.sum() for k, v in arms.items()}
print("  per-close-window dispersion (the unit the trial randomises on):")
print("  %-16s %10s %12s %12s" % ("arm", "windows", "mean $/win", "SD $/win"))
for k, s in w.items():
    print("  %-16s %10d %12.4f %12.4f" % (k, len(s), s.mean(), s.std(ddof=1)))

print("\n  windows per arm needed to detect a given per-window difference")
print("  at 80%% power, 5%% two-sided (paired on the window):")
d60 = (w["PTC_60"] - w["PROBE_ONLY_60"])
d120 = (w["PTC_120"] - w["PROBE_ONLY_60"])
dc60 = (w["PTC_60"] - w["CONTROL"])
for nm, d in (("PTC_60 - PROBE_ONLY", d60), ("PTC_120 - PROBE_ONLY", d120),
              ("PTC_60 - CONTROL", dc60)):
    sd = d.std(ddof=1)
    obs = d.mean()
    n_obs = (Z80 * sd / obs) ** 2 if obs != 0 else float("inf")
    print(f"    {nm:24s} observed {obs:+7.4f}/win  SD {sd:7.4f}  "
          f"n = {n_obs:,.0f} windows" if np.isfinite(n_obs)
          else f"    {nm:24s} observed {obs:+7.4f}  n = infinite")
print("\n  At 4 close windows per hour and ~24h coverage, one calendar day")
print("  supplies at most 96 windows per arm if every window is assigned to")
print("  every arm; with 4 arms randomised per window it is ~24/day/arm.")
for nm, d in (("PTC_60 - PROBE_ONLY", d60), ("PTC_120 - PROBE_ONLY", d120)):
    sd, obs = d.std(ddof=1), d.mean()
    if obs > 0:
        n = (Z80 * sd / obs) ** 2
        print(f"    {nm:24s} -> {n/24:,.0f} calendar days at 24 windows/day/arm")
