"""Does a pause-after-loss breaker actually MAKE MONEY?

WHERE THIS COMES FROM
  Window edge on the live book is serially correlated beyond the time-of-day
  cycle: residual lag-1 r = +0.2468, CI [+0.0595, +0.4282], and the clean
  control confirms it is time-local rather than a daily pattern -

      adjacent windows, same day        r = +0.3610  [+0.1733, +0.5335]
      same clock time, different day    r = -0.0581  [-0.3653, +0.2819]

  Next-window edge is -10.17c after a losing window and +13.83c after a winning
  one. That looks like a circuit breaker should pay.

WHY THAT SPLIT IS NOT EVIDENCE
  It was computed on the same data that produced the autocorrelation, and it
  conditions on the sign of the previous window - which is exactly what the
  autocorrelation already says. Restating a correlation as a conditional mean
  does not test whether acting on it is profitable.

  Two things can break it:
    1. the losing windows are also the ones we hold the most capital in, so
       skipping the next window may skip a large winner
    2. a breaker forfeits entries permanently. Every filter tested today died
       on exactly this - the hours filter raised edge per contract and lost
       $34-96/day because the discarded entries were still positive

THE TEST
  Simulate the rule forward through the actual window sequence. It only ever
  uses information available before each window, so no look-ahead is possible.
  Compare total dollars against trading everything, with a moving-block
  bootstrap over the sequence to respect the very autocorrelation being tested.

  A breaker only earns its keep if the windows it SKIPS are negative in
  aggregate - the same bar every other filter had to clear.
"""
import glob
import json
from datetime import datetime

import numpy as np
import pandas as pd

from acct import api

RNG = np.random.default_rng(20261045)


def ts(x):
    try:
        return datetime.fromisoformat((x or "").replace("Z", "+00:00"))
    except Exception:
        return None


PX = {}
for fp in sorted(glob.glob("lsm_*.jsonl")) + sorted(glob.glob("lsm_*.log")):
    try:
        for ln in open(fp, errors="ignore"):
            ln = ln.strip()
            if not ln.startswith("{"):
                continue
            try:
                d = json.loads(ln)
            except Exception:
                continue
            if d.get("ev") == "post" and d.get("ticker") and d.get("price"):
                PX[d["ticker"]] = float(d["price"])
    except Exception:
        pass
rows, cur, seen = [], None, set()
for _ in range(12):
    p = {"limit": 200, "settlement_status": "settled"}
    if cur:
        p["cursor"] = cur
    c, j = api("/portfolio/settlements", p)
    if c != 200 or not j:
        break
    b = j.get("settlements") or []
    if not b:
        break
    for s in b:
        if s.get("ticker") and s["ticker"] not in seen:
            seen.add(s["ticker"])
            rows.append(s)
    cur = j.get("cursor")
    if not cur:
        break
o = []
for s in rows:
    t = s["ticker"]
    if t not in PX:
        continue
    n = float(s.get("yes_count_fp") or 0) + float(s.get("no_count_fp") or 0)
    cost = float(s.get("yes_total_cost_dollars") or 0) + float(
        s.get("no_total_cost_dollars") or 0)
    if n <= 0 or cost <= 0:
        continue
    st = ts(s.get("settled_time"))
    if st is None:
        continue
    o.append(dict(t=st, n=n,
                  pnl=float(s.get("revenue") or 0) / 100.0 - cost
                      - float(s.get("fee_cost") or 0)))
D = pd.DataFrame(o)
W = (D.assign(k=D.t.dt.floor("15min"))
       .groupby("k").agg(pnl=("pnl", "sum"), n=("n", "sum")).sort_index())
P = W.pnl.to_numpy()
N = W.n.to_numpy()
G = np.array([str(d.date()) for d in W.index])
print(f"windows {len(W)}   total ${P.sum():+.2f}   "
      f"({P.sum()/N.sum()*100:+.2f}c/ct)\n")


def run(pnl, nn, grp, pause):
    """Skip the next `pause` windows after any losing window. Day-local."""
    keep = np.ones(len(pnl), dtype=bool)
    cool = 0
    for i in range(len(pnl)):
        if i > 0 and grp[i] != grp[i - 1]:
            cool = 0                      # each day starts fresh
        if cool > 0:
            keep[i] = False
            cool -= 1
        if pnl[i] < 0 and keep[i]:
            cool = pause
        elif pnl[i] < 0:
            cool = max(cool, pause)
    return keep


print("=" * 76)
print("PAUSE-AFTER-LOSS, SIMULATED FORWARD THROUGH THE ACTUAL SEQUENCE")
print("=" * 76)
print(f"{'rule':>22} {'windows':>9} {'$ total':>10} {'c/ct':>9} "
      f"{'vs trade-all':>13} {'skipped $':>11}")
base = P.sum()
print(f"{'trade every window':>22} {len(P):>9} {base:>10.2f} "
      f"{P.sum()/N.sum()*100:>+8.2f}c {'-':>13} {'-':>11}")
res = {}
for pause in (1, 2, 3, 4):
    k = run(P, N, G, pause)
    tot = P[k].sum()
    res[pause] = k
    print(f"{'pause ' + str(pause) + ' after a loss':>22} {k.sum():>9} "
          f"{tot:>10.2f} {P[k].sum()/max(N[k].sum(),1)*100:>+8.2f}c "
          f"{tot-base:>+12.2f} {P[~k].sum():>+11.2f}")

print("\n" + "=" * 76)
print("THE BAR: are the SKIPPED windows actually negative?")
print("=" * 76)
for pause in (1, 2, 3):
    k = res[pause]
    sk = P[~k]
    if len(sk) < 5:
        continue
    skn = N[~k]
    print(f"  pause {pause}: skipped {len(sk):>3} windows, "
          f"${sk.sum():+8.2f} total, {sk.sum()/max(skn.sum(),1)*100:+7.2f}c/ct "
          f"({(sk<0).mean():.2f} of them losers)")
print("\n  A breaker only pays if this is clearly negative. Every filter tested")
print("  today failed here - the discards kept turning out positive.")

print("\n" + "=" * 76)
print("MOVING-BLOCK BOOTSTRAP (blocks preserve the autocorrelation)")
print("=" * 76)
L = 8
nb = int(np.ceil(len(P) / L))
for pause in (1, 2, 3):
    diffs = []
    for _ in range(4000):
        st = RNG.integers(0, max(len(P) - L, 1), nb)
        idx = np.concatenate([np.arange(s, min(s + L, len(P))) for s in st])[:len(P)]
        pp, nn2, gg = P[idx], N[idx], G[idx]
        kk = run(pp, nn2, gg, pause)
        diffs.append(pp[kk].sum() - pp.sum())
    d = np.sort(np.array(diffs))
    k = res[pause]
    print(f"  pause {pause}: observed {P[k].sum()-base:+8.2f}   "
          f"bootstrap mean {d.mean():+8.2f}   "
          f"95% CI [{d[100]:+8.2f}, {d[3899]:+8.2f}]   "
          f"P(<=0) {(d <= 0).mean():.4f}")

print("\n" + "=" * 76)
print("HONEST LIMITS")
print("=" * 76)
print("  111 windows over 2 days, one regime, one coin mix. The")
print("  autocorrelation itself is solid and the clock has been ruled out, but")
print("  a breaker is a change to trading behaviour and would need the user's")
print("  approval regardless of what this shows.")
print()
print("  Note also what a breaker cannot do: it reacts AFTER a loss, and the")
print("  losses here arrive as whole-window wipeouts of 3 positions at once.")
print("  The first loss is never avoided - only the follow-on.")
