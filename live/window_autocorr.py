"""Is window-to-window edge genuinely autocorrelated, or is it the clock?

WHY THIS MATTERS OPERATIONALLY
  Earlier work concluded that halting after losses is a pure tax, because
  window P&L was serially independent - stopping cannot help if the next window
  is unaffected by the last. That conclusion is the reason no drawdown-triggered
  circuit breaker was ever added beyond the absolute floor.

  On the live book the lag-1 autocorrelation of window edge is +0.2899 with a
  CI of [+0.1041, +0.4681], which excludes zero. Next-window edge is -10.17c
  after a losing window and +13.83c after a winning one. If that is real, the
  earlier conclusion is wrong and a breaker has value.

THE CONFOUND THAT PROBABLY EXPLAINS IT
  Today established that edge varies smoothly with time of day - amplitude
  22.09c, split-half r=+0.6885, permutation p=0.0000. Consecutive windows are
  15 minutes apart, so they sit at almost the same point on that curve and
  therefore share almost the same expected edge.

  That ALONE produces positive lag-1 autocorrelation with no momentum
  whatsoever. A slow deterministic cycle sampled at short intervals looks
  autocorrelated because it is smooth, not because the past predicts the future
  in any way that can be traded.

  The distinction is not academic:
    - genuine momentum  -> a breaker helps, because stopping after losses
                           avoids a genuinely worse period
    - smooth time cycle -> a breaker is still a tax, because the low-edge hours
                           are KNOWN IN ADVANCE and are still positive; you
                           would size down, never stop

HOW IT IS SEPARATED
  1. remove the fitted time-of-day curve from each window's edge, then re-test
     the autocorrelation on the residual
  2. compare adjacent-in-time pairs against pairs matched on time-of-day but
     drawn from a DIFFERENT day - if the effect is the clock, the different-day
     pairs correlate just as strongly
  3. test whether the runs of losses are longer than a shuffle produces

  Only if the residual autocorrelation survives is this a real finding.
"""
import glob
import json
from datetime import datetime

import numpy as np
import pandas as pd

from acct import api

RNG = np.random.default_rng(20261044)
KAPPA = 12.0
GRID = np.linspace(0, 2 * np.pi, 145)[:-1]


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
W["e"] = W.pnl / W.n * 100
W["tod"] = W.index.hour * 60 + W.index.minute
W["day"] = W.index.date
a = W.e.to_numpy()
print(f"windows {len(W)}   days {W.day.nunique()}   "
      f"mean edge {a.mean():+.2f}c   sd {a.std():.2f}c\n")
if len(W) < 30:
    raise SystemExit("too few windows")


def ac1(x):
    return np.corrcoef(x[:-1], x[1:])[0, 1]


def ci(x, f, n=6000):
    b = np.sort(np.array([f(x[RNG.integers(0, len(x), len(x))])
                          for _ in range(n)]))
    return b[int(.025 * n)], b[int(.975 * n)]


print("=" * 76)
print("STEP 1 - RAW AUTOCORRELATION (what prompted this)")
print("=" * 76)
r = ac1(a)
pairs = np.stack([a[:-1], a[1:]], axis=1)
lo, hi = ci(pairs, lambda p: np.corrcoef(p[:, 0], p[:, 1])[0, 1])
print(f"  lag-1 r = {r:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")

print("\n" + "=" * 76)
print("STEP 2 - REMOVE THE TIME-OF-DAY CURVE, THEN RE-TEST")
print("=" * 76)
th = W.tod.to_numpy() / 1440 * 2 * np.pi
Wk = np.exp(KAPPA * np.cos(GRID[:, None] - th[None, :]))
curve = (Wk @ a) / Wk.sum(axis=1)
fit = np.interp(th, GRID, curve)
resid = a - fit
print(f"  time-of-day curve: min {curve.min():+.2f}c  max {curve.max():+.2f}c  "
      f"amplitude {curve.max()-curve.min():.2f}c")
rr = ac1(resid)
pr = np.stack([resid[:-1], resid[1:]], axis=1)
lo2, hi2 = ci(pr, lambda p: np.corrcoef(p[:, 0], p[:, 1])[0, 1])
print(f"  residual lag-1 r = {rr:+.4f}   95% CI [{lo2:+.4f}, {hi2:+.4f}]")
print(f"  the clock explains {(1 - rr/r)*100:.1f}% of the raw autocorrelation"
      if r != 0 else "")

print("\n" + "=" * 76)
print("STEP 3 - SAME TIME OF DAY, DIFFERENT DAY")
print("=" * 76)
print("  If the effect is the clock, windows at the same time on DIFFERENT days")
print("  correlate as strongly as windows 15 minutes apart on the same day.")
same_day, diff_day = [], []
idx = list(range(len(W)))
tod = W.tod.to_numpy()
day = np.array([str(d) for d in W.day])
for i in idx:
    for j in idx:
        if i >= j:
            continue
        if day[i] == day[j] and abs(tod[i] - tod[j]) == 15:
            same_day.append((a[i], a[j]))
        if day[i] != day[j] and tod[i] == tod[j]:
            diff_day.append((a[i], a[j]))
for lbl, pl in (("adjacent, same day", same_day),
                ("same clock time, different day", diff_day)):
    if len(pl) >= 12:
        p = np.array(pl)
        rv = np.corrcoef(p[:, 0], p[:, 1])[0, 1]
        l3, h3 = ci(p, lambda q: np.corrcoef(q[:, 0], q[:, 1])[0, 1])
        print(f"  {lbl:>32}: n {len(p):>4}  r {rv:+.4f}  "
              f"CI [{l3:+.4f}, {h3:+.4f}]")
    else:
        print(f"  {lbl:>32}: n {len(pl)} - too few")

print("\n" + "=" * 76)
print("STEP 4 - ARE LOSING RUNS LONGER THAN CHANCE?")
print("=" * 76)


def maxrun(x):
    best = cur = 0
    for v in x:
        cur = cur + 1 if v < 0 else 0
        best = max(best, cur)
    return best


obs = maxrun(a)
sh = np.array([maxrun(RNG.permutation(a)) for _ in range(6000)])
print(f"  longest observed run of losing windows: {obs}")
print(f"  shuffled: median {np.median(sh):.0f}  "
      f"P(shuffle >= observed) {(sh >= obs).mean():.4f}")

print("\n" + "=" * 76)
print("WHAT THIS MEANS FOR A CIRCUIT BREAKER")
print("=" * 76)
if lo2 > 0:
    print("  Residual autocorrelation SURVIVES the clock. Losing windows do")
    print("  predict losing windows beyond time-of-day, so the earlier")
    print("  'halting is a tax' conclusion needs revisiting.")
else:
    print("  Residual autocorrelation does NOT survive. The raw +0.29 is the")
    print("  time-of-day cycle being sampled every 15 minutes, not momentum.")
    print()
    print("  That leaves the earlier conclusion intact: halting after losses is")
    print("  still a tax. The low-edge hours are known IN ADVANCE and are")
    print("  positive on average, so the response to them is sizing, never")
    print("  stopping - and the hour tilt that would do that is still")
    print("  unconfirmed on live data.")
