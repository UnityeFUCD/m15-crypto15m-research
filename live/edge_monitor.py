"""Sequential edge-health monitor. REPORTS ONLY - never touches trading.

WHY THIS IS A MONITOR AND NOT A BREAKER
  A simulated SPRT breaker was tested against the deployed static 30%-of-peak
  drawdown envelope across three regimes. No calibration dominated it: static
  costs $156 when the edge is alive against the SPRT's $240-297, with a lower
  false-stop rate (1.99% vs 2.91-3.66%), and saves essentially the same when
  the edge is dead ($2,451 vs $2,378-2,488). Automating a second kill path that
  does not dominate would add risk, not remove it.

  The ONE thing the sequential test does better is SPEED: it identifies a truly
  dead edge in ~30 windows against the drawdown rule's ~43, roughly three hours
  earlier. That is worth having as information, with a human deciding - which
  costs nothing and cannot false-stop the strategy.

WHAT IT COMPUTES
  A log-likelihood ratio over settled positions, testing
      H_ok   : edge = the historical +9.95c/contract
      H_dead : edge = -5c/contract
  LLR rises toward H_dead as evidence accumulates against us. A bad LUCK run
  nudges it and decays; a genuinely dead edge drives it monotonically.
"""
import glob
import json

import numpy as np

from acct import api

MU_OK = 9.95      # cents/contract, historical
MU_DEAD = -5.0    # cents/contract, the "it has died" hypothesis
BOUND = np.log((1 - 0.05) / 0.02)

# LSM tickers
MINE = set()
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
            if d.get("ev") in ("post", "opportunity") and d.get("ticker"):
                MINE.add(d["ticker"])
    except Exception:
        pass

s, cur = [], None
while True:
    p = {"limit": 200}
    if cur:
        p["cursor"] = cur
    c, j = api("/portfolio/settlements", p)
    if c != 200 or not j:
        break
    b = j.get("settlements") or []
    s += b
    cur = j.get("cursor")
    if not cur or not b or len(s) >= 4000:
        break
M = [x for x in s if x.get("ticker") in MINE]
M.sort(key=lambda x: x.get("settled_time") or "")

per = []
for x in M:
    n = abs(float(x.get("yes_count_fp") or 0)) + abs(float(x.get("no_count_fp") or 0))
    if n <= 0:
        continue
    rev = float(x.get("revenue") or 0) / 100.0
    cst = (float(x.get("yes_total_cost_dollars") or 0) +
           float(x.get("no_total_cost_dollars") or 0) + float(x.get("fee_cost") or 0))
    per.append((rev - cst) / n * 100.0)      # cents per contract
per = np.array(per)
if len(per) < 20:
    raise SystemExit(f"only {len(per)} settlements")

sd = per.std(ddof=1)
llr = np.cumsum((-(per - MU_DEAD) ** 2 + (per - MU_OK) ** 2) / (2 * sd ** 2))
print(f"settlements {len(per)}   per-settlement sd {sd:.2f}c")
print(f"cumulative edge {per.mean():+.2f}c/settlement-weighted")
print(f"\nSPRT  H_ok {MU_OK:+.2f}c  vs  H_dead {MU_DEAD:+.2f}c   "
      f"alarm boundary {BOUND:+.2f}")
print(f"  current LLR = {llr[-1]:+.3f}")
if llr[-1] > BOUND:
    print("  *** EVIDENCE FAVOURS A DEAD EDGE - escalate to the user ***")
elif llr[-1] > BOUND * 0.5:
    print("  ** LLR past halfway to the alarm - worth watching **")
else:
    print("  edge health: no evidence of death")
k = 40
if len(llr) > k:
    print(f"\n  LLR trajectory (last {k} settlements, every 5th):")
    for i in range(len(llr) - k, len(llr), 5):
        bar = "#" * max(0, int((llr[i] + 5) * 2))
        print(f"    n={i+1:>4}  LLR {llr[i]:>+7.2f}  {bar}")
print(f"\n  rolling edge, last 30 settlements: {per[-30:].mean():+.2f}c")
print(f"  rolling edge, last 60 settlements: {per[-60:].mean():+.2f}c")
print(f"  all-time                         : {per.mean():+.2f}c")
print("\n  REPORTS ONLY. This process never writes LSM_KILL and never touches"
      "\n  the runner. The deployed breaker remains the static 30% envelope.")
