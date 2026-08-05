"""Should we resume after a halt? Makes the restart rule explicit and testable.

WHY A RULE IS NEEDED AT ALL
  The halt was defined; the resume never was. In practice it had been "whenever
  someone notices", which is not a policy and is not reproducible.

WHAT THE MEASUREMENTS SAY THE RULE SHOULD BE
  Window P&L is serially INDEPENDENT - autocorrelation null at every lag
  (permutation p = 0.87, 0.28, 0.095, 0.68, 0.77). A drawdown therefore carries
  no information about the next window, which splits halts into two kinds
  needing opposite responses:

    BAD LUCK -> nothing changed -> resume IMMEDIATELY. Waiting is a pure tax on
                a positive-expectation edge. Simulated over 20 days, halting at
                20% of peak:
                    resume immediately   $12,213   (edge alive)
                    wait 1 day           $11,447
                    wait 3 days          $10,613
                    never resume          $9,598
    BAD EDGE -> something changed -> stay out. Same simulation, edge dead:
                    resume immediately   -$3,146
                    never resume            +$448

  Neither elapsed time nor equity recovery carries information about which case
  we are in. Only the accumulated EVIDENCE on the edge does. Combining a
  fractional halt with an evidence-gated resume captured 97.6% of the
  immediate-resume upside ($11,924 of $12,213) while completely avoiding the
  -$3,140 catastrophe.

THE RULE
    resume immediately, UNLESS the sequential test says the edge is dead.

  This program reports that decision. It does NOT resume automatically -
  restarting live trading stays a human action.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

from acct import api
import risk

HERE = Path(__file__).resolve().parent
KILL = HERE / "LSM_KILL"
MU_OK = 9.95        # cents/contract, historical
MU_DEAD = -5.0      # cents/contract, the "it died" hypothesis
BOUND = np.log((1 - 0.05) / 0.02)

print("=" * 66)
print("RESUME CHECK")
print("=" * 66)

halted = KILL.exists()
print(f"\n1. HALT STATE")
print(f"   kill file present : {halted}")
if halted:
    try:
        print(f"   reason            : {KILL.read_text().strip()[:80]}")
    except Exception:
        pass

c, j = api("/portfolio/balance")
bal = float((j or {}).get("balance_dollars") or 0)
c2, j2 = api("/portfolio/positions", {"limit": 200})
cost = 0.0
for p in ((j2 or {}).get("market_positions") or []):
    try:
        if float(p.get("position_fp") or 0) != 0:
            cost += abs(float(p.get("total_traded_dollars") or 0))
    except Exception:
        pass
eq = bal + cost
s = risk.summary(eq)
print(f"\n2. RISK STATE")
for k, v in s.items():
    print(f"   {k:<16}: {v}")

# --- the evidence test ---
MINE = set()
for fp in sorted(glob.glob(str(HERE / "lsm_*.jsonl"))) + \
        sorted(glob.glob(str(HERE / "lsm_*.log"))):
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

st, cur = [], None
while True:
    p = {"limit": 200}
    if cur:
        p["cursor"] = cur
    c3, j3 = api("/portfolio/settlements", p)
    if c3 != 200 or not j3:
        break
    b = j3.get("settlements") or []
    st += b
    cur = j3.get("cursor")
    if not cur or not b or len(st) >= 4000:
        break
M = [x for x in st if x.get("ticker") in MINE]
M.sort(key=lambda x: x.get("settled_time") or "")
per = []
for x in M:
    n = abs(float(x.get("yes_count_fp") or 0)) + abs(float(x.get("no_count_fp") or 0))
    if n <= 0:
        continue
    rev = float(x.get("revenue") or 0) / 100.0
    cst = (float(x.get("yes_total_cost_dollars") or 0) +
           float(x.get("no_total_cost_dollars") or 0) + float(x.get("fee_cost") or 0))
    per.append((rev - cst) / n * 100.0)
per = np.array(per)

print(f"\n3. EDGE EVIDENCE")
if len(per) < 20:
    print(f"   only {len(per)} settlements - too few to test. Treat as UNKNOWN.")
    edge_dead = None
else:
    sd = per.std(ddof=1)
    llr = float(np.sum((-(per - MU_DEAD) ** 2 + (per - MU_OK) ** 2) / (2 * sd ** 2)))
    edge_dead = llr > BOUND
    print(f"   settlements       : {len(per)}")
    print(f"   edge, all-time    : {per.mean():+.2f}c/contract")
    print(f"   edge, last 30     : {per[-30:].mean():+.2f}c")
    print(f"   edge, last 60     : {per[-60:].mean():+.2f}c")
    print(f"   LLR (dead vs ok)  : {llr:+.3f}   alarm at {BOUND:+.3f}")
    print(f"   verdict           : "
          f"{'EDGE LOOKS DEAD' if edge_dead else 'no evidence of death'}")

print(f"\n4. DECISION")
if not halted:
    print("   Not halted - nothing to resume. Running normally.")
elif edge_dead is True:
    print("   *** DO NOT RESUME ***")
    print("   The sequential test favours a dead edge. Resuming here is the")
    print("   -$3,146 branch of the simulation. Investigate before restarting.")
elif edge_dead is None:
    print("   INSUFFICIENT EVIDENCE. Too few settlements to judge the edge.")
    print("   Default to resuming only if the halt had a known benign cause")
    print("   (a routine wipeout window, which is 3.0% of all windows).")
else:
    print("   RESUME NOW.")
    print("   No evidence the edge has died, and window outcomes are serially")
    print("   independent - so the windows skipped while waiting are as good as")
    print("   any other. Waiting is a pure cost.")
    print(f"\n   command:  rm {KILL}")
print("=" * 66)
sys.exit(0)
