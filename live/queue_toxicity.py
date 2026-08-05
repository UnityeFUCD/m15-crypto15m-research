"""Are deep-queue fills adversely selected?

THE QUESTION, AND WHY IT IS PURE EFFICIENCY
  Fill rate falls hard with queue depth: 0.939 with under 100 ahead, 0.748 with
  over 3,000. That much is already measured. What is NOT measured is whether
  the fills we DO get from deep in the queue are worth having.

  There is a specific reason to fear they are not. Sitting behind 3,000
  contracts, our order only reaches the front if that whole queue gets consumed
  - which happens when the market is trading through our price, i.e. when the
  favourite is being sold off. So deep-queue fills may be concentrated in
  exactly the windows where the favourite is about to flip. That is textbook
  adverse selection, and it would mean deep-queue orders are worse than useless:
  they tie up reserved capital, usually do not fill, and when they do fill it is
  because we were picked off.

  If true, skipping deep-queue entries raises edge AND frees capital - a pure
  efficiency gain with no P&L given up. If false, queue depth costs us only
  fill rate, which is harmless.

METHOD
  Join the runner's own order log (which records queue_ahead at submission)
  to the exchange settlement record, and compare realised P&L per contract
  across queue-depth buckets. Live data only - no backtest can answer this,
  because no historical order-book depth overlaps the discovery period.
"""
import glob
import json
from collections import defaultdict

import numpy as np

from acct import api

RNG = np.random.default_rng(20260826)

# queue depth at submission, per ticker, from the runner's own logs
QA, PX = {}, {}
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
            if d.get("ev") in ("opportunity", "terminal"):
                t = d.get("ticker")
                q = d.get("queue_ahead")
                if t and q is not None:
                    QA[t] = float(q)
                    if d.get("post_at") is not None:
                        PX[t] = float(d["post_at"])
                    elif d.get("price") is not None:
                        PX[t] = float(d["price"])
    except Exception:
        pass
print(f"tickers with a recorded queue depth: {len(QA):,}")

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

rows = []
for x in s:
    t = x.get("ticker")
    if t not in QA:
        continue
    n = abs(float(x.get("yes_count_fp") or 0)) + abs(float(x.get("no_count_fp") or 0))
    if n <= 0:
        continue
    rev = float(x.get("revenue") or 0) / 100.0
    cst = (float(x.get("yes_total_cost_dollars") or 0) +
           float(x.get("no_total_cost_dollars") or 0) + float(x.get("fee_cost") or 0))
    rows.append((t, QA[t], n, rev - cst, 1 if rev > 0 else 0, PX.get(t)))
print(f"settled positions matched to a queue depth: {len(rows):,}\n")
if len(rows) < 20:
    raise SystemExit("not enough matched settlements yet")

qa = np.array([r[1] for r in rows])
nc = np.array([r[2] for r in rows])
pnl = np.array([r[3] for r in rows])
won = np.array([r[4] for r in rows])

print("=== REALISED P&L BY QUEUE DEPTH AT SUBMISSION ===")
print(f"{'queue ahead':>14} {'positions':>10} {'contracts':>10} {'win':>8} "
      f"{'P&L':>10} {'c/contract':>11}")
BUCK = [(0, 100), (100, 400), (400, 1000), (1000, 3000), (3000, 10 ** 9)]
LAB = ["<100", "100-400", "400-1k", "1k-3k", ">3k"]
for (lo, hi), lab in zip(BUCK, LAB):
    m = (qa >= lo) & (qa < hi)
    if m.sum() == 0:
        continue
    print(f"{lab:>14} {m.sum():>10} {nc[m].sum():>10,.0f} {won[m].mean():>8.4f} "
          f"{pnl[m].sum():>+10.2f} {pnl[m].sum()/nc[m].sum()*100:>+11.2f}")

print(f"\n{'ALL':>14} {len(rows):>10} {nc.sum():>10,.0f} {won.mean():>8.4f} "
      f"{pnl.sum():>+10.2f} {pnl.sum()/nc.sum()*100:>+11.2f}")

print("\n=== DEEP vs SHALLOW, bootstrapped over positions ===")
for cut in (1000, 3000):
    deep, shal = qa >= cut, qa < cut
    if deep.sum() < 5 or shal.sum() < 10:
        print(f"  cut {cut}: too few ({deep.sum()} deep)")
        continue
    ed = pnl[deep].sum() / nc[deep].sum() * 100
    es = pnl[shal].sum() / nc[shal].sum() * 100
    bs = []
    for _ in range(8000):
        i = RNG.integers(0, deep.sum(), deep.sum())
        k = RNG.integers(0, shal.sum(), shal.sum())
        a = pnl[deep][i].sum() / max(nc[deep][i].sum(), 1) * 100
        b = pnl[shal][k].sum() / max(nc[shal][k].sum(), 1) * 100
        bs.append(a - b)
    bs = np.sort(np.array(bs))
    print(f"  queue >= {cut}: {deep.sum()} pos, {ed:+.2f}c   "
          f"vs < {cut}: {shal.sum()} pos, {es:+.2f}c")
    print(f"    difference {ed-es:+.2f}c   95% CI "
          f"[{bs[200]:+.2f},{bs[7799]:+.2f}]   "
          f"p {2*min((bs<=0).mean(),(bs>=0).mean()):.4f}")

print("\n=== THE COMBINED COST OF DEPTH (fill rate x edge) ===")
print("  a deep order reserves capital, usually fails to fill, and its fills")
print("  may be toxic. Expected value per SUBMITTED order is what matters:\n")
FILL = {"<100": 0.939, "100-400": 0.980, "400-1k": 0.874, "1k-3k": 0.900,
        ">3k": 0.748}
print(f"{'queue ahead':>14} {'fill rate':>10} {'edge/ct':>9} "
      f"{'EV per submitted contract':>27}")
for (lo, hi), lab in zip(BUCK, LAB):
    m = (qa >= lo) & (qa < hi)
    if m.sum() == 0:
        continue
    e = pnl[m].sum() / nc[m].sum() * 100
    f = FILL.get(lab, 0.9)
    print(f"{lab:>14} {f:>10.3f} {e:>+9.2f} {f*e:>+26.2f}c")
