"""Capital TURNOVER: the efficiency lever I missed by stopping at "capital-bound".

THE MISTAKE I MADE
  I established the strategy is capital-constrained (177 order refusals per 36
  fills) and concluded nothing more could be done from inside the strategy.
  That skips the consequence. Under a binding capital constraint the objective
  is not edge per DOLLAR, it is edge per DOLLAR-MINUTE. Two entries with equal
  edge are not equal if one ties up capital for 15 minutes and the other for 9.

WHERE THE CAPITAL-MINUTES GO
  Kalshi reserves cash the moment an order RESTS, not when it fills. So the
  meter starts at submission:

      post -> (resting, earning nothing) -> fill -> (held) -> close -> settle

  We deliberately JOIN the bid, which puts us at the BACK of the queue. That
  maximises price but also maximises resting time - and resting time is pure
  capital burn at zero edge. Every minute spent resting is a minute that
  capital could have been in a different position.

THE TRADEOFF THIS MEASURES
  Posting one tick higher would put us ahead of the queue: faster fill, higher
  fill rate, fewer capital-minutes - at 1c worse entry price. At a ~73c entry
  with ~+11c edge, giving up 1c costs ~9% of edge. If it cuts capital-time by
  more than 9%, it is a net win under a binding constraint. That is an
  empirical question and this measures the inputs to it.

Live data only: no backtest has order-book depth or our own queue position.
"""
import glob
import json
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

from acct import api


def ts(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


# our own order submissions: when we posted, at what queue depth and price
POST = {}
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
            if d.get("ev") == "opportunity" and d.get("ticker"):
                POST[d["ticker"]] = dict(t=ts(d.get("utc")),
                                         qa=d.get("queue_ahead"),
                                         px=d.get("post_at"),
                                         left=d.get("min_left"))
    except Exception:
        pass
print(f"orders with a recorded submission: {len(POST):,}")

c, j = api("/portfolio/fills", {"limit": 1000})
fills = defaultdict(list)
for f in ((j or {}).get("fills") or []):
    t = f.get("ticker")
    if t in POST:
        fills[t].append(dict(t=ts(f.get("created_time")),
                             n=float(f.get("count_fp") or 0),
                             taker=bool(f.get("is_taker"))))
print(f"of those, filled: {len(fills):,}\n")

rows = []
for tk, ff in fills.items():
    p = POST[tk]
    if not p["t"] or not ff:
        continue
    tot = sum(x["n"] for x in ff)
    if tot <= 0:
        continue
    # quantity-weighted mean fill time -> how long the capital sat idle
    wt = sum((x["t"] - p["t"]).total_seconds() * x["n"] for x in ff if x["t"]) / tot
    rows.append(dict(tk=tk, qa=p["qa"] or 0, px=p["px"] or 0, left=p["left"] or 0,
                     wait=wt, n=tot,
                     taker=any(x["taker"] for x in ff)))
if len(rows) < 10:
    raise SystemExit("not enough matched fills yet")

wait = np.array([r["wait"] for r in rows])
qa = np.array([r["qa"] for r in rows])
left = np.array([r["left"] for r in rows])
px = np.array([r["px"] for r in rows])
ncon = np.array([r["n"] for r in rows])
print("=== TIME FROM POST TO FILL (capital resting, earning nothing) ===")
print(f"  n {len(rows)}   median {np.median(wait):.0f}s   mean {wait.mean():.0f}s"
      f"   p90 {np.percentile(wait,90):.0f}s   max {wait.max():.0f}s")
print(f"  taker fills: {sum(r['taker'] for r in rows)} (must be 0)\n")

print("=== RESTING TIME BY QUEUE DEPTH ===")
print(f"{'queue ahead':>14} {'n':>5} {'median wait':>12} {'mean wait':>11}")
for lo, hi, lab in ((0, 100, "<100"), (100, 400, "100-400"),
                    (400, 1000, "400-1k"), (1000, 3000, "1k-3k"),
                    (3000, 10**9, ">3k")):
    m = (qa >= lo) & (qa < hi)
    if m.sum() == 0:
        continue
    print(f"{lab:>14} {m.sum():>5} {np.median(wait[m]):>11.0f}s "
          f"{wait[m].mean():>10.0f}s")

print("\n=== THE FULL CAPITAL-TIME BUDGET PER ENTRY ===")
print("  reserved from POST until SETTLEMENT. Entry happens at `min_left`")
print("  minutes before close, so:")
print("      total lock = rest time + (min_left*60 - rest) + settle lag")
print("                 = min_left*60 + settle lag,  regardless of rest time\n")
hold = left * 60.0
print(f"  median min_left at entry : {np.median(left):.2f} min")
print(f"  median total lock        : {np.median(hold)/60:.2f} min (+~45s settle)")
print(f"  of which RESTING         : {np.median(wait)/60:.2f} min "
       f"({np.median(wait)/np.median(hold)*100:.0f}% of the lock)")
print("\n  KEY POINT: resting time does NOT extend the lock - the position")
print("  settles at the window close either way. Filling faster does NOT")
print("  free capital earlier. What resting costs is the RISK of not filling")
print("  at all, having reserved the capital regardless.")

print("\n=== SO WHERE IS THE REAL TURNOVER LEVER? ===")
print("  Capital is locked for min_left minutes no matter what. Entering")
print("  LATER in the 8-14 window mechanically shortens the lock:\n")
print(f"{'entry min_left':>15} {'n':>5} {'lock (min)':>11} {'contracts':>10}")
for lo, hi in ((8, 10), (10, 12), (12, 15)):
    m = (left >= lo) & (left < hi)
    if m.sum() == 0:
        continue
    print(f"{f'{lo}-{hi}':>15} {m.sum():>5} {left[m].mean():>10.2f} "
          f"{ncon[m].sum():>10,.0f}")
lo_m, hi_m = left < 11, left >= 11
if lo_m.sum() and hi_m.sum():
    r = left[hi_m].mean() / left[lo_m].mean()
    print(f"\n  entering at 8-11 instead of 11-14 cuts the capital lock by "
          f"{(1-1/r)*100:.0f}%")
    print(f"  which at a binding constraint means ~{(r-1)*100:.0f}% MORE "
          f"entries from the same capital")
print("\n  Whether that pays depends on whether edge survives entering later,")
print("  which is a research question the grid can answer directly.")
