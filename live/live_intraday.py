"""Does the intraday edge pattern appear in the LIVE book?

WHY THIS IS THE TEST THAT MATTERS
  The intraday effect is established on grid.parquet: coin-adjusted +8.19c with
  CI [+5.38,+11.35], P=0.0000, surviving permutation, split-half and a
  walk-forward contract-neutral sizing tilt worth +$25.57/day.

  Every one of those numbers comes from PRE-SHIFT data on XRP/SOL/DOGE. The
  live book is a different regime and a different coin mix - ETH/XRP/HYPE plus
  DOGE and SOL, the last two now carrying NEGATIVE post-shift edge. Two changes
  fitted pre-shift were deployed and reversed earlier today for exactly this
  reason.

  So before the tilt goes anywhere near production it has to show up in real
  fills, at real prices, in the current regime.

WHAT IS MEASURED
  Every settled live position, its actual entry price, its actual settlement,
  and the UTC hour of its window. Edge per contract by time block, and the
  smooth curve where there is enough data.

HONEST ABOUT POWER
  ~200 settled positions over 2 days is thin for a 24-hour profile. This cannot
  confirm the effect on its own. What it CAN do is contradict it - if the live
  book shows the opposite sign in the hours the pre-shift fit calls best, the
  tilt is dead and no further days are needed.
"""
import collections
import glob
import json
from datetime import datetime

import numpy as np
import pandas as pd

from acct import api


def ts(s):
    try:
        return datetime.fromisoformat((s or "").replace("Z", "+00:00"))
    except Exception:
        return None


# entry price per ticker from our own posts
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
                PX.setdefault(d["ticker"], []).append(
                    (float(d["price"]), d.get("side")))
    except Exception:
        pass
print(f"tickers with a logged post price: {len(PX)}")

# settled positions from the exchange
rows, cur = [], None
seen = set()
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
        t = s.get("ticker")
        if not t or t in seen:
            continue
        seen.add(t)
        rows.append(s)
    cur = j.get("cursor")
    if not cur:
        break
print(f"settlements pulled: {len(rows)}")

out = []
for s in rows:
    t = s.get("ticker")
    if t not in PX:
        continue
    n = float(s.get("yes_count_fp") or 0) + float(s.get("no_count_fp") or 0)
    if n <= 0:
        continue
    # revenue is an integer in CENTS; cost splits across yes_/no_ and the
    # field is *_dollars, NOT *_fp - using _fp silently yields 0 and makes
    # every winner look like +100c/contract.
    rev = float(s.get("revenue") or 0) / 100.0
    cost = float(s.get("yes_total_cost_dollars") or 0) + float(
        s.get("no_total_cost_dollars") or 0)
    fee = float(s.get("fee_cost") or 0)
    if cost <= 0:
        continue
    px = float(np.mean([p for p, _ in PX[t]]))
    st = ts(s.get("settled_time")) or ts(s.get("determined_time"))
    if st is None:
        continue
    # window close is at most ~15 min before settlement; hour of the window
    hr = st.hour
    out.append(dict(ticker=t, n=n, px=px, rev=rev, cost=cost,
                    pnl=rev - cost - fee, hour=hr,
                    edge=(rev - cost - fee) / n,
                    day=str(st.date())))
D = pd.DataFrame(out)
if len(D) < 40:
    raise SystemExit(f"only {len(D)} matched settlements - not enough")
print(f"matched settled positions: {len(D)}   contracts {D.n.sum():.0f}   "
      f"net ${D.pnl.sum():+.2f}\n")

print("=" * 74)
print("LIVE EDGE BY TIME BLOCK  (UTC)")
print("=" * 74)
D["blk"] = np.where(D.hour < 8, "00-07", np.where(D.hour < 14, "08-13", "14-23"))
print(f"{'block':>8} {'pos':>5} {'contracts':>11} {'avg px':>8} "
      f"{'edge/ct':>9} {'net $':>10}")
for k in ("00-07", "08-13", "14-23"):
    g = D[D.blk == k]
    if not len(g):
        continue
    print(f"{k:>8} {len(g):>5} {g.n.sum():>11.0f} {g.px.mean()*100:>7.1f}c "
          f"{g.pnl.sum()/g.n.sum()*100:>+8.2f}c {g.pnl.sum():>+10.2f}")

a = D[D.blk == "00-07"]
r = D[D.blk != "00-07"]
if len(a) >= 10 and len(r) >= 10:
    ea = a.pnl.sum() / a.n.sum() * 100
    er = r.pnl.sum() / r.n.sum() * 100
    RNG = np.random.default_rng(20261038)
    bs = np.sort(np.array([
        (lambda x, y: x.pnl.sum() / max(x.n.sum(), 1) * 100
         - y.pnl.sum() / max(y.n.sum(), 1) * 100)(
            a.sample(len(a), replace=True), r.sample(len(r), replace=True))
        for _ in range(8000)]))
    print(f"\n  00-07 minus rest: {ea-er:+.2f}c   "
          f"95% CI [{bs[200]:+.2f}, {bs[7799]:+.2f}]   "
          f"P(gap<=0) {(bs <= 0).mean():.4f}")
    print(f"  pre-shift predicted this gap at roughly +8 to +11c")

print("\n" + "=" * 74)
print("HOURLY DETAIL (thin - shown for sign, not for magnitude)")
print("=" * 74)
print(f"{'hr':>3} {'pos':>5} {'ct':>7} {'edge/ct':>9}")
for h in range(24):
    g = D[D.hour == h]
    if not len(g):
        continue
    print(f"{h:>3} {len(g):>5} {g.n.sum():>7.0f} "
          f"{g.pnl.sum()/g.n.sum()*100:>+8.2f}c")

print("\n" + "=" * 74)
print("THE DECISION THIS FEEDS")
print("=" * 74)
print("  A contract-neutral hour tilt at strength 0.25 was worth +$25.57/day")
print("  walk-forward pre-shift, CI [+17,+33], with zero clipping and mean qty")
print("  exactly 20.00. It is NOT deployed because it was fitted pre-shift on")
print("  coins whose edge has since gone negative.")
print()
print("  If the live sign here agrees with pre-shift, the tilt stays a")
print("  candidate and accumulates evidence. If it disagrees, it is dead.")
