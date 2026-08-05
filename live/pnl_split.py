"""Split settled P&L by day so the LSM strategy is separated from the earlier
15-minute experiments that share the same series tickers.

The all-time number mixes every experiment run on KX*15M this week. Only the
window since the LSM runner went live describes the strategy in question.
"""
import time
from collections import defaultdict

from acct import api

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
S = [x for x in s if x.get("ticker", "").startswith(
    ("KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M", "KXDOGE15M"))]
print(f"settlements fetched {len(s):,}   crypto-15M {len(S):,}\n")

by = defaultdict(lambda: [0, 0, 0.0, 0.0, 0.0])   # n, won, rev, cost, contracts
for x in S:
    d = (x.get("settled_time") or "")[:10]
    n = abs(float(x.get("yes_count_fp") or 0)) + abs(float(x.get("no_count_fp") or 0))
    rev = float(x.get("revenue") or 0) / 100.0
    cst = (float(x.get("yes_total_cost_dollars") or 0) +
           float(x.get("no_total_cost_dollars") or 0) + float(x.get("fee_cost") or 0))
    r = by[d]
    r[0] += 1
    r[1] += 1 if rev > 0 else 0
    r[2] += rev
    r[3] += cst
    r[4] += n

print(f"{'day':>12} {'settled':>8} {'won':>6} {'win%':>7} {'contracts':>10} "
      f"{'cost':>10} {'revenue':>10} {'P&L':>10} {'c/contract':>11}")
tot = [0, 0, 0.0, 0.0, 0.0]
for d in sorted(by):
    n, w, rev, cst, nc = by[d]
    pnl = rev - cst
    print(f"{d:>12} {n:>8} {w:>6} {w/n:>7.4f} {nc:>10,.0f} {cst:>10.2f} "
          f"{rev:>10.2f} {pnl:>+10.2f} {pnl/max(nc,1)*100:>+11.2f}")
    for i, v in enumerate((n, w, rev, cst, nc)):
        tot[i] += v
n, w, rev, cst, nc = tot
print(f"{'ALL TIME':>12} {n:>8} {w:>6} {w/n:>7.4f} {nc:>10,.0f} {cst:>10.2f} "
      f"{rev:>10.2f} {rev-cst:>+10.2f} {(rev-cst)/max(nc,1)*100:>+11.2f}")

# the LSM strategy specifically: it went live on 2026-08-04
LSM = [d for d in sorted(by) if d >= "2026-08-04"]
if LSM:
    n = sum(by[d][0] for d in LSM)
    w = sum(by[d][1] for d in LSM)
    rev = sum(by[d][2] for d in LSM)
    cst = sum(by[d][3] for d in LSM)
    nc = sum(by[d][4] for d in LSM)
    print(f"\n=== LSM ERA ONLY ({LSM[0]} onward) ===")
    print(f"  settled {n}   won {w} ({w/n:.4f})   contracts {nc:,.0f}")
    print(f"  P&L ${rev-cst:+.2f}   per contract {(rev-cst)/max(nc,1)*100:+.2f}c")
    pre = [d for d in sorted(by) if d < "2026-08-04"]
    if pre:
        prv = sum(by[d][2] for d in pre) - sum(by[d][3] for d in pre)
        pnc = sum(by[d][4] for d in pre)
        print(f"\n=== EVERYTHING BEFORE THAT (earlier experiments) ===")
        print(f"  P&L ${prv:+.2f} over {pnc:,.0f} contracts "
              f"({prv/max(pnc,1)*100:+.2f}c/contract)")
