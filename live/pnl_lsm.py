"""Attribute P&L to the LSM strategy by TICKER, not by date.

Date slicing is wrong: on 2026-08-04 the older `ptt` runner was still trading
until 06:58 UTC, so that day's settlements mix two strategies. The only clean
attribution is the set of tickers the LSM runner actually posted orders on,
read out of its own logs, intersected with the exchange settlement record.
"""
import glob
import json
import os
from collections import defaultdict

from acct import api

# every ticker the LSM runner posted an order on
LSM_T = set()
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
            if d.get("ev") in ("post", "order_placed", "submit_ok", "placed",
                               "fill", "filled"):
                t = d.get("ticker") or d.get("tk")
                if t:
                    LSM_T.add(t)
    except Exception:
        pass
print(f"LSM posted on {len(LSM_T):,} distinct tickers "
      f"(from {len(glob.glob('lsm_*.jsonl')) + len(glob.glob('lsm_*.log'))} logs)")

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

M = [x for x in s if x.get("ticker") in LSM_T]
print(f"of {len(s):,} settlements, {len(M):,} are LSM tickers\n")


def summ(rows, label):
    if not rows:
        print(f"{label:>26}  (none)")
        return
    rev = sum(float(x.get("revenue") or 0) / 100.0 for x in rows)
    cst = sum(float(x.get("yes_total_cost_dollars") or 0) +
              float(x.get("no_total_cost_dollars") or 0) +
              float(x.get("fee_cost") or 0) for x in rows)
    won = sum(1 for x in rows if float(x.get("revenue") or 0) > 0)
    nc = sum(abs(float(x.get("yes_count_fp") or 0)) +
             abs(float(x.get("no_count_fp") or 0)) for x in rows)
    pnl = rev - cst
    print(f"{label:>26} {len(rows):>5} settled  win {won/len(rows):>6.4f}  "
          f"{nc:>7,.0f} contracts  P&L ${pnl:>+8.2f}  "
          f"{pnl/max(nc,1)*100:>+6.2f}c/contract")


by = defaultdict(list)
for x in M:
    by[(x.get("settled_time") or "")[:10]].append(x)
for d in sorted(by):
    summ(by[d], d)
print()
summ(M, "LSM TOTAL")

nonlsm = [x for x in s if x.get("ticker") not in LSM_T
          and x.get("ticker", "").startswith(
              ("KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M", "KXDOGE15M"))]
summ(nonlsm, "earlier experiments")

print("\n=== MODEL vs LIVE, the comparison that matters ===")
if M:
    rev = sum(float(x.get("revenue") or 0) / 100.0 for x in M)
    cst = sum(float(x.get("yes_total_cost_dollars") or 0) +
              float(x.get("no_total_cost_dollars") or 0) +
              float(x.get("fee_cost") or 0) for x in M)
    nc = sum(abs(float(x.get("yes_count_fp") or 0)) +
             abs(float(x.get("no_count_fp") or 0)) for x in M)
    won = sum(1 for x in M if float(x.get("revenue") or 0) > 0)
    live_c = (rev - cst) / max(nc, 1) * 100
    print(f"  backtest (65-85/8-14/v500/cap2):  win 0.798   edge  +8.34c/contract")
    print(f"  live                           :  win {won/len(M):.3f}   "
          f"edge {live_c:>+6.2f}c/contract")
    print(f"  shortfall                      :  {live_c - 8.34:+.2f}c/contract")
