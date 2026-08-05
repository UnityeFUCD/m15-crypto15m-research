"""Live P&L per coin.

Why this matters: the backtest grid has 3,515 DOGE / 3,994 SOL / 3,723 XRP rows
in the 8-14 minute band but only 256 ETH and ONE BTC row. So the +11.29c edge
that justifies the live configuration was measured on DOGE, SOL and XRP.
BTC and ETH are running live on an assumption, not on evidence.
"""
from collections import defaultdict

import numpy as np

from acct import api
from pnl_lsm import LSM_T

RNG = np.random.default_rng(4242)
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

by = defaultdict(list)
for x in M:
    t = x.get("ticker") or ""
    coin = t.split("15M")[0].replace("KX", "")
    n = abs(float(x.get("yes_count_fp") or 0)) + abs(float(x.get("no_count_fp") or 0))
    rev = float(x.get("revenue") or 0) / 100.0
    cst = (float(x.get("yes_total_cost_dollars") or 0) +
           float(x.get("no_total_cost_dollars") or 0) + float(x.get("fee_cost") or 0))
    if n > 0:
        by[coin].append((rev - cst, n))

print(f"{'coin':>6} {'settled':>8} {'contracts':>10} {'win':>7} {'P&L':>10} "
      f"{'c/contract':>11} {'95% CI (settlement bootstrap)':>32} {'backtested?':>12}")
HAVE = {"DOGE": "yes 3,515 rows", "SOL": "yes 3,994 rows", "XRP": "yes 3,723 rows",
        "ETH": "THIN 256 rows", "BTC": "NO   1 row"}
tot = []
for coin in sorted(by, key=lambda k: -len(by[k])):
    v = by[coin]
    pnl = np.array([a for a, _ in v])
    nc = np.array([b for _, b in v])
    tot += v
    pt = pnl.sum() / nc.sum() * 100
    if len(pnl) >= 5:
        idx = RNG.integers(0, len(pnl), size=(8000, len(pnl)))
        bs = np.sort(pnl[idx].sum(1) / nc[idx].sum(1) * 100)
        ci = f"[{bs[200]:+7.2f}, {bs[7799]:+7.2f}]"
    else:
        ci = "  (too few)"
    print(f"{coin:>6} {len(pnl):>8} {nc.sum():>10,.0f} "
          f"{(pnl > 0).mean():>7.4f} {pnl.sum():>+10.2f} {pt:>+11.2f} "
          f"{ci:>32} {HAVE.get(coin,'?'):>12}")

pnl = np.array([a for a, _ in tot])
nc = np.array([b for _, b in tot])
print(f"{'ALL':>6} {len(pnl):>8} {nc.sum():>10,.0f} {(pnl > 0).mean():>7.4f} "
      f"{pnl.sum():>+10.2f} {pnl.sum()/nc.sum()*100:>+11.2f}")

tested = [c for c in by if c in ("DOGE", "SOL", "XRP")]
untested = [c for c in by if c in ("BTC", "ETH")]
for label, group in (("BACKTESTED (DOGE/SOL/XRP)", tested),
                     ("UNTESTED  (BTC/ETH)", untested)):
    v = [x for c in group for x in by[c]]
    if not v:
        print(f"\n  {label}: no settlements")
        continue
    pnl = np.array([a for a, _ in v])
    nc = np.array([b for _, b in v])
    idx = RNG.integers(0, len(pnl), size=(8000, len(pnl)))
    bs = np.sort(pnl[idx].sum(1) / nc[idx].sum(1) * 100)
    print(f"\n  {label}: {len(pnl)} settled, {nc.sum():,.0f} contracts, "
          f"${pnl.sum():+.2f}, {pnl.sum()/nc.sum()*100:+.2f}c/contract "
          f"CI [{bs[200]:+.2f}, {bs[7799]:+.2f}]")
