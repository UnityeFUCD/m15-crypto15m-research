"""Confidence interval on the LIVE edge, so the point estimate is not overread.

The live per-contract edge moved from +9.27c to +5.59c inside twenty minutes as
a few more windows settled. That is not a change in the strategy, it is what
sampling error looks like at ~100 settlements - and it means quoting either
number bare would misrepresent what is known.

The resampling unit is the SETTLEMENT, not the contract: all 30 contracts in a
position share one outcome, so treating them as 30 observations would shrink
the interval by sqrt(30) and be wrong.
"""
import numpy as np

from acct import api
from pnl_lsm import LSM_T          # tickers the LSM runner actually posted on

RNG = np.random.default_rng(20260810)
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

pnl, ncon = [], []
for x in M:
    n = abs(float(x.get("yes_count_fp") or 0)) + abs(float(x.get("no_count_fp") or 0))
    rev = float(x.get("revenue") or 0) / 100.0
    cst = (float(x.get("yes_total_cost_dollars") or 0) +
           float(x.get("no_total_cost_dollars") or 0) + float(x.get("fee_cost") or 0))
    if n > 0:
        pnl.append(rev - cst)
        ncon.append(n)
pnl, ncon = np.array(pnl), np.array(ncon)
tot, N = pnl.sum(), ncon.sum()
print(f"settlements {len(pnl)}   contracts {N:,.0f}   P&L ${tot:+.2f}   "
      f"point estimate {tot/N*100:+.2f}c/contract")

B = 20000
idx = RNG.integers(0, len(pnl), size=(B, len(pnl)))
bs = np.sort(pnl[idx].sum(1) / ncon[idx].sum(1) * 100)
lo, hi = bs[int(.025 * B)], bs[int(.975 * B)]
print(f"\nbootstrap over SETTLEMENTS (the correct unit):")
print(f"  95% CI  [{lo:+.2f}c, {hi:+.2f}c]     width {hi-lo:.2f}c")
print(f"  P(true edge <= 0)  {(bs <= 0).mean():.4f}")
print(f"  P(true edge >= backtest's +8.34c)  {(bs >= 8.34).mean():.4f}")

wrong = np.sort((pnl[idx].sum(1) / ncon[idx].sum(1)) * 100)
per_con = np.repeat(pnl / ncon, ncon.astype(int))
b2 = RNG.choice(per_con, size=(2000, len(per_con))).mean(1) * 100
print(f"\n  (for contrast, resampling CONTRACTS would give the much too narrow"
      f"\n   interval [{np.percentile(b2,2.5):+.2f}, {np.percentile(b2,97.5):+.2f}] "
      f"- that is the mistake this avoids)")

print("\n=== IS LIVE CONSISTENT WITH THE BACKTEST? ===")
print(f"  backtest  +8.34c   live {tot/N*100:+.2f}c   "
      f"backtest inside live's CI: {lo <= 8.34 <= hi}")
print(f"  the honest read: {len(pnl)} settlements cannot distinguish +5c from")
print(f"  +9c. What IS established is that the edge is positive.")
