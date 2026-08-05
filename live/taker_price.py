"""Price the taker alternative on the orders we actually posted.

WHERE THIS COMES FROM
  The reachability control found that we fill the two cohorts at very different
  rates: markets whose price comes back to our bid fill 51/53 = 96%, and those
  we lose money on (-12.75c). Markets whose price runs away fill only 53/67 =
  79%, and those are the ones that win 100% (+30.72c).

  So our fill process is biased toward the losing side, and the 14 unreachable
  markets we MISSED were worth +30.86c each. A taker crosses the spread and
  captures all of them, because taking removes the requirement that someone
  come and sell to you.

WHY IT IS NOT OBVIOUSLY FREE
  1. crossing costs the spread on EVERY order, not just the missed ones
  2. taker fees are real: ceil(0.07*C*P*(1-P)*1e4)/1e4 per fill block, against
     ZERO for maker
  3. the edge thesis is the longshot premium. Part of that premium is
     compensation for providing liquidity - it is not obvious how much survives
     when you stop providing it

THE HONEST FRAME
  This is a STRATEGY CHANGE, which the standing constraints forbid deploying.
  Nothing here gets deployed. It is priced because it is the one structural
  alternative never costed, and because if it is clearly worse that closes the
  question permanently rather than leaving it as a maybe.

METHOD
  Real orders, real posted prices, real logged fav_ask, real settled outcomes.
  No resampling of a hypothetical book. The only counterfactual assumption is
  that crossing at the logged ask would have filled - which is what crossing
  means, and the logged ask is the size-1 top of book we could see.
"""
import glob
import json
import math
from datetime import datetime

import numpy as np
import pandas as pd

from acct import api

RNG = np.random.default_rng(20261031)
QTY = 20


def ts(s):
    try:
        return datetime.fromisoformat((s or "").replace("Z", "+00:00"))
    except Exception:
        return None


def taker_fee(c, p):
    """Kalshi taker fee for a block of c contracts at price p (dollars)."""
    return math.ceil(0.07 * c * p * (1.0 - p) * 1e4) / 1e4


ORD = {}
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
            t = d.get("ticker")
            if not t:
                continue
            if d.get("ev") == "opportunity":
                ORD.setdefault(t, {}).update(side=d.get("side"),
                                             bid=d.get("fav_bid"),
                                             ask=d.get("fav_ask"),
                                             spread=d.get("spread_c"))
            if d.get("ev") == "terminal":
                ORD.setdefault(t, {}).update(px=d.get("price"),
                                             target=d.get("target"),
                                             filled=d.get("FILLED"))
    except Exception:
        pass
A = {t: o for t, o in ORD.items()
     if o.get("px") and o.get("side") and o.get("ask")}
print(f"orders with posted price, side AND a logged ask: {len(A)}")

rows = []
for t, o in A.items():
    c, j = api("/markets", {"tickers": t})
    m = ((j or {}).get("markets") or [None])[0]
    if not m or m.get("result") not in ("yes", "no"):
        continue
    won = 1 if o["side"] == m["result"] else 0
    rows.append(dict(ticker=t, side=o["side"], won=won,
                     bid=float(o["px"]), ask=float(o["ask"]),
                     spread=float(o.get("spread") or 0),
                     filled=int(o.get("filled") or 0),
                     target=int(o.get("target") or QTY),
                     day=str(ts(m.get("close_time")).date())))
D = pd.DataFrame(rows)
print(f"resolved: {len(D)}   days {D.day.nunique()}\n")
if len(D) < 60:
    raise SystemExit("too few")

print("=" * 78)
print("THE SPREAD WE WOULD HAVE TO PAY")
print("=" * 78)
D["cross_c"] = (D.ask - D.bid) * 100
print(f"  median {D.cross_c.median():.2f}c   mean {D.cross_c.mean():.2f}c   "
      f"p90 {D.cross_c.quantile(0.90):.2f}c")
print(f"  one tick (<=1c): {(D.cross_c <= 1.01).mean():.3f} of orders")

print("\n" + "=" * 78)
print("MAKER (what actually happened) vs TAKER (cross on every order)")
print("=" * 78)
mk_pnl = ((D.won - D.bid) * D.filled).sum()
mk_ct = int(D.filled.sum())
tk_gross = ((D.won - D.ask) * D.target).sum()
tk_fees = sum(taker_fee(int(r.target), float(r.ask)) for r in D.itertuples())
tk_pnl = tk_gross - tk_fees
tk_ct = int(D.target.sum())
print(f"{'':>10} {'contracts':>11} {'gross $':>11} {'fees $':>9} {'net $':>11} "
      f"{'per-ct':>9}")
print(f"{'MAKER':>10} {mk_ct:>11} {mk_pnl:>11.2f} {0.0:>9.2f} {mk_pnl:>11.2f} "
      f"{mk_pnl/max(mk_ct,1)*100:>+8.2f}c")
print(f"{'TAKER':>10} {tk_ct:>11} {tk_gross:>11.2f} {tk_fees:>9.2f} "
      f"{tk_pnl:>11.2f} {tk_pnl/max(tk_ct,1)*100:>+8.2f}c")
print(f"\n  difference: ${tk_pnl - mk_pnl:+.2f} over {D.day.nunique()} days "
      f"= ${(tk_pnl-mk_pnl)/D.day.nunique():+.2f}/day")

print("\n" + "=" * 78)
print("WHERE THE DIFFERENCE COMES FROM")
print("=" * 78)
miss = D[D.filled < D.target]
cap = ((miss.won - miss.ask) * (miss.target - miss.filled)).sum()
print(f"  contracts the maker MISSED         : "
      f"{int((miss.target-miss.filled).sum()):>8}")
print(f"  value of capturing them at the ask : ${cap:>+8.2f}")
print(f"  extra spread paid on orders we already filled: "
      f"${-((D.ask-D.bid)*D.filled).sum():>+8.2f}")
print(f"  taker fees on the whole book       : ${-tk_fees:>+8.2f}")
print(f"  {'-'*54}")
print(f"  net                                : ${tk_pnl-mk_pnl:>+8.2f}")

print("\n" + "=" * 78)
print("IS THE DIFFERENCE REAL OR NOISE? (day-clustered bootstrap)")
print("=" * 78)
days = sorted(D.day.unique())
g = {d: gg for d, gg in D.groupby("day")}


def day_diff(dd):
    s = g[dd]
    mk = ((s.won - s.bid) * s.filled).sum()
    tk = ((s.won - s.ask) * s.target).sum() - sum(
        taker_fee(int(r.target), float(r.ask)) for r in s.itertuples())
    return tk - mk


base = np.array([day_diff(d) for d in days])
bs = np.sort(np.array([base[RNG.integers(0, len(base), len(base))].mean()
                       for _ in range(8000)]))
print(f"  taker minus maker: ${base.mean():+.2f}/day   "
      f"95% CI [{bs[200]:+.2f}, {bs[7799]:+.2f}]")
print(f"  taker better on {(base > 0).sum()}/{len(base)} days   "
      f"P(taker worse) {(bs <= 0).mean():.4f}")

print("\n" + "=" * 78)
print("THE ASSUMPTION THAT COULD BREAK THIS")
print("=" * 78)
print("  Crossing at the logged ask assumes the ask had enough size for the")
print("  full order. Kalshi's top of book is often thin, so a 20-lot may walk")
print("  the book and pay worse than the quoted ask. That makes the taker")
print("  number here OPTIMISTIC. If taker still loses on an optimistic")
print("  accounting, it loses.")
