"""Are the six untraded 15-minute coins worth adding?

WHY THIS IS THE ONLY LEVER LEFT
  The economics are now fully decomposed and there is no leak:

      population edge          +3.65c
      our price                0.86c BETTER than population
      our win rate             2.97pp worse - exactly what fill bias predicts
      our realised edge        +1.54c

  Fill bias explains the entire shortfall (predicted realised win 0.714 vs
  observed 0.7063), it is structural, and posting price cannot change it -
  fill|LOSE is 1.0000 at every rung of the ladder.

  So edge per contract is at its achievable limit. What is NOT at its limit is
  the NUMBER of independent bets. Daily P&L has mean +$36.59 and SD $157.51,
  a signal-to-noise of 0.232, which is why two consecutive days look like
  different strategies. Edge scales with n; noise scales with sqrt(n). Doubling
  the coin count is worth about 41% on that ratio - roughly halving the time to
  know whether the edge is real.

  Kalshi lists twelve single-coin 15-minute series. We trade six.

WHAT HAS TO BE TRUE FOR THIS TO WORK
  1. LIQUIDITY. MIN_VOLUME=2000 exists because thin markets do not fill. If
     these series trade a tenth of BTC's volume they will never qualify and
     adding them is a no-op.
  2. THE SAME PREMIUM. The edge is a longshot premium paid by impatient
     retail. A coin nobody watches may have no such flow - or a much larger
     one, if it is thinner and more retail-driven.
  3. GENUINE DIVERSIFICATION. Crypto is highly correlated. If ADA and BNB move
     with BTC tick for tick, six more coins add six more copies of the same
     bet and the sqrt(n) benefit never materialises. This is the one that
     usually disappoints.

  All three are measured. Failing any one of them kills it.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from acct import api

LIVE = Path(__file__).resolve().parent
CACHE = LIVE / "new_series.parquet"
NEW = ["KXADA15M", "KXBCH15M", "KXBNB15M", "KXNEAR15M", "KXTON15M", "KXZEC15M"]
OLD = ["KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M", "KXDOGE15M", "KXHYPE15M"]

rows, seen = [], set()
if CACHE.exists():
    old = pd.read_parquet(CACHE)
    rows, seen = old.to_dict("records"), set(old.ticker)
    print(f"cache: {len(rows):,}")

for s in NEW:
    cur, got, pages = None, 0, 0
    while pages < 12:
        p = {"series_ticker": s, "limit": 1000, "status": "settled"}
        if cur:
            p["cursor"] = cur
        try:
            c, j = api("/markets", p)
        except Exception:
            break
        if c != 200 or not j:
            break
        b = j.get("markets") or []
        if not b:
            break
        for m in b:
            t = m.get("ticker")
            if not t or t in seen:
                continue
            fs, ev = m.get("floor_strike"), m.get("expiration_value")
            if fs is None or ev in (None, ""):
                continue
            try:
                a0, a1 = float(fs), float(ev)
            except (TypeError, ValueError):
                continue
            if a0 <= 0 or m.get("result") not in ("yes", "no"):
                continue
            seen.add(t)
            rows.append(dict(ticker=t, series=s,
                             coin=s.replace("KX", "").replace("15M", ""),
                             wkey=t.split("-")[1], a0=a0, a1=a1,
                             ret=(a1 - a0) / a0, result=m["result"],
                             volume=float(m.get("volume_fp") or 0),
                             oi=float(m.get("open_interest_fp") or 0)))
            got += 1
        cur = j.get("cursor")
        pages += 1
        if not cur:
            break
        time.sleep(0.05)
    print(f"  {s:>12}: +{got}")

N = pd.DataFrame(rows).drop_duplicates(subset=["ticker"])
N.to_parquet(CACHE)
O = pd.read_parquet(LIVE / "underlying.parquet")
print(f"\nnew series: {len(N):,} settled markets   "
      f"existing: {len(O):,}")

print("\n" + "=" * 78)
print("TEST 1 - LIQUIDITY.  MIN_VOLUME=2000 is the gate")
print("=" * 78)
print(f"{'coin':>7} {'markets':>9} {'median vol':>12} {'p75':>10} "
       f"{'share >=2000':>14} {'status':>10}")
for lbl, df in (("CURRENT", O), ("NEW", N)):
    print(f"  --- {lbl} ---")
    for c, g in df.groupby("coin"):
        sh = (g.volume >= 2000).mean()
        print(f"{c:>7} {len(g):>9} {g.volume.median():>12.0f} "
              f"{g.volume.quantile(.75):>10.0f} {sh:>14.3f} "
              f"{'usable' if sh > 0.15 else 'TOO THIN':>10}")

print("\n" + "=" * 78)
print("TEST 2 - DO THEY MOVE ENOUGH TO HAVE A REAL MARKET?")
print("=" * 78)
print(f"{'coin':>7} {'median |move|':>15} {'YES rate':>10}")
for lbl, df in (("CURRENT", O), ("NEW", N)):
    print(f"  --- {lbl} ---")
    for c, g in df.groupby("coin"):
        print(f"{c:>7} {g.ret.abs().median()*1e4:>13.2f}bp "
              f"{(g.result=='yes').mean():>10.4f}")

print("\n" + "=" * 78)
print("TEST 3 - CORRELATION.  Do they add independent bets or copies?")
print("=" * 78)
A = pd.concat([O[["coin", "wkey", "ret"]], N[["coin", "wkey", "ret"]]])
P = A.pivot_table(index="wkey", columns="coin", values="ret")
P = P.dropna(thresh=int(len(P.columns) * 0.6))
C = P.corr()
cur = [c for c in C.columns if c in
       [x.replace("KX", "").replace("15M", "") for x in OLD]]
new = [c for c in C.columns if c in
       [x.replace("KX", "").replace("15M", "") for x in NEW]]
print(f"  windows with broad coverage: {len(P):,}\n")
print(f"{'':>7}" + "".join(f"{c:>7}" for c in cur))
for c in new:
    print(f"{c:>7}" + "".join(f"{C.loc[c, k]:>7.3f}" if c in C.index and k in C.columns
                              else f"{'-':>7}" for k in cur))
if cur and new:
    within = C.loc[cur, cur].values
    within = within[~np.eye(len(cur), dtype=bool)].mean()
    across = C.loc[new, cur].values.mean()
    print(f"\n  average correlation AMONG the coins we already trade: {within:.3f}")
    print(f"  average correlation of NEW coins to current ones:     {across:.3f}")
    n_now, n_new = len(cur), len(cur) + len(new)
    def eff(n, rho):
        return n / (1 + (n - 1) * max(rho, 0))
    print(f"\n  effective independent bets now:  {eff(n_now, within):.2f}")
    print(f"  effective independent bets after: "
          f"{eff(n_new, (within*len(cur)+across*len(new))/n_new):.2f}")
    print("\n  If the effective count barely moves, the coins are copies of each")
    print("  other and adding them buys nothing but more of the same risk.")

print("\n" + "=" * 78)
print("VERDICT INPUTS")
print("=" * 78)
usable = [c for c, g in N.groupby("coin") if (g.volume >= 2000).mean() > 0.15]
print(f"  new coins clearing the volume gate: {usable or 'NONE'}")
print("  next step if any clear: reconstruct their premium from candlesticks")
print("  before adding anything. Liquidity alone is not edge.")
