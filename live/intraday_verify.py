"""Triple-check the live intraday result before it is allowed near production.

WHAT IS BEING CHECKED
  The live book showed 00-07 UTC at +12.00c against -5.51c (08-13) and -0.99c
  (14-23), a gap of +14.58c with a bootstrap CI of [+1.92, +26.64]. Pre-shift
  analysis predicted +8 to +11c BEFORE the live data was examined, so this is a
  genuine out-of-sample confirmation of a pre-registered direction.

  That is the strongest evidence produced today. It is also exactly when to be
  most careful, because two changes deployed earlier today were reversed.

THE FLAW IN THAT CI
  It resampled POSITIONS. Positions inside one window are correlated - the
  agreement rule exists because windows tend to go one direction, and the
  correlated-loss work measured variance +25% from exactly this. Resampling
  correlated units as if independent makes the interval too narrow.

  Redone here by resampling WINDOWS, which is the independent unit.

FOUR WAYS IT COULD STILL BE AN ARTIFACT
  1. one coin - ETH carries +12.97c post-shift while DOGE and SOL are negative.
     If ETH happens to trade in 00-07, the block effect is the coin effect.
  2. one day - only 2 days of live fills exist.
  3. a few windows - 70 positions in 00-07 is perhaps 25 windows.
  4. price - 00-07 averaged 68.9c against 71.0c for 08-13, and cheaper entries
     carry more edge structurally.

  All four are tested. Any one of them surviving is a reason not to deploy.
"""
import glob
import json
from datetime import datetime

import numpy as np
import pandas as pd

from acct import api

RNG = np.random.default_rng(20261039)


def ts(s):
    try:
        return datetime.fromisoformat((s or "").replace("Z", "+00:00"))
    except Exception:
        return None


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
                PX.setdefault(d["ticker"], []).append(float(d["price"]))
    except Exception:
        pass

rows, cur, seen = [], None, set()
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
        if s.get("ticker") and s["ticker"] not in seen:
            seen.add(s["ticker"])
            rows.append(s)
    cur = j.get("cursor")
    if not cur:
        break

out = []
for s in rows:
    t = s.get("ticker")
    if t not in PX:
        continue
    n = float(s.get("yes_count_fp") or 0) + float(s.get("no_count_fp") or 0)
    cost = float(s.get("yes_total_cost_dollars") or 0) + float(
        s.get("no_total_cost_dollars") or 0)
    if n <= 0 or cost <= 0:
        continue
    rev = float(s.get("revenue") or 0) / 100.0
    fee = float(s.get("fee_cost") or 0)
    st = ts(s.get("settled_time")) or ts(s.get("determined_time"))
    if st is None:
        continue
    out.append(dict(ticker=t, n=n, pnl=rev - cost - fee,
                    px=cost / n, hour=st.hour,
                    coin=t.split("-")[0].replace("KX", "").replace("15M", ""),
                    window=t.split("-")[1],   # TIME key, shared across coins
                    day=str(st.date())))
D = pd.DataFrame(out)
D["blk"] = np.where(D.hour < 8, "00-07", np.where(D.hour < 14, "08-13", "14-23"))
D["good"] = D.blk == "00-07"
print(f"settled positions {len(D)}   windows {D.window.nunique()}   "
      f"days {D.day.nunique()}   net ${D.pnl.sum():+.2f} "
      f"({D.pnl.sum()/D.n.sum()*100:+.2f}c/ct)\n")


def gap(df):
    a, b = df[df.good], df[~df.good]
    if not len(a) or not len(b) or a.n.sum() == 0 or b.n.sum() == 0:
        return np.nan
    return (a.pnl.sum() / a.n.sum() - b.pnl.sum() / b.n.sum()) * 100


OBS = gap(D)
print("=" * 74)
print("CHECK 1 - WINDOW-CLUSTERED BOOTSTRAP (the correct unit)")
print("=" * 74)
wins = sorted(D.window.unique())
gw = {w: g for w, g in D.groupby("window")}
bs = []
for _ in range(8000):
    pick = [gw[wins[i]] for i in RNG.integers(0, len(wins), len(wins))]
    bs.append(gap(pd.concat(pick)))
bs = np.sort(np.array([x for x in bs if not np.isnan(x)]))
print(f"  observed gap {OBS:+.2f}c")
print(f"  window-clustered 95% CI [{bs[int(.025*len(bs))]:+.2f}, "
      f"{bs[int(.975*len(bs))]:+.2f}]   P(gap<=0) {(bs <= 0).mean():.4f}")
print(f"  (position-clustered was [+1.92, +26.64], P=0.0114 - too narrow)")

print("\n" + "=" * 74)
print("CHECK 2 - IS IT ONE COIN?")
print("=" * 74)
print(f"{'coin':>7} {'n 00-07':>9} {'n rest':>8} {'00-07':>9} {'rest':>9} "
      f"{'gap':>9}")
gaps = []
for c, g in D.groupby("coin"):
    a, b = g[g.good], g[~g.good]
    if len(a) < 3 or len(b) < 3:
        continue
    ea = a.pnl.sum() / a.n.sum() * 100
    eb = b.pnl.sum() / b.n.sum() * 100
    gaps.append(ea - eb)
    print(f"{c:>7} {len(a):>9} {len(b):>8} {ea:>+8.2f}c {eb:>+8.2f}c "
          f"{ea-eb:>+8.2f}c")
if gaps:
    print(f"\n  positive in {sum(1 for x in gaps if x>0)}/{len(gaps)} coins   "
          f"median gap {np.median(gaps):+.2f}c")
D["resid"] = D.pnl / D.n - D.groupby("coin").apply(
    lambda g: g.pnl.sum() / g.n.sum()).reindex(D.coin).values
ra = D[D.good].resid.mean() * 100
rb = D[~D.good].resid.mean() * 100
print(f"  coin-adjusted gap {ra-rb:+.2f}c  (raw was {OBS:+.2f}c)")

print("\n" + "=" * 74)
print("CHECK 3 - IS IT ONE DAY, OR A FEW WINDOWS?")
print("=" * 74)
for d, g in D.groupby("day"):
    print(f"  {d}: n {len(g):>4}  gap {gap(g):+.2f}c")
lo = [gap(D[D.window != w]) for w in wins]
lo = np.array([x for x in lo if not np.isnan(x)])
print(f"\n  leave-one-window-out gap: min {lo.min():+.2f}c  "
       f"max {lo.max():+.2f}c  ({(lo>0).sum()}/{len(lo)} stay positive)")
top = D.groupby("window").pnl.sum().abs().nlargest(3).index
print(f"  dropping the 3 largest-|P&L| windows: "
      f"{gap(D[~D.window.isin(top)]):+.2f}c")

print("\n" + "=" * 74)
print("CHECK 4 - IS IT JUST PRICE?")
print("=" * 74)
D["pb"] = pd.cut(D.px, [0.60, 0.68, 0.72, 0.76, 0.90])
print(f"{'price bucket':>16} {'n 00-07':>9} {'n rest':>8} {'gap':>9}")
wsum, wtot = 0.0, 0.0
for b, g in D.groupby("pb", observed=True):
    a, r = g[g.good], g[~g.good]
    if len(a) < 3 or len(r) < 3:
        continue
    gg = (a.pnl.sum() / a.n.sum() - r.pnl.sum() / r.n.sum()) * 100
    w = g.n.sum()
    wsum += gg * w
    wtot += w
    print(f"{str(b):>16} {len(a):>9} {len(r):>8} {gg:>+8.2f}c")
if wtot:
    print(f"\n  price-matched gap {wsum/wtot:+.2f}c  (raw {OBS:+.2f}c)")

print("\n" + "=" * 74)
print("VERDICT")
print("=" * 74)
p = (bs <= 0).mean()
print(f"  window-clustered P(gap<=0) = {p:.4f}")
if p < 0.05 and (wtot and wsum / wtot > 0) and (ra - rb) > 0:
    print("  Survives clustering, the coin confound and the price confound.")
    print("  Pre-registered direction, confirmed out of sample.")
else:
    print("  Does NOT survive. The position-clustered CI was overstating it.")
print()
print("  Either way this is 2 days of live fills. The pre-shift curve SHAPE is")
print("  still fitted on coins whose edge has gone negative - only the block")
print("  DIRECTION is confirmed live.")
