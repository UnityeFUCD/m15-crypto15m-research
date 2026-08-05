"""Does MAX_PER_WINDOW=3 still earn its keep? Tested on live fills.

THE ARGUMENT FOR THE CAP, AND WHY IT MAY HAVE EXPIRED
  Pre-shift, rank-3 entries carried +16.65c with CI [+12.23, +21.40] and cap 3
  beat cap 2 on 9 of 10 days. The mechanism was selection: a 3rd entry only
  exists when three separate coins independently clear the volume floor in the
  same window, which happens only when the whole window is unusually active.

  Post-shift the volume floor keeps 58.2% of entries instead of 31.1%. If three
  coins now clear it routinely rather than exceptionally, the selection that
  made a 3rd entry valuable is gone, and the cap is simply admitting ordinary
  entries. That is precisely how the dead-entry filter and the edge-weighted
  sizing model went stale - both were validated pre-shift and both had to be
  reversed today.

WHY LIVE DATA IS THE RIGHT TEST NOW
  The earlier post-shift check used the market-data panel, where 'rank' is a
  reconstruction. On the live book rank is observed directly: the runner enters
  in a fixed series order, so the sequence of our own posts within a window IS
  the rank the cap acts on. 239 fills across 111 windows.

  It also carries the fill bias with it, which the panel does not - and that
  bias is worth 27.6-58.9% of edge, so panel ranks flatter every rank equally.

THE BAR
  The cap only earns its keep if the marginal entry it ADMITS is positive.
  Lowering the cap to 2 is only justified if rank-3 entries are verifiably
  NEGATIVE - not merely smaller than rank-1. And the comparison must be in
  dollars, because dropping rank 3 removes entries that are not replaced.
"""
import glob
import json
from datetime import datetime

import numpy as np
import pandas as pd

from acct import api

RNG = np.random.default_rng(20261046)


def ts(x):
    try:
        return datetime.fromisoformat((x or "").replace("Z", "+00:00"))
    except Exception:
        return None


# post time gives the true within-window entry order the cap acts on
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
            if d.get("ev") == "post" and d.get("ticker") and d.get("price"):
                t = d["ticker"]
                u = ts(d.get("utc"))
                if t not in POST or (u and u < POST[t][1]):
                    POST[t] = (float(d["price"]), u)
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

o = []
for s in rows:
    t = s["ticker"]
    if t not in POST:
        continue
    n = float(s.get("yes_count_fp") or 0) + float(s.get("no_count_fp") or 0)
    cost = float(s.get("yes_total_cost_dollars") or 0) + float(
        s.get("no_total_cost_dollars") or 0)
    if n <= 0 or cost <= 0 or POST[t][1] is None:
        continue
    o.append(dict(ticker=t, window=t.split("-")[1], posted=POST[t][1],
                  n=n, px=cost / n,
                  coin=t.split("-")[0].replace("KX", "").replace("15M", ""),
                  pnl=float(s.get("revenue") or 0) / 100.0 - cost
                      - float(s.get("fee_cost") or 0)))
D = pd.DataFrame(o).sort_values(["window", "posted"])
D["rank"] = D.groupby("window").cumcount() + 1
print(f"fills {len(D)}   windows {D.window.nunique()}   "
      f"net ${D.pnl.sum():+.2f} ({D.pnl.sum()/D.n.sum()*100:+.2f}c/ct)\n")
if len(D) < 60:
    raise SystemExit("too few")

print("=" * 76)
print("EDGE BY WITHIN-WINDOW RANK (the order the cap actually admits them)")
print("=" * 76)
nw = D.window.nunique()
print(f"{'rank':>5} {'fills':>7} {'share of windows':>18} {'avg px':>8} "
      f"{'edge/ct':>9} {'net $':>10}")
for r, g in D.groupby("rank"):
    if r > 4:
        continue
    print(f"{int(r):>5} {len(g):>7} {len(g)/nw:>18.3f} "
          f"{g.px.mean()*100:>7.1f}c {g.pnl.sum()/g.n.sum()*100:>+8.2f}c "
          f"{g.pnl.sum():>+10.2f}")
print("\n  'share of windows' at rank 3 is the selection the cap relies on.")
print("  Pre-shift it was rare and therefore informative.")

print("\n" + "=" * 76)
print("IS THE MARGINAL (rank-3) ENTRY VERIFIABLY NEGATIVE?")
print("=" * 76)
m3 = D[D["rank"] == 3]
if len(m3) >= 8:
    wins = sorted(m3.window.unique())
    gw = {w: g for w, g in m3.groupby("window")}
    bs = np.sort(np.array([
        (lambda s: s.pnl.sum() / max(s.n.sum(), 1) * 100)(
            pd.concat([gw[wins[i]] for i in RNG.integers(0, len(wins), len(wins))]))
        for _ in range(8000)]))
    e = m3.pnl.sum() / m3.n.sum() * 100
    print(f"  rank-3 cohort: n {len(m3)}  edge {e:+.2f}c  "
          f"95% CI [{bs[200]:+.2f}, {bs[7799]:+.2f}]")
    print(f"  P(rank-3 is negative) {(bs < 0).mean():.4f}")
    print(f"  at qty 20 those {len(m3)} entries produced ${m3.pnl.sum():+.2f}")
else:
    print(f"  only {len(m3)} rank-3 fills - too few")

print("\n" + "=" * 76)
print("CAP 2 vs CAP 3 IN DOLLARS (window-clustered)")
print("=" * 76)
wins = sorted(D.window.unique())
gw = {w: g for w, g in D.groupby("window")}
for cap in (1, 2, 3, 4):
    k = D[D["rank"] <= cap]
    print(f"  cap {cap}: fills {len(k):>4}  net ${k.pnl.sum():>+8.2f}  "
          f"{k.pnl.sum()/max(k.n.sum(),1)*100:>+7.2f}c/ct")
d23 = []
for _ in range(8000):
    s = pd.concat([gw[wins[i]] for i in RNG.integers(0, len(wins), len(wins))])
    d23.append(s[s["rank"] <= 3].pnl.sum() - s[s["rank"] <= 2].pnl.sum())
d23 = np.sort(np.array(d23))
obs = D[D["rank"] <= 3].pnl.sum() - D[D["rank"] <= 2].pnl.sum()
print(f"\n  cap3 minus cap2: observed ${obs:+.2f}   "
      f"95% CI [{d23[200]:+.2f}, {d23[7799]:+.2f}]   "
      f"P(cap3 worse) {(d23 < 0).mean():.4f}")

print("\n" + "=" * 76)
print("DOES THE 3rd ENTRY STILL MARK AN ACTIVE WINDOW?")
print("=" * 76)
sz = D.groupby("window").size()
print(f"{'entries in window':>19} {'windows':>9} {'edge/ct of that window':>24}")
for k in sorted(sz.unique()):
    ws = sz[sz == k].index
    g = D[D.window.isin(ws)]
    print(f"{int(k):>19} {len(ws):>9} "
          f"{g.pnl.sum()/g.n.sum()*100:>23.2f}c")
print("\n  Pre-shift, 3-entry windows were the best ones - that was the whole")
print("  mechanism. If they are no longer better, the cap is admitting")
print("  ordinary entries and the selection argument has expired.")

print("\n" + "=" * 76)
print("VERDICT")
print("=" * 76)
if len(m3) >= 8:
    if bs[7799] < 0:
        print("  rank-3 is verifiably NEGATIVE - the cap should come down to 2.")
    elif bs[200] > 0:
        print("  rank-3 is verifiably POSITIVE - keep cap 3.")
    else:
        print("  UNRESOLVED - rank-3 interval spans zero. The bar requires the")
        print("  discarded cohort to be verifiably negative before cutting it,")
        print("  so cap 3 stays. Lowering it on an ambiguous slice would repeat")
        print("  the mistake the hours filter made: cutting positive entries.")
