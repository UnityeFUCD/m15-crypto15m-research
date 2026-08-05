"""Is the agreement rule - the ONLY filter deployed - still earning its keep?

WHY THIS ONE MATTERS MOST RIGHT NOW
  Three filters were deployed today. The dead-entry filter was reversed (it was
  costing $14.89/day) and the edge-weighted sizing model was reversed (it had
  stopped ranking). The agreement rule is the sole survivor, so it is the only
  thing currently discarding entries. If it is stale too, the system is giving
  up trades for nothing while sitting $127 off its peak.

  Its evidence was pre-shift: discarded cohort -19.80c, CI [-33.47,-5.00],
  +$18.12/day, better on 8/10 days. Everything else fitted pre-shift has failed
  when re-tested on current data. This has never been checked on LIVE fills.

WHAT THE RULE DOES
  Within a window, the first entry sets the direction. Any later candidate on
  the OPPOSITE side is skipped, and the slot is burned rather than backfilled.
  The runner logs each one as direction_disagree_skip with the ticker, the side
  we would have taken, and the bid we would have paid.

  That is enough to price the counterfactual exactly: look up how each skipped
  market actually settled, and compute the edge we declined.

THE BAR
  A filter only earns its keep if what it DISCARDS is verifiably negative. Not
  smaller than what we keep - negative. If the discarded cohort is positive,
  the rule is destroying money regardless of how good it looked pre-shift.

  18 live skips is thin. This test can therefore reject the rule if the
  discards are clearly positive, or fail to resolve - it cannot confirm it.
"""
import glob
import json

import numpy as np
import pandas as pd

from acct import api

RNG = np.random.default_rng(20261043)

SK = {}
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
            if d.get("ev") == "direction_disagree_skip" and d.get("ticker"):
                SK[d["ticker"]] = d
    except Exception:
        pass
print(f"distinct markets skipped by the agreement rule: {len(SK)}")
if len(SK) < 8:
    raise SystemExit("too few skips to say anything")

rows = []
for t, d in SK.items():
    c, j = api("/markets", {"tickers": t})
    m = ((j or {}).get("markets") or [None])[0]
    if not m or m.get("result") not in ("yes", "no"):
        continue
    side = d.get("side")
    bid = d.get("fav_bid")
    if side is None or bid is None:
        continue
    won = 1 if side == m["result"] else 0
    rows.append(dict(ticker=t, side=side, px=float(bid), won=won,
                     edge=won - float(bid),
                     wdir=d.get("window_dir"),
                     window=str(d.get("window"))[:16]))
D = pd.DataFrame(rows)
print(f"resolved: {len(D)}   windows touched: {D.window.nunique()}\n")
if len(D) < 8:
    raise SystemExit("too few resolved")

print("=" * 76)
print("THE COHORT THE RULE THREW AWAY")
print("=" * 76)
print(f"  n {len(D)}   avg price {D.px.mean()*100:.1f}c   "
      f"win {D.won.mean():.4f}   edge {D.edge.mean()*100:+.2f}c/contract")
wins = sorted(D.window.unique())
gw = {w: g for w, g in D.groupby("window")}
bs = np.sort(np.array([
    pd.concat([gw[wins[i]] for i in RNG.integers(0, len(wins), len(wins))]).edge.mean() * 100
    for _ in range(8000)]))
print(f"  window-clustered 95% CI [{bs[200]:+.2f}, {bs[7799]:+.2f}]   "
      f"P(discards were POSITIVE) {(bs > 0).mean():.4f}")
print(f"\n  at qty 20 those {len(D)} skips were worth "
      f"${D.edge.sum()*20:+.2f} in total")

print("\n" + "=" * 76)
print("HOW THE SKIPPED TRADES ACTUALLY WENT")
print("=" * 76)
print(f"{'ticker':>34} {'side':>5} {'bid':>7} {'result':>7} {'edge':>9}")
for r in D.sort_values("edge").itertuples():
    print(f"{r.ticker:>34} {r.side:>5} {r.px*100:>6.0f}c "
          f"{'WON' if r.won else 'lost':>7} {r.edge*100:>+8.0f}c")

print("\n" + "=" * 76)
print("WHAT ABOUT THE ENTRY THAT SET THE DIRECTION?")
print("=" * 76)
print("  The rule assumes the FIRST entry's direction is the right one. If the")
print("  first entries in these same windows did well, the rule is protecting")
print("  a good signal. If they did badly, it is anchoring on noise.")
FIRST = {}
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
                FIRST.setdefault(d["ticker"], float(d["price"]))
    except Exception:
        pass
same = []
for r in D.itertuples():
    stem = r.ticker.split("-")[1]
    for t2, p2 in FIRST.items():
        if t2.split("-")[1] != stem or t2 == r.ticker:
            continue
        c, j = api("/markets", {"tickers": t2})
        m = ((j or {}).get("markets") or [None])[0]
        if not m or m.get("result") not in ("yes", "no"):
            continue
        w = 1 if r.wdir == m["result"] else 0
        same.append(w - p2)
if len(same) >= 5:
    s = np.array(same)
    print(f"\n  entries taken in those same windows: n {len(s)}   "
          f"edge {s.mean()*100:+.2f}c")
    print(f"  entries the rule SKIPPED:            n {len(D)}   "
          f"edge {D.edge.mean()*100:+.2f}c")
    print(f"  difference {(s.mean()-D.edge.mean())*100:+.2f}c")

print("\n" + "=" * 76)
print("VERDICT")
print("=" * 76)
e = D.edge.mean() * 100
if bs[200] > 0:
    print(f"  The discarded cohort is clearly POSITIVE ({e:+.2f}c, CI excludes")
    print("  zero). The rule is destroying money and should come off.")
elif bs[7799] < 0:
    print(f"  The discarded cohort is clearly NEGATIVE ({e:+.2f}c, CI excludes")
    print("  zero). The rule is earning its keep - leave it deployed.")
else:
    print(f"  UNRESOLVED. Discards are {e:+.2f}c with the interval spanning")
    print("  zero, so this cannot confirm or reject the rule. The default is to")
    print("  leave a validated filter in place rather than churn it on 18")
    print("  observations - reversing it would need evidence, not an absence.")
