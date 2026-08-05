"""The deep structure: what do we give up to ADVERSE SELECTION at the fill?

THE MECHANISM
  We post a maker bid at the favourite's bid, then wait. Two things can happen:

    favourite STRENGTHENS -> price walks up away from our bid -> NO FILL.
                             We miss a market that was going to win.
    favourite WEAKENS     -> sellers come down and hit our bid -> FILL.
                             We buy at 73c something the market now says is 68c.

  So fills are selected toward the losing side and non-fills toward the winning
  side. That is not a flaw to be removed - it is the price of being a maker,
  and the strategy is profitable precisely because the longshot premium we
  collect EXCEEDS this cost. Understanding the balance is understanding why
  the strategy works.

WHY IT MATTERS RIGHT NOW
  Two live flags may be one phenomenon:
    - F_Q fell to 0.8626, and the degradation is in SHALLOW queues, which is
      not a queue-depth story. It is what "price walked away" looks like.
    - the 80-85c bucket went negative. At 85c the favourite is already strong,
      so there is little room left to strengthen and plenty to fall. Adverse
      selection should bite hardest exactly there, while the longshot premium
      is at its smallest.

  If both are the same mechanism, the strategy has a natural price ceiling that
  is structural rather than statistical.

THE TEST
  Every order we placed - FILLED OR NOT - names a market that later settled.
  Compare how the markets we filled settled against the markets we did not.
  Unfilled orders have no position and therefore no settlement record, so the
  outcome is taken from the market object itself.
"""
import glob
import json
from collections import defaultdict

import numpy as np

from acct import api

RNG = np.random.default_rng(20260901)

# every order we submitted, with its outcome at the exchange
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
                ORD.setdefault(t, {})["px"] = d.get("post_at")
                ORD[t]["side"] = d.get("side")
                ORD[t]["qa"] = d.get("queue_ahead")
            if d.get("ev") == "terminal":
                ORD.setdefault(t, {})["target"] = d.get("target")
                ORD[t]["filled"] = d.get("FILLED")
    except Exception:
        pass
print(f"orders seen: {len(ORD):,}")

# settlement result per market, straight from the market object
RES = {}
for ser in ("KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M", "KXDOGE15M",
            "KXHYPE15M", "KXBNB15M"):
    cur = None
    for _ in range(4):
        p = {"series_ticker": ser, "status": "settled", "limit": 100}
        if cur:
            p["cursor"] = cur
        c, j = api("/markets", p)
        if c != 200 or not j:
            break
        b = j.get("markets") or []
        for m in b:
            if m.get("result") in ("yes", "no"):
                RES[m.get("ticker")] = m["result"]
        cur = j.get("cursor")
        if not cur or not b:
            break
print(f"settled markets fetched: {len(RES):,}\n")

rows = []
for t, o in ORD.items():
    r = RES.get(t)
    if r is None or o.get("px") is None or o.get("side") is None:
        continue
    tgt = o.get("target")
    fil = o.get("filled")
    if tgt is None or fil is None:
        continue
    # we bought the favourite on `side`; it wins if the result matches
    won = 1 if ((o["side"] == "yes" and r == "yes") or
                (o["side"] == "no" and r == "no")) else 0
    rows.append(dict(t=t, px=float(o["px"]), won=won,
                     frac=float(fil) / max(float(tgt), 1),
                     filled=float(fil) > 0, qa=o.get("qa") or 0))
if len(rows) < 25:
    raise SystemExit(f"only {len(rows)} matched orders - too few")

F = [r for r in rows if r["filled"]]
U = [r for r in rows if not r["filled"]]
print(f"=== DID THE ORDERS WE MISSED WIN MORE THAN THE ONES WE GOT? ===")
print(f"  matched orders {len(rows)}   filled {len(F)}   never filled {len(U)}")
if len(U) < 5:
    print("  too few unfilled orders to conclude")
else:
    wf = np.mean([r["won"] for r in F])
    wu = np.mean([r["won"] for r in U])
    print(f"  FILLED     win rate {wf:.4f}  (n {len(F)}, avg px "
          f"{np.mean([r['px'] for r in F])*100:.1f}c)")
    print(f"  NOT FILLED win rate {wu:.4f}  (n {len(U)}, avg px "
          f"{np.mean([r['px'] for r in U])*100:.1f}c)")
    bs = []
    for _ in range(8000):
        a = RNG.integers(0, len(F), len(F))
        b = RNG.integers(0, len(U), len(U))
        bs.append(np.mean([U[i]["won"] for i in b])
                  - np.mean([F[i]["won"] for i in a]))
    bs = np.sort(np.array(bs))
    print(f"  MISSED minus FILLED: {(wu-wf)*100:+.2f}pp   95% CI "
          f"[{bs[200]*100:+.2f},{bs[7799]*100:+.2f}]   "
          f"p {2*min((bs<=0).mean(),(bs>=0).mean()):.4f}")
    print("\n  Positive means the markets that walked away from us were the")
    print("  winners - the cost of being a maker, paid in missed upside.")

print("\n=== ADVERSE SELECTION BY PRICE - is it worse where the premium is thin? ===")
print(f"{'price':>10} {'orders':>7} {'filled':>7} {'fill rate':>10} "
      f"{'win|filled':>11} {'win|missed':>11} {'gap':>8}")
for lo, hi in ((0.65, 0.70), (0.70, 0.75), (0.75, 0.80), (0.80, 0.90)):
    g = [r for r in rows if lo <= r["px"] < hi]
    if len(g) < 6:
        continue
    gf = [r for r in g if r["filled"]]
    gu = [r for r in g if not r["filled"]]
    wf = np.mean([r["won"] for r in gf]) if gf else float("nan")
    wu = np.mean([r["won"] for r in gu]) if gu else float("nan")
    gap = (wu - wf) * 100 if (gf and gu) else float("nan")
    print(f"{f'{lo*100:.0f}-{hi*100:.0f}c':>10} {len(g):>7} {len(gf):>7} "
          f"{len(gf)/len(g):>10.3f} {wf:>11.4f} "
          f"{wu if gu else float('nan'):>11.4f} {gap:>+8.2f}")

print("\n=== THE ECONOMICS THIS IMPLIES ===")
print("  realised edge = longshot premium collected - adverse selection paid")
print("  The premium shrinks as price rises (less longshot left to sell), while")
print("  adverse selection does not. Where the two cross, the bucket turns")
print("  negative - which is a STRUCTURAL ceiling on price, not bad luck.")
for lo, hi in ((0.65, 0.75), (0.75, 0.90)):
    g = [r for r in rows if lo <= r["px"] < hi and r["filled"]]
    if len(g) < 5:
        continue
    px = np.mean([r["px"] for r in g])
    w = np.mean([r["won"] for r in g])
    print(f"    {lo*100:.0f}-{hi*100:.0f}c: filled win {w:.4f} vs price {px:.4f} "
          f"-> realised excess {(w-px)*100:+.2f}c")
