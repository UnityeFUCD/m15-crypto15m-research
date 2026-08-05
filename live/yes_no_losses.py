"""Are our losses disproportionately on the NO side, and is there a reason?

THE OBSERVATION
  "A lot of our losses are NO."

THE MECHANISM THAT WOULD EXPLAIN IT
  Settlement is A1 >= A0. The inequality is NOT strict, so an exact tie pays
  YES. A NO position therefore has to win outright; a YES position wins on a
  tie as well.

  Whether that matters depends entirely on how often ties actually happen, and
  that depends on the PRECISION of the underlying index. If the CF Benchmarks
  RTI is carried to enough decimals, an exact tie between two 60-second means
  is vanishingly rare and the tie-break is worth nothing. If it is rounded
  coarsely - and DOGE trades near $0.07, XRP near $2 - ties could be common
  enough to matter, and NO would carry a hidden structural disadvantage the
  quoted price does not reflect.

  A backtest asymmetry test was run earlier (script 32) and found NO slightly
  BETTER, with the sign flipping across buckets - inconclusive. That was the
  pre-shift regime and it did not look at ties at all.

WHAT THIS CHECKS
  1. the raw split of our wins and losses by side, live
  2. whether NO loses more than its price implies (the only fair comparison,
     since NO positions may simply be priced differently)
  3. how often YES settles overall - a tie-break advantage would show up as
     YES settling more often than the market's average price implies
  4. per coin, because the tie hypothesis predicts it is worst where the index
     has the fewest significant figures
"""
import glob
import json
from collections import defaultdict

import numpy as np

from acct import api

RNG = np.random.default_rng(20261012)

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
            if d.get("ev") == "opportunity" and d.get("ticker"):
                ORD[d["ticker"]] = dict(side=d.get("side"),
                                        px=d.get("post_at"))
    except Exception:
        pass

RES = {}
for ser in ("KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M", "KXDOGE15M",
            "KXHYPE15M"):
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
                RES[m["ticker"]] = m["result"]
        cur = j.get("cursor")
        if not cur or not b:
            break

rows = []
for t, o in ORD.items():
    r = RES.get(t)
    if r is None or not o.get("side") or o.get("px") is None:
        continue
    won = 1 if o["side"] == r else 0
    rows.append(dict(ticker=t, coin=t.split("15M")[0].replace("KX", ""),
                     side=o["side"], px=float(o["px"]), won=won,
                     result=r, edge=(1 - float(o["px"])) if won
                     else -float(o["px"])))
if len(rows) < 20:
    raise SystemExit(f"only {len(rows)} matched positions")
import pandas as pd
D = pd.DataFrame(rows)
print(f"live positions matched to a settlement: {len(D)}\n")

print("=" * 70)
print("1. THE RAW SPLIT - is the observation right?")
print("=" * 70)
L = D[D.won == 0]
W = D[D.won == 1]
print(f"  all positions : {len(D)}   YES {(D.side=='yes').sum()} "
      f"({(D.side=='yes').mean():.3f})   NO {(D.side=='no').sum()} "
      f"({(D.side=='no').mean():.3f})")
print(f"  our LOSSES    : {len(L)}   YES {(L.side=='yes').sum()} "
      f"({(L.side=='yes').mean():.3f})   NO {(L.side=='no').sum()} "
      f"({(L.side=='no').mean():.3f})")
print(f"  our WINS      : {len(W)}   YES {(W.side=='yes').sum()} "
      f"({(W.side=='yes').mean():.3f})   NO {(W.side=='no').sum()} "
      f"({(W.side=='no').mean():.3f})")
print(f"\n  -> losses are {(L.side=='no').mean():.1%} NO against a "
      f"{(D.side=='no').mean():.1%} NO book")

print("\n" + "=" * 70)
print("2. THE FAIR TEST - does each side beat ITS OWN price?")
print("=" * 70)
print(f"{'side':>6} {'n':>5} {'avg px':>8} {'required':>10} {'actual win':>12} "
      f"{'SURPLUS':>10} {'edge/ct':>9}")
for s in ("yes", "no"):
    g = D[D.side == s]
    if not len(g):
        continue
    print(f"{s:>6} {len(g):>5} {g.px.mean()*100:>7.1f}c {g.px.mean()*100:>9.1f}% "
          f"{g.won.mean()*100:>11.1f}% {(g.won.mean()-g.px.mean())*100:>+9.1f}pp "
          f"{g.edge.mean()*100:>+8.2f}c")
a, b = D[D.side == "yes"], D[D.side == "no"]
if len(a) > 10 and len(b) > 10:
    real = ((b.won.mean() - b.px.mean()) - (a.won.mean() - a.px.mean())) * 100
    bs = []
    for _ in range(8000):
        x = a.sample(len(a), replace=True)
        y = b.sample(len(b), replace=True)
        bs.append(((y.won.mean() - y.px.mean())
                   - (x.won.mean() - x.px.mean())) * 100)
    bs = np.sort(np.array(bs))
    print(f"\n  NO surplus minus YES surplus: {real:+.2f}pp   "
          f"95% CI [{bs[200]:+.2f}, {bs[7799]:+.2f}]")
    print(f"  P(NO genuinely worse) = {(bs < 0).mean():.4f}")

print("\n" + "=" * 70)
print("3. THE TIE-BREAK TEST - does YES settle more than prices imply?")
print("=" * 70)
print("  Every market we looked at, regardless of which side we took.\n")
allm = defaultdict(lambda: [0, 0])
for t, r in RES.items():
    coin = t.split("15M")[0].replace("KX", "")
    allm[coin][0] += 1
    allm[coin][1] += 1 if r == "yes" else 0
print(f"{'coin':>6} {'settled':>8} {'YES share':>11}")
tot_n = tot_y = 0
for c, (n, y) in sorted(allm.items()):
    if n < 30:
        continue
    tot_n += n
    tot_y += y
    print(f"{c:>6} {n:>8} {y/n:>11.4f}")
print(f"{'ALL':>6} {tot_n:>8} {tot_y/tot_n:>11.4f}")
se = np.sqrt(0.25 / tot_n)
z = (tot_y / tot_n - 0.5) / se
print(f"\n  vs a fair coin: {tot_y/tot_n:.4f} +/- {se:.4f}  (z = {z:+.2f})")
print("  A tie-break advantage for YES would push this ABOVE 0.5.")
print("  At/below 0.5 the tie-break is worth nothing in practice - ties are")
print("  too rare at the index precision Kalshi uses.")

print("\n" + "=" * 70)
print("4. PER COIN - our own side performance")
print("=" * 70)
print(f"{'coin':>6} {'side':>5} {'n':>4} {'avg px':>8} {'win':>7} {'surplus':>9}")
for (c, s), g in D.groupby(["coin", "side"]):
    if len(g) < 4:
        continue
    print(f"{c:>6} {s:>5} {len(g):>4} {g.px.mean()*100:>7.1f}c "
          f"{g.won.mean():>7.3f} {(g.won.mean()-g.px.mean())*100:>+8.1f}pp")
