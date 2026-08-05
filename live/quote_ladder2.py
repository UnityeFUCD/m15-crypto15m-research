"""The quote ladder, with taker fees charged where the rung crosses the spread.

THE FLAW IN THE FIRST VERSION
  It reported bid+1 beating bid+0 by +0.63c/posted with P(<=0)=0.0049. That is
  not a quote-improvement result. In 60% of the sample the spread is exactly
  one tick, so "bid + 1 tick" IS the ask - the order crosses and executes as a
  TAKER, which incurs a fee the simulation never charged.

  Kalshi's taker fee is ceil(0.07*C*p*(1-p)*1e4)/1e4 per block. At p=0.70 on 20
  contracts that is 1.47c per contract, more than the entire claimed gain. So
  the headline was an artifact of free crossing.

WHAT IS ACTUALLY BEING ASKED
  Two separate questions were tangled together:

    1. QUOTE IMPROVEMENT - posting strictly INSIDE the spread, still as a
       maker, still fee-free. Only possible when the spread is 2+ ticks, which
       is 39.7% of markets.
    2. CROSSING - posting at or through the ask, which is taking, and pays the
       fee. Already tested at full aggression and rejected (-$5.58/day).

  Only (1) is new. This prices it correctly: maker rungs pay nothing, crossing
  rungs pay the real fee, and the comparison is restricted to markets where
  improvement is actually possible.

WHY IT STILL MATTERS EVEN IF SMALL
  The mechanism is confirmed and it is the cleanest result of the day:
  fill|LOSE is 1.0000 at every rung while fill|WIN is 0.856 at the bid and
  0.984 four ticks up. We ALWAYS fill the losers. The only question is how much
  it costs to stop missing the winners.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

LIVE = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261059)
TICK, QTY = 0.01, 20


def taker_fee_per_contract(p):
    return math.ceil(0.07 * QTY * p * (1.0 - p) * 1e4) / 1e4 / QTY


D = pd.read_parquet(LIVE / "ladder_paths.parquet").drop_duplicates(
    subset=["ticker"])
print(f"paths: {len(D):,}")

out = []
for r in D.itertuples():
    p = sorted(json.loads(r.path), key=lambda x: -x["ml"])
    entry = next((k for k in p if 8 <= k["ml"] <= 14), None)
    if entry is None:
        continue
    later = [k for k in p if k["ml"] < entry["ml"]]
    if len(later) < 4:
        continue
    yb, ya = entry["bc"], entry["ac"]
    if yb <= 0 or ya >= 1 or yb >= ya:
        continue
    side, bid, ask = (("yes", yb, ya) if yb >= 0.5
                      else ("no", 1.0 - ya, 1.0 - yb))
    if not (0.65 <= bid < 0.80):
        continue
    won = 1 if side == r.result else 0
    best_ask = min(k["al"] for k in later) if side == "yes" else None
    best_bid = max(k["bh"] for k in later) if side == "no" else None
    room = int(round((ask - bid) / TICK))
    rung = {}
    for t in range(0, 5):
        px = bid + t * TICK
        if px > ask + 1e-9:
            break
        crosses = px >= ask - 1e-9
        if crosses:
            filled, fee = True, taker_fee_per_contract(px)
        else:
            fee = 0.0
            filled = (best_ask <= px + 1e-9) if side == "yes" \
                else (best_bid >= (1.0 - px) - 1e-9)
        rung[t] = (px, bool(filled), fee, bool(crosses))
    out.append(dict(ticker=r.ticker, date=r.date, side=side, bid=bid, ask=ask,
                    room=room, won=won, rung=rung))
O = pd.DataFrame(out)
print(f"entries in band: {len(O):,}   days {O.date.nunique()}")
print(f"spread distribution: 1 tick {(O.room == 1).mean():.3f}   "
      f"2 ticks {(O.room == 2).mean():.3f}   "
      f"3+ {(O.room >= 3).mean():.3f}\n")


def ladder(sub, label):
    print("=" * 78)
    print(label)
    print("=" * 78)
    print(f"{'rung':>16} {'n':>6} {'crosses':>9} {'fill':>7} {'fill|WIN':>10} "
          f"{'fee':>7} {'edge/posted':>13}")
    vals = {}
    for t in range(0, 5):
        s = sub[[t in r for r in sub.rung]]
        if len(s) < 80:
            continue
        px = np.array([r[t][0] for r in s.rung])
        fl = np.array([r[t][1] for r in s.rung])
        fe = np.array([r[t][2] for r in s.rung])
        cr = np.array([r[t][3] for r in s.rung])
        wn = s.won.to_numpy()
        v = (wn - px - fe) * fl
        vals[t] = (s, v)
        print(f"{'bid+' + str(t):>16} {len(s):>6} {cr.mean():>9.3f} "
              f"{fl.mean():>7.4f} {fl[wn == 1].mean():>10.4f} "
              f"{fe.mean()*100:>6.2f}c {v.mean()*100:>+12.2f}c")
    return vals


def compare(vals, base=0):
    if base not in vals:
        return
    print(f"\n{'comparison':>22} {'diff':>9} {'95% CI':>20} {'days better':>13} "
          f"{'P(<=0)':>9}")
    sb, vb = vals[base]
    bmap = dict(zip(sb.ticker, vb))
    for t in sorted(vals):
        if t == base:
            continue
        st, vt = vals[t]
        rows = [(d, v - bmap[tk]) for tk, d, v in zip(st.ticker, st.date, vt)
                if tk in bmap]
        if len(rows) < 100:
            continue
        DF = pd.DataFrame(rows, columns=["date", "d"])
        per = DF.groupby("date").d.mean() * 100
        bs = np.sort(np.array([RNG.choice(per.values, len(per), True).mean()
                               for _ in range(8000)]))
        print(f"{'bid+' + str(t) + ' vs bid+' + str(base):>22} "
              f"{per.mean():>+8.2f}c [{bs[200]:>+7.2f},{bs[7799]:>+7.2f}] "
              f"{(per > 0).sum():>6}/{len(per):<6} {(bs <= 0).mean():>9.4f}")


v_all = ladder(O, "ALL MARKETS (crossing rungs now pay the real taker fee)")
compare(v_all)

print()
W = O[O.room >= 2]
v_w = ladder(W, f"QUOTE IMPROVEMENT ONLY - spread 2+ ticks "
                f"({len(W)} of {len(O)}, {len(W)/len(O):.1%})")
compare(v_w)

print("\n" + "=" * 78)
print("THE STRICTLY-INSIDE-THE-SPREAD RESULT")
print("=" * 78)
print("  On wide-spread markets, bid+1 is a genuine maker improvement: still")
print("  inside the spread, still no fee. That is the only rung that is both")
print("  new and free.\n")
sub = O[O.room >= 2].copy()
r0 = np.array([(r[0][0], r[0][1]) for r in sub.rung])
r1 = np.array([(r[1][0], r[1][1]) for r in sub.rung])
wn = sub.won.to_numpy()
e0 = ((wn - r0[:, 0]) * r0[:, 1]).mean() * 100
e1 = ((wn - r1[:, 0]) * r1[:, 1]).mean() * 100
print(f"  post at bid   : fill {r0[:,1].mean():.4f}  "
      f"fill|WIN {r0[:,1][wn==1].mean():.4f}  edge/posted {e0:+.2f}c")
print(f"  post at bid+1 : fill {r1[:,1].mean():.4f}  "
      f"fill|WIN {r1[:,1][wn==1].mean():.4f}  edge/posted {e1:+.2f}c")
print(f"  gain {e1-e0:+.2f}c per posted order, on {len(sub)/len(O):.1%} of markets")
print(f"  blended across all markets: {(e1-e0)*len(sub)/len(O):+.3f}c/posted")
per = (pd.DataFrame({"date": sub.date,
                     "d": ((wn - r1[:, 0]) * r1[:, 1]
                           - (wn - r0[:, 0]) * r0[:, 1])})
       .groupby("date").d.mean() * 100)
bs = np.sort(np.array([RNG.choice(per.values, len(per), True).mean()
                       for _ in range(8000)]))
print(f"  day-clustered 95% CI [{bs[200]:+.2f}, {bs[7799]:+.2f}]   "
      f"P(<=0) {(bs <= 0).mean():.4f}   better {(per > 0).sum()}/{len(per)} days")

print("\n" + "=" * 78)
print("WHAT THIS IS WORTH IN DOLLARS")
print("=" * 78)
print("  The runner posts ~130 orders/day at qty 20 = ~2,600 contracts.")
g = (e1 - e0) * len(sub) / len(O) / 100.0
print(f"  blended gain {g*100:+.3f}c/contract -> ${g*2600:+.2f}/day")
print("\n  And the caveat that will not go away: these fill rates assume our")
print("  resting order is at the FRONT of the queue at that price. Size ahead")
print("  of us absorbs some of these fills, so this is an upper bound.")
