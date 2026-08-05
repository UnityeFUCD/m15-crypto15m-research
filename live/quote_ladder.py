"""What is the OPTIMAL posting price? The full fill-vs-edge ladder.

WHY THIS REOPENS
  Quote improvement was killed early on the grounds that 84% of spreads are one
  tick, leaving no room to post better than the bid. That figure came from a
  2-day live sample - and that sample only contains markets WE CHOSE TO TRADE,
  which are the liquid, tight-spread ones. It was selection-biased.

  Measured properly across 73 days: mean spread 2.08c, p90 4.00c, and only
  52.9% are one tick or less. Roughly half the time there is real room between
  the bid and the ask, and we have never tested posting into it.

WHY IT ATTACKS THE ACTUAL LEAK
  The largest identified loss is not a bad filter or a bad hour - it is that we
  fill 96.2% of losers and only 84.3% of winners. That gap turns a population
  premium of ~+8.7c into a realised ~+2.0c.

  The mechanism is mechanical: a favourite that goes on to WIN strengthens and
  runs away from our resting bid, so we never fill it. One that goes on to LOSE
  weakens, comes back through our price, and fills us. Resting deeper is worse;
  resting closer to the ask is the only direction that can capture more winners.

  Taker - the extreme version - was tested and lost $5.58/day, because crossing
  a 2.08c average spread is too expensive. But taker is the LAST rung of a
  ladder. The rungs in between have never been priced.

HOW FILL IS DETERMINED EXACTLY, NOT MODELLED
  Candlesticks carry yes_bid high/low and yes_ask high/low per minute. A
  resting order fills when someone crosses to it:

      buying YES at P   fills if the yes_ask ever drops to P or below
      buying NO  at P   fills if the yes_bid ever rises to (1-P) or above

  So for every candidate price the fill is computed from the observed book, not
  assumed. Combined with the settled result, that gives the complete
  fill-rate-and-edge curve as a function of how aggressively we post.

WHAT WOULD MAKE THIS WRONG
  Queue position. Seeing the ask touch our price means someone sold there; it
  does not guarantee they reached US rather than the size ahead. So these fill
  rates are an UPPER bound, and the improvement from posting higher is
  overstated - though the bias applies to every rung, so the SHAPE of the
  ladder is more trustworthy than its level.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from acct import api

LIVE = Path(__file__).resolve().parent
CACHE = LIVE / "ladder_paths.parquet"
NSAMPLE = 5000
RNG = np.random.default_rng(20261058)

U = pd.read_parquet(LIVE / "underlying.parquet")
_exp = pd.to_datetime(U.close, format="mixed", utc=True)
U["close"] = (pd.to_datetime(U.wkey, format="%y%b%d%H%M", utc=True)
              + pd.Timedelta(hours=4))
U = U[((_exp - U.close).dt.total_seconds() / 60).round(1) == 5.0].copy()
U["slot"] = U.close.dt.hour * 4 + U.close.dt.minute // 15
POOL = U[U.slot % 2 == 0].reset_index(drop=True)
if len(POOL) > NSAMPLE:
    POOL = POOL.iloc[RNG.choice(len(POOL), NSAMPLE, replace=False)]
print(f"sampling {len(POOL):,} markets over {POOL.close.dt.date.nunique()} days")

rows, done = [], set()
if CACHE.exists():
    old = pd.read_parquet(CACHE)
    rows, done = old.to_dict("records"), set(old.ticker)
    print(f"cache: {len(rows):,}")
todo = POOL[~POOL.ticker.isin(done)]
print(f"to fetch: {len(todo):,}\n")


def fetch(series, ticker, cts, tries=3):
    for a in range(tries):
        try:
            return api(f"/series/{series}/markets/{ticker}/candlesticks",
                       {"period_interval": 1, "start_ts": cts - 16 * 60,
                        "end_ts": cts})
        except Exception:
            if a == tries - 1:
                return None, None
            time.sleep(1.5 * (a + 1))
    return None, None


t0, errs = time.time(), 0
for i, r in enumerate(todo.itertuples()):
    cts = int(r.close.timestamp())
    c, j = fetch(r.series, r.ticker, cts)
    if c != 200 or not j:
        errs += 1
        continue
    path = []
    for k in (j.get("candlesticks") or []):
        ml = (cts - int(k.get("end_period_ts") or 0)) / 60.0
        try:
            path.append(dict(
                ml=ml,
                bc=float(k["yes_bid"]["close_dollars"]),
                bh=float(k["yes_bid"]["high_dollars"]),
                ac=float(k["yes_ask"]["close_dollars"]),
                al=float(k["yes_ask"]["low_dollars"])))
        except (KeyError, TypeError, ValueError):
            continue
    if len(path) >= 10:
        rows.append(dict(ticker=r.ticker, coin=r.coin,
                         date=str(r.close.date()), result=r.result,
                         path=json.dumps(path)))
    if (i + 1) % 200 == 0:
        el = time.time() - t0
        print(f"  {i+1}/{len(todo)} kept {len(rows)} errs {errs} "
              f"eta {el/(i+1)*(len(todo)-i-1)/60:.1f}min", flush=True)
        pd.DataFrame(rows).drop_duplicates(subset=["ticker"]).to_parquet(CACHE)

D = pd.DataFrame(rows).drop_duplicates(subset=["ticker"])
D.to_parquet(CACHE)
print(f"\npaths stored: {len(D):,}\n")

TICK = 0.01
out = []
for r in D.itertuples():
    p = json.loads(r.path)
    p.sort(key=lambda x: -x["ml"])
    entry = next((k for k in p if 8 <= k["ml"] <= 14), None)
    if entry is None:
        continue
    later = [k for k in p if k["ml"] < entry["ml"]]
    if len(later) < 4:
        continue
    yb, ya = entry["bc"], entry["ac"]
    if yb <= 0 or ya >= 1 or yb >= ya:
        continue
    if yb >= 0.5:
        side, bid, ask = "yes", yb, ya
    else:
        side, bid, ask = "no", 1.0 - ya, 1.0 - yb
    if not (0.65 <= bid < 0.80):
        continue
    won = 1 if side == r.result else 0
    # exact fill test at each rung of the ladder
    if side == "yes":
        best_ask = min(k["al"] for k in later)
    else:
        best_bid = max(k["bh"] for k in later)
    room = int(round((ask - bid) / TICK))
    rung = {}
    for t in range(0, 5):
        px = bid + t * TICK
        if px > ask + 1e-9:
            break
        if side == "yes":
            filled = best_ask <= px + 1e-9
        else:
            filled = best_bid >= (1.0 - px) - 1e-9
        rung[t] = (px, bool(filled))
    out.append(dict(ticker=r.ticker, coin=r.coin, date=r.date, side=side,
                    bid=bid, ask=ask, spread_c=(ask - bid) * 100, room=room,
                    won=won, rung=rung))
print(f"entries in band with a usable path: {len(out):,}")
if len(out) < 200:
    raise SystemExit("too few - rerun to continue fetching")

print("\n" + "=" * 78)
print("THE LADDER - how fill rate and edge change as we post more aggressively")
print("=" * 78)
print(f"{'rung':>18} {'n':>6} {'fill':>7} {'fill|WIN':>10} {'fill|LOSE':>11} "
      f"{'avg px':>8} {'edge/posted':>13}")
days = sorted({o["date"] for o in out})
res = {}
for t in range(0, 5):
    sub = [o for o in out if t in o["rung"]]
    if len(sub) < 100:
        continue
    fl = np.array([o["rung"][t][1] for o in sub])
    px = np.array([o["rung"][t][0] for o in sub])
    wn = np.array([o["won"] for o in sub])
    e = ((wn - px) * fl).mean() * 100          # per POSTED order
    fw = fl[wn == 1].mean()
    fll = fl[wn == 0].mean()
    res[t] = dict(sub=sub, fl=fl, px=px, wn=wn, e=e)
    print(f"{'bid + ' + str(t) + ' tick':>18} {len(sub):>6} {fl.mean():>7.4f} "
          f"{fw:>10.4f} {fll:>11.4f} {px.mean()*100:>7.2f}c {e:>+12.2f}c")
print("\n  'edge/posted' counts an unfilled order as zero, which is what it is.")
print("  fill|WIN vs fill|LOSE is the adverse selection: the gap should NARROW")
print("  as we post higher, which is the entire point.")

print("\n" + "=" * 78)
print("IS THE BEST RUNG BETTER THAN THE BID, DAY-CLUSTERED?")
print("=" * 78)
if 0 in res:
    base = res[0]
    for t in sorted(res):
        if t == 0:
            continue
        common = {o["ticker"] for o in res[t]["sub"]} & {o["ticker"] for o in base["sub"]}
        a = {o["ticker"]: o for o in res[t]["sub"]}
        b = {o["ticker"]: o for o in base["sub"]}
        rowsd = []
        for tk in common:
            oa, ob = a[tk], b[tk]
            va = (oa["won"] - oa["rung"][t][0]) * oa["rung"][t][1]
            vb = (ob["won"] - ob["rung"][0][0]) * ob["rung"][0][1]
            rowsd.append((oa["date"], va - vb))
        if len(rowsd) < 100:
            continue
        DF = pd.DataFrame(rowsd, columns=["date", "d"])
        per = DF.groupby("date").d.mean() * 100
        bs = np.sort(np.array([RNG.choice(per.values, len(per), True).mean()
                               for _ in range(8000)]))
        print(f"  bid+{t} minus bid: {per.mean():+.2f}c/posted   "
              f"95% CI [{bs[200]:+.2f}, {bs[7799]:+.2f}]   "
              f"better {(per > 0).sum()}/{len(per)} days   "
              f"P(<=0) {(bs <= 0).mean():.4f}")

print("\n" + "=" * 78)
print("ONLY WHERE THERE IS ROOM (spread > 1 tick)")
print("=" * 78)
wide = [o for o in out if o["room"] >= 2]
print(f"  markets with a 2+ tick spread: {len(wide)} of {len(out)} "
      f"({len(wide)/len(out):.3f})")
if len(wide) >= 150:
    print(f"{'rung':>18} {'n':>6} {'fill':>7} {'fill|WIN':>10} "
          f"{'edge/posted':>13}")
    for t in range(0, 4):
        sub = [o for o in wide if t in o["rung"]]
        if len(sub) < 60:
            continue
        fl = np.array([o["rung"][t][1] for o in sub])
        px = np.array([o["rung"][t][0] for o in sub])
        wn = np.array([o["won"] for o in sub])
        print(f"{'bid + ' + str(t) + ' tick':>18} {len(sub):>6} "
              f"{fl.mean():>7.4f} {fl[wn == 1].mean():>10.4f} "
              f"{((wn - px) * fl).mean()*100:>+12.2f}c")
    print("\n  This is the deployable subset: when the spread is one tick there")
    print("  is nothing to do, so any rule would only act on these.")
