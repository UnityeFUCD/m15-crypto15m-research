"""Maker vs taker, priced against 73 days instead of 2.

WHY THIS IS WORTH REDOING
  This morning the comparison came out UNDECIDABLE: break-even sat at
  P(fill|win) = 0.8710, inside that parameter's own CI of [0.7703, 0.9100], and
  P(taker better) was 0.77. Two days of live orders could not separate them.

  Two things have changed since:

  1. THE PREMIUM IS BIGGER THAN BELIEVED. Reconstructing 73 days of
     candlesticks puts the population premium at +5.01c over the recent half
     and +8.74c over the last 14 days, with no decay trend. The "78% decay
     after Aug 1" was an artifact of comparing two differently-built datasets.

  2. THAT CHANGES THE ARITHMETIC, NOT JUST THE INPUTS. The maker's handicap is
     that it fills 96.2% of losers but only 84.3% of winners - it
     systematically misses the good ones. The COST of missing a winner is
     proportional to the edge. The taker's handicap is the spread plus fees,
     which is a fixed toll per contract regardless of edge.

     So a larger premium favours the taker, and the break-even moves. A
     comparison run against an understated edge was biased toward the maker.

WHAT IS MEASURED
  Entries reconstructed exactly as the runner takes them - favourite side,
  8-14 minutes before close, 65-80c - across 73 days, now storing BOTH sides of
  the book so the taker's actual cost is the real ask, not an assumed spread.

    maker: pays the bid, fills P(fill|win)/P(fill|lose) of the time, zero fees
    taker: pays the ask, fills always, pays ceil(0.07*C*p*(1-p)*1e4)/1e4

WHAT WOULD STILL KILL IT
  - the quoted ask is top-of-book. A 20-lot may walk it, so the taker number
    here is OPTIMISTIC and is reported as an upper bound.
  - taking removes us from the liquidity-provision role. If any part of the
    longshot premium is payment for providing liquidity rather than for
    absorbing longshot demand, that part does not survive the switch. Nothing
    in this data can settle that, and it is stated rather than assumed away.
  - it is a STRATEGY CHANGE and will not be deployed on research alone.
"""
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from acct import api

LIVE = Path(__file__).resolve().parent
CACHE = LIVE / "book_history.parquet"
P_FILL_WIN, P_FILL_LOSE = 0.843, 0.962
QTY = 20


def taker_fee(c, p):
    return math.ceil(0.07 * c * p * (1.0 - p) * 1e4) / 1e4


U = pd.read_parquet(LIVE / "underlying.parquet")
_exp = pd.to_datetime(U.close, format="mixed", utc=True)
U["close"] = (pd.to_datetime(U.wkey, format="%y%b%d%H%M", utc=True)
              + pd.Timedelta(hours=4))
U = U[((_exp - U.close).dt.total_seconds() / 60).round(1) == 5.0].copy()
U["slot"] = U.close.dt.hour * 4 + U.close.dt.minute // 15
SAMPLE = U[U.slot % 12 == 0].sort_values(["coin", "close"]).reset_index(drop=True)

rows, done = [], set()
if CACHE.exists():
    old = pd.read_parquet(CACHE)
    rows, done = old.to_dict("records"), set(old.ticker)
    print(f"cache: {len(rows):,}")
todo = SAMPLE[~SAMPLE.ticker.isin(done)]
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
    best = None
    for k in (j.get("candlesticks") or []):
        ml = (cts - int(k.get("end_period_ts") or 0)) / 60.0
        if not (8 <= ml <= 14):
            continue
        try:
            yb = float(k["yes_bid"]["close_dollars"])
            ya = float(k["yes_ask"]["close_dollars"])
        except (KeyError, TypeError, ValueError):
            continue
        if yb <= 0 or ya >= 1 or yb >= ya:
            continue
        if yb >= 0.5:
            side, bid, ask = "yes", yb, ya
        else:
            side, bid, ask = "no", 1.0 - ya, 1.0 - yb
        if not (0.65 <= bid < 0.80):
            continue
        if best is None or ml > best["ml"]:
            best = dict(ml=ml, side=side, bid=bid, ask=ask,
                        vol=float(k.get("volume_fp") or 0))
    if best:
        won = 1 if best["side"] == r.result else 0
        rows.append(dict(ticker=r.ticker, coin=r.coin,
                         date=str(r.close.date()), side=best["side"],
                         bid=best["bid"], ask=best["ask"], won=won,
                         vol=best["vol"], ml=best["ml"]))
    if (i + 1) % 400 == 0:
        el = time.time() - t0
        print(f"  {i+1}/{len(todo)} kept {len(rows)} errs {errs} "
              f"eta {el/(i+1)*(len(todo)-i-1)/60:.1f}min", flush=True)
        pd.DataFrame(rows).drop_duplicates(subset=["ticker"]).to_parquet(CACHE)

D = pd.DataFrame(rows).drop_duplicates(subset=["ticker"])
D.to_parquet(CACHE)
days = sorted(D.date.unique())
print(f"\nentries {len(D):,} over {len(days)} days")

D["spread"] = D.ask - D.bid
print("\n" + "=" * 78)
print("THE SPREAD WE WOULD PAY, MEASURED OVER 73 DAYS")
print("=" * 78)
print(f"  median {D.spread.median()*100:.2f}c   mean {D.spread.mean()*100:.2f}c "
      f"  p90 {D.spread.quantile(.9)*100:.2f}c   "
      f"one tick or less {(D.spread <= 0.0101).mean():.3f}")
print(f"  (live 2-day sample said median 1.00c, mean 1.21c, 80.8% one tick)")

print("\n" + "=" * 78)
print("MAKER (with measured fill bias) vs TAKER (crosses, pays fees)")
print("=" * 78)
D["pf"] = np.where(D.won == 1, P_FILL_WIN, P_FILL_LOSE)
D["mk"] = (D.won - D.bid) * QTY * D.pf
D["tk"] = (D.won - D.ask) * QTY - [taker_fee(QTY, a) for a in D.ask]
D["naive"] = (D.won - D.bid) * QTY
mk = D.groupby("date").mk.sum().reindex(days, fill_value=0)
tk = D.groupby("date").tk.sum().reindex(days, fill_value=0)
nv = D.groupby("date").naive.sum().reindex(days, fill_value=0)
print(f"{'accounting':>34} {'$/day':>10} {'per contract':>14}")
print(f"{'BACKTEST (100% fill, no cost)':>34} {nv.mean():>10.2f} "
      f"{nv.sum()/(len(D)*QTY)*100:>+13.2f}c")
print(f"{'MAKER with fill bias':>34} {mk.mean():>10.2f} "
      f"{mk.sum()/(D.pf.sum()*QTY)*100:>+13.2f}c")
print(f"{'TAKER':>34} {tk.mean():>10.2f} "
      f"{tk.sum()/(len(D)*QTY)*100:>+13.2f}c")

RNG = np.random.default_rng(20261056)
d = (tk - mk).values
bs = np.sort(np.array([RNG.choice(d, len(d), True).mean() for _ in range(8000)]))
print(f"\n  TAKER minus MAKER: ${d.mean():+.2f}/day   "
      f"95% CI [{bs[200]:+.2f}, {bs[7799]:+.2f}]   "
      f"better {(tk > mk).sum()}/{len(days)} days   "
      f"P(taker worse) {(bs <= 0).mean():.4f}")

print("\n" + "=" * 78)
print("BREAK-EVEN, AND IS IT STILL INSIDE THE PARAMETER'S OWN CI?")
print("=" * 78)
lo, hi = 0.50, 1.30
for _ in range(60):
    mid = (lo + hi) / 2
    pf = np.where(D.won == 1, mid, P_FILL_LOSE)
    m = ((D.won - D.bid) * QTY * pf).groupby(D.date).sum().reindex(
        days, fill_value=0)
    if m.mean() < tk.mean():
        lo = mid
    else:
        hi = mid
be = (lo + hi) / 2
print(f"  taker wins while P(fill|win) < {be:.4f}")
print(f"  measured P(fill|win) = 0.8426, 95% CI [0.7703, 0.9100]")
print(f"  break-even is {'INSIDE' if 0.7703 < be < 0.9100 else 'OUTSIDE'} "
      f"that interval")
print(f"  (this morning, against the understated edge, break-even was 0.8710)")

print("\n" + "=" * 78)
print("SENSITIVITY")
print("=" * 78)
print(f"{'P(fill|win)':>12} {'maker $/day':>13} {'taker $/day':>13} "
      f"{'taker-maker':>13}")
for pw in (0.78, 0.843, 0.88, 0.91, 0.95, 1.00):
    pf = np.where(D.won == 1, pw, P_FILL_LOSE)
    m = ((D.won - D.bid) * QTY * pf).groupby(D.date).sum().reindex(
        days, fill_value=0)
    print(f"{pw:>12.3f} {m.mean():>13.2f} {tk.mean():>13.2f} "
          f"{tk.mean()-m.mean():>+13.2f}")

print("\n" + "=" * 78)
print("RECENT PERIOD ONLY - where the premium is largest")
print("=" * 78)
for n in (30, 14):
    dd = days[-n:]
    m2, t2 = mk.reindex(dd), tk.reindex(dd)
    sub = D[D.date.isin(dd)]
    print(f"  last {n:>2} days: maker ${m2.mean():>7.2f}/day   "
          f"taker ${t2.mean():>7.2f}/day   diff ${t2.mean()-m2.mean():>+7.2f}   "
          f"taker better {(t2 > m2).sum()}/{n}   "
          f"(population {(sub.won - sub.bid).mean()*100:+.2f}c)")
