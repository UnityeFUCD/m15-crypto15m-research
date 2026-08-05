"""Measure the actual EV of every traded coin - including the two I added blind.

WHY THIS EXISTS
  BNB and HYPE were added to the live series list on volume alone. There are
  ZERO local trade files for either, so the edge on them was never measured -
  it was assumed to generalise, the same assumption already carried by BTC and
  ETH. That assumption is now testable: Kalshi serves settled markets with
  their result, and the trade feed per market, so the entry logic can be
  replayed exactly as the runner applies it.

METHOD (identical to the research extraction, so numbers are comparable)
  For each settled market: replay trades chronologically, track cumulative
  volume, and take the FIRST trade with 8-14 minutes remaining whose favourite
  maker price sits in 65-85c and whose cumulative volume has cleared 2,000.
  Score it against the actual settlement.

  DOGE is included as a CONTROL. Its backtest edge is known (+11.29c region),
  so if this API-derived method reproduces that, the BNB/HYPE numbers from the
  same pipeline can be trusted. If DOGE comes out wildly different, the method
  is broken and no conclusion should be drawn about anything.
"""
import sys
import time
from datetime import datetime, timezone

import numpy as np

from acct import api

RNG = np.random.default_rng(20260831)
SERIES = ["KXBNB15M", "KXHYPE15M", "KXDOGE15M", "KXBTC15M", "KXETH15M",
          "KXSOL15M", "KXXRP15M"]
MAXMKT = int(sys.argv[1]) if len(sys.argv) > 1 else 120


def ts(s):
    try:
        return datetime.fromisoformat((s or "").replace("Z", "+00:00"))
    except Exception:
        return None


def all_trades(tk, maxpages=6):
    out, cur = [], None
    for _ in range(maxpages):
        p = {"ticker": tk, "limit": 1000}
        if cur:
            p["cursor"] = cur
        c, j = api("/markets/trades", p)
        if c != 200 or not j:
            break
        b = j.get("trades") or []
        out += b
        cur = j.get("cursor")
        if not cur or not b:
            break
    return out


print(f"{'series':>12} {'mkts':>5} {'entries':>8} {'win':>8} {'avg px':>8} "
      f"{'edge/ct':>9} {'95% CI (day-clustered)':>26} {'P(edge<=0)':>11}")
RES = {}
for ser in SERIES:
    mk, cur = [], None
    while len(mk) < MAXMKT:
        p = {"series_ticker": ser, "status": "settled", "limit": 100}
        if cur:
            p["cursor"] = cur
        c, j = api("/markets", p)
        if c != 200 or not j:
            break
        b = j.get("markets") or []
        mk += b
        cur = j.get("cursor")
        if not cur or not b:
            break
    mk = mk[:MAXMKT]
    rows = []
    for m in mk:
        res = m.get("result")
        if res not in ("yes", "no"):
            continue
        y = 1 if res == "yes" else 0
        close = ts(m.get("close_time"))
        if not close:
            continue
        tr = all_trades(m.get("ticker"))
        seq = []
        for t in tr:
            tt = ts(t.get("created_time"))
            if not tt:
                continue
            s = (close - tt).total_seconds()
            if s <= 0 or s > 900:
                continue
            try:
                yp = float(t["yes_price_dollars"])
                n = float(t["count_fp"])
                sd = t.get("taker_side")
            except Exception:
                continue
            if sd not in ("yes", "no") or n <= 0:
                continue
            mp = yp if sd == "no" else 1.0 - yp     # favourite's maker price
            seq.append((s, mp, sd == "no", n))
        if len(seq) < 10:
            continue
        seq.sort(key=lambda x: -x[0])               # chronological
        cum = 0.0
        for (s, mp, my, n) in seq:
            if (480 <= s <= 840 and 0.65 <= mp < 0.85 and cum >= 2000):
                won = 1 if (my == bool(y)) else 0
                rows.append((close.strftime("%Y-%m-%d"), mp, won,
                             (1 - mp) if won else -mp))
                break
            cum += n
    if len(rows) < 8:
        print(f"{ser:>12} {len(mk):>5} {len(rows):>8}   (too few qualifying entries)")
        RES[ser] = None
        continue
    days = sorted(set(r[0] for r in rows))
    edge = np.array([r[3] for r in rows])
    won = np.array([r[2] for r in rows])
    px = np.array([r[1] for r in rows])
    byday = {d: np.array([r[3] for r in rows if r[0] == d]) for d in days}
    bs = np.sort(np.array([
        np.concatenate([byday[days[k]]
                        for k in RNG.integers(0, len(days), len(days))]).mean() * 100
        for _ in range(4000)]))
    RES[ser] = dict(n=len(rows), win=won.mean(), edge=edge.mean() * 100,
                    lo=bs[100], hi=bs[3899], p0=(bs <= 0).mean())
    print(f"{ser:>12} {len(mk):>5} {len(rows):>8} {won.mean():>8.4f} "
          f"{px.mean()*100:>7.1f}c {edge.mean()*100:>+9.2f} "
          f"[{bs[100]:>+10.2f},{bs[3899]:>+10.2f}] {(bs <= 0).mean():>11.4f}")

print("\n=== CONTROL CHECK ===")
d = RES.get("KXDOGE15M")
if d:
    print(f"  DOGE via this API pipeline: {d['edge']:+.2f}c on {d['n']} entries")
    print(f"  DOGE in the historical backtest: around +11c")
    print("  If these disagree badly, distrust every row above.")

print("\n=== THE TWO ADDED BLIND ===")
for s in ("KXBNB15M", "KXHYPE15M"):
    r = RES.get(s)
    if not r:
        print(f"  {s}: insufficient data - it is being traded on NO evidence")
        continue
    verdict = ("edge established" if r["p0"] < 0.05 else
               "NOT established - CI includes zero")
    print(f"  {s}: {r['edge']:+.2f}c, win {r['win']:.4f}, n {r['n']}  -> {verdict}")
