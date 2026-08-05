"""IS THE LONGSHOT PREMIUM DECAYING? The one question that is existential.

THE DECOMPOSITION THIS TESTS
  realised edge = longshot premium collected - adverse selection paid

  Those two have opposite implications and must not be confused:

    selection cost  is structural. It is what a maker pays for letting the
                    market choose when to trade with them. It cannot be
                    removed, only managed, and it does not get worse over time.
    premium decay   is competition. If other makers arrive and bid the
                    favourite up, the premium we collect shrinks toward zero
                    and the strategy dies. This is existential.

  The live shortfall (backtest +11.29c vs live ~+9c) is currently unattributed
  between the two. This attributes it.

THE KEY PROPERTY OF THIS MEASUREMENT
  The premium is measured at TRADE prices with NO fill conditioning - i.e. what
  a maker WOULD have earned if filled at the printed price. Selection cannot
  contaminate it, because no fill decision enters. So a downward trend here is
  premium decay and nothing else.

  A suggestive prior: replay over the last ~42h gave +2 to +6c per coin, while
  the 10-day backtest ending Aug 1 gives ~+11c on the same measure. If that is
  a real trend rather than sampling noise, it is the most important fact about
  this strategy.

DISCIPLINE
  window-equal-weighted; the trend is tested with a day-clustered bootstrap
  slope, not eyeballed; and the null (a flat premium with noisy days) is
  stated so it can actually be rejected or not.
"""
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

from acct import api

RNG = np.random.default_rng(20260902)
SERIES = ["KXDOGE15M", "KXSOL15M", "KXXRP15M", "KXETH15M"]   # the ones with history
MAXMKT = int(sys.argv[1]) if len(sys.argv) > 1 else 400


def ts(s):
    try:
        return datetime.fromisoformat((s or "").replace("Z", "+00:00"))
    except Exception:
        return None


def all_trades(tk, maxpages=4):
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


rows = []
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
    n0 = len(rows)
    for m in mk[:MAXMKT]:
        res = m.get("result")
        if res not in ("yes", "no"):
            continue
        y = 1 if res == "yes" else 0
        close = ts(m.get("close_time"))
        if not close:
            continue
        seq = []
        for t in all_trades(m.get("ticker")):
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
            mp = yp if sd == "no" else 1.0 - yp
            seq.append((s, mp, sd == "no", n))
        if len(seq) < 10:
            continue
        seq.sort(key=lambda x: -x[0])
        cum = 0.0
        for (s, mp, my, n) in seq:
            if 480 <= s <= 840 and 0.65 <= mp < 0.85 and cum >= 2000:
                won = 1 if (my == bool(y)) else 0
                rows.append((close, ser, mp, won, (1 - mp) if won else -mp))
                break
            cum += n
    print(f"  {ser}: {len(rows)-n0} entries", flush=True)

if len(rows) < 60:
    raise SystemExit(f"only {len(rows)} entries - not enough")
rows.sort(key=lambda r: r[0])
print(f"\nentries {len(rows):,}   period "
      f"{rows[0][0]:%Y-%m-%d %H:%M} .. {rows[-1][0]:%Y-%m-%d %H:%M} UTC")

byday = defaultdict(list)
for (c, ser, px, won, e) in rows:
    byday[c.strftime("%Y-%m-%d")].append(e)
dd = sorted(byday)
print("\n=== PREMIUM BY DAY (trade prices, NO fill conditioning) ===")
print(f"{'day':>12} {'n':>5} {'edge/ct':>9} {'cumulative':>12}")
allv = []
for d in dd:
    v = np.array(byday[d])
    allv += list(v)
    print(f"{d:>12} {len(v):>5} {v.mean()*100:>+9.2f} "
          f"{np.mean(allv)*100:>+11.2f}")

print("\n=== IS THERE A TREND? (day-clustered bootstrap of the slope) ===")
x = np.arange(len(dd), dtype=float)
ymu = np.array([np.mean(byday[d]) * 100 for d in dd])
nper = np.array([len(byday[d]) for d in dd])
if len(dd) < 4:
    print("  too few distinct days to fit a trend")
else:
    slope = np.polyfit(x, ymu, 1)[0]
    bs = []
    for _ in range(6000):
        pick = RNG.integers(0, len(dd), len(dd))
        yy = np.array([np.mean(RNG.choice(byday[dd[k]], len(byday[dd[k]]), True)) * 100
                       for k in pick])
        xx = np.sort(x[pick])
        order = np.argsort(x[pick])
        try:
            bs.append(np.polyfit(xx, yy[order], 1)[0])
        except Exception:
            pass
    bs = np.sort(np.array(bs))
    print(f"  slope {slope:+.2f} cents per day   95% CI "
          f"[{bs[int(.025*len(bs))]:+.2f}, {bs[int(.975*len(bs))]:+.2f}]")
    print(f"  P(slope >= 0) = {(bs >= 0).mean():.4f}")
    print("\n  A clearly negative slope = the premium is being competed away.")
    print("  A CI spanning zero = no decay detectable; the live shortfall is")
    print("  then attributable to adverse selection, which is structural.")

    half = len(dd) // 2
    a = np.concatenate([byday[d] for d in dd[:half]])
    b = np.concatenate([byday[d] for d in dd[half:]])
    print(f"\n  first half  ({dd[0]}..{dd[half-1]}): n {len(a):>4}  "
          f"{a.mean()*100:+.2f}c")
    print(f"  second half ({dd[half]}..{dd[-1]}): n {len(b):>4}  "
          f"{b.mean()*100:+.2f}c")
    print(f"  change {(b.mean()-a.mean())*100:+.2f}c")

print("\n=== ATTRIBUTION OF THE LIVE SHORTFALL ===")
prem = np.mean([r[4] for r in rows]) * 100
print(f"  premium, measured at trade prices, recent period : {prem:+.2f}c")
print(f"  10-day backtest on the same measure              : ~+11.29c")
print(f"  live realised, conditional on fill               : ~+9.12c")
print(f"\n  If the recent premium is well BELOW the backtest, the shortfall is")
print(f"  premium decay, not selection - and the backtest is stale as a guide.")
