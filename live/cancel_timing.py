"""Should we hold unfilled orders past the 7-minute cancel?

WHY THIS IS THE LAST FILL-RATE LEVER
  Unfilled orders are lost EDGE, not merely lost volume: markets we failed to
  fill went on to win 92.86% against 77.67% for the ones we caught. 16 of 122
  orders never filled at all.

  The other three ways to improve fill are closed:
    quote improvement  - 84% of spreads are exactly one tick, so quoting better
                         would cross; and the 16% that are wider already fill
                         at 0.8942
    earlier entry      - already at the top of the band, median 13.75 min
    smaller orders     - correlates with fill but is confounded with the
                         regime shift and cannot be separated

  That leaves HOLDING LONGER. The runner cancels every unfilled remainder once
  the window passes MIN_LEFT[0] - 1 = 7 minutes. Anything that would have
  filled at 6, 5 or 4 minutes is thrown away.

I PREVIOUSLY CALLED THIS UNTESTABLE. IT IS NOT.
  We never observe our own fill past the cancel, but we DO observe the market.
  If a market traded at or through our posted price after we cancelled, a
  resting order would have filled - we join the bid, so any trade at our price
  or better on our side clears the level we were sitting on. That is an upper
  bound on what holding would have captured (queue position ahead of us could
  still have absorbed it), and an upper bound is exactly what is needed: if
  even the optimistic case does not pay, the idea is dead.

THE CATCH THAT MAKES THIS INTERESTING
  A fill that arrives late arrives because the price came DOWN to us - the
  favourite weakened. That is adverse selection by construction. So the
  question is not "would we have filled" but "would those fills have WON".
"""
import glob
import json
from datetime import datetime, timezone

import numpy as np

from acct import api

RNG = np.random.default_rng(20261029)


def ts(s):
    try:
        return datetime.fromisoformat((s or "").replace("Z", "+00:00"))
    except Exception:
        return None


# our orders. 'terminal' carries price/target/FILLED/unfilled/reason directly;
# 'opportunity' supplies the side, which terminal does not log.
ORD, REASON = {}, __import__("collections").Counter()
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
                ORD.setdefault(t, {}).update(side=d.get("side"),
                                             post_at=d.get("post_at"),
                                             qa=d.get("queue_ahead"))
            if d.get("ev") == "terminal":
                REASON[d.get("reason")] += 1
                ORD.setdefault(t, {}).update(
                    px=d.get("price"), target=d.get("target"),
                    filled=d.get("FILLED"), unf=d.get("unfilled"),
                    fq=d.get("F_Q"), reason=d.get("reason"))
    except Exception:
        pass
print("terminal reasons:", dict(REASON))
unf = {t: o for t, o in ORD.items()
       if o.get("target") and (o.get("filled") or 0) < o["target"]
       and o.get("px") and o.get("side")}
print(f"orders with an unfilled remainder: {len(unf)}")
never = {t: o for t, o in unf.items() if (o.get("filled") or 0) == 0}
print(f"  of which NEVER filled at all: {len(never)}\n")

rows = []
for t, o in unf.items():
    c, j = api("/markets", {"tickers": t})
    m = ((j or {}).get("markets") or [None])[0]
    if not m or m.get("result") not in ("yes", "no"):
        continue
    close = ts(m.get("close_time"))
    if not close:
        continue
    res = m["result"]
    out, cur = [], None
    for _ in range(3):
        p = {"ticker": t, "limit": 1000}
        if cur:
            p["cursor"] = cur
        cc, jj = api("/markets/trades", p)
        if cc != 200 or not jj:
            break
        b = jj.get("trades") or []
        out += b
        cur = jj.get("cursor")
        if not cur or not b:
            break
    our_px = float(o["px"])
    side = o["side"]
    would = False
    best = None
    for tr in out:
        tt = ts(tr.get("created_time"))
        if not tt:
            continue
        s = (close - tt).total_seconds()
        # strictly AFTER our cancel at 7 minutes, before close
        if not (0 < s < 420):
            continue
        try:
            yp = float(tr["yes_price_dollars"])
        except Exception:
            continue
        # price of OUR side at this trade
        px_side = yp if side == "yes" else 1.0 - yp
        if px_side <= our_px + 1e-9:
            would = True
            best = px_side if best is None else min(best, px_side)
    won = 1 if side == res else 0
    rows.append(dict(ticker=t, our_px=our_px, side=side, res=res, won=won,
                     would_fill=would, best=best, reason=o.get("reason"),
                     unfilled=o["target"] - (o.get("filled") or 0)))
if len(rows) < 10:
    raise SystemExit(f"only {len(rows)} matched - too few")
import pandas as pd
D = pd.DataFrame(rows)
print("=" * 74)
print("WOULD HOLDING PAST 7 MINUTES HAVE FILLED THEM?")
print("=" * 74)
print(f"  unfilled orders examined      : {len(D)}")
print(f"  market traded THROUGH our price after cancel: "
      f"{D.would_fill.sum()} ({D.would_fill.mean():.3f})")
print(f"  never touched our price again : {(~D.would_fill).sum()}")

print("\n" + "=" * 74)
print("AND WOULD THOSE LATE FILLS HAVE WON?")
print("=" * 74)
w = D[D.would_fill]
n = D[~D.would_fill]
if len(w) >= 5:
    print(f"  WOULD have filled late : n {len(w):>3}  win {w.won.mean():.4f}  "
          f"avg price {w.our_px.mean()*100:.1f}c")
    exp = (w.won.mean() - w.our_px.mean()) * 100
    print(f"    -> edge if we had held: {exp:+.2f}c/contract")
    bs = np.sort(np.array([
        (lambda s: (s.won.mean() - s.our_px.mean()) * 100)(
            w.sample(len(w), replace=True)) for _ in range(8000)]))
    print(f"    95% CI [{bs[200]:+.2f}, {bs[7799]:+.2f}]   "
          f"P(<=0) {(bs <= 0).mean():.4f}")
if len(n) >= 5:
    print(f"  never reachable        : n {len(n):>3}  win {n.won.mean():.4f}  "
          f"(these are unreachable regardless - the price never came back)")

print("\n" + "=" * 74)
print("THE DOLLAR ESTIMATE")
print("=" * 74)
if len(w) >= 5:
    contracts = w.unfilled.sum()
    exp = (w.won.mean() - w.our_px.mean())
    print(f"  contracts we could have captured: {contracts:.0f}")
    print(f"  at {exp*100:+.2f}c each -> ${contracts*exp:+.2f} over the whole "
          f"live period")
    print("\n  UPPER BOUND, deliberately: it assumes our resting order was at")
    print("  the front of the queue at that price level. Queue ahead of us")
    print("  could have absorbed the trade entirely, so the true capture is")
    print("  lower. If the upper bound is small or negative, holding is dead.")
