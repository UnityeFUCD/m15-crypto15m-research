"""Why does live realise +9.12c when the replay says the premium is +2.32c?

THE DISCREPANCY
  Replay measures the premium at TRADE prices with no fill conditioning:
  +2.32c over the recent period, on ~1,200 observations. Live realises +9.12c
  on 141 settlements. Both cannot be a fair description of the same market.

  Candidate explanations, which have very different consequences:
    (a) SELECTION - live applies filters the replay does not (the dead-entry
        conjunction especially), so it trades a genuinely better subset. If so
        the filters are worth far more than measured and the premium figure is
        not the number that matters.
    (b) SMALL SAMPLE - live's 141 settlements carry a CI of [-2.36, +14.81];
        +2.32c sits inside it. If so there is no real discrepancy and live has
        simply been lucky.
    (c) ENTRY-RULE MISMATCH - the replay's "first qualifying print" is not what
        the runner does, so they are measuring different strategies.

THE DECISIVE TEST
  Run the replay rule on EXACTLY the markets the runner traded. Same markets,
  same period, one measures at trade prices and the other at our fill.

    replay on our markets ~= +9c  -> SELECTION. Live picks better markets, and
                                     the filters deserve the credit.
    replay on our markets ~= +2c  -> the difference is in EXECUTION or luck,
                                     not selection, and the +9.12c is fragile.
"""
import glob
import json
from datetime import datetime, timezone

import numpy as np

from acct import api

RNG = np.random.default_rng(20260907)


def ts(s):
    try:
        return datetime.fromisoformat((s or "").replace("Z", "+00:00"))
    except Exception:
        return None


# markets the runner actually posted on
MINE = {}
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
            if d.get("ev") == "post" and d.get("ticker") and \
                    d.get("status") in (200, 201):
                MINE[d["ticker"]] = dict(px=d.get("price"), side=d.get("side"))
    except Exception:
        pass
print(f"markets the runner posted on: {len(MINE):,}")

# settlement result for each
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
                RES[m["ticker"]] = (m["result"], ts(m.get("close_time")))
        cur = j.get("cursor")
        if not cur or not b:
            break


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


rep, ours = [], []
for tk, o in MINE.items():
    if tk not in RES:
        continue
    res, close = RES[tk]
    if not close:
        continue
    y = 1 if res == "yes" else 0
    seq = []
    for t in all_trades(tk):
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
        seq.append((s, yp if sd == "no" else 1.0 - yp, sd == "no", n))
    if len(seq) < 10:
        continue
    seq.sort(key=lambda x: -x[0])
    cum = 0.0
    hit = None
    for (s, mp, my, n) in seq:
        if 480 <= s <= 840 and 0.65 <= mp < 0.85 and cum >= 2000:
            hit = (mp, my)
            break
        cum += n
    if hit:
        mp, my = hit
        won = 1 if (my == bool(y)) else 0
        rep.append((1 - mp) if won else -mp)
    # our own economics on the same market
    if o.get("px") is not None and o.get("side"):
        px = float(o["px"])
        won = 1 if ((o["side"] == "yes" and res == "yes") or
                    (o["side"] == "no" and res == "no")) else 0
        ours.append((1 - px) if won else -px)

rep = np.array(rep)
ours = np.array(ours)
print(f"\n=== SAME MARKETS, TWO MEASUREMENTS ===")
print(f"  replay rule on OUR markets : n {len(rep):>4}  "
      f"edge {rep.mean()*100:+.2f}c")
print(f"  our own posted economics   : n {len(ours):>4}  "
      f"edge {ours.mean()*100:+.2f}c")
if len(rep) > 20:
    bs = np.sort(np.array([RNG.choice(rep, len(rep), True).mean() * 100
                           for _ in range(8000)]))
    print(f"  replay-on-our-markets 95% CI [{bs[200]:+.2f}, {bs[7799]:+.2f}]")
print(f"\n  replay on ALL recent markets (measured earlier): +2.32c")
print(f"  live realised (all settlements)                 : +9.12c")

print("\n=== VERDICT ===")
if len(rep) > 20:
    if rep.mean() * 100 > 6:
        print("  Replay on OUR markets is high -> SELECTION. The markets the")
        print("  runner chooses are genuinely better than the average")
        print("  qualifying market, so the filters carry the difference and the")
        print("  +2.32c population figure is not the number that governs us.")
    elif rep.mean() * 100 < 4:
        print("  Replay on OUR markets is LOW -> the markets we choose are")
        print("  ordinary. The +9.12c is then luck or execution, not selection,")
        print("  and should be expected to decay toward the population figure.")
    else:
        print("  Intermediate - selection explains part of the gap but not all.")
    print(f"\n  Selection premium = {rep.mean()*100 - 2.32:+.2f}c "
          f"(our markets vs all markets, same measure)")
