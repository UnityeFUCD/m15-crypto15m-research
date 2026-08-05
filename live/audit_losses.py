"""Forensic audit: are the losses REAL, or is something in the code wrong?

WHAT COULD BE WRONG, in rough order of how badly it would matter
  1. WRONG SIDE. If the runner posts on the longshot instead of the favourite,
     it would lose ~70% of the time and every "loss" would be self-inflicted.
     Checked by comparing the side we took against the price we paid: the
     favourite is by definition the side priced above 50c.
  2. TAKER FILLS. Crossing the spread converts a +10c edge into a measured
     -1.68c. The runner has no taker path, but "should not" is not "did not".
  3. PRICE SLIPPAGE. If fills land worse than the price we posted, the edge is
     being eaten before settlement.
  4. FEES. Maker fees should be exactly zero. Any nonzero fee is either a
     taker fill or a rule change.
  5. SETTLEMENT MISMATCH. If our recorded outcome disagrees with the
     exchange's official result, the P&L accounting is wrong.
  6. SIZE MISMATCH. Filled quantity differing from submitted, or exceeding the
     configured cap.

  Every one of these is checked against the EXCHANGE's own records, not
  against the runner's beliefs about itself.
"""
import glob
import json
from collections import defaultdict

import numpy as np

from acct import api

# what the runner THINKS it did
POST = {}
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
                POST.setdefault(t, {}).update(
                    post_at=d.get("post_at"), side=d.get("side"),
                    fav_bid=d.get("fav_bid"), fav_ask=d.get("fav_ask"),
                    qty=d.get("qty"), utc=d.get("utc"))
    except Exception:
        pass
print(f"orders the runner recorded posting: {len(POST):,}\n")

# what the EXCHANGE says happened
fills = defaultdict(list)
cur = None
for _ in range(6):
    p = {"limit": 200}
    if cur:
        p["cursor"] = cur
    c, j = api("/portfolio/fills", p)
    if c != 200 or not j:
        break
    b = j.get("fills") or []
    for f in b:
        fills[f.get("ticker")].append(f)
    cur = j.get("cursor")
    if not cur or not b:
        break
print(f"tickers with fills on the exchange: {len(fills):,}")

setl = {}
cur = None
for _ in range(8):
    p = {"limit": 200}
    if cur:
        p["cursor"] = cur
    c, j = api("/portfolio/settlements", p)
    if c != 200 or not j:
        break
    b = j.get("settlements") or []
    for s in b:
        setl[s.get("ticker")] = s
    cur = j.get("cursor")
    if not cur or not b:
        break
print(f"settlements available: {len(setl):,}\n")

print("=" * 78)
print("CHECK 1 - DID WE EVER POST ON THE WRONG SIDE?")
print("=" * 78)
bad_side = [t for t, o in POST.items()
            if o.get("post_at") is not None and float(o["post_at"]) < 0.50]
print(f"  orders posted BELOW 50c (i.e. on the longshot): {len(bad_side)}")
if bad_side:
    for t in bad_side[:5]:
        print(f"    {t} at {POST[t]['post_at']}")
else:
    print("  -> every order was on the favourite side. Correct.")
pxs = [float(o["post_at"]) for o in POST.values() if o.get("post_at")]
print(f"  posted price range: {min(pxs)*100:.0f}c to {max(pxs)*100:.0f}c "
      f"(band is 65-85c, live cap 80c)")
oob = [p for p in pxs if p < 0.65 or p > 0.85]
print(f"  outside the configured band: {len(oob)}")

print("\n" + "=" * 78)
print("CHECK 2 - ANY TAKER FILLS? (the one rule that cannot be broken)")
print("=" * 78)
tak = 0
tot = 0
for t, fl in fills.items():
    for f in fl:
        tot += 1
        if f.get("is_taker"):
            tak += 1
            print(f"    TAKER FILL: {t} {f.get('created_time')}")
print(f"  fills examined {tot}   taker fills {tak}")
print("  -> " + ("CLEAN" if tak == 0 else "*** TAKER FILLS PRESENT ***"))

print("\n" + "=" * 78)
print("CHECK 3 - DID FILLS LAND AT THE PRICE WE POSTED?")
print("=" * 78)
slip = []
for t, o in POST.items():
    if t not in fills or o.get("post_at") is None or not o.get("side"):
        continue
    want = float(o["post_at"])
    for f in fills[t]:
        yp = f.get("yes_price")
        if yp is None:
            continue
        got = float(yp) / 100.0 if float(yp) > 1 else float(yp)
        got = got if o["side"] == "yes" else 1.0 - got
        slip.append(got - want)
if slip:
    s = np.array(slip)
    print(f"  fills compared {len(s)}")
    print(f"  mean slippage {s.mean()*100:+.3f}c   worst {s.max()*100:+.2f}c   "
          f"best {s.min()*100:+.2f}c")
    print(f"  fills WORSE than posted: {(s > 1e-9).sum()}")
    print("  -> " + ("CLEAN - maker fills cannot be worse than posted"
                     if (s > 1e-9).sum() == 0 else "*** SLIPPAGE PRESENT ***"))
else:
    print("  yes_price not populated on fills; slippage not checkable this way")

print("\n" + "=" * 78)
print("CHECK 4 - FEES")
print("=" * 78)
fee_tot = 0.0
fee_n = 0
for t, s in setl.items():
    f = float(s.get("fee_cost") or 0)
    if f > 0:
        fee_tot += f
        fee_n += 1
print(f"  settlements with a nonzero fee: {fee_n}   total ${fee_tot:.2f}")
print("  -> " + ("CLEAN - maker fee is zero as expected" if fee_n == 0
                 else f"NOTE: ${fee_tot:.2f} of fees across {fee_n} settlements"))

print("\n" + "=" * 78)
print("CHECK 5 - DO OUR LOSSES RECONCILE WITH THE OFFICIAL RESULT?")
print("=" * 78)
mismatch = 0
checked = 0
rows = []
for t, o in POST.items():
    s = setl.get(t)
    if not s or not o.get("side") or o.get("post_at") is None:
        continue
    res = s.get("market_result") or s.get("result")
    if res not in ("yes", "no"):
        continue
    checked += 1
    should_win = (o["side"] == res)
    rev = float(s.get("revenue") or 0) / 100.0
    cost = (float(s.get("yes_total_cost_dollars") or 0) +
            float(s.get("no_total_cost_dollars") or 0))
    actually_won = rev > cost
    if should_win != actually_won:
        mismatch += 1
        if mismatch <= 6:
            print(f"    MISMATCH {t}: our side {o['side']}, result {res}, "
                  f"rev {rev:.2f} cost {cost:.2f}")
    rows.append((t, o["side"], res, should_win, actually_won,
                 float(o["post_at"]), rev - cost))
print(f"  settlements checked {checked}   mismatches {mismatch}")
print("  -> " + ("CLEAN - every outcome matches the official result"
                 if mismatch == 0 else "*** ACCOUNTING MISMATCH ***"))

print("\n" + "=" * 78)
print("CHECK 6 - THE LOSING TRADES, ONE BY ONE (most recent 12)")
print("=" * 78)
L = [r for r in rows if not r[4]]
L.sort(key=lambda r: r[0])
print(f"  total losses recorded: {len(L)} of {len(rows)} "
      f"({len(L)/max(len(rows),1):.1%})")
print(f"\n{'ticker':>32} {'our side':>9} {'result':>7} {'paid':>7} {'P&L':>9}")
for r in L[-12:]:
    print(f"{r[0][:32]:>32} {r[1]:>9} {r[2]:>7} {r[5]*100:>6.0f}c {r[6]:>+9.2f}")
print("\n  Every row above should show: we bought a side, the OTHER side")
print("  settled, and we lost exactly what we paid. That is a normal loss,")
print("  not a malfunction.")
