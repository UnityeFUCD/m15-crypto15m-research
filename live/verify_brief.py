"""Verify the external brief's central claims against our own data.

THE BRIEF'S CENTRAL CORRECTION (which looks right and should be adopted)
  It says we conflated two different quantities:
      delta = q_unfilled - q_filled          the raw outcome GAP  (14.00pp)
      s     = q_all - q_filled = (1-f)*delta the actual SELECTION COST
  Algebraically that is exact:
      q_all = f*q_F + (1-f)*q_U
      q_all - q_F = (1-f)*(q_U - q_F) = (1-f)*delta
  So the cost of being filled is much smaller than the raw gap. Correct, and
  worth adopting.

THE BRIEF'S CENTRAL ERROR (which drives its whole recommendation)
  It computes the current edge as
      2.32c post-shift premium - 2.08c selection = +0.24c
  and concludes that quantity 30 is unjustified. But 2.32c is the premium over
  ALL qualifying markets. It is not the premium on the markets the runner
  actually selects. We measured that separately, AFTER the data pack was
  generated, so the brief could not have seen it: replaying the same rule on
  only the markets we traded gives +8.54c, a selection premium of +6.22c.

  This script recomputes the reconciliation both ways and checks which one
  matches realised P&L.

ALSO CHECKED
  - the brief uses F_Q (a CONTRACT-weighted fill fraction) together with a
    delta computed ORDER-weighted. Those units do not match, and the brief
    itself warns against exactly this.
  - its claim that our "window-equal" weighting is ambiguous.
  - its claim that Q30 risks ~15% of equity in one correlated window.
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from acct import api

RNG = np.random.default_rng(20260913)
HERE = Path(__file__).resolve().parent
RES = HERE.parent / "research" / "final_minute_favorite_maker_validation"

# ---------- 1. the fill/no-fill gap, computed at the ORDER level ----------
ORD = {}
for fp in sorted(glob.glob(str(HERE / "lsm_*.jsonl"))) + \
        sorted(glob.glob(str(HERE / "lsm_*.log"))):
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
                ORD.setdefault(t, {}).update(px=d.get("post_at"),
                                             side=d.get("side"))
            if d.get("ev") == "terminal":
                ORD.setdefault(t, {}).update(target=d.get("target"),
                                             filled=d.get("FILLED"))
    except Exception:
        pass

RESULT = {}
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
                RESULT[m["ticker"]] = m["result"]
        cur = j.get("cursor")
        if not cur or not b:
            break

rows = []
for t, o in ORD.items():
    r = RESULT.get(t)
    if r is None or o.get("px") is None or o.get("side") is None:
        continue
    if o.get("target") is None or o.get("filled") is None:
        continue
    won = 1 if ((o["side"] == "yes" and r == "yes") or
                (o["side"] == "no" and r == "no")) else 0
    frac = float(o["filled"]) / max(float(o["target"]), 1)
    rows.append(dict(px=float(o["px"]), won=won, frac=frac,
                     filled=frac > 0, qty=float(o["target"])))
R = pd.DataFrame(rows)
F, U = R[R.filled], R[~R.filled]
q_F, q_U = F.won.mean(), U.won.mean()
delta = (q_U - q_F) * 100
f_order = len(F) / len(R)
f_qty = (R.frac * R.qty).sum() / R.qty.sum()
q_all = R.won.mean()

print("=== 1. THE SELECTION DECOMPOSITION, DONE IN CONSISTENT UNITS ===")
print(f"  orders matched to a settlement : {len(R)}")
print(f"  q_filled   (order-weighted)    : {q_F:.4f}")
print(f"  q_unfilled (order-weighted)    : {q_U:.4f}")
print(f"  delta = q_U - q_F              : {delta:+.2f} pp")
print(f"  f, ORDER-weighted              : {f_order:.4f}")
print(f"  f, CONTRACT-weighted (F_Q)     : {f_qty:.4f}")
print()
print(f"  brief's s = (1-F_Q)*delta      : {(1-f_qty)*delta:+.2f}c   "
      f"<- mixes contract-weighted f with order-weighted delta")
print(f"  consistent s = (1-f_order)*delta: {(1-f_order)*delta:+.2f}c   "
      f"<- both order-weighted")
print(f"  DIRECT check, q_all - q_F      : {(q_all - q_F)*100:+.2f}c   "
      f"<- no algebra, measured")
print("  (the direct measurement is the arbiter; if it matches the")
print("   consistent formula, the decomposition is verified)")

# ---------- 2. the reconciliation the brief could not do ----------
print("\n=== 2. RECONCILIATION: population premium vs OUR-MARKETS premium ===")
s_use = (q_all - q_F) * 100
print(f"  selection cost (measured)              : {s_use:+.2f}c")
print()
print("  BRIEF'S VERSION (population premium):")
print(f"    post-shift premium, ALL markets      : +2.32c")
print(f"    minus selection                      : {2.32 - s_use:+.2f}c  "
      f"<- the brief's basis for cutting to Q1")
print()
print("  WITH THE SELECTION PREMIUM WE MEASURED (brief could not see this):")
print(f"    post-shift premium, OUR markets      : +8.54c")
print(f"    minus selection                      : {8.54 - s_use:+.2f}c")
print()
print(f"  ACTUAL REALISED live edge              : see below")

# realised
MINE = set(ORD)
st, cur = [], None
while True:
    p = {"limit": 200}
    if cur:
        p["cursor"] = cur
    c, j = api("/portfolio/settlements", p)
    if c != 200 or not j:
        break
    b = j.get("settlements") or []
    st += b
    cur = j.get("cursor")
    if not cur or not b or len(st) >= 4000:
        break
M = [x for x in st if x.get("ticker") in MINE]
pnl = ncon = 0.0
for x in M:
    n = abs(float(x.get("yes_count_fp") or 0)) + abs(float(x.get("no_count_fp") or 0))
    if n <= 0:
        continue
    rev = float(x.get("revenue") or 0) / 100.0
    cst = (float(x.get("yes_total_cost_dollars") or 0) +
           float(x.get("no_total_cost_dollars") or 0) + float(x.get("fee_cost") or 0))
    pnl += rev - cst
    ncon += n
live = pnl / max(ncon, 1) * 100
print(f"    realised: {len(M)} settlements, {ncon:,.0f} contracts, "
      f"${pnl:+.2f} = {live:+.2f}c/contract")
print()
print(f"  Which prediction matches reality?")
print(f"    brief's  {2.32 - s_use:+.2f}c   vs realised {live:+.2f}c   "
      f"error {abs((2.32-s_use)-live):.2f}c")
print(f"    ours     {8.54 - s_use:+.2f}c   vs realised {live:+.2f}c   "
      f"error {abs((8.54-s_use)-live):.2f}c")

# ---------- 3. window-equal ambiguity ----------
print("\n=== 3. IS OUR 'WINDOW-EQUAL' AMBIGUOUS? (brief section 1.3) ===")
G = pd.read_parquet(RES / "grid.parquet")
q = G[(G.px >= 0.65) & (G.px < 0.85) & (G.minute >= 8) & (G.minute <= 14)
      & (G.vol >= 2000)]
E = (q.sort_values("minute", ascending=False)
       .groupby(["wkey", "coin"], as_index=False).first())
E = (E.sort_values(["wkey", "minute"], ascending=[True, False])
       .groupby("wkey", as_index=False).head(3))
per_entry = E.edge.mean() * 100
per_window = E.groupby("wkey").edge.mean().mean() * 100
print(f"  mean over ENTRIES (what we have reported): {per_entry:+.2f}c")
print(f"  mean over WINDOWS (each wkey weight 1)   : {per_window:+.2f}c")
print(f"  difference                                : {per_entry-per_window:+.2f}c")
print("  The brief is RIGHT that these differ and that we were not explicit")
print("  about which one 'window-equal-weighted' meant. Ours is per-entry")
print("  after capping at 3/window, which is the ECONOMIC quantity; the")
print("  per-window mean is the SCIENTIFIC one. Both should be reported.")

# ---------- 4. risk exposure ----------
print("\n=== 4. DOES Q30 RISK ~15% OF EQUITY IN ONE WINDOW? (brief 3.4) ===")
c, j = api("/portfolio/balance")
bal = float((j or {}).get("balance_dollars") or 0)
c2, j2 = api("/portfolio/positions", {"limit": 200})
cost = sum(abs(float(p.get("total_traded_dollars") or 0))
           for p in ((j2 or {}).get("market_positions") or [])
           if float(p.get("position_fp") or 0) != 0)
eq = bal + cost
W = E.copy()
W["pnl"] = np.where(W.won == 1, (1 - W.px) * 30, -W.px * 30)
wp = W.groupby("wkey").pnl.sum()
print(f"  current equity                    : ${eq:.2f}")
print(f"  worst window OBSERVED at Q30      : ${wp.min():+.2f} "
      f"= {abs(wp.min())/eq*100:.1f}% of equity")
print(f"  theoretical worst (3 x 30 x 0.85) : ${-3*30*0.85:.2f} "
      f"= {3*30*0.85/eq*100:.1f}% of equity")
print(f"  1st-percentile window             : ${np.percentile(wp,1):+.2f} "
      f"= {abs(np.percentile(wp,1))/eq*100:.1f}%")
print("  The brief's 15% figure uses the theoretical maximum at an 85c fill.")
print("  The OBSERVED worst is the more relevant number for sizing.")
