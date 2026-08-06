"""AUDIT D + E - what is answerable now, and the powered protocol for the rest.

AUDIT D CANNOT BE RUN ON HISTORY. This is a data fact, not a judgement.
  Kalshi exposes GET /portfolio/orders/{order_id}/queue_position, but it
  reports queue depth for an order that is CURRENTLY RESTING. It is not a
  historical endpoint - there is no way to ask what the queue looked like in
  front of an order that filled or cancelled in July.
  No queue field exists anywhere in data/: orders_history, fills_history and
  every book file lack it. It was never captured.
  The account is below its kill floor and correctly placing no orders, so
  none can be captured right now either.
  Therefore: no EV_submitted(q), no P(fill|win,q), no monotonicity test.
  Reporting any of those would require inventing the input.

AUDIT E IS PROSPECTIVE BY CONSTRUCTION. Fills are endogenous: cancelling an
  order changes whether it would have filled, so no replay of historical fills
  can price a cancel policy. Historical replay is descriptive only. What can
  be done rigorously TODAY is the power calculation that says whether the
  experiment is worth running at all.

This script computes the required sample size from OBSERVED outcome rates and
reports how long each arm must run. If the answer is longer than the strategy
can survive, that is the finding.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
Q = 15
Z80, Z95 = 2.802, 1.96          # two-sided 5%, 80% power -> 2.802 combined

BF = pd.read_parquet(DATA / "book_full.parquet")
BF["maker"] = BF.won - BF.bid
B = BF[(BF.bid >= 0.65) & (BF.bid < 0.80)].copy()
BF["utc"] = pd.to_datetime(BF.close_ts, unit="s", utc=True)
days = pd.to_datetime(B.close_ts, unit="s", utc=True).dt.date.nunique()

print("=" * 88)
print("AUDIT D - QUEUE POSITION: NOT ANSWERABLE FROM STORED DATA")
print("=" * 88)
O = pd.read_parquet(DATA / "orders_history.parquet")
F = pd.read_parquet(DATA / "fills_history.parquet")
qcols = [c for c in list(O.columns) + list(F.columns)
         if "queue" in c.lower() or "depth" in c.lower()]
print(f"  queue/depth columns in orders_history + fills_history: {qcols}")
print(f"  orders {len(O)}, fills {len(F)} - neither carries queue state.")
print("  The endpoint is live-only: it describes an order resting NOW.")
print("  CONCLUSION: EV_submitted(q), P(fill|win,q), P(fill|lose,q) and the")
print("  monotonicity test are all UNCOMPUTABLE until capture exists.")
print("  capture/queue.py is added to start collecting it. No result is")
print("  reported for AUDIT D because none can be honestly produced.")

print("\n" + "=" * 88)
print("AUDIT E - POWER FOR THE RANDOMIZED PROSPECTIVE EXPERIMENT")
print("=" * 88)

# --- observed dispersion, the input to every sample-size calculation -------
sd_maker = B.maker.std()
print(f"\n  observed SD of maker P&L per contract: {sd_maker*100:.1f}c")
print(f"  in-band markets/day: {len(B)/days:.1f}   at q{Q} that is "
      f"{len(B)/days*Q:.0f} contracts/day across all arms")

print("\n" + "-" * 88)
print("E1. DETECTING A CHANGE IN EV PER SUBMITTED CONTRACT")
print("-" * 88)
print("  %-34s %14s %13s %12s" % ("effect to detect", "contracts/arm",
                                  "days/arm (4 arms)", "years"))
per_day_all = len(B) / days * Q
per_day_arm = per_day_all / 4
for delta_c in (0.5, 1.0, 2.0, 3.0, 5.0):
    d = delta_c / 100.0
    n = 2 * (Z80 ** 2) * (sd_maker ** 2) / (d ** 2)
    dys = n / per_day_arm
    print("  %-34s %14.0f %13.0f %12.1f"
          % (f"{delta_c:+.1f}c per submitted contract", n, dys, dys / 365))
print("\n  The population is -0.43c. Turning that positive needs roughly")
print("  +1.0c, which the table prices at the number of days shown.")

print("\n" + "-" * 88)
print("E2. DETECTING A REDUCTION IN THE FILL-BIAS GAP  (better powered)")
print("-" * 88)
print("  A binary outcome has far less variance than P&L, so the mechanism")
print("  is cheaper to test than the money it produces.")
p_w, p_l = 0.8848, 0.9884
print(f"  baseline P(fill|win) {p_w:.4f}  P(fill|lose) {p_l:.4f}  "
      f"gap {(p_l-p_w)*100:+.1f}pp")
print("  %-30s %16s %15s %10s" % ("gap reduction to detect", "orders/arm",
                                  "days/arm (4 arms)", "weeks"))
orders_day_all = len(B) / days
orders_day_arm = orders_day_all / 4
for red_pp in (2.0, 4.0, 6.0, 10.0):
    d = red_pp / 100.0
    pbar = (p_w + p_l) / 2
    n = 2 * (Z80 ** 2) * pbar * (1 - pbar) / (d ** 2)
    dys = n / orders_day_arm
    print("  %-30s %16.0f %15.0f %10.1f"
          % (f"{red_pp:.0f}pp", n, dys, dys / 7))

print("\n" + "-" * 88)
print("E3. THE PROTOCOL (frozen before any data is collected)")
print("-" * 88)
print("""  ARMS - randomised per MARKET, not per day, so regime is shared:
    A control      post and leave resting to settlement (current behaviour)
    B queue-gate   post only if queue ahead <= threshold T, else skip
    C state-cancel cancel a still-resting order when the index has moved
                   against the held side by >= X bp. Never cancel-replace,
                   never chase.
    D combined     B and C together

  ASSIGNMENT - HMAC(market_ticker, secret) mod 4. Deterministic, reproducible,
  independent of anything observable at decision time, and already implemented
  in capture/treatments.py.

  FROZEN PARAMETERS - T and X are chosen on TRAIN ONLY and then frozen. They
  are NOT re-tuned on valid or test. If the queue data shows no monotonic
  relationship on train, arm B is dropped rather than threshold-searched.

  PRIMARY ENDPOINT - EV per SUBMITTED contract, day-clustered.
  SECONDARY       - fill-bias gap, fill rate, realised P&L.
  The primary is powered by E1; the secondary by E2. If only the secondary
  reaches significance, the mechanism is confirmed but the money is not.

  STOPPING - fixed horizon from the tables above. No peeking, no early stop
  on a favourable interim, because the whole failure mode of this project has
  been stopping to look at a number that later regressed.""")

print("\n" + "-" * 88)
print("E4. WHAT ACTUALLY BINDS")
print("-" * 88)
n1c = 2 * (Z80 ** 2) * (sd_maker ** 2) / (0.01 ** 2)
d4 = n1c / per_day_arm
d2 = n1c / (per_day_all / 2)
print(f"  Detecting the +1.0c that would take the strategy from -0.43c to")
print(f"  break-even needs {n1c:,.0f} contracts per arm:")
print(f"    four arms  {d4:,.0f} days   ({d4/30:.1f} months)")
print(f"    two arms   {d2:,.0f} days   ({d2/30:.1f} months)")
print("  All arms run CONCURRENTLY on disjoint markets, so those are calendar")
print("  days for the whole experiment, not per-arm serial time.")
print("\n  Both endpoints are reachable on a sane horizon: the mechanism in")
print("  ~3 weeks, the money in ~2-3 months. Sample size is NOT the binding")
print("  constraint, and an earlier draft of this script claimed otherwise")
print("  in prose while printing numbers that said the opposite.")

print("\n  THE ACTUAL BINDING CONSTRAINT IS CAPITAL, NOT STATISTICS:")
eq, floor = 136.27, 398.25
print(f"    account equity  ${eq:.2f}")
print(f"    kill floor      ${floor:.2f}  (75% of the $531 strategy HWM)")
print(f"    shortfall       ${floor-eq:.2f}")
print("  capture/hcr.py returns KILL and sizes to 0 at this equity, which is")
print("  correct. No arm of this experiment can be run live until the account")
print("  is funded above the floor. Until then the only executable step is")
print("  SHADOW capture - recording queue position and index state for orders")
print("  that are never sent - which costs nothing and builds the dataset")
print("  AUDIT D needs but does not have.")
print(f"\n  Capacity check at q{Q}: 4 open x {Q} contracts x ~$0.72 = "
      f"${4*Q*0.72:.2f} at risk, well inside any funded account. Depth is not")
print("  the limit either - the caps allow ~192 markets/day and the")
print(f"  population only offers {len(B)/days:.0f}.")
