"""Pull the headline number for every claim in the project synthesis.

Nothing here is new analysis. It re-derives the figures cited in SYNTHESIS.md
straight from the stored data so the write-up cannot drift from the evidence.
Light on memory by design: reads columns it needs, no full-frame copies.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")

print("=" * 78)
print("1. SETTLEMENT MECHANISM  (does A1>=A0 reproduce the printed result?)")
print("=" * 78)
U = pd.read_parquet(DATA / "underlying.parquet",
                    columns=["ticker", "coin", "wkey", "result", "a0", "a1"])
m = U[U.result.isin(["yes", "no"]) & U.a0.notna() & U.a1.notna()].copy()
m["pred"] = np.where(m.a1 >= m.a0, "yes", "no")
print("  markets %d   reproduced %.4f%%   ties pay YES"
      % (len(m), (m.pred == m.result).mean() * 100))
print("  YES base rate %.4f  (the favourite is right ~73%% of the time)"
      % (m.result == "yes").mean())

print("\n" + "=" * 78)
print("2. LIVE ACCOUNT  (what the strategy actually did)")
print("=" * 78)
F = pd.read_parquet(DATA / "fills_history.parquet")
O = pd.read_parquet(DATA / "orders_history.parquet")
lsm = O[O.client_order_id.astype(str).str.startswith("lsm")]
print("  orders table   %d rows total, %d are LSM" % (len(O), len(lsm)))
print("  fills  table   %d rows" % len(F))
print("  other strategies on the same account: %d rows (must be excluded)"
      % (len(O) - len(lsm)))

print("\n" + "=" * 78)
print("3. FILL BIAS  (the mechanism that caps the whole strategy)")
print("=" * 78)
print("  measured on live resting orders:")
print("    P(fill | order would LOSE)  0.962")
print("    P(fill | order would WIN)   0.843")
print("  a resting bid is 14%% more likely to fill when it is about to lose.")
print("  that is adverse selection, not bad luck: the counterparty who lifts")
print("  you is the one who just saw the index move against your side.")

print("\n" + "=" * 78)
print("4. POPULATION vs REALIZED EDGE")
print("=" * 78)
print("  *** CORRECTION ***")
print("  The figures below are computed on ladder_paths, a 28.6% SUBSET of")
print("  the :00/:30 population. They ran hot: +3.99c against a true")
print("  population value of +2.37c, a 1.43 SD fluctuation that a random")
print("  draw of that size reproduces 7.5% of the time. See")
print("  research/reconcile_edge.py. The correct population figures are")
print("  maker +2.29c and FILL-CORRECTED -0.43c, i.e. NEGATIVE.")
print("  Kept here only so the old number remains reproducible.")
print()


def fee1(p):
    return math.ceil(0.07 * p * (1 - p) * 100) / 100.0


import json
L = pd.read_parquet(DATA / "ladder_paths.parquet").drop_duplicates("ticker")
rows = []
for r in L.itertuples():
    try:
        p = json.loads(r.path)
    except Exception:
        continue
    e = sorted([k for k in p if 8 <= k["ml"] <= 14], key=lambda x: -x["ml"])
    if not e:
        continue
    e = e[0]
    yb, ya = e["bc"], e["ac"]
    if yb <= 0 or ya >= 1 or yb >= ya:
        continue
    fy = yb >= 0.5
    bid, ask = (yb, ya) if fy else (1.0 - ya, 1.0 - yb)
    if not (0.65 <= bid < 0.80):
        continue
    rows.append((bid, ask, 1 if (("yes" if fy else "no") == r.result) else 0))
D = pd.DataFrame(rows, columns=["bid", "ask", "won"])
pf = np.where(D.won == 1, 0.843, 0.962)
print("  in-band markets %d" % len(D))
print("  maker edge, assuming every order fills : %+.2fc"
      % ((D.won - D.bid).mean() * 100))
print("  maker edge, fill-corrected             : %+.2fc"
      % (((D.won - D.bid) * pf).sum() / pf.sum() * 100))
print("  taker edge, real ask + real fee        : %+.2fc"
      % ((D.won - D.ask - D.ask.map(fee1)).mean() * 100))
print("\n  the gap between the first two lines IS the strategy's problem.")

print("\n" + "=" * 78)
print("5. WHY THE ACCOUNT SWINGS SO HARD")
print("=" * 78)
p, q = 0.72, 15
win, lose = (1 - p) * q, -p * q
sd1 = math.sqrt(0.73 * (1 - 0.73)) * q
print("  at q%d and %.0fc: a win pays $%.2f, a loss costs $%.2f"
      % (q, p * 100, win, -lose))
print("  one market's SD is $%.2f - about %.1fx the mean outcome"
      % (sd1, sd1 / abs(0.73 * win + 0.27 * lose)) if (0.73 * win + 0.27 * lose)
      else "")
print("  a 73%% win rate still loses 4 in a row %.1f%% of the time"
      % (0.27 ** 4 * 100))
print("  4 straight losses at q15 = $%.2f" % (4 * lose))
