"""PROSPECTIVE EXECUTION TEST - held separate from the population audit.

WHY SEPARATE, AND WHAT 'PROSPECTIVE' MEANS HERE

  The population audit asks: what is the edge, IF an order fills?
  This test asks:            do orders fill in a way that preserves that edge?

  Those are different questions on different data, and pooling them is what
  produced the wrong answers earlier in this project. The population study
  cannot observe fills; the live record cannot identify a signal. Each is
  evidence only about its own question.

  HONESTY ABOUT 'PROSPECTIVE': the live orders span 2026-08-04 to 08-05, which
  OVERLAPS the population window. So this is NOT out-of-sample in time, and it
  cannot serve as a forward test of HCR. It is prospective only in the sense
  that the fill-bias model was estimated first and is checked here against
  what execution actually did. Any HCR figure computed on 303 orders across
  2 days is reported for completeness and must not be read as validation.

PRE-REGISTERED PREDICTIONS, stated before the numbers are computed. Each one
follows from the population model; each can fail.

  P1  fill rate on eventual LOSERS exceeds fill rate on eventual WINNERS
      (this is the fill-bias mechanism; if it fails, the mechanism is wrong)
  P2  realized edge per contract lands near the FILL-CORRECTED population
      figure (+1.36c), not the raw maker figure (+3.99c)
  P3  canceled orders skew toward eventual WINNERS - the ones that got away
  P4  the strategy's net result is near zero, well below any backtest that
      assumes full fills

POWER WARNING computed up front: with n orders over 2 days, state what this
test could and could not have detected. A null here is not evidence of
absence if the test never had the power to see the effect.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
POP_RAW_MAKER = 3.99      # cents, population, assumes every order fills
POP_CORRECTED = 1.36      # cents, population, fill-corrected
P_FILL_LOSE, P_FILL_WIN = 0.962, 0.843

print("=" * 78)
print("PROSPECTIVE EXECUTION TEST - live orders only, held apart from")
print("the population audit")
print("=" * 78)

O = pd.read_parquet(DATA / "orders_history.parquet")
U = pd.read_parquet(DATA / "underlying.parquet",
                    columns=["ticker", "coin", "result"])
U = U[U.result.isin(["yes", "no"])].drop_duplicates("ticker")

# CRITICAL: underlying.parquet was snapshotted before some LSM markets closed.
# Those markets are NOT missing at random - the 17 absent here were worth
# -$134 and previously moved the measured result from +$143.38 to +$9.38.
# Dropping them silently is the single most expensive bug in this project,
# and it has now occurred twice. Merge the resolved outcomes back in.
_mx = DATA / "lsm_missing_outcomes.parquet"
if _mx.exists():
    extra = pd.read_parquet(_mx)
    extra = extra[~extra.ticker.isin(U.ticker)].copy()
    extra["coin"] = extra.ticker.str.extract(r"^KX([A-Z]+)15M")[0]
    U = pd.concat([U, extra[["ticker", "coin", "result"]]], ignore_index=True)
    print(f"  [recovered {len(extra)} late-closing outcomes that "
          f"underlying.parquet is missing]")
else:
    print("  [WARNING] lsm_missing_outcomes.parquet absent - run "
          "research/fetch_lsm_outcomes.py first, or results are "
          "survivorship-biased")

lsm = O[O.client_order_id.astype(str).str.startswith("lsm")].copy()
print(f"\n  LSM orders {len(lsm)}   (of {len(O)} on the account - the other "
      f"{len(O)-len(lsm)} belong to different strategies)")
print(f"  window {lsm.created_time.min()[:10]} to {lsm.created_time.max()[:10]}"
      f"  = 2 days")

# ---- derive the side actually HELD ----------------------------------------
# orders table: action=sell on side=yes means the position held is NO.
lsm["held"] = np.where(lsm.action.eq("sell"),
                       np.where(lsm.side.eq("yes"), "no", "yes"),
                       lsm.side)
lsm["filled"] = pd.to_numeric(lsm.fill_count_fp, errors="coerce").fillna(0)
lsm["initial"] = pd.to_numeric(lsm.initial_count_fp, errors="coerce").fillna(0)
lsm["did_fill"] = lsm.filled > 0
lsm = lsm.merge(U, on="ticker", how="left")
lsm = lsm.dropna(subset=["result"])
lsm["would_win"] = (lsm.held == lsm.result)
print(f"  with a settled outcome: {len(lsm)}")

# price actually paid for the held side
yp = pd.to_numeric(lsm.yes_price_dollars, errors="coerce")
npx = pd.to_numeric(lsm.no_price_dollars, errors="coerce")
lsm["px"] = np.where(lsm.held.eq("yes"), yp, npx)

print("\n" + "-" * 78)
print("POWER CHECK - what could this test have detected?")
print("-" * 78)
n = len(lsm)
nw, nl = int(lsm.would_win.sum()), int((~lsm.would_win).sum())
se = math.sqrt(0.9 * 0.1 / max(nw, 1) + 0.9 * 0.1 / max(nl, 1))
print(f"  n = {n} orders ({nw} would-win, {nl} would-lose)")
print(f"  SE on a fill-rate difference ~ {se*100:.1f}pp")
print(f"  smallest difference detectable at 80% power ~ {2.8*se*100:.1f}pp")
print(f"  the population model predicts a {(P_FILL_LOSE-P_FILL_WIN)*100:.1f}pp "
      f"difference")
print("  -> " + ("adequately powered" if 2.8 * se < (P_FILL_LOSE - P_FILL_WIN)
                 else "UNDERPOWERED: cannot resolve the predicted effect"))

print("\n" + "-" * 78)
print("P1  fill rate on eventual LOSERS > fill rate on eventual WINNERS")
print("-" * 78)
fw = lsm[lsm.would_win].did_fill.mean()
fl = lsm[~lsm.would_win].did_fill.mean()
print(f"  P(fill | would WIN)  {fw:.4f}   n {nw}")
print(f"  P(fill | would LOSE) {fl:.4f}   n {nl}")
print(f"  difference {(fl-fw)*100:+.1f}pp   "
      f"(population model predicts {(P_FILL_LOSE-P_FILL_WIN)*100:+.1f}pp)")
rng = np.random.default_rng(7)
w, l = lsm[lsm.would_win].did_fill.values, lsm[~lsm.would_win].did_fill.values
bs = np.sort(np.array([rng.choice(l, len(l), True).mean()
                       - rng.choice(w, len(w), True).mean()
                       for _ in range(8000)]) * 100)
print(f"  95% CI [{bs[200]:+.1f}, {bs[7799]:+.1f}]pp   "
      f"P(<=0) {(bs<=0).mean():.4f}")
print("  VERDICT: " + ("direction CONFIRMED" if fl > fw else "direction FAILED"))

print("\n" + "-" * 78)
print("P2  realized edge per contract sits near the FILL-CORRECTED figure")
print("-" * 78)
f = lsm[lsm.did_fill].copy()
f["pnl_ct"] = np.where(f.would_win, 1.0 - f.px, -f.px)
realized = (f.pnl_ct * f.filled).sum() / f.filled.sum() * 100
print(f"  filled orders {len(f)}   contracts {f.filled.sum():,.0f}")
print(f"  realized edge          {realized:+.2f}c per contract")
print(f"  population raw maker   {POP_RAW_MAKER:+.2f}c   "
      f"(what a full-fill backtest would claim)")
print(f"  population corrected   {POP_CORRECTED:+.2f}c   (what the model says)")
d_raw, d_corr = abs(realized - POP_RAW_MAKER), abs(realized - POP_CORRECTED)
print(f"  |realized - raw| {d_raw:.2f}c   vs   |realized - corrected| "
      f"{d_corr:.2f}c")
print("  VERDICT: " + ("closer to CORRECTED - model supported"
                       if d_corr < d_raw else
                       "closer to RAW - fill correction not supported here"))

print("\n" + "-" * 78)
print("P3  canceled orders skew toward eventual WINNERS")
print("-" * 78)
nf = lsm[~lsm.did_fill]
if len(nf) >= 5:
    print(f"  never-filled orders {len(nf)}")
    print(f"  of those, would have WON: {nf.would_win.mean():.4f}")
    print(f"  base rate among all orders: {lsm.would_win.mean():.4f}")
    print("  VERDICT: " + ("CONFIRMED - the ones that got away were winners"
                           if nf.would_win.mean() > lsm.would_win.mean()
                           else "FAILED"))
else:
    print(f"  only {len(nf)} never-filled orders - no test possible")

print("\n" + "-" * 78)
print("P4  net result is near zero, far below a full-fill backtest")
print("-" * 78)
net = (f.pnl_ct * f.filled).sum()
hypo = (lsm.assign(p=np.where(lsm.would_win, 1.0 - lsm.px, -lsm.px))
        .eval("p * initial").sum())
print(f"  actual net over 2 days          {net:+.2f}")
print(f"  same orders if ALL had filled   {hypo:+.2f}")
print(f"  cost of fill bias               {net-hypo:+.2f}")
print("  VERDICT: " + ("CONFIRMED - full-fill overstates"
                       if hypo > net else "FAILED - full fill was not better"))

print("\n" + "=" * 78)
print("REPORTED FOR COMPLETENESS ONLY - NOT A TEST OF HCR")
print("=" * 78)
print("  303 orders across 2 days cannot validate a signal. Any split below")
print("  is descriptive. The population audit is the only evidence on HCR.")
print(f"  filled orders by held side: {f.held.value_counts().to_dict()}")
print(f"  win rate on filled orders : {f.would_win.mean():.4f}")
print(f"  win rate on ALL orders    : {lsm.would_win.mean():.4f}")
print(f"  the gap between those two lines IS the fill bias, measured live.")
