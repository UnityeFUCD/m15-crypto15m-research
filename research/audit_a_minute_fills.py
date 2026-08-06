"""AUDIT A - minute-specific execution. Are :00 fills different from :30 fills?

WHY THIS MUST COME FIRST
  Every fill correction in this repo has used ONE global pair - 0.843/0.962,
  later 0.8848/0.9884 - applied to every market. If fill behaviour differs by
  close minute, that global number is a blend, and applying a blended rate to
  a single minute biases that minute's economics in an unknown direction.

  The :00 candidate cannot be priced until its OWN fill rate is measured.

WHAT IS COMPUTED, PER CLOSE MINUTE
  order count, submitted qty, filled qty, fill fraction
  P(fill | would win), P(fill | would lose), and the gap between them
  first-fill latency
  edge per FILLED contract      (what the strategy earns when it trades)
  edge per SUBMITTED contract   (what posting an order is worth ex ante)
  P&L per day
  day-clustered intervals

THE DISTINCTION THAT MATTERS
  edge_per_filled  = E[maker | filled]
  edge_per_submitted = E[maker * 1{filled}] = edge_per_filled * fill_rate

  Only the second is the value of a decision to post. The first overstates it
  by 1/fill_rate, and because fill_rate is itself higher on losers, the two
  can point in opposite directions.

POWER IS STATED BEFORE ANY NULL IS INTERPRETED. With 303 orders over 2 days
split four ways, most cells are tiny, and a null in a tiny cell is not
evidence of absence.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
RNG = np.random.default_rng(20260806)

O = pd.read_parquet(DATA / "orders_history.parquet")
F = pd.read_parquet(DATA / "fills_history.parquet")
U = pd.read_parquet(DATA / "underlying.parquet",
                    columns=["ticker", "coin", "result"])
U = U[U.result.isin(["yes", "no"])].drop_duplicates("ticker")
mx = DATA / "lsm_missing_outcomes.parquet"
if mx.exists():
    extra = pd.read_parquet(mx)
    extra = extra[~extra.ticker.isin(U.ticker)].copy()
    extra["coin"] = extra.ticker.str.extract(r"^KX([A-Z]+)15M")[0]
    U = pd.concat([U, extra[["ticker", "coin", "result"]]], ignore_index=True)

lsm = O[O.client_order_id.astype(str).str.startswith("lsm")].copy()
# Close minute from the ticker. KXBTC15M-26AUG051915-15 carries the wkey
# 26AUG051915 (yyMMMddHHMM, ET) and a trailing -15 suffix. Derive from the
# wkey the way every other script here does; ET->UTC is +4h and does not
# change the minute, so the suffix is an independent cross-check.
_wk = lsm.ticker.str.extract(r"-(\d{2}[A-Z]{3}\d{6})-")[0]
lsm["minute"] = (pd.to_datetime(_wk, format="%y%b%d%H%M", utc=True)
                 + pd.Timedelta(hours=4)).dt.minute
_suffix = pd.to_numeric(lsm.ticker.str.extract(r"-(\d{2})$")[0],
                        errors="coerce")
_agree = (lsm.minute == _suffix).mean()
print(f"[minute derivation cross-check: wkey vs ticker suffix agree "
      f"{_agree:.4f} on {len(lsm)} orders]")
assert _agree > 0.99, "minute derivation disagrees with the ticker suffix"
lsm["held"] = np.where(lsm.action.eq("sell"),
                       np.where(lsm.side.eq("yes"), "no", "yes"), lsm.side)
lsm["filled"] = pd.to_numeric(lsm.fill_count_fp, errors="coerce").fillna(0)
lsm["submitted"] = pd.to_numeric(lsm.initial_count_fp,
                                 errors="coerce").fillna(0)
lsm["did_fill"] = lsm.filled > 0
lsm = lsm.merge(U, on="ticker", how="left").dropna(subset=["result", "minute"])
lsm["minute"] = lsm.minute.astype(int)
lsm["would_win"] = lsm.held == lsm.result
yp = pd.to_numeric(lsm.yes_price_dollars, errors="coerce")
npx = pd.to_numeric(lsm.no_price_dollars, errors="coerce")
lsm["px"] = np.where(lsm.held.eq("yes"), yp, npx)
lsm["edge_if_filled"] = np.where(lsm.would_win, 1.0 - lsm.px, -lsm.px)
lsm["ct"] = pd.to_datetime(lsm.created_time, format="mixed", utc=True)
lsm["day"] = lsm.ct.dt.date

# first-fill latency from the fills table
ff = (F.assign(ft=pd.to_datetime(F.created_time, format="mixed", utc=True))
      .groupby("order_id").ft.min().rename("first_fill"))
lsm = lsm.merge(ff, left_on="order_id", right_index=True, how="left")
lsm["latency_s"] = (lsm.first_fill - lsm.ct).dt.total_seconds()

print("=" * 92)
print("AUDIT A - MINUTE-SPECIFIC EXECUTION")
print("=" * 92)
print(f"  LSM orders with a settled outcome and a decodable minute: {len(lsm)}")
print(f"  window: {lsm.day.min()} to {lsm.day.max()}  ({lsm.day.nunique()} days)")

print("\n" + "-" * 92)
print("POWER, STATED BEFORE ANY RESULT IS INTERPRETED")
print("-" * 92)
print("  %-8s %7s %9s %9s %28s" % ("minute", "orders", "wouldwin",
                                   "wouldlose", "min. detectable gap @80%"))
for mn, g in lsm.groupby("minute"):
    nw, nl = int(g.would_win.sum()), int((~g.would_win).sum())
    if nw < 1 or nl < 1:
        print("  :%02d      %7d %9d %9d   %26s" % (mn, len(g), nw, nl,
                                                   "no test possible"))
        continue
    se = math.sqrt(0.9 * 0.1 / nw + 0.9 * 0.1 / nl)
    print("  :%02d      %7d %9d %9d   %24.1fpp" % (mn, len(g), nw, nl,
                                                   2.8 * se * 100))
print("  reference: the pooled estimate found a +10.4pp gap.")
print("  Any per-minute cell whose detectable gap exceeds ~10pp CANNOT")
print("  resolve the effect, and a null there means nothing.")

print("\n" + "-" * 92)
print("FILL BEHAVIOUR BY CLOSE MINUTE")
print("-" * 92)
print("  %-7s %6s %7s %7s %8s %10s %10s %8s %9s"
      % ("minute", "orders", "subQty", "filQty", "fillFrac", "P(fil|win)",
         "P(fil|lose)", "gap", "lat(s)"))
rowsA = []
for mn, g in lsm.groupby("minute"):
    sub, fil = g.submitted.sum(), g.filled.sum()
    w, l = g[g.would_win], g[~g.would_win]
    pw = w.did_fill.mean() if len(w) else np.nan
    pl = l.did_fill.mean() if len(l) else np.nan
    lat = g.latency_s.median()
    print("  :%02d     %6d %7.0f %7.0f %8.4f %10s %11s %8s %9s"
          % (mn, len(g), sub, fil, fil / sub if sub else np.nan,
             f"{pw:.4f}" if pw == pw else "-",
             f"{pl:.4f}" if pl == pl else "-",
             f"{(pl-pw)*100:+.1f}pp" if (pw == pw and pl == pl) else "-",
             f"{lat:.0f}" if lat == lat else "-"))
    rowsA.append((mn, pw, pl))
sub, fil = lsm.submitted.sum(), lsm.filled.sum()
w, l = lsm[lsm.would_win], lsm[~lsm.would_win]
print("  %-7s %6d %7.0f %7.0f %8.4f %10.4f %11.4f %8s %9.0f"
      % ("ALL", len(lsm), sub, fil, fil / sub, w.did_fill.mean(),
         l.did_fill.mean(),
         f"{(l.did_fill.mean()-w.did_fill.mean())*100:+.1f}pp",
         lsm.latency_s.median()))

print("\n" + "-" * 92)
print("EDGE PER FILLED vs PER SUBMITTED CONTRACT  (the decision-relevant one)")
print("-" * 92)
print("  %-7s %8s %11s %14s %14s %12s"
      % ("minute", "orders", "fillRate", "c/FILLED", "c/SUBMITTED", "$/day"))
days = lsm.day.nunique()
for mn, g in lsm.groupby("minute"):
    f_ = g[g.did_fill]
    if not len(f_):
        continue
    per_filled = (f_.edge_if_filled * f_.filled).sum() / f_.filled.sum() * 100
    per_sub = (f_.edge_if_filled * f_.filled).sum() / g.submitted.sum() * 100
    fr = g.filled.sum() / g.submitted.sum()
    pnl = (f_.edge_if_filled * f_.filled).sum() / days
    print("  :%02d     %8d %11.4f %+13.2fc %+13.2fc %+11.2f"
          % (mn, len(g), fr, per_filled, per_sub, pnl))
fl = lsm[lsm.did_fill]
print("  %-7s %8d %11.4f %+13.2fc %+13.2fc %+11.2f"
      % ("ALL", len(lsm), lsm.filled.sum() / lsm.submitted.sum(),
         (fl.edge_if_filled * fl.filled).sum() / fl.filled.sum() * 100,
         (fl.edge_if_filled * fl.filled).sum() / lsm.submitted.sum() * 100,
         (fl.edge_if_filled * fl.filled).sum() / days))

print("\n" + "-" * 92)
print("DAY-CLUSTERED INTERVAL ON EDGE PER SUBMITTED CONTRACT")
print("-" * 92)
for mn, g in list(lsm.groupby("minute")) + [("ALL", lsm)]:
    gd = {k: v for k, v in g.groupby("day")}
    ds = sorted(gd)
    if len(ds) < 2:
        print("  %-7s only %d day(s) - no clustered interval possible"
              % (f":{mn:02d}" if mn != "ALL" else "ALL", len(ds)))
        continue
    bs = []
    for _ in range(4000):
        s = pd.concat([gd[ds[i]] for i in RNG.integers(0, len(ds), len(ds))])
        if s.submitted.sum() > 0:
            bs.append((s[s.did_fill].edge_if_filled
                       * s[s.did_fill].filled).sum() / s.submitted.sum() * 100)
    bs = np.sort(np.array(bs))
    pt = (g[g.did_fill].edge_if_filled
          * g[g.did_fill].filled).sum() / g.submitted.sum() * 100
    print("  %-7s %+6.2fc  95%% CI [%+7.2f, %+7.2f]  P(<=0) %.4f  (%d days)"
          % (f":{mn:02d}" if mn != "ALL" else "ALL", pt,
             bs[100], bs[3899], (bs <= 0).mean(), len(ds)))

print("\n" + "=" * 92)
print("WHAT THIS CAN AND CANNOT SUPPORT")
print("=" * 92)
n00 = int((lsm.minute == 0).sum())
print(f"  minute :00 has {n00} live orders across {lsm.day.nunique()} days.")
print("  A minute-specific fill model estimated on that is far too thin to")
print("  replace the pooled estimate. The honest position is that AUDIT A")
print("  CANNOT deliver a :00-specific fill rate from this record, and the")
print("  economics in minute_zero.md must therefore be reported as a RANGE")
print("  over the pooled model, with the sensitivity shown explicitly.")
