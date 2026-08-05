"""ALLOCATION, NOT GATING: size each entry by its predicted edge.

WHY THIS IS THE RIGHT LEVER AND THE OTHERS ARE NOT
  Four signals genuinely predict edge - volume, longshot share, cross-coin
  price, and entry rank within the window. Every attempt to use them as GATES
  lost money, because discarding an entry buys nothing back when opportunity
  is the binding constraint.

  The obvious alternative was quote-price control: quote more aggressively when
  the signal is strong. That is now measured and mostly IMPOSSIBLE here - 84%
  of observed spreads are exactly one tick, so quoting one tick better would
  cross and be rejected. Only 16% of opportunities have any room at all.

  That leaves SIZE. Sizing is the one allocation channel with no capacity
  limit, no opportunity cost, and no execution risk: every entry is still
  taken, the total contracts deployed can be held constant, and only the
  distribution across entries changes.

THE TEST THAT MATTERS
  Hold TOTAL CONTRACTS PER DAY constant and redistribute them toward
  higher-predicted-edge entries. If realised edge per contract rises, the
  signals are worth something in a form we can actually use - and the gain is
  free, because nothing was given up to get it.

  The weights are built from PAST DAYS ONLY and applied to the next day, so
  this is a walk-forward test, not a description of the sample.

KNOWN EDGE PREDICTORS, from earlier work
  rank within window : rank1 +10.20c, rank2 +11.28c, rank3 +16.65c
  volume             : monotone, 2000-floor was selected 7/7 walk-forward days
  longshot share     : +4.39c high vs low inside the live filter, p 0.0000
  others_px          : +5.31c out of sample, p 0.0000
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260930)
D = pd.read_parquet(OUT / "grid.parquet")
BASE_QTY = 15

GX = D.groupby(["wkey", "minute"]).agg(s_px=("px", "sum"),
                                       n=("px", "size")).reset_index()
D2 = D.merge(GX, on=["wkey", "minute"], how="left")
D2["others_px"] = np.where(D2.n >= 3, (D2.s_px - D2.px) / (D2.n - 1).clip(lower=1),
                           np.nan)
q = D2[(D2.px >= 0.65) & (D2.px < 0.80) & (D2.minute >= 8) & (D2.minute <= 14)
       & (D2.vol >= 2000)]
E = (q.sort_values("minute", ascending=False)
       .groupby(["wkey", "coin"], as_index=False).first())
E = (E.sort_values(["wkey", "minute"], ascending=[True, False])).copy()
E["rank"] = E.groupby("wkey").cumcount() + 1
E = E[E["rank"] <= 3].copy()
E["ec"] = np.where(E.won == 1, (1 - E.px), -E.px)          # edge per contract
E["others_px"] = E.others_px.fillna(E.others_px.median())
days = sorted(E.day.unique())
print(f"entries {len(E):,}   windows {E.wkey.nunique():,}   days {len(days)}")
print(f"flat-{BASE_QTY} baseline: {len(E)/len(days):.1f} entries/day, "
      f"{len(E)*BASE_QTY/len(days):,.0f} contracts/day, "
      f"edge {E.ec.mean()*100:+.2f}c\n")

FEATS = ["rank", "vol", "frac_long", "others_px", "px"]


def fit_weights(train):
    """Linear edge model on standardised features, ridge-regularised.
    Deliberately simple: 5 features, ~800 rows, and a complex model would fit
    noise that the walk-forward would then expose."""
    X = train[FEATS].values.astype(float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xs = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    y = train.ec.values
    lam = 10.0
    A = Xs.T @ Xs + lam * np.eye(Xs.shape[1])
    A[0, 0] -= lam
    w = np.linalg.solve(A, Xs.T @ y)
    return w, mu, sd


def predict_edge(df, w, mu, sd):
    X = df[FEATS].values.astype(float)
    Xs = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    return Xs @ w


print("=== WALK-FORWARD: weights from prior days, applied to the next ===\n")
print(f"{'day':>12} {'entries':>8} {'flat $':>10} {'weighted $':>12} "
      f"{'diff':>9} {'contracts flat':>15} {'contracts wtd':>14}")
tot_f = tot_w = 0.0
cf = cw = 0
rows = []
for i in range(3, len(days)):
    tr = E[E.day.isin(days[:i])]
    te = E[E.day == days[i]].copy()
    if len(te) < 10 or len(tr) < 100:
        continue
    w, mu, sd = fit_weights(tr)
    pe = predict_edge(te, w, mu, sd)
    # map predicted edge to a size multiplier, then RESCALE so the day deploys
    # exactly the same contracts as flat sizing - the gain must come from
    # distribution, not from quietly trading more
    r = np.clip(pe / max(tr.ec.mean(), 1e-6), 0.3, 2.0)
    r = r / r.mean()
    qty = np.clip(np.round(BASE_QTY * r), 5, 30)
    qty = qty * (BASE_QTY * len(te)) / qty.sum()      # exact contract match
    qty = np.round(qty).astype(int)
    f_pnl = (te.ec.values * BASE_QTY).sum()
    w_pnl = (te.ec.values * qty).sum()
    tot_f += f_pnl
    tot_w += w_pnl
    cf += BASE_QTY * len(te)
    cw += qty.sum()
    rows.append((f_pnl, w_pnl))
    print(f"{days[i]:>12} {len(te):>8} {f_pnl:>+10.2f} {w_pnl:>+12.2f} "
          f"{w_pnl-f_pnl:>+9.2f} {BASE_QTY*len(te):>15,} {qty.sum():>14,}")

nd = len(rows)
print(f"\n  flat {BASE_QTY}      : ${tot_f/nd:+.2f}/day   {cf/nd:,.0f} contracts/day")
print(f"  edge-weighted : ${tot_w/nd:+.2f}/day   {cw/nd:,.0f} contracts/day")
print(f"  difference    : ${(tot_w-tot_f)/nd:+.2f}/day   "
      f"better on {sum(1 for a, b in rows if b > a)}/{nd} days")
print(f"  contracts deployed differ by {abs(cw-cf)/max(cf,1)*100:.2f}% "
      f"- the comparison is matched")

diff = np.array([b - a for a, b in rows])
bs = np.sort(np.array([RNG.choice(diff, len(diff), True).mean()
                       for _ in range(8000)]))
print(f"\n  day-clustered 95% CI on the daily gain: "
      f"[{bs[200]:+.2f}, {bs[7799]:+.2f}]")
print(f"  P(edge-weighting is WORSE) = {(bs < 0).mean():.4f}")

print("\n=== WHAT THE MODEL LEARNED (last fit, standardised coefficients) ===")
w, mu, sd = fit_weights(E[E.day.isin(days[:-1])])
for nm, c in zip(["intercept"] + FEATS, w):
    print(f"  {nm:>11}: {c*100:>+8.3f}c")
print("\n  Positive means more size where that feature is higher.")

print("\n=== SANITY: does the predicted edge actually rank realised edge? ===")
tr = E[E.day.isin(days[:-3])]
te = E[E.day.isin(days[-3:])].copy()
w, mu, sd = fit_weights(tr)
te["pe"] = predict_edge(te, w, mu, sd)
te["q"] = pd.qcut(te.pe.rank(method="first"), 4, labels=False, duplicates="drop")
print(f"{'predicted quartile':>20} {'n':>6} {'predicted':>11} {'REALISED':>10}")
for qq, g in te.groupby("q"):
    print(f"{int(qq):>20} {len(g):>6} {g.pe.mean()*100:>+10.2f}c "
          f"{g.ec.mean()*100:>+9.2f}c")
print("\n  If realised rises with predicted, the model ranks entries correctly")
print("  and sizing by it is justified. If flat, the weights are noise.")
