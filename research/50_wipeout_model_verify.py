"""Verify the wipeout model, and test the only thing that matters: DOLLARS.

WHAT WAS FOUND
  No single feature predicts a correlated wipeout - fifteen were tested and all
  failed. But a regularised logistic model on all twenty, cross-validated
  leave-one-day-out, reached AUC 0.6972 against a permutation null of 0.5061
  (95th percentile 0.6324). So the signal is a CONJUNCTION: it exists only in
  combination, which is exactly why every univariate test missed it.

WHY THAT IS NOT YET A RESULT
  Four things have to hold before this is worth anything:
    1. STABILITY - not an artifact of one seed, one regularisation, or one
       feature set.
    2. HONEST SAMPLE - 20 wipeout events is very few. AUC can look respectable
       while resting on a handful of windows.
    3. IT MUST BEAT THE SIMPLE THING - if a plain price threshold does as well,
       the model adds nothing.
    4. IT MUST MAKE MONEY - every previously "real" signal in this project died
       here, because skipping entries costs opportunity we cannot replace. A
       wipeout filter is different in kind (it removes THREE losing positions at
       once, not one marginal entry), so it deserves the test rather than the
       assumption.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260923)
exec(open(OUT / "49_wipeout_definitive.py").read().split(
    'print("\\n=== OUT-OF-SAMPLE AUC')[0])

print("\n" + "=" * 70)
print("1. STABILITY - across seeds, regularisation, and feature subsets")
print("=" * 70)
print(f"{'setting':>28} {'AUC':>8}")
for lam in (0.5, 1.0, 2.0, 5.0, 10.0):
    def fl(X, y, lam=lam, iters=600, lr=0.25):
        n, d = X.shape
        w = np.zeros(d + 1)
        Xb = np.hstack([np.ones((n, 1)), X])
        pos = max(y.sum(), 1)
        neg = max(n - y.sum(), 1)
        sw = np.where(y == 1, n / (2.0 * pos), n / (2.0 * neg))
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))
            g = Xb.T @ (sw * (p - y)) / n
            g[1:] += lam * w[1:] / n
            w -= lr * g
        return w
    globals()["fit_logit"] = fl
    print(f"{'lambda ' + str(lam):>28} {cv_auc(X, y, dd):>8.4f}")

print("\n=== single-feature AUCs, for comparison ===")
best_single = 0
for i, f in enumerate(FE):
    a = auc(y, X[:, i])
    a = max(a, 1 - a)          # direction-free
    if a > best_single:
        best_single, best_name = a, f
print(f"  best single feature: {best_name} at AUC {best_single:.4f}")
print(f"  full model         : 0.6972")
print("  The gap is the conjunction - the part no univariate test could see.")

print("\n" + "=" * 70)
print("2. HOW MANY EVENTS IS THIS RESTING ON?")
print("=" * 70)
print(f"  windows {len(y):,}   wipeouts {int(y.sum())}   base rate {y.mean():.4f}")
print(f"  an AUC of 0.70 on {int(y.sum())} events has a standard error near")
print(f"  {np.sqrt(0.70*0.30/min(y.sum(), len(y)-y.sum())):.3f}, so the honest interval is roughly "
      f"[{0.6972-2*np.sqrt(0.70*0.30/y.sum()):.2f}, {min(1,0.6972+2*np.sqrt(0.70*0.30/y.sum())):.2f}]")

print("\n" + "=" * 70)
print("3. THE DOLLAR TEST - walk-forward, refit on past days only")
print("=" * 70)
G2 = pd.read_parquet(OUT / "grid.parquet")
q2 = G2[(G2.px >= 0.65) & (G2.px < 0.85) & (G2.minute >= 8) & (G2.minute <= 14)
        & (G2.vol >= 2000)]
E2 = (q2.sort_values("minute", ascending=False)
        .groupby(["wkey", "coin"], as_index=False).first())
E2 = (E2.sort_values(["wkey", "minute"], ascending=[True, False])
        .groupby("wkey", as_index=False).head(3)).copy()
E2["pnl"] = np.where(E2.won == 1, (1 - E2.px) * 15, -E2.px * 15)
WP = E2.groupby("wkey").pnl.sum().rename("pnl")
WW = W.merge(WP, on="wkey", how="left")

def fl2(X, y, lam=2.0, iters=600, lr=0.25):
    n, d = X.shape
    w = np.zeros(d + 1)
    Xb = np.hstack([np.ones((n, 1)), X])
    pos = max(y.sum(), 1)
    neg = max(n - y.sum(), 1)
    sw = np.where(y == 1, n / (2.0 * pos), n / (2.0 * neg))
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (sw * (p - y)) / n
        g[1:] += lam * w[1:] / n
        w -= lr * g
    return w

print(f"{'day':>12} {'windows':>8} {'all $':>10} {'skip top decile $':>19} "
      f"{'skipped':>8} {'wipes skipped':>14}")
tot_all = tot_sel = 0.0
n_sk = n_wipe_sk = 0
rows = []
for i in range(3, len(days)):
    tr = WW[WW.day.isin(days[:i])]
    te = WW[WW.day == days[i]]
    if len(te) < 5 or tr.wipe.sum() < 3:
        continue
    Xtr = tr[FE].values.astype(float)
    Xte = te[FE].values.astype(float)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    w = fl2((Xtr - mu) / sd, tr.wipe.values.astype(float))
    sc = predict((Xte - mu) / sd, w)
    thr = np.quantile(sc, 0.90)                # skip the riskiest 10%
    keep = sc < thr
    a = te.pnl.sum()
    b = te.pnl[keep].sum()
    tot_all += a
    tot_sel += b
    n_sk += (~keep).sum()
    n_wipe_sk += te.wipe.values[~keep].sum()
    rows.append((a, b))
    print(f"{days[i]:>12} {len(te):>8} {a:>+10.2f} {b:>+19.2f} "
          f"{(~keep).sum():>8} {int(te.wipe.values[~keep].sum()):>14}")
nd = len(rows)
print(f"\n  ALL windows       : ${tot_all/nd:+.2f}/day")
print(f"  skip riskiest 10% : ${tot_sel/nd:+.2f}/day")
print(f"  difference        : ${(tot_sel-tot_all)/nd:+.2f}/day   "
      f"better on {sum(1 for a, b in rows if b > a)}/{nd} days")
print(f"  skipped {n_sk} windows containing {int(n_wipe_sk)} of "
      f"{int(WW[WW.day.isin(days[3:])].wipe.sum())} wipeouts")

print("\n=== SWEEP THE SKIP THRESHOLD ===")
print(f"{'skip top':>10} {'$/day':>10} {'vs all':>10} {'windows skipped':>17} "
      f"{'wipes caught':>14}")
for pct in (0.95, 0.90, 0.80, 0.70, 0.50):
    ta = ts = 0.0
    nsk = nwp = 0
    nd2 = 0
    for i in range(3, len(days)):
        tr = WW[WW.day.isin(days[:i])]
        te = WW[WW.day == days[i]]
        if len(te) < 5 or tr.wipe.sum() < 3:
            continue
        Xtr = tr[FE].values.astype(float)
        Xte = te[FE].values.astype(float)
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1.0
        w = fl2((Xtr - mu) / sd, tr.wipe.values.astype(float))
        sc = predict((Xte - mu) / sd, w)
        keep = sc < np.quantile(sc, pct)
        ta += te.pnl.sum()
        ts += te.pnl[keep].sum()
        nsk += (~keep).sum()
        nwp += te.wipe.values[~keep].sum()
        nd2 += 1
    print(f"{1-pct:>9.0%} {ts/nd2:>+10.2f} {(ts-ta)/nd2:>+10.2f} "
          f"{nsk:>17} {int(nwp):>14}")
