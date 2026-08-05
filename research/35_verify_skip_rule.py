"""Verification before this goes anywhere near live money.

FOUR WAYS THIS COULD BE FOOLING ME, each tested
  1. SELECTION. I picked the worst of four cells and then tested it. That is a
     multiple comparison, and the p-value ignores it. A permutation test that
     shuffles outcomes within (coin, price bucket) - destroying any real signal
     while preserving all structure - says how often a cell this bad appears by
     chance alone.
  2. REDUNDANCY. If others_px alone does the work, frac_long is dead weight and
     the live implementation gets much simpler AND free: the runner already
     scans all five series, whereas frac_long needs an extra trade-feed call
     per candidate.
  3. THRESHOLD FRAGILITY. Per-(coin, bucket) medians are estimated from thin
     data and are awkward to embed in the runner. If a single GLOBAL threshold
     works as well, the rule is simpler and far more robust.
  4. COIN CONCENTRATION. Three prior "findings" this session turned out to live
     entirely in one coin or one bucket.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260824)
D = pd.read_parquet(OUT / "grid.parquet")
G = D.groupby(["wkey", "minute"]).agg(s_px=("px", "sum"),
                                      n=("px", "size")).reset_index()
D2 = D.merge(G, on=["wkey", "minute"], how="left")
D2["others_px"] = np.where(D2.n >= 3, (D2.s_px - D2.px) / (D2.n - 1), np.nan)
q = D2[(D2.px >= 0.65) & (D2.px < 0.85) & (D2.minute >= 8) & (D2.minute <= 14)
       & (D2.vol >= 2000)]
E = (q.sort_values("minute", ascending=False)
       .groupby(["wkey", "coin"], as_index=False).first()).copy()
E = (E.sort_values(["wkey", "minute"], ascending=[True, False])
       .groupby("wkey", as_index=False).head(3)).copy()
E["pb"] = (E.px * 20).astype(int)
E = E.dropna(subset=["others_px"]).copy()
days = sorted(E.day.unique())
print(f"entries with the signal: {len(E):,} over {len(days)} days "
      f"(base edge {E.edge.mean()*100:+.2f}c, win {E.won.mean():.4f})\n")


def rolling(rule, label):
    """Walk-forward: thresholds from all prior days, applied to the next."""
    W, R = [], []
    for i in range(3, len(days)):
        fit = E[E.day.isin(days[:i])]
        te = E[E.day == days[i]].copy()
        m = rule(fit, te)
        if m is None or m.sum() < 2 or (~m).sum() < 5:
            continue
        W.append(te[m])
        R.append(te[~m])
    if not W:
        return None
    W, R = pd.concat(W), pd.concat(R)
    ds = sorted(set(W.day) | set(R.day))
    gw = {d: g for d, g in W.groupby("day")}
    gr = {d: g for d, g in R.groupby("day")}
    bs = []
    for _ in range(6000):
        pick = RNG.choice(len(ds), len(ds), True)
        a = [gw[ds[k]] for k in pick if ds[k] in gw]
        b = [gr[ds[k]] for k in pick if ds[k] in gr]
        if a and b:
            a, b = pd.concat(a), pd.concat(b)
            if len(a) > 8 and len(b) > 20:
                bs.append((a.edge.mean() - b.edge.mean()) * 100)
    bs = np.sort(np.array(bs))
    ndays = sum(1 for x, y in zip(W.groupby("day"), R.groupby("day"))
                if x[1].edge.mean() < y[1].edge.mean())
    return dict(label=label, nW=len(W), share=len(W) / (len(W) + len(R)),
                eW=W.edge.mean() * 100, eR=R.edge.mean() * 100,
                winW=W.won.mean(), diff=(W.edge.mean() - R.edge.mean()) * 100,
                lo=bs[int(.025 * len(bs))], hi=bs[int(.975 * len(bs))],
                p=2 * min((bs <= 0).mean(), (bs >= 0).mean()),
                days_worse=ndays, ndays=W.day.nunique())


def r_both_pb(fit, te):
    to = fit.groupby(["coin", "pb"]).others_px.median()
    tf = fit.groupby(["coin", "pb"]).frac_long.median()
    a = np.array([to.get(k, np.nan) for k in zip(te.coin, te.pb)])
    b = np.array([tf.get(k, np.nan) for k in zip(te.coin, te.pb)])
    return (te.others_px.values <= a) & (te.frac_long.values <= b)


def r_others_pb(fit, te):
    to = fit.groupby(["coin", "pb"]).others_px.median()
    a = np.array([to.get(k, np.nan) for k in zip(te.coin, te.pb)])
    return te.others_px.values <= a


def r_flong_pb(fit, te):
    tf = fit.groupby(["coin", "pb"]).frac_long.median()
    b = np.array([tf.get(k, np.nan) for k in zip(te.coin, te.pb)])
    return te.frac_long.values <= b


def r_both_global(fit, te):
    a, b = fit.others_px.median(), fit.frac_long.median()
    return (te.others_px.values <= a) & (te.frac_long.values <= b)


def r_others_global(fit, te):
    return te.others_px.values <= fit.others_px.median()


print("=== 2. WHICH SIGNAL IS DOING THE WORK? (rolling walk-forward) ===")
print(f"{'rule':>34} {'n skip':>7} {'share':>7} {'edge skip':>10} "
      f"{'edge keep':>10} {'DIFF':>8} {'95% CI':>19} {'p':>7} {'days':>6}")
res = {}
for fn, lbl in ((r_both_pb, "both LO, per (coin,bucket)"),
                (r_others_pb, "others_px LO only, per cell"),
                (r_flong_pb, "frac_long LO only, per cell"),
                (r_both_global, "both LO, GLOBAL threshold"),
                (r_others_global, "others_px LO only, GLOBAL")):
    r = rolling(fn, lbl)
    if not r:
        continue
    res[lbl] = r
    print(f"{lbl:>34} {r['nW']:>7} {r['share']:>7.3f} {r['eW']:>+10.2f} "
          f"{r['eR']:>+10.2f} {r['diff']:>+8.2f} "
          f"[{r['lo']:>+7.2f},{r['hi']:>+7.2f}] {r['p']:>7.4f} "
          f"{r['days_worse']}/{r['ndays']:>2}")

print("\n=== 1. PERMUTATION TEST: is this just picking the worst of four? ===")
print("   outcomes shuffled WITHIN (coin, price bucket), preserving all")
print("   structure and destroying only the signal. 2,000 permutations.\n")
real = res.get("both LO, per (coin,bucket)")
null = []
for it in range(2000):
    P = E.copy()
    P["edge"] = P.groupby(["coin", "pb"]).edge.transform(
        lambda s: s.sample(frac=1, random_state=RNG.integers(1 << 30)).values)
    W, R = [], []
    for i in range(3, len(days)):
        fit = P[P.day.isin(days[:i])]
        te = P[P.day == days[i]]
        to = fit.groupby(["coin", "pb"]).others_px.median()
        tf = fit.groupby(["coin", "pb"]).frac_long.median()
        a = np.array([to.get(k, np.nan) for k in zip(te.coin, te.pb)])
        b = np.array([tf.get(k, np.nan) for k in zip(te.coin, te.pb)])
        m = (te.others_px.values <= a) & (te.frac_long.values <= b)
        if m.sum() >= 2 and (~m).sum() >= 5:
            W.append(te[m])
            R.append(te[~m])
    if W:
        W, R = pd.concat(W), pd.concat(R)
        null.append((W.edge.mean() - R.edge.mean()) * 100)
null = np.array(null)
print(f"  real difference        {real['diff']:+.2f}c")
print(f"  null mean {null.mean():+.2f}c, sd {null.std():.2f}c, "
      f"2.5th pct {np.percentile(null,2.5):+.2f}c")
print(f"  permutation p-value    {(null <= real['diff']).mean():.4f}   "
      f"({(null <= real['diff']).sum()} of {len(null)} shuffles this extreme)")

print("\n=== 4. COIN CONCENTRATION: does it live in one coin? ===")
W, R = [], []
for i in range(3, len(days)):
    fit = E[E.day.isin(days[:i])]
    te = E[E.day == days[i]].copy()
    m = r_both_pb(fit, te)
    if m.sum() >= 2:
        W.append(te[m])
        R.append(te[~m])
W, R = pd.concat(W), pd.concat(R)
print(f"{'coin':>6} {'n skip':>7} {'edge skip':>10} {'edge keep':>10} {'DIFF':>8}")
for c in sorted(E.coin.unique()):
    a, b = W[W.coin == c], R[R.coin == c]
    if len(a) < 5 or len(b) < 15:
        continue
    print(f"{c:>6} {len(a):>7} {a.edge.mean()*100:>+10.2f} "
          f"{b.edge.mean()*100:>+10.2f} {(a.edge.mean()-b.edge.mean())*100:>+8.2f}")

print("\n=== 3. THRESHOLD VALUES a live runner would need ===")
print(f"  global others_px median : {E.others_px.median():.4f}")
print(f"  global frac_long median : {E.frac_long.median():.4f}")
print(f"  per-day stability of the others_px median:")
s = E.groupby("day").others_px.median()
print(f"    min {s.min():.4f}  max {s.max():.4f}  sd {s.std():.4f}")
s2 = E.groupby("day").frac_long.median()
print(f"  per-day stability of the frac_long median:")
print(f"    min {s2.min():.4f}  max {s2.max():.4f}  sd {s2.std():.4f}")
