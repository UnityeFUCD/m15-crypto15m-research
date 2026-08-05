"""Make the dead-entry cuts ADAPTIVE, so a regime shift cannot un-aim them.

THE DEFECT
  The filter uses two ABSOLUTE cut points, others_px <= 0.71 and
  frac_long <= 0.5046, chosen as the medians of the pre-shift distribution so
  that the conjunction would catch roughly a quarter of entries.

  Post-shift the distributions moved - volume median +141%, longshot share
  median +18% - and the same absolute numbers now sit near the 85th percentile.
  The conjunction catches 72.0% of entries instead of 25.1%. It is discarding
  three quarters of the book on a rule designed to discard a quarter.

  This is exactly the failure the volume floor did NOT have: 2000 happened to
  remain the total-edge optimum post-shift, so it survived by luck rather than
  design. The dead-entry cuts did not.

THE FIX
  Express the cuts as PERCENTILES of a rolling recent window rather than as
  fixed numbers. Then the filter catches the same share of entries whatever the
  distribution does, and a future regime shift cannot silently un-aim it.

  This is not a new signal and not an attempt to find a better optimum on three
  days of data. It is making an existing, validated filter robust to the thing
  that has already broken it once.

TESTED
  - does percentile-targeting reproduce the pre-shift behaviour? (it must, or
    it is a different filter)
  - does it repair the post-shift behaviour?
  - what target share is right, and how sensitive is the answer?
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261019)
QTY = 15


def prep_pre():
    PRE = pd.read_parquet(OUT / "grid.parquet")
    G = PRE.groupby(["wkey", "minute"]).agg(s=("px", "sum"),
                                            k=("px", "size")).reset_index()
    Q = PRE.merge(G, on=["wkey", "minute"], how="left")
    Q["others_px"] = np.where(Q.k >= 3, (Q.s - Q.px) / (Q.k - 1).clip(lower=1),
                              np.nan)
    q = Q[(Q.px >= 0.65) & (Q.px < 0.85) & (Q.minute >= 8) & (Q.minute <= 14)
          & (Q.vol >= 2000)]
    e = (q.sort_values("minute", ascending=False)
           .groupby(["wkey", "coin"], as_index=False).first())
    return e.dropna(subset=["others_px"]).copy()


def prep_post():
    D = pd.read_parquet(OUT / "postshift_nofilter.parquet")
    P = D[D.vol >= 2000].copy()
    G = P.groupby("wkey").agg(s=("px", "sum"), k=("px", "size")).reset_index()
    P = P.merge(G, on="wkey", how="left")
    P["others_px"] = np.where(P.k >= 3, (P.s - P.px) / (P.k - 1).clip(lower=1),
                              np.nan)
    return P.dropna(subset=["others_px"]).copy()


A, P = prep_pre(), prep_post()
print(f"pre-shift entries {len(A):,} over {A.day.nunique()} days")
print(f"post-shift entries {len(P):,} over {P.day.nunique()} days\n")

print("=" * 76)
print("ABSOLUTE cuts vs PERCENTILE cuts - what share does each catch?")
print("=" * 76)
print(f"{'regime':>7} {'rule':>26} {'share caught':>13} {'dead edge':>11} "
      f"{'kept edge':>11} {'gap':>8}")
for nm, df in (("PRE", A), ("POST", P)):
    for lbl, om, fm in (("ABSOLUTE 0.71/0.5046", 0.71, 0.5046),
                        ("PERCENTILE 50th/50th",
                         df.others_px.median(), df.frac_long.median())):
        d = df[(df.others_px <= om) & (df.frac_long <= fm)]
        k = df[~((df.others_px <= om) & (df.frac_long <= fm))]
        if not len(d) or not len(k):
            continue
        print(f"{nm:>7} {lbl:>26} {len(d)/len(df):>13.3f} "
              f"{d.edge.mean()*100:>+11.2f} {k.edge.mean()*100:>+11.2f} "
              f"{(k.edge.mean()-d.edge.mean())*100:>+8.2f}")
print("\n  The percentile rule catches ~25% in BOTH regimes by construction -")
print("  that is the whole point. The absolute rule drifts to 72% post-shift.")

print("\n" + "=" * 76)
print("WALK-FORWARD: percentiles estimated from PRIOR days only")
print("=" * 76)


def walk(df, mode, pct=0.50, om=0.71, fm=0.5046):
    days = sorted(df.day.unique())
    kept, dead = [], []
    for i in range(1, len(days)):
        tr = df[df.day.isin(days[:i])]
        te = df[df.day == days[i]]
        if len(tr) < 60 or len(te) < 10:
            continue
        if mode == "abs":
            o, f = om, fm
        else:
            o = tr.others_px.quantile(pct)
            f = tr.frac_long.quantile(pct)
        m = (te.others_px <= o) & (te.frac_long <= f)
        dead.append(te[m])
        kept.append(te[~m])
    if not kept:
        return None
    K = pd.concat(kept)
    Dd = pd.concat(dead) if any(len(x) for x in dead) else pd.DataFrame()
    nd = K.day.nunique()
    return dict(kept=K, dead=Dd, per_day=K.edge.sum() * QTY / nd,
                share=len(Dd) / (len(K) + len(Dd)) if len(Dd) else 0.0,
                dead_edge=Dd.edge.mean() * 100 if len(Dd) else float("nan"),
                kept_edge=K.edge.mean() * 100, nd=nd)


for nm, df in (("PRE-SHIFT", A), ("POST-SHIFT", P)):
    print(f"\n  {nm}")
    print(f"{'rule':>26} {'share cut':>10} {'dead edge':>11} {'kept edge':>11} "
          f"{'$/day':>9}")
    r = walk(df, "abs")
    if r:
        print(f"{'ABSOLUTE (deployed)':>26} {r['share']:>10.3f} "
              f"{r['dead_edge']:>+11.2f} {r['kept_edge']:>+11.2f} "
              f"{r['per_day']:>+9.2f}")
    for pct in (0.35, 0.50, 0.65):
        r = walk(df, "pct", pct=pct)
        if r:
            print(f"{'PERCENTILE ' + f'{pct:.2f}':>26} {r['share']:>10.3f} "
                  f"{r['dead_edge']:>+11.2f} {r['kept_edge']:>+11.2f} "
                  f"{r['per_day']:>+9.2f}")

print("\n" + "=" * 76)
print("PAIRED TEST, pre-shift (10 days - the only place with enough days)")
print("=" * 76)
days = sorted(A.day.unique())


def daily(df, mode, pct=0.50):
    days_ = sorted(df.day.unique())
    out = []
    for i in range(1, len(days_)):
        tr = df[df.day.isin(days_[:i])]
        te = df[df.day == days_[i]]
        if len(tr) < 60 or len(te) < 10:
            continue
        if mode == "abs":
            o, f = 0.71, 0.5046
        else:
            o, f = tr.others_px.quantile(pct), tr.frac_long.quantile(pct)
        k = te[~((te.others_px <= o) & (te.frac_long <= f))]
        out.append(k.edge.sum() * QTY)
    return np.array(out)


a = daily(A, "abs")
for pct in (0.35, 0.50, 0.65):
    b = daily(A, "pct", pct)
    if len(b) != len(a):
        continue
    d = b - a
    bs = np.sort(np.array([RNG.choice(d, len(d), True).mean()
                           for _ in range(8000)]))
    print(f"  percentile {pct:.2f} minus absolute: ${d.mean():+7.2f}/day   "
          f"95% CI [{bs[200]:+7.2f}, {bs[7799]:+7.2f}]   "
          f"better {(b > a).sum()}/{len(a)}")
print("\n  A percentile rule that MATCHES the absolute one pre-shift while")
print("  repairing it post-shift is the result worth having - it means the")
print("  change costs nothing where the filter already worked.")
