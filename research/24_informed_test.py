"""Are the longshot buyers informed - and can any pre-entry feature predict a flip?

THE CORE TEST
  Price is the market's own forecast. So the question is not "does the longshot
  sometimes win" - it is whether, HOLDING PRICE FIXED, heavier longshot buying
  predicts the favourite flipping MORE often than that price implies.

    informed flow  -> heavy longshot buying predicts flips -> our edge FALLS
    noise flow     -> no relationship                      -> our edge is safe
                      (and the earlier volume result says it may even RISE)

  Everything is measured inside price buckets, because price mechanically
  drives the win rate and an unconditional comparison would just rediscover it.

MULTIPLE COMPARISONS
  Eight features are screened. At p<0.05 one false positive is EXPECTED by
  chance alone, so nothing here is treated as real on its p-value; anything
  that survives the screen is re-fitted on the first half of days and re-tested
  on the second, which is the only claim that counts.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260811)
D = pd.read_parquet(OUT / "adverse.parquet")
D = D[(D.px >= 0.55) & (D.px < 0.90)].copy()
D["pb"] = (D.px * 20).astype(int)          # 5c price buckets
days = sorted(D.day.unique())
print(f"entries {len(D):,}  windows {D.wkey.nunique():,}  days {len(days)}  "
      f"coins {sorted(D.coin.unique())}")
print(f"base win {D.won.mean():.4f}   edge {D.edge.mean()*100:+.2f}c\n")

FEATS = ["frac_long", "ls_recent", "ls_accel", "max_ls", "max_ls_share",
         "n_ls", "mean_ls", "flips", "secs_since_flip", "px_vol", "px_drift",
         "vol"]


def dayboot(df, col, n=4000):
    """Day-clustered bootstrap of the high-minus-low edge difference."""
    ds = sorted(df.day.unique())
    g = {d: gg for d, gg in df.groupby("day")}
    out = []
    for _ in range(n):
        s = pd.concat([g[ds[k]] for k in RNG.choice(len(ds), len(ds), True)])
        a, b = s[~s[col]], s[s[col]]
        if len(a) > 30 and len(b) > 30:
            out.append((b.edge.mean() - a.edge.mean()) * 100)
    return np.sort(np.array(out))


print("=== SCREEN: does the feature predict OUR outcome, holding price fixed? ===")
print("   split at the within-(coin,price-bucket) median\n")
print(f"{'feature':>16} {'low edge':>9} {'high edge':>10} {'DIFF':>8} "
      f"{'95% CI':>20} {'p':>7} {'lo win':>7} {'hi win':>7}")
hits = []
for f in FEATS:
    d = D.dropna(subset=[f]).copy()
    if len(d) < 200:
        continue
    d["hi"] = d.groupby(["coin", "pb"])[f].transform(lambda s: s > s.median())
    a, b = d[~d.hi], d[d.hi]
    if len(a) < 50 or len(b) < 50:
        continue
    diff = (b.edge.mean() - a.edge.mean()) * 100
    bs = dayboot(d, "hi")
    lo, hi = bs[int(.025 * len(bs))], bs[int(.975 * len(bs))]
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    print(f"{f:>16} {a.edge.mean()*100:>+9.2f} {b.edge.mean()*100:>+10.2f} "
          f"{diff:>+8.2f} [{lo:>+8.2f},{hi:>+8.2f}] {p:>7.4f} "
          f"{a.won.mean():>7.4f} {b.won.mean():>7.4f}")
    if p < 0.05:
        hits.append((f, diff, p))

print("\n=== THE INFORMED-TRADER QUESTION, STATED DIRECTLY ===")
print("  If longshot buyers knew something, heavy longshot buying would predict")
print("  the favourite FLIPPING. Win rate by frac_long quartile, inside price")
print("  buckets so price is held fixed:\n")
D["fq"] = D.groupby(["coin", "pb"]).frac_long.transform(
    lambda s: pd.qcut(s.rank(method="first"), 4, labels=False, duplicates="drop"))
print(f"{'quartile':>9} {'n':>6} {'mean frac_long':>15} {'avg px':>8} "
      f"{'win':>8} {'implied':>9} {'excess':>8} {'edge':>9}")
for q, g in D.dropna(subset=["fq"]).groupby("fq"):
    implied = g.px.mean()
    print(f"{int(q):>9} {len(g):>6,} {g.frac_long.mean():>15.4f} "
          f"{g.px.mean()*100:>7.1f}c {g.won.mean():>8.4f} {implied:>9.4f} "
          f"{(g.won.mean()-implied)*100:>+7.2f}c {g.edge.mean()*100:>+9.2f}")
print("\n  'excess' = actual win rate minus the rate the entry price implies.")
print("  Informed longshot flow would make excess FALL as frac_long rises.")

if hits:
    print("\n=== WALK-FORWARD ON WHAT SURVIVED THE SCREEN ===")
    print("   in-sample significance is not evidence; this is the real test.\n")
    CUT = days[len(days) // 2]
    for f, diff, p in hits:
        d = D.dropna(subset=[f]).copy()
        fit = d[d.day < CUT]
        thr = fit.groupby(["coin", "pb"])[f].median()          # rule from PAST only
        te = d[d.day >= CUT].copy()
        key = list(zip(te.coin, te.pb))
        te["thr"] = [thr.get(k, np.nan) for k in key]
        te = te.dropna(subset=["thr"])
        te["hi"] = te[f] > te.thr
        a, b = te[~te.hi], te[te.hi]
        if len(a) < 30 or len(b) < 30:
            print(f"  {f:>16}  too few out-of-sample")
            continue
        d2 = (b.edge.mean() - a.edge.mean()) * 100
        keep = b if diff > 0 else a
        print(f"  {f:>16}  in-sample {diff:>+6.2f}c  ->  OUT-OF-SAMPLE "
              f"{d2:>+6.2f}c   (trading only the good half: "
              f"{keep.edge.mean()*100:>+6.2f}c on {len(keep)} entries vs "
              f"{te.edge.mean()*100:+.2f}c for all)")
else:
    print("\n=== NOTHING SURVIVED THE SCREEN ===")
    print("  No pre-entry flow feature predicts our outcome once price is held")
    print("  fixed. That is evidence the flow is UNINFORMED - which is the")
    print("  good outcome: it means the premium we collect is not adverse")
    print("  selection waiting to happen, and there is no sniping rule to find")
    print("  in these features.")
