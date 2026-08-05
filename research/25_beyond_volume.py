"""Does ANY flow feature add edge once the volume filter is already applied?

WHY THE PREVIOUS SCREEN OVERSTATED ITS CASE
  Seven features came up significant and all seven survived walk-forward. That
  is not seven signals. frac_long and ls_recent produced IDENTICAL bucket edges
  (+3.61 / +8.53), and px_vol, vol, n_ls and flips were within a tenth of a
  cent of them. They are all measuring the same latent thing - how active the
  window is - which is exactly what the deployed volume floor already selects.

  So the honest question is not "does frac_long predict?" It is: conditional on
  volume >= 2000, which the runner already enforces, does anything add MORE?

TESTS
  1. correlation matrix of the candidate features - are they one thing?
  2. each feature tested INSIDE the live filter (vol >= 2000, band 65-85c)
  3. a horse race: feature quartile x volume quartile, to see whether the
     feature has any effect at fixed volume
  4. walk-forward on anything that still stands
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260812)
D = pd.read_parquet(OUT / "adverse.parquet")
D = D[(D.px >= 0.55) & (D.px < 0.90)].copy()
D["pb"] = (D.px * 20).astype(int)
days = sorted(D.day.unique())

FE = ["frac_long", "ls_recent", "n_ls", "flips", "px_vol", "vol",
      "secs_since_flip"]
print("=== ARE THEY ONE THING? (Spearman correlation) ===")
C = D[FE].corr(method="spearman")
print(f"{'':>16}" + "".join(f"{c[:9]:>11}" for c in FE))
for a in FE:
    print(f"{a:>16}" + "".join(f"{C.loc[a,b]:>+11.3f}" for b in FE))
print("\n  values near +/-1 mean the 'separate' features are the same variable.")

LIVE = D[(D.vol >= 2000) & (D.px >= 0.65) & (D.px < 0.85)].copy()
print(f"\n=== INSIDE THE LIVE FILTER: {len(LIVE):,} entries, "
      f"{LIVE.wkey.nunique():,} windows, win {LIVE.won.mean():.4f}, "
      f"edge {LIVE.edge.mean()*100:+.2f}c ===")


def dayboot(df, col, n=4000):
    ds = sorted(df.day.unique())
    g = {d: gg for d, gg in df.groupby("day")}
    out = []
    for _ in range(n):
        s = pd.concat([g[ds[k]] for k in RNG.choice(len(ds), len(ds), True)])
        a, b = s[~s[col]], s[s[col]]
        if len(a) > 20 and len(b) > 20:
            out.append((b.edge.mean() - a.edge.mean()) * 100)
    return np.sort(np.array(out)) if out else np.array([0.0])


print(f"\n{'feature':>16} {'n':>6} {'low edge':>9} {'high edge':>10} {'DIFF':>8} "
      f"{'95% CI':>20} {'p':>7}")
survivors = []
for f in FE:
    d = LIVE.dropna(subset=[f]).copy()
    if len(d) < 120:
        continue
    d["hi"] = d.groupby(["coin", "pb"])[f].transform(lambda s: s > s.median())
    a, b = d[~d.hi], d[d.hi]
    if len(a) < 40 or len(b) < 40:
        continue
    diff = (b.edge.mean() - a.edge.mean()) * 100
    bs = dayboot(d, "hi")
    lo, hi = bs[int(.025 * len(bs))], bs[int(.975 * len(bs))]
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    flag = "  <-- survives" if p < 0.05 else ""
    print(f"{f:>16} {len(d):>6,} {a.edge.mean()*100:>+9.2f} "
          f"{b.edge.mean()*100:>+10.2f} {diff:>+8.2f} "
          f"[{lo:>+8.2f},{hi:>+8.2f}] {p:>7.4f}{flag}")
    if p < 0.05:
        survivors.append(f)

print("\n=== HORSE RACE: frac_long vs volume, both quartiled ===")
print("   read DOWN a column: does longshot share help at FIXED volume?")
print("   read ACROSS a row:  does volume help at FIXED longshot share?\n")
d = D.dropna(subset=["frac_long", "vol"]).copy()
d["vq"] = d.groupby(["coin", "pb"]).vol.transform(
    lambda s: pd.qcut(s.rank(method="first"), 3, labels=False, duplicates="drop"))
d["lq"] = d.groupby(["coin", "pb"]).frac_long.transform(
    lambda s: pd.qcut(s.rank(method="first"), 3, labels=False, duplicates="drop"))
print(f"{'':>18}" + "".join(f"{'vol q' + str(i):>13}" for i in range(3)))
for lq in range(3):
    row = ""
    for vq in range(3):
        g = d[(d.lq == lq) & (d.vq == vq)]
        row += f"{g.edge.mean()*100:>+13.2f}" if len(g) > 25 else f"{'-':>13}"
    print(f"  frac_long q{lq}     {row}")
print()
for lq in range(3):
    row = ""
    for vq in range(3):
        g = d[(d.lq == lq) & (d.vq == vq)]
        row += f"{len(g):>13,}" if len(g) > 25 else f"{'-':>13}"
    print(f"  n         q{lq}     {row}")

if survivors:
    print(f"\n=== WALK-FORWARD, inside the live filter ===")
    CUT = days[len(days) // 2]
    for f in survivors:
        d = LIVE.dropna(subset=[f]).copy()
        thr = d[d.day < CUT].groupby(["coin", "pb"])[f].median()
        te = d[d.day >= CUT].copy()
        te["thr"] = [thr.get(k, np.nan) for k in zip(te.coin, te.pb)]
        te = te.dropna(subset=["thr"])
        te["hi"] = te[f] > te.thr
        a, b = te[~te.hi], te[te.hi]
        if len(a) < 20 or len(b) < 20:
            print(f"  {f:>16}  too few out-of-sample ({len(a)}/{len(b)})")
            continue
        print(f"  {f:>16}  OUT-OF-SAMPLE diff "
              f"{(b.edge.mean()-a.edge.mean())*100:>+6.2f}c   "
              f"good half {max(a.edge.mean(),b.edge.mean())*100:+.2f}c "
              f"vs {te.edge.mean()*100:+.2f}c for all "
              f"({len(b)} hi / {len(a)} lo)")
else:
    print("\n=== NOTHING ADDS TO THE VOLUME FILTER ===")
    print("  Every feature that looked predictive on the full sample stops")
    print("  predicting once volume >= 2000 is imposed. The volume floor was")
    print("  already capturing the whole effect; these are restatements of it,")
    print("  not additional signals.")
