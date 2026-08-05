"""Verify the 80-85c band trim to the same bar the dead-entry filter had to clear.

THE CANDIDATE
  On the capital-normalised score - score = F_Q * (win - p) / p, i.e. expected
  edge per DOLLAR RESERVED, which is the right objective because Kalshi
  reserves qty*p - the 80-85c slice scores 0.0973 against 0.1499 for 65-80c,
  p = 0.0200. It looks acceptable on raw edge (+8.82c) but ties up 82c to earn
  it, which raw edge cannot see.

THE BAR IT MUST CLEAR (the same one 35 applied)
  1. rolling walk-forward, never using the test day
  2. permutation test - outcomes shuffled to see how often this appears by luck
  3. not concentrated in one coin
  4. the DOLLAR consequence, not just the ratio, since trimming removes entries

WHY THIS COULD BE PROFIT AND NOT JUST RISK
  Unlike the dead-entry filter, the trimmed entries are NOT worthless - they
  earn +8.82c. Trimming only pays if the freed capital and gross-cap headroom
  get reused by better entries. That is a real mechanism (the cap binds) but it
  is not automatic, so the dollar test below is the one that decides.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260825)
D = pd.read_parquet(OUT / "grid.parquet")
QTY, MAX_GROSS, FQ = 30, 110.0, 0.906
days = sorted(D.day.unique())


def pick(band_hi, cap=3):
    q = D[(D.px >= 0.65) & (D.px < band_hi) & (D.minute >= 8) & (D.minute <= 14)
          & (D.vol >= 2000)]
    e = (q.sort_values("minute", ascending=False)
           .groupby(["wkey", "coin"], as_index=False).first())
    return (e.sort_values(["wkey", "minute"], ascending=[True, False])
              .groupby("wkey", as_index=False).head(cap)).copy()


def chrono(e):
    """Apply the gross cap chronologically; returns realised entries."""
    keep = []
    for _, g in e.groupby("wkey"):
        gross = 0.0
        for r in g.itertuples():
            c = QTY * r.px
            if gross + c > MAX_GROSS:
                continue
            gross += c
            keep.append((r.day, r.px, r.won, r.edge))
    return pd.DataFrame(keep, columns=["day", "px", "won", "edge"])


A, B = chrono(pick(0.85)), chrono(pick(0.80))
print(f"65-85c: {len(A):,} entries   65-80c: {len(B):,} entries "
      f"({(1-len(B)/len(A))*100:.1f}% removed)\n")

print("=== THE DOLLAR TEST: does trimming actually make more money? ===")
print(f"{'band':>8} {'entries/day':>12} {'edge/ct':>9} {'contracts':>10} "
      f"{'$/day @30':>10} {'score':>8}")
for lbl, K in (("65-85c", A), ("65-80c", B)):
    sc = FQ * (K.won.mean() - K.px.mean()) / K.px.mean()
    print(f"{lbl:>8} {len(K)/len(days):>12.1f} {K.edge.mean()*100:>+9.2f} "
          f"{len(K)*QTY:>10,} {K.edge.sum()*QTY/len(days):>+10.2f} {sc:>8.4f}")
d_day = (B.edge.sum() - A.edge.sum()) * QTY / len(days)
print(f"\n  dollar difference: {d_day:+.2f}/day   "
      f"{'TRIM WINS' if d_day > 0 else 'TRIM COSTS MONEY'}")

print("\n=== ROLLING WALK-FORWARD (the trim rule is fixed, so this is a")
print("    per-day replay rather than a fitted threshold) ===")
print(f"{'day':>12} {'65-85 $':>9} {'65-80 $':>9} {'diff':>8}")
w = 0
tot85 = tot80 = 0.0
for d in days:
    a = A[A.day == d].edge.sum() * QTY
    b = B[B.day == d].edge.sum() * QTY
    tot85 += a
    tot80 += b
    w += (b > a)
    print(f"{d:>12} {a:>+9.2f} {b:>+9.2f} {b-a:>+8.2f}")
print(f"\n  trim better on {w}/{len(days)} days   "
      f"mean ${tot80/len(days):+.2f} vs ${tot85/len(days):+.2f}")

print("\n=== IS THE 80-85c SLICE ACTUALLY WORSE, day-clustered? ===")
S = pick(0.85)
S["top"] = S.px >= 0.80
a, b = S[~S.top], S[S.top]
g = {d: x for d, x in S.groupby("day")}
ds = sorted(S.day.unique())
bs_e, bs_s = [], []
for _ in range(6000):
    s = pd.concat([g[ds[k]] for k in RNG.choice(len(ds), len(ds), True)])
    x, y = s[~s.top], s[s.top]
    if len(x) > 30 and len(y) > 15:
        bs_e.append((y.edge.mean() - x.edge.mean()) * 100)
        bs_s.append(FQ * (y.won.mean() - y.px.mean()) / y.px.mean()
                    - FQ * (x.won.mean() - x.px.mean()) / x.px.mean())
bs_e, bs_s = np.sort(np.array(bs_e)), np.sort(np.array(bs_s))
print(f"  80-85c n {len(b)}  edge {b.edge.mean()*100:+.2f}c  "
      f"win {b.won.mean():.4f}  score "
      f"{FQ*(b.won.mean()-b.px.mean())/b.px.mean():.4f}")
print(f"  65-80c n {len(a)}  edge {a.edge.mean()*100:+.2f}c  "
      f"win {a.won.mean():.4f}  score "
      f"{FQ*(a.won.mean()-a.px.mean())/a.px.mean():.4f}")
print(f"  EDGE  diff {(b.edge.mean()-a.edge.mean())*100:+.2f}c  "
      f"CI [{bs_e[150]:+.2f},{bs_e[5849]:+.2f}]  "
      f"p {2*min((bs_e<=0).mean(),(bs_e>=0).mean()):.4f}")
print(f"  SCORE diff {bs_s.mean():+.4f}  "
      f"CI [{bs_s[150]:+.4f},{bs_s[5849]:+.4f}]  "
      f"p {2*min((bs_s<=0).mean(),(bs_s>=0).mean()):.4f}")

print("\n=== PERMUTATION: how often does a slice look this bad by luck? ===")
null = []
for _ in range(2000):
    P = S.copy()
    P["edge"] = P.groupby("coin").edge.transform(
        lambda s: s.sample(frac=1, random_state=RNG.integers(1 << 30)).values)
    x, y = P[~P.top], P[P.top]
    null.append((y.edge.mean() - x.edge.mean()) * 100)
null = np.array(null)
real = (b.edge.mean() - a.edge.mean()) * 100
print(f"  real {real:+.2f}c   null mean {null.mean():+.2f}c sd {null.std():.2f}")
print(f"  permutation p = {(null <= real).mean():.4f}")

print("\n=== PER COIN ===")
print(f"{'coin':>6} {'n 80-85':>8} {'edge 80-85':>11} {'edge 65-80':>11} {'diff':>8}")
for c, gg in S.groupby("coin"):
    x, y = gg[~gg.top], gg[gg.top]
    if len(y) < 12 or len(x) < 25:
        continue
    print(f"{c:>6} {len(y):>8} {y.edge.mean()*100:>+11.2f} "
          f"{x.edge.mean()*100:>+11.2f} {(y.edge.mean()-x.edge.mean())*100:>+8.2f}")
