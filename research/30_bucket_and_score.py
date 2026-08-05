"""The five questions, plus the capital-normalised ranking score.

Q1  which price bucket has the highest observed settlement win rate?
Q2  which has the highest outcome-minus-price (excess) relationship?
Q3  does z predict settlement beyond EXACT price, coin and time?
Q4  does the bucket relationship survive across days and coins?
Q5  is the 75-85c plateau distinguishable from 65-75c?

THE SCORE
      score = [ P(fill) * (P(win|fill) - p) - costs ] / worst-case capital

  Maker fees are zero, so costs vanish. Kalshi reserves qty*p on rest, so
  worst-case reserved capital per contract is exactly p. That reduces to

      score = F_Q * (win - p) / p

  which is expected edge per DOLLAR RESERVED, not per contract. This is the
  right objective whenever capital or drawdown headroom binds - and it can
  rank cheap buckets above expensive ones even at equal edge, because a 68c
  contract ties up 20% less capital than an 85c one for the same $1 payout.

DISCIPLINE
  window-equal-weighted; day-clustered bootstrap; contract weighting has faked
  three findings in this project.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260817)
D = pd.read_parquet(OUT / "grid.parquet")
days = sorted(D.day.unique())

q = D[(D.minute >= 8) & (D.minute <= 14) & (D.vol >= 2000)
      & (D.px >= 0.55) & (D.px < 0.95)]
E = (q.sort_values("minute", ascending=False)
       .groupby(["wkey", "coin"], as_index=False).first()).copy()
E["b"] = pd.cut(E.px, [.55, .65, .70, .75, .80, .85, .95],
                labels=["55-65", "65-70", "70-75", "75-80", "80-85", "85-95"])
# live fill rate, measured: overall F_Q 0.906 with a queue-depth gradient
FQ = 0.906
print(f"entries {len(E):,}  windows {E.wkey.nunique():,}  days {len(days)}  "
      f"vmin 2000, 8-14 min\n")

print("=== Q1/Q2: WIN RATE AND EXCESS BY BUCKET (window-equal-weighted) ===")
print(f"{'bucket':>8} {'n':>6} {'avg px':>8} {'win':>8} {'excess':>9} "
      f"{'edge/ct':>9} {'SCORE = FQ*(win-p)/p':>22} {'$/day @30':>10}")
rows = []
for b, g in E.groupby("b", observed=True):
    if len(g) < 25:
        continue
    p = g.px.mean()
    win = g.won.mean()
    exc = win - p
    score = FQ * exc / p
    rows.append(dict(b=str(b), n=len(g), px=p, win=win, exc=exc,
                     edge=g.edge.mean(), score=score,
                     day=g.edge.sum() * 30 / len(days)))
    print(f"{str(b):>8} {len(g):>6,} {p*100:>7.1f}c {win:>8.4f} "
          f"{exc*100:>+8.2f}c {g.edge.mean()*100:>+8.2f}c {score:>+21.4f} "
          f"{g.edge.sum()*30/len(days):>+10.2f}")
R = pd.DataFrame(rows)
print(f"\n  highest win rate : {R.loc[R.win.idxmax(),'b']} "
      f"({R.win.max():.4f})")
print(f"  highest excess   : {R.loc[R.exc.idxmax(),'b']} "
      f"({R.exc.max()*100:+.2f}c)")
print(f"  highest SCORE    : {R.loc[R.score.idxmax(),'b']} "
      f"({R.score.max():+.4f} per dollar reserved)  <-- capital-normalised")
print(f"  most $/day       : {R.loc[R.day.idxmax(),'b']} "
      f"(${R.day.max():+.2f}) - this is a VOLUME effect, not a quality one")


def dayboot(a, b, n=6000):
    """Day-clustered bootstrap of mean(b)-mean(a) on the excess measure."""
    ds = sorted(set(a.day) | set(b.day))
    ga = {d: g for d, g in a.groupby("day")}
    gb = {d: g for d, g in b.groupby("day")}
    out = []
    for _ in range(n):
        pick = RNG.choice(len(ds), len(ds), True)
        sa = pd.concat([ga[ds[k]] for k in pick if ds[k] in ga])
        sb = pd.concat([gb[ds[k]] for k in pick if ds[k] in gb])
        if len(sa) > 20 and len(sb) > 20:
            out.append(((sb.won.mean() - sb.px.mean())
                        - (sa.won.mean() - sa.px.mean())) * 100)
    return np.sort(np.array(out))


print("\n=== Q5: IS 75-85c DISTINGUISHABLE FROM 65-75c? ===")
lo = E[(E.px >= 0.65) & (E.px < 0.75)]
hi = E[(E.px >= 0.75) & (E.px < 0.85)]
bs = dayboot(lo, hi)
d = ((hi.won.mean() - hi.px.mean()) - (lo.won.mean() - lo.px.mean())) * 100
print(f"  65-75c: n {len(lo):,}  win {lo.won.mean():.4f}  "
      f"excess {(lo.won.mean()-lo.px.mean())*100:+.2f}c  "
      f"score {FQ*(lo.won.mean()-lo.px.mean())/lo.px.mean():+.4f}")
print(f"  75-85c: n {len(hi):,}  win {hi.won.mean():.4f}  "
      f"excess {(hi.won.mean()-hi.px.mean())*100:+.2f}c  "
      f"score {FQ*(hi.won.mean()-hi.px.mean())/hi.px.mean():+.4f}")
print(f"  DIFF in excess {d:+.2f}c   95% CI [{bs[int(.025*len(bs))]:+.2f}, "
      f"{bs[int(.975*len(bs))]:+.2f}]   "
      f"p {2*min((bs<=0).mean(),(bs>=0).mean()):.4f}")
print("  -> if the CI spans zero the 'plateau' is not established and the")
print("     buckets should be treated as one population.")

print("\n=== Q4: DOES THE BUCKET RELATIONSHIP SURVIVE ACROSS DAYS AND COINS? ===")
print("  excess (win - price), in cents, by bucket:\n")
piv = E.assign(exc=(E.won - E.px) * 100).pivot_table(
    index="coin", columns="b", values="exc", aggfunc="mean", observed=True)
cnt = E.pivot_table(index="coin", columns="b", values="won",
                    aggfunc="size", observed=True)
print(piv.round(2).to_string())
print("\n  cell counts:")
print(cnt.fillna(0).astype(int).to_string())
byday = E.assign(exc=(E.won - E.px) * 100).pivot_table(
    index="day", columns="b", values="exc", aggfunc="mean", observed=True)
print(f"\n  days where 75-85c beats 65-75c on excess: ", end="")
w = 0
for dd in byday.index:
    a = byday.loc[dd, ["65-70", "70-75"]].mean()
    b2 = byday.loc[dd, ["75-80", "80-85"]].mean()
    if pd.notna(a) and pd.notna(b2) and b2 > a:
        w += 1
print(f"{w}/{len(byday)}")

print("\n=== Q3: DOES z ADD BEYOND EXACT PRICE, COIN AND TIME? ===")
try:
    M = pd.read_parquet(OUT / "mechanism.parquet")
    M = M[(M.px >= 0.55) & (M.px < 0.95)].dropna(subset=["z"])
    M["pxc"] = (M.px * 100).round().astype(int)      # EXACT price in cents
    M["tb"] = pd.cut(M.secs, [0, 300, 480, 660, 900], labels=False)
    print(f"  {len(M):,} observations, {M.pxc.nunique()} distinct prices")
    # within each (coin, exact price, time bucket) cell, split on z
    M["zhi"] = M.groupby(["coin", "pxc", "tb"], observed=True).z.transform(
        lambda s: s > s.median() if len(s) >= 8 else np.nan)
    U = M.dropna(subset=["zhi"])
    a, b = U[U.zhi == 0], U[U.zhi == 1]
    print(f"  usable cells cover {len(U):,} observations "
          f"({U.groupby(['coin','pxc','tb'], observed=True).ngroups} cells)")
    print(f"  low  z: win {a.won.mean():.4f}   high z: win {b.won.mean():.4f}   "
          f"DIFF {(b.won.mean()-a.won.mean())*100:+.2f}pp")
    ds = sorted(U.day.unique())
    ga = {d: g for d, g in U.groupby("day")}
    bs = []
    for _ in range(4000):
        s = pd.concat([ga[ds[k]] for k in RNG.choice(len(ds), len(ds), True)])
        x, y = s[s.zhi == 0], s[s.zhi == 1]
        if len(x) > 30 and len(y) > 30:
            bs.append((y.won.mean() - x.won.mean()) * 100)
    bs = np.sort(np.array(bs))
    pv = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    print(f"  day-clustered 95% CI [{bs[int(.025*len(bs))]:+.2f}, "
          f"{bs[int(.975*len(bs))]:+.2f}] pp   p {pv:.4f}")
    print(f"\n  and BELOW 80c specifically (the user's stated case):")
    L = U[U.px < 0.80]
    x, y = L[L.zhi == 0], L[L.zhi == 1]
    print(f"    n {len(L):,}   low z win {x.won.mean():.4f}   "
          f"high z win {y.won.mean():.4f}   "
          f"DIFF {(y.won.mean()-x.won.mean())*100:+.2f}pp")
except Exception as e:
    print(f"  could not run: {e}")
