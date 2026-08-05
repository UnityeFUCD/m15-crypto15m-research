"""DO REVERSALS CLUSTER? The variable I never tested.

WHAT I TESTED BEFORE AND WHY IT MISSED THIS
  I tested whether the settled DIRECTION persists (does YES follow YES) - null
  at every lag. And I tested whether the WIPEOUT EVENT clusters in time - also
  null.

  Neither of those is the variable that actually decides our P&L.

  We lose when the FAVOURITE FLIPS. That is not a direction property, it is a
  REVERSAL property - a market can trend hard in either direction and our
  favourite holds; a market can chop sideways with no net direction at all and
  our favourite flips constantly. Direction and reversal-frequency are
  different variables and one can be i.i.d. while the other is highly
  persistent. Volatility clustering is one of the most robust facts in all of
  finance, and reversal frequency is a volatility property.

  So: does the recent FLIP RATE predict the next window's flip rate?

WHY THIS ONE COULD BE TRADEABLE WHERE THE OTHERS WERE NOT
  If flip rate is persistent, then after a run of losses we are in a choppy
  regime and the edge is genuinely lower for a while - which is exactly the
  "edge decay" idea. And critically, a regime signal does not have to gate
  entries: it can drive SIZE, which is now a live channel with no opportunity
  cost.

TESTED HERE
  1. autocorrelation of our own win/loss, per coin, chronological
  2. does a rolling recent loss rate predict the NEXT window's outcome
  3. does it predict at the market level (all coins pooled)
  4. and the only thing that matters - does it convert to dollars via sizing
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261003)
D = pd.read_parquet(OUT / "grid.parquet")

q = D[(D.px >= 0.65) & (D.px < 0.85) & (D.minute >= 8) & (D.minute <= 14)
      & (D.vol >= 2000)]
E = (q.sort_values("minute", ascending=False)
       .groupby(["wkey", "coin"], as_index=False).first()).copy()
E["close"] = pd.to_datetime(E.wkey, format="%y%b%d%H%M", utc=True,
                            errors="coerce")
E = E.dropna(subset=["close"]).sort_values("close").reset_index(drop=True)
E["ec"] = np.where(E.won == 1, (1 - E.px), -E.px)
E["loss"] = 1 - E.won
print(f"entries {len(E):,}   windows {E.wkey.nunique():,}   "
      f"days {E.day.nunique()}   base loss rate {E.loss.mean():.4f}\n")

print("=" * 72)
print("1. DOES OUR WIN/LOSS AUTOCORRELATE, PER COIN?")
print("=" * 72)
print(f"{'coin':>6} {'n':>6} {'loss rate':>10} " +
      "".join(f"{'lag'+str(l):>9}" for l in (1, 2, 3, 4)))
for coin, g in E.groupby("coin"):
    g = g.sort_values("close")
    v = g.loss.to_numpy().astype(float)
    if len(v) < 80 or v.std() == 0:
        continue
    acs = []
    for l in (1, 2, 3, 4):
        acs.append(np.corrcoef(v[:-l], v[l:])[0, 1] if len(v) > l + 10 else np.nan)
    print(f"{coin:>6} {len(v):>6} {v.mean():>10.4f} " +
          "".join(f"{a:>+9.4f}" for a in acs))

print("\n" + "=" * 72)
print("2. DOES A ROLLING LOSS RATE PREDICT THE NEXT WINDOW? (market level)")
print("=" * 72)
W = E.groupby("wkey").agg(n=("loss", "size"), loss=("loss", "mean"),
                          ec=("ec", "mean"), close=("close", "first"),
                          day=("day", "first")).reset_index()
W = W.sort_values("close").reset_index(drop=True)
for k in (2, 3, 5, 8, 12):
    W[f"roll{k}"] = W.loss.shift(1).rolling(k, min_periods=k).mean()
print(f"{'lookback':>10} {'n':>6} {'low-half next loss':>20} "
      f"{'high-half next loss':>21} {'diff':>8} {'p (day-clustered)':>19}")
hits = []
for k in (2, 3, 5, 8, 12):
    d = W.dropna(subset=[f"roll{k}"])
    if len(d) < 80:
        continue
    med = d[f"roll{k}"].median()
    lo = d[d[f"roll{k}"] <= med]
    hi = d[d[f"roll{k}"] > med]
    real = hi.loss.mean() - lo.loss.mean()
    null = []
    for _ in range(4000):
        S = d.copy()
        S["loss"] = S.groupby("day").loss.transform(
            lambda s: RNG.permutation(s.values))
        null.append(S[S[f"roll{k}"] > med].loss.mean()
                    - S[S[f"roll{k}"] <= med].loss.mean())
    null = np.array(null)
    p = (np.abs(null) >= abs(real)).mean()
    print(f"{k:>10} {len(d):>6} {lo.loss.mean():>20.4f} {hi.loss.mean():>21.4f} "
          f"{real:>+8.4f} {p:>19.4f}")
    if p < 0.05:
        hits.append(k)

print("\n=== THE SAME THING IN EDGE TERMS (what we actually earn) ===")
print(f"{'lookback':>10} {'edge after LOW loss':>21} {'edge after HIGH loss':>22} "
      f"{'diff':>9}")
for k in (2, 3, 5, 8, 12):
    d = W.dropna(subset=[f"roll{k}"])
    if len(d) < 80:
        continue
    med = d[f"roll{k}"].median()
    lo = d[d[f"roll{k}"] <= med]
    hi = d[d[f"roll{k}"] > med]
    print(f"{k:>10} {lo.ec.mean()*100:>+20.2f}c {hi.ec.mean()*100:>+21.2f}c "
          f"{(hi.ec.mean()-lo.ec.mean())*100:>+8.2f}c")

print("\n" + "=" * 72)
print("3. QUINTILES OF THE ROLLING LOSS RATE - is it monotone?")
print("=" * 72)
best_k = hits[0] if hits else 5
d = W.dropna(subset=[f"roll{best_k}"]).copy()
d["q"] = pd.qcut(d[f"roll{best_k}"].rank(method="first"), 5, labels=False,
                 duplicates="drop")
print(f"  using lookback {best_k}\n")
print(f"{'quintile':>9} {'n':>6} {'prior loss rate':>17} {'NEXT loss rate':>16} "
      f"{'NEXT edge':>11}")
for qq, g in d.groupby("q"):
    print(f"{int(qq):>9} {len(g):>6} {g[f'roll{best_k}'].mean():>17.4f} "
          f"{g.loss.mean():>16.4f} {g.ec.mean()*100:>+10.2f}c")

print("\n" + "=" * 72)
print("4. THE DOLLAR TEST - size by the regime signal, walk-forward")
print("=" * 72)
E2 = E.merge(W[["wkey"] + [f"roll{k}" for k in (2, 3, 5, 8, 12)]],
             on="wkey", how="left")
days = sorted(E2.day.unique())
BASE = 15
for k in (3, 5, 8):
    col = f"roll{k}"
    tot_f = tot_w = 0.0
    rows = []
    for i in range(3, len(days)):
        tr = E2[E2.day.isin(days[:i])].dropna(subset=[col])
        te = E2[E2.day == days[i]].dropna(subset=[col]).copy()
        if len(te) < 10 or len(tr) < 100:
            continue
        med = tr[col].median()
        # size DOWN in a high-recent-loss regime, UP in a calm one
        qty = np.where(te[col].values > med, BASE * 0.7, BASE * 1.3)
        qty = np.clip(np.round(qty), 8, 22)
        qty = np.round(qty * (BASE * len(te)) / qty.sum()).astype(int)
        f = (te.ec.values * BASE).sum()
        v = (te.ec.values * qty).sum()
        tot_f += f
        tot_w += v
        rows.append((f, v))
    if rows:
        nd = len(rows)
        print(f"  lookback {k:>2}: flat ${tot_f/nd:+.2f}/day   "
              f"regime-sized ${tot_w/nd:+.2f}/day   "
              f"diff ${(tot_w-tot_f)/nd:+.2f}   "
              f"better {sum(1 for a, b in rows if b > a)}/{nd}")
