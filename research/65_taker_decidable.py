"""Is the maker-vs-taker question decidable with the data we have?

THE SITUATION
  Applying the measured fill bias to 13 days of history made taker look better
  by $13.95/day pre-shift with a tight-looking CI. But the sensitivity table
  showed the sign flipping between P(fill|win)=0.843 and 0.900:

      P(fill|win)   taker - maker
            0.800         +39.81
            0.843         +13.95      <- the point estimate
            0.900         -14.40
            0.950         -39.21

  That CI was computed holding the fill model FIXED and resampling days. It
  therefore measures day-to-day noise and completely ignores the uncertainty in
  the parameter the answer actually depends on.

  P(fill|win) is built from two live proportions:
      P(fill | price ran away) = 53/67 = 0.791   SE ~ 0.050
      P(fill | price came back) = 51/53 = 0.962  SE ~ 0.026
  and 0.791 with SE 0.050 has a 95% interval of roughly [0.69, 0.89].

  If the sign-flip point lies INSIDE that interval, the honest answer is not
  'taker wins' - it is 'this data cannot decide', and no amount of day
  bootstrapping will change that.

WHAT THIS SCRIPT DOES
  Propagates the real parameter uncertainty. Resamples the 120 live orders to
  get a distribution over the fill model, resamples days for the market noise,
  and reports P(taker better) integrating over BOTH. Then finds the exact
  break-even P(fill|win) and asks whether it sits inside the parameter's own
  confidence interval.

THIS IS THE TRIPLE-CHECK, and it is designed to be able to say no.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261033)
QTY = 20
SPREAD = 0.0121

# the live 2x2 that produced the fill model
N_REACH, N_REACH_FILL, N_REACH_WIN = 53, 51, 29
N_AWAY, N_AWAY_FILL, N_AWAY_WIN = 67, 53, 67


def taker_fee(c, p):
    return math.ceil(0.07 * c * p * (1.0 - p) * 1e4) / 1e4


D = pd.read_parquet(OUT / "grid.parquet")
q = D[(D.px >= 0.65) & (D.px < 0.80) & (D.minute >= 8) & (D.minute <= 14)
      & (D.vol >= 2000)]
E = (q.sort_values("minute", ascending=False)
       .groupby(["wkey", "coin"], as_index=False).first())
E = (E.sort_values(["wkey", "minute"], ascending=[True, False])
       .groupby("wkey", as_index=False).head(3)).copy()
days = sorted(E.day.unique())
dayix = pd.Categorical(E.day, categories=days).codes
won = E.won.to_numpy().astype(bool)
px = E.px.to_numpy()
pay = np.where(won, 1.0, 0.0)
nd = len(days)
print(f"pre-shift entries {len(E):,} over {nd} days\n")

ask = np.minimum(px + SPREAD, 0.99)
tk_day = np.zeros(nd)
np.add.at(tk_day, dayix,
          (pay - ask) * QTY - np.array([taker_fee(QTY, a) for a in ask]))
TK = tk_day.mean()


def maker_per_day(pw, pl):
    """Expected maker $/day under a given fill model - analytic, no sampling."""
    v = (pay - px) * QTY * np.where(won, pw, pl)
    acc = np.zeros(nd)
    np.add.at(acc, dayix, v)
    return acc


print("=" * 78)
print("STEP 1 - WHERE EXACTLY DOES THE SIGN FLIP?")
print("=" * 78)
lo, hi = 0.60, 1.00
for _ in range(60):
    mid = (lo + hi) / 2
    if maker_per_day(mid, 0.962).mean() < TK:
        lo = mid
    else:
        hi = mid
BREAK = (lo + hi) / 2
print(f"  taker beats maker only while P(fill|win) < {BREAK:.4f}")
print(f"  point estimate is 0.8426")

print("\n" + "=" * 78)
print("STEP 2 - HOW WELL DO WE KNOW P(fill|win)?")
print("=" * 78)
NB = 20000
pw_bs = np.empty(NB)
for i in range(NB):
    rf = RNG.binomial(N_REACH, N_REACH_FILL / N_REACH) / N_REACH
    af = RNG.binomial(N_AWAY, N_AWAY_FILL / N_AWAY) / N_AWAY
    rw = RNG.binomial(N_REACH, N_REACH_WIN / N_REACH)
    p_reach_win = rw / max(rw + N_AWAY_WIN, 1)
    pw_bs[i] = p_reach_win * rf + (1 - p_reach_win) * af
s = np.sort(pw_bs)
print(f"  P(fill|win) = {pw_bs.mean():.4f}   95% CI "
      f"[{s[int(0.025*NB)]:.4f}, {s[int(0.975*NB)]:.4f}]")
inside = (s[int(0.025 * NB)] < BREAK < s[int(0.975 * NB)])
print(f"  break-even {BREAK:.4f} lies "
      f"{'INSIDE' if inside else 'OUTSIDE'} that interval")

print("\n" + "=" * 78)
print("STEP 3 - P(taker better), INTEGRATING OVER BOTH SOURCES OF NOISE")
print("=" * 78)
NS = 8000
better = 0
diffs = np.empty(NS)
for i in range(NS):
    pw = pw_bs[RNG.integers(0, NB)]
    rf = RNG.binomial(N_REACH, N_REACH_FILL / N_REACH) / N_REACH
    mk = maker_per_day(pw, rf)
    pick = RNG.integers(0, nd, nd)
    d = (tk_day - mk)[pick].mean()
    diffs[i] = d
    better += d > 0
sd = np.sort(diffs)
print(f"  taker minus maker: ${diffs.mean():+.2f}/day   "
      f"95% CI [{sd[200]:+.2f}, {sd[7799]:+.2f}]")
print(f"  P(taker better) = {better/NS:.4f}")

print("\n" + "=" * 78)
print("STEP 4 - AND THE SPREAD IS ALSO AN ESTIMATE")
print("=" * 78)
print(f"{'spread':>9} {'break-even P(fill|win)':>24} {'inside CI?':>12}")
for sp in (0.0100, 0.0121, 0.0150, 0.0200):
    a2 = np.minimum(px + sp, 0.99)
    t2 = np.zeros(nd)
    np.add.at(t2, dayix,
              (pay - a2) * QTY - np.array([taker_fee(QTY, a) for a in a2]))
    T2 = t2.mean()
    lo, hi = 0.50, 1.20
    for _ in range(60):
        mid = (lo + hi) / 2
        if maker_per_day(mid, 0.962).mean() < T2:
            lo = mid
        else:
            hi = mid
    b = (lo + hi) / 2
    ins = s[int(0.025 * NB)] < b < s[int(0.975 * NB)]
    print(f"{sp*100:>8.2f}c {b:>24.4f} {'yes' if ins else 'no':>12}")

print("\n" + "=" * 78)
print("VERDICT")
print("=" * 78)
if inside or better / NS < 0.95:
    print("  NOT DECIDABLE with current data.")
    print()
    print("  The comparison turns on a parameter estimated from 67 live orders,")
    print("  and the value at which the answer reverses sits inside that")
    print("  parameter's own confidence interval. The earlier tight-looking CI")
    print("  was an artifact of holding the fill model fixed while bootstrapping")
    print("  days - it measured the wrong uncertainty.")
    print()
    print("  Taker is NOT recommended, and equally is not ruled out. What would")
    print("  settle it is more live orders: the fill model sharpens as ~1/sqrt(n),")
    print("  so roughly 4x the current book would halve the interval.")
else:
    print("  Taker survives the propagated uncertainty. Still a STRATEGY CHANGE")
    print("  and therefore not deployable without explicit approval.")
