"""Is our strategy secretly short mean-reversion? 41,334 windows, true index.

THE IDEA
  Volatility turned out to be priced correctly - edge is flat across regimes
  (calm +13.68c, wild +13.03c in the 65-70c band). So the loss mechanism is not
  "the market moves too much".

  But there is a directional assumption buried in the strategy that has never
  been tested. We enter 8-14 minutes before close and buy the FAVOURITE. A
  favourite exists because the index has already moved away from A0 in the
  early part of the window. Buying it is therefore a bet that the move
  PERSISTS for the remaining minutes.

  If 15-minute index returns mean-revert, that bet is systematically taxed. And
  if the strength of mean reversion varies through the day, it would explain
  the intraday edge pattern with a mechanism instead of a calendar - which is
  exactly what has been missing.

WHAT MAKES THIS TESTABLE NOW
  A1 of one window is the same instant as A0 of the next, so the market
  metadata gives a true BRTI series at 15-minute resolution: 41,334 windows
  across 6 coins over 73 days. That is 4x the history of any dataset used so
  far and it is the actual index, not a proxy.

THE TESTS
  1. do consecutive 15-minute returns mean-revert? (sign persistence and
     autocorrelation, with enormous power)
  2. does reversion vary by hour of day - the mechanism candidate
  3. does it vary by volatility regime
  4. and the one that decides anything: is the effect big enough to matter at
     the ~30bp scale that separates a winning favourite from a losing one

WHAT WOULD MAKE THIS A DEAD END
  Window-to-window reversion is not the same as WITHIN-window reversion, and
  it is the latter that decides our contracts. Consecutive windows are the
  closest observable proxy. If reversion is absent between windows it is
  unlikely to be present within them, but the converse is not guaranteed.
"""
from pathlib import Path

import numpy as np
import pandas as pd

LIVE = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261054)

U = pd.read_parquet(LIVE / "underlying.parquet")
U["close"] = pd.to_datetime(U.close, format="mixed", utc=True)
U = U.sort_values(["coin", "close"]).reset_index(drop=True)
U["prev"] = U.groupby("coin").ret.shift(1)
U["prev2"] = U.groupby("coin").ret.shift(2)
U["hour"] = U.close.dt.hour
U["gap_ok"] = (U.groupby("coin").close.diff().dt.total_seconds() == 900)
D = U.dropna(subset=["prev"])
D = D[D.gap_ok].copy()          # only genuinely consecutive windows
print(f"consecutive window pairs: {len(D):,}   coins {D.coin.nunique()}   "
      f"days {D.close.dt.date.nunique()}\n")

print("=" * 78)
print("TEST 1 - DO 15-MINUTE RETURNS MEAN-REVERT?")
print("=" * 78)
print(f"{'coin':>7} {'pairs':>8} {'autocorr':>10} {'P(same sign)':>14} "
      f"{'verdict':>14}")
for c, g in D.groupby("coin"):
    r = g.ret.corr(g.prev)
    same = ((np.sign(g.ret) == np.sign(g.prev)) & (g.prev != 0)).mean()
    v = "reverts" if r < -0.02 else ("trends" if r > 0.02 else "~random")
    print(f"{c:>7} {len(g):>8,} {r:>+10.4f} {same:>14.4f} {v:>14}")
r_all = D.ret.corr(D.prev)
same_all = ((np.sign(D.ret) == np.sign(D.prev)) & (D.prev != 0)).mean()
n = len(D)
se = 1 / np.sqrt(n)
print(f"\n  POOLED: autocorr {r_all:+.4f}   (se ~{se:.4f}, so |r|>{2*se:.4f} "
      f"is significant)")
print(f"  P(next window same sign as previous) = {same_all:.4f}   "
      f"(0.50 = no information)")
z = (same_all - 0.5) / np.sqrt(0.25 / n)
print(f"  z = {z:+.2f}")

print("\n" + "=" * 78)
print("TEST 2 - DOES REVERSION VARY BY HOUR? (the mechanism candidate)")
print("=" * 78)
print(f"{'UTC':>5} {'ET':>5} {'pairs':>7} {'autocorr':>10} {'P(same sign)':>14}  ")
rows = []
for h in range(24):
    g = D[D.hour == h]
    if len(g) < 200:
        continue
    r = g.ret.corr(g.prev)
    same = ((np.sign(g.ret) == np.sign(g.prev)) & (g.prev != 0)).mean()
    rows.append((h, r, same, len(g)))
    bar = "#" * int(abs(r) * 400)
    sign = "-" if r < 0 else "+"
    print(f"{h:>3}:00 {(h-4)%24:>3}:00 {len(g):>7,} {r:>+10.4f} "
          f"{same:>14.4f}  {sign}{bar}")
if rows:
    R = pd.DataFrame(rows, columns=["h", "r", "same", "n"])
    print(f"\n  most reverting hour  {int(R.loc[R.r.idxmin(),'h']):02d}:00 UTC  "
          f"r={R.r.min():+.4f}")
    print(f"  most trending hour   {int(R.loc[R.r.idxmax(),'h']):02d}:00 UTC  "
          f"r={R.r.max():+.4f}")
    print(f"  spread {R.r.max()-R.r.min():.4f}   "
          f"typical se per hour ~{1/np.sqrt(R.n.mean()):.4f}")

print("\n" + "=" * 78)
print("TEST 3 - DOES REVERSION VARY WITH VOLATILITY?")
print("=" * 78)
D["rv"] = (D.groupby("coin").ret
             .transform(lambda s: s.abs().shift(1).rolling(8).mean()))
Dv = D.dropna(subset=["rv"]).copy()
Dv["vq"] = Dv.groupby("coin").rv.transform(
    lambda s: pd.qcut(s, 4, labels=[0, 1, 2, 3], duplicates="drop"))
print(f"{'vol quartile':>14} {'pairs':>8} {'autocorr':>10} {'P(same sign)':>14}")
for k, g in Dv.groupby("vq", observed=True):
    same = ((np.sign(g.ret) == np.sign(g.prev)) & (g.prev != 0)).mean()
    print(f"{str(k):>14} {len(g):>8,} {g.ret.corr(g.prev):>+10.4f} "
          f"{same:>14.4f}")

print("\n" + "=" * 78)
print("TEST 4 - IS IT BIG ENOUGH TO MATTER?")
print("=" * 78)
print("  A favourite priced ~70c is roughly 0.5 standard deviations ahead.")
print("  For reversion to flip outcomes it has to be worth a meaningful")
print("  fraction of a typical move.\n")
big = D[D.prev.abs() > D.prev.abs().quantile(0.75)]
print(f"  after a LARGE prior move (top 25%, n {len(big):,}):")
print(f"    mean next return, signed by prior direction: "
      f"{(big.ret * np.sign(big.prev)).mean()*1e4:+.3f}bp")
print(f"    median |next move|: {big.ret.abs().median()*1e4:.2f}bp")
cont = (big.ret * np.sign(big.prev)).mean() / big.ret.abs().median()
print(f"    continuation as a fraction of a typical move: {cont:+.4f}")
small = D[D.prev.abs() < D.prev.abs().quantile(0.25)]
print(f"\n  after a SMALL prior move (bottom 25%, n {len(small):,}):")
print(f"    mean next return, signed by prior direction: "
      f"{(small.ret * np.sign(small.prev)).mean()*1e4:+.3f}bp")

print("\n" + "=" * 78)
print("VERDICT")
print("=" * 78)
crit = 2 * se
if abs(r_all) < crit:
    print(f"  Pooled autocorrelation {r_all:+.4f} is inside +/-{crit:.4f}, the")
    print(f"  2-sigma band for {n:,} observations. Fifteen-minute index returns")
    print("  are a random walk to the precision this data can measure.")
    print()
    print("  That is a strong negative: with 41,334 windows there is enough")
    print("  power to detect an autocorrelation of 1%, and there is none. The")
    print("  strategy is NOT systematically short mean-reversion, so that is")
    print("  not where the losses come from either.")
else:
    print(f"  Autocorrelation {r_all:+.4f} EXCEEDS the {crit:.4f} threshold.")
    print("  Direction: " + ("MEAN REVERSION - the strategy is structurally"
                             " taxed" if r_all < 0 else
                             "MOMENTUM - continuation is favoured"))
