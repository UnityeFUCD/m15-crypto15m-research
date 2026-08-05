"""THE USER'S HYPOTHESIS: is there a persistent winning DIRECTION, and do
correlated losses mark the moment it flips?

THE CLAIM, STATED PRECISELY
  "Most of our wins are one direction - say YES about 80% of the time. Then a
  regime shift happens, the correlated loss hits, and after that NO starts
  winning for a while. Then we lose again when it flips back."

  That is a REGIME-PERSISTENCE claim about the settlement outcome itself, and
  it makes three sharp, separable predictions:

    P1  the settled direction is AUTOCORRELATED - YES windows follow YES
        windows more often than chance. Equivalently, there are runs.
    P2  wipeouts occur AT the transitions, not uniformly.
    P3  after a wipeout, the direction has FLIPPED and stays flipped for a
        while.

  Each is tested separately, because the hypothesis can be right about one and
  wrong about another - and the interesting case is precisely that.

WHY THIS IS WORTH TESTING EVEN THOUGH DIRECTION SHOULD BE UNPREDICTABLE
  Earlier work showed the price path is a random walk and no feature predicts a
  wipeout. But NONE of that tested the settled direction's own serial
  structure. A random walk in PRICE is entirely compatible with runs in the
  SIGN of 15-minute returns if there is drift or trend persistence. So this is
  a real gap, not a re-test.

DERIVING THE SETTLED DIRECTION
  We know which side the maker held (maker_yes) and whether it won, so
      yes_settled = (maker_yes == won)
  which recovers the market's actual outcome regardless of which side we took.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260928)
Y = pd.read_parquet(OUT / "yesno.parquet")

q = Y[(Y.minute >= 8) & (Y.minute <= 14) & (Y.vol >= 2000)
      & (Y.px >= 0.55) & (Y.px < 0.95)]
C = (q.sort_values("minute", ascending=False)
       .groupby(["wkey", "coin"], as_index=False).first()).copy()
# the market's own outcome, independent of which side we took
C["yes_settled"] = (C.maker_yes == (C.won == 1)).astype(int)
C["close"] = pd.to_datetime(C.wkey, format="%y%b%d%H%M", utc=True,
                            errors="coerce")
C = C.dropna(subset=["close"])
print(f"observations {len(C):,}   windows {C.wkey.nunique():,}   "
      f"coins {sorted(C.coin.unique())}")
print(f"overall P(YES settles) = {C.yes_settled.mean():.4f}\n")

print("=" * 74)
print("P1 - IS THE SETTLED DIRECTION PERSISTENT? (per coin, chronological)")
print("=" * 74)
print(f"{'coin':>6} {'windows':>8} {'P(YES)':>8} {'lag-1 autocorr':>15} "
      f"{'runs p-value':>13} {'P(same as prev)':>16}")
allac = []
for coin, g in C.groupby("coin"):
    g = g.sort_values("close")
    v = g.yes_settled.to_numpy()
    if len(v) < 60:
        continue
    ac = np.corrcoef(v[:-1], v[1:])[0, 1] if v.std() > 0 else np.nan
    same = (v[:-1] == v[1:]).mean()
    # runs test: how many runs vs expected under independence
    runs = 1 + int((v[1:] != v[:-1]).sum())
    n1, n0 = v.sum(), len(v) - v.sum()
    exp_r = 2 * n1 * n0 / len(v) + 1
    sd_r = np.sqrt(2 * n1 * n0 * (2 * n1 * n0 - len(v)) /
                   (len(v) ** 2 * (len(v) - 1))) if len(v) > 1 else np.nan
    zr = (runs - exp_r) / sd_r if sd_r and sd_r > 0 else np.nan
    from math import erfc
    pr = erfc(abs(zr) / np.sqrt(2)) if np.isfinite(zr) else np.nan
    allac.append(ac)
    print(f"{coin:>6} {len(v):>8} {v.mean():>8.4f} {ac:>+15.4f} "
          f"{pr:>13.4f} {same:>16.4f}")
print(f"\n  mean lag-1 autocorrelation across coins: {np.nanmean(allac):+.4f}")
print("  Under the hypothesis this should be clearly POSITIVE - YES following")
print("  YES. Near zero means the direction is a coin flip each window.")

print("\n=== POOLED, ACROSS THE WHOLE MARKET PER WINDOW ===")
Wd = C.groupby("wkey").agg(yes_frac=("yes_settled", "mean"),
                           n=("yes_settled", "size"),
                           close=("close", "first"),
                           day=("day", "first")).reset_index()
Wd = Wd.sort_values("close").reset_index(drop=True)
Wd["dir"] = (Wd.yes_frac > 0.5).astype(int)
v = Wd.dir.to_numpy()
ac = np.corrcoef(v[:-1], v[1:])[0, 1]
null = np.array([np.corrcoef(p[:-1], p[1:])[0, 1]
                 for p in (RNG.permutation(v) for _ in range(6000))])
print(f"  windows {len(v)}   P(market YES) {v.mean():.4f}")
print(f"  lag-1 autocorrelation of market direction: {ac:+.4f}")
print(f"  permutation p = {(np.abs(null) >= abs(ac)).mean():.4f}")
for lag in (2, 3, 4, 8):
    a2 = np.corrcoef(v[:-lag], v[lag:])[0, 1]
    print(f"  lag {lag}: {a2:+.4f}")

print("\n" + "=" * 74)
print("P2 - DO WIPEOUTS HAPPEN AT DIRECTION TRANSITIONS?")
print("=" * 74)
# our entries and wipeouts
qq = Y[(Y.px >= 0.65) & (Y.px < 0.85) & (Y.minute >= 8) & (Y.minute <= 14)
       & (Y.vol >= 2000)]
E = (qq.sort_values("minute", ascending=False)
       .groupby(["wkey", "coin"], as_index=False).first())
E = (E.sort_values(["wkey", "minute"], ascending=[True, False])
       .groupby("wkey", as_index=False).head(3))
WE = E.groupby("wkey").agg(n=("won", "size"),
                           losses=("won", lambda s: int((s == 0).sum()))).reset_index()
WE = WE[WE.n >= 2]
WE["wipe"] = (WE.losses == WE.n).astype(int)
Z = Wd.merge(WE[["wkey", "wipe"]], on="wkey", how="inner").sort_values("close")
Z = Z.reset_index(drop=True)
Z["prev_dir"] = Z.dir.shift(1)
Z["flip"] = (Z.dir != Z.prev_dir).astype(float)
Z.loc[0, "flip"] = np.nan
zz = Z.dropna(subset=["flip"])
print(f"  windows analysed {len(zz)}   wipeouts {int(zz.wipe.sum())}   "
      f"direction flips {int(zz.flip.sum())} ({zz.flip.mean():.3f})")
a = zz[zz.flip == 0].wipe.mean()
b = zz[zz.flip == 1].wipe.mean()
print(f"\n  wipeout rate when direction HELD    : {a:.4f} "
      f"({int(zz[zz.flip==0].wipe.sum())}/{int((zz.flip==0).sum())})")
print(f"  wipeout rate when direction FLIPPED : {b:.4f} "
      f"({int(zz[zz.flip==1].wipe.sum())}/{int((zz.flip==1).sum())})")
real = b - a
null2 = []
for _ in range(6000):
    S = zz.copy()
    S["wipe"] = S.groupby("day").wipe.transform(lambda s: RNG.permutation(s.values))
    null2.append(S[S.flip == 1].wipe.mean() - S[S.flip == 0].wipe.mean())
null2 = np.array(null2)
print(f"  difference {real:+.4f}   day-clustered permutation p = "
      f"{(np.abs(null2) >= abs(real)).mean():.4f}")

print("\n" + "=" * 74)
print("P3 - AFTER A WIPEOUT, HAS THE DIRECTION FLIPPED AND DOES IT STICK?")
print("=" * 74)
idx = np.where(zz.wipe.values == 1)[0]
dirs = zz.dir.values
print(f"{'k windows after':>16} {'P(dir same as pre-wipeout)':>28} {'n':>6}")
for k in (1, 2, 3, 4, 8):
    same = []
    for i in idx:
        if i - 1 >= 0 and i + k < len(dirs):
            same.append(int(dirs[i + k] == dirs[i - 1]))
    if len(same) >= 8:
        print(f"{k:>16} {np.mean(same):>28.4f} {len(same):>6}")
print("\n  Under the hypothesis this should be clearly BELOW 0.5 - the")
print("  direction flipped at the wipeout and stayed flipped. Near 0.5 means")
print("  the wipeout carried no information about the direction that follows.")

print("\n=== AND THE PART THAT WOULD BE WORTH MONEY ===")
print("  If direction persisted, the FAVOURITE would inherit that persistence")
print("  and our win rate should depend on whether the previous window settled")
print("  the same way as our current favourite points.\n")
E2 = E.merge(Wd[["wkey", "dir", "close"]], on="wkey", how="left").dropna(subset=["dir"])
E2 = E2.sort_values("close")
prev = Wd.set_index("wkey").dir.shift(1)
E2["prev_dir"] = E2.wkey.map(prev)
E2 = E2.dropna(subset=["prev_dir"])
E2["agrees"] = (E2.maker_yes.astype(int) == E2.prev_dir).astype(int)
for lbl, g in (("favourite AGREES with last window", E2[E2.agrees == 1]),
               ("favourite OPPOSES last window", E2[E2.agrees == 0])):
    print(f"  {lbl:>34}: n {len(g):>5}  win {g.won.mean():.4f}  "
          f"px {g.px.mean()*100:.1f}c  edge {(g.won.mean()-g.px.mean())*100:+.2f}c")
