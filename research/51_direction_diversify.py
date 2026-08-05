"""DIRECTION-DIVERSIFIED SELECTION: kill the correlation without losing entries.

THE REFRAME THAT MAKES THIS POSSIBLE
  Every previous attempt asked "which windows should we SKIP" - and every one
  failed, because skipping costs opportunity we cannot replace.

  This asks a different question: given that several coins qualify in the same
  window and we can take THREE, WHICH three should they be?

  That is free. It changes composition, not count.

THE MECHANISM
  We buy the FAVOURITE on every coin, so the favourite's direction is whatever
  the market currently favours. When three coins all have YES favourites, one
  market-wide move down loses all three at once - that is the correlated
  wipeout, and 19 of the 20 observed wipeouts had every favourite pointing the
  same way (aligned 5.65% wipeout rate vs mixed 1.54%).

  But if we hold BTC-YES and SOL-NO, the SAME market-wide move makes one lose
  and the other WIN. A directional move cannot kill both. Only idiosyncratic
  moves can, and those are what diversification is supposed to handle.

  So: when the qualifying set contains both directions, prefer a MIX.

WHY THIS IS NOT THE FILTER I ALREADY REJECTED
  Trading only mixed WINDOWS discards 84% of opportunity. Choosing a mixed
  BASKET inside a window discards nothing - the same number of positions is
  taken, only the identity changes. If two coins are interchangeable on edge,
  taking the one that hedges is strictly better.

WHAT MUST BE TRUE FOR THIS TO WORK
  1. entries/day essentially unchanged
  2. wipeout rate materially lower
  3. dollars NOT lower - if the diversifying pick has systematically worse edge,
     the hedge is being paid for and the whole point is lost
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260924)
QTY = 15

Y = pd.read_parquet(OUT / "yesno.parquet")
q = Y[(Y.px >= 0.65) & (Y.px < 0.85) & (Y.minute >= 8) & (Y.minute <= 14)
      & (Y.vol >= 2000)]
C = (q.sort_values("minute", ascending=False)
       .groupby(["wkey", "coin"], as_index=False).first()).copy()
C["pnl"] = np.where(C.won == 1, (1 - C.px) * QTY, -C.px * QTY)
SER = {"BTC": 0, "ETH": 1, "SOL": 2, "XRP": 3, "DOGE": 4}
C["ord"] = C.coin.map(SER).fillna(9)
days = sorted(C.day.unique())
print(f"candidates {len(C):,}  windows {C.wkey.nunique():,}  days {len(days)}")
per_win = C.groupby("wkey").size()
print(f"candidates per window: mean {per_win.mean():.2f}  "
      f"2+: {(per_win >= 2).sum()}  3+: {(per_win >= 3).sum()}\n")


def select(g, mode, cap=3):
    """Choose up to `cap` candidates from one window."""
    g = g.sort_values(["ord", "minute"], ascending=[True, False])
    if mode == "current":                      # series list order, as deployed
        return g.head(cap)
    if mode == "diversify":
        # take the first, then PREFER the opposite direction while any remains
        picks = [g.iloc[0]]
        rest = g.iloc[1:]
        while len(picks) < cap and len(rest):
            want = not picks[-1].maker_yes
            opp = rest[rest.maker_yes == want]
            take = opp.iloc[0] if len(opp) else rest.iloc[0]
            picks.append(take)
            rest = rest.drop(take.name)
        return pd.DataFrame(picks)
    if mode == "maxmix":
        # balance the basket as evenly as possible across directions
        yes = g[g.maker_yes]
        no = g[~g.maker_yes]
        picks = []
        i = j = 0
        while len(picks) < cap and (i < len(yes) or j < len(no)):
            if len(yes) - i >= len(no) - j and i < len(yes):
                picks.append(yes.iloc[i]); i += 1
            elif j < len(no):
                picks.append(no.iloc[j]); j += 1
            elif i < len(yes):
                picks.append(yes.iloc[i]); i += 1
        return pd.DataFrame(picks)
    raise ValueError(mode)


print("=" * 74)
print("HEAD TO HEAD - same windows, same cap, different basket composition")
print("=" * 74)
res = {}
for mode in ("current", "diversify", "maxmix"):
    keep = []
    for wk, g in C.groupby("wkey"):
        keep.append(select(g, mode))
    K = pd.concat(keep)
    W = K.groupby("wkey").agg(n=("won", "size"),
                              losses=("won", lambda s: int((s == 0).sum())),
                              pnl=("pnl", "sum"),
                              nyes=("maker_yes", "sum"),
                              day=("day", "first")).reset_index()
    M = W[W.n >= 2]
    wipe = (M.losses == M.n).mean()
    aligned = ((M.nyes == M.n) | (M.nyes == 0)).mean()
    res[mode] = dict(K=K, W=W, M=M)
    print(f"\n  {mode.upper()}")
    print(f"    entries        {len(K):,}  ({len(K)/len(days):.1f}/day)")
    print(f"    windows 2+     {len(M)}")
    print(f"    aligned share  {aligned:.4f}")
    print(f"    WIPEOUT rate   {wipe:.4f}  ({int((M.losses == M.n).sum())} events)")
    print(f"    edge/contract  {K.pnl.sum()/(len(K)*QTY)*100:+.2f}c")
    print(f"    $/day          {K.pnl.sum()/len(days):+.2f}")
    print(f"    worst window   {W.pnl.min():+.2f}")

print("\n" + "=" * 74)
print("THE THREE CONDITIONS")
print("=" * 74)
a, b = res["current"], res["diversify"]
ka, kb = a["K"], b["K"]
ma, mb = a["M"], b["M"]
wa = (ma.losses == ma.n).mean()
wb = (mb.losses == mb.n).mean()
print(f"  1. entries preserved : {len(ka):,} -> {len(kb):,} "
      f"({(len(kb)/len(ka)-1)*100:+.2f}%)")
print(f"  2. wipeouts reduced  : {wa:.4f} -> {wb:.4f} "
      f"({(wb/wa-1)*100:+.1f}%)")
print(f"  3. dollars preserved : ${ka.pnl.sum()/len(days):+.2f} -> "
      f"${kb.pnl.sum()/len(days):+.2f} "
      f"({(kb.pnl.sum()-ka.pnl.sum())/len(days):+.2f}/day)")

print("\n=== DAY BY DAY (walk-forward is not needed: the rule is FIXED, ===")
print("=== it fits nothing and has no parameters to tune)              ===\n")
print(f"{'day':>12} {'current $':>11} {'diversify $':>13} {'diff':>9} "
      f"{'cur wipes':>10} {'div wipes':>10}")
w1 = w2 = 0
for d in days:
    x = ka[ka.day == d].pnl.sum()
    y_ = kb[kb.day == d].pnl.sum()
    mx = ma[ma.day == d]
    my = mb[mb.day == d]
    cw = int((mx.losses == mx.n).sum())
    dw = int((my.losses == my.n).sum())
    w1 += cw
    w2 += dw
    print(f"{d:>12} {x:>+11.2f} {y_:>+13.2f} {y_-x:>+9.2f} {cw:>10} {dw:>10}")
print(f"\n  total wipeouts: current {w1}  diversified {w2}")

print("\n=== DAY-CLUSTERED TEST ON THE P&L DIFFERENCE ===")
pd_diff = np.array([kb[kb.day == d].pnl.sum() - ka[ka.day == d].pnl.sum()
                    for d in days])
bs = np.sort(np.array([RNG.choice(pd_diff, len(pd_diff), True).mean()
                       for _ in range(8000)]))
print(f"  mean P&L difference {pd_diff.mean():+.2f}/day   "
      f"95% CI [{bs[200]:+.2f}, {bs[7799]:+.2f}]")
print(f"  P(diversifying is WORSE) = {(bs < 0).mean():.4f}")
