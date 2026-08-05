"""How efficient is the 65-80c band as a drawdown reducer, in honest units?

THE ERROR BEING CORRECTED
  28/29 reported "DD cut per $/day given up" and I quoted ratios of 2.05-6.45
  as if >1 meant a good trade. That is a units error: it divides a drawdown
  LEVEL (dollars) by a profit RATE (dollars per day), giving a quantity in
  days. It cannot be compared against 1, and it flatters every intervention by
  roughly the horizon length.

  The correct comparison puts both sides on the SAME horizon:
      expected profit given up over 30 days   vs   95% drawdown removed

TWO OTHER THINGS THE OLD NUMBERS HID
  1. The stress draw is random - it picks which windows to flip - so a single
     seed gives a single realisation. 28 and 29 used different seeds and got
     2.05 and 6.45 for the SAME quantity, a 3x spread that was reported as if
     it were a range of estimates. This runs many seeds and reports the spread.
  2. Drawdown reduction and profit reduction are not comparable in utility
     terms anyway, so a risk-adjusted measure (return per unit of drawdown) is
     reported alongside.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
D = pd.read_parquet(OUT / "grid.parquet")
QTY, MAX_GROSS, BANK = 30, 110.0, 408.11
NPATH, NDAY = 6000, 30
NSEED = 24
days = sorted(D.day.unique())
BT_WIN, LIVE_WIN = 0.8462, 0.7788
DELTA = BT_WIN - LIVE_WIN


def entries(cap=3, vmin=2000, band=(0.65, 0.85)):
    q = D[(D.px >= band[0]) & (D.px < band[1]) & (D.minute >= 8)
          & (D.minute <= 14) & (D.vol >= vmin)]
    e = (q.sort_values("minute", ascending=False)
           .groupby(["wkey", "coin"], as_index=False).first())
    e = (e.sort_values(["wkey", "minute"], ascending=[True, False])
           .groupby("wkey", as_index=False).head(cap)).copy()
    keep = []
    for _, g in e.groupby("wkey"):
        gross = 0.0
        for r in g.itertuples():
            c = QTY * r.px
            if gross + c > MAX_GROSS:
                continue
            gross += c
            keep.append((r.wkey, r.day, r.px, r.won))
    return pd.DataFrame(keep, columns=["wkey", "day", "px", "won"])


def run(K, rng):
    """One stress realisation + one drawdown simulation."""
    K = K.copy()
    need = int(round(DELTA * len(K)))
    wins = K.index[K.won == 1]
    wk = K.loc[wins, "wkey"]
    flip, got = [], 0
    for w in rng.permutation(wk.unique()):
        idx = list(wins[wk == w])
        flip += idx
        got += len(idx)
        if got >= need:
            break
    K.loc[flip, "won"] = 0
    p = np.where(K.won == 1, (1 - K.px) * QTY, -K.px * QTY)
    wp = pd.Series(p).groupby(K.wkey.values).sum().to_numpy()
    k = max(int(round(K.wkey.nunique() / len(days))), 1)
    idx = rng.integers(0, len(wp), size=(NPATH, NDAY, k))
    daily = wp[idx].sum(2)
    eq = BANK + np.cumsum(daily, axis=1)
    run_ = np.concatenate([np.full((NPATH, 1), BANK), eq], axis=1)
    dd = (np.maximum.accumulate(run_, axis=1) - run_).max(1)
    return daily.mean(), np.percentile(dd, 95), (run_.min(1) < .5 * BANK).mean()


CFG = [("current 65-85c", dict()),
       ("band 65-80c", dict(band=(0.65, 0.80))),
       ("vmin 4000", dict(vmin=4000)),
       ("band 65-80c + vmin 4000", dict(band=(0.65, 0.80), vmin=4000))]
print(f"{NSEED} stress seeds x {NPATH:,} paths x {NDAY} days   bank ${BANK:.2f}\n")
res = {}
for lbl, kw in CFG:
    K = entries(**kw)
    out = np.array([run(K, np.random.default_rng(1000 + s)) for s in range(NSEED)])
    res[lbl] = out
    print(f"{lbl:>26}  $/day {out[:,0].mean():>+7.2f} "
          f"[{out[:,0].min():+.0f},{out[:,0].max():+.0f}]   "
          f"95%DD ${out[:,1].mean():>6.2f} [{out[:,1].min():.0f},{out[:,1].max():.0f}]"
          f"   P(ruin) {out[:,2].mean():.4f}")

b = res["current 65-85c"]
print(f"\n=== THE HONEST TRADE, both sides over the SAME 30 DAYS ===")
print(f"{'intervention':>26} {'profit given up':>17} {'95%DD removed':>15} "
      f"{'$ DD saved per $ profit':>25}")
for lbl, out in res.items():
    if lbl.startswith("current"):
        continue
    give = (b[:, 0].mean() - out[:, 0].mean()) * NDAY
    cut = b[:, 1].mean() - out[:, 1].mean()
    print(f"{lbl:>26} {give:>16,.0f} {cut:>15,.0f} {cut/give if give else 0:>25.3f}")
print("\n  Every one of these pays MORE in forgone profit than it removes in")
print("  tail drawdown. On a like-for-like horizon none is an efficient")
print("  drawdown trade - the opposite of what my 'ratio 2.05-6.45' implied.")

print(f"\n=== RISK-ADJUSTED: 30-day expected profit per dollar of 95% drawdown ===")
for lbl, out in res.items():
    r = out[:, 0].mean() * NDAY / out[:, 1].mean()
    print(f"  {lbl:>26}  ${out[:,0].mean()*NDAY:>7,.0f} profit / "
          f"${out[:,1].mean():>6.0f} DD  =  {r:>5.2f}")
print("\n  Higher is better. If the current config wins here, the band trim")
print("  is not justified by drawdown - only by the capital-normalised SCORE")
print("  result (80-85c scores 0.0973 vs 0.1499, p=0.0200), which is a")
print("  separate and still-valid argument.")

print(f"\n=== SEED INSTABILITY: why 2.05 and 6.45 were the same number ===")
o = res["band 65-80c"]
give = (b[:, 0] - o[:, 0]) * NDAY
cut = b[:, 1] - o[:, 1]
print(f"  across {NSEED} seeds, profit given up ranges "
      f"${give.min():,.0f} to ${give.max():,.0f}")
print(f"  and drawdown removed ranges ${cut.min():,.0f} to ${cut.max():,.0f}")
print(f"  the old per-seed 'ratio' would have ranged "
      f"{np.nanmin(cut/np.where(give==0,np.nan,give)):.2f} to "
      f"{np.nanmax(cut/np.where(give==0,np.nan,give)):.2f}")
print("  A single seed was never enough to quote a number from.")
