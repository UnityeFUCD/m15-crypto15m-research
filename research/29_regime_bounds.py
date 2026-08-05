"""How much of the drawdown answer is the data, and how much is my assumption?

TWO ASSUMPTIONS DRIVE EVERYTHING
  1. IS the live shortfall real? Live win rate is 0.7788 on 104 settlements
     against a backtest 0.8462. But the live edge CI is [-4.24c, +15.03c] and
     the backtest sits inside it, so the "degradation" may be pure sampling
     noise.
  2. IF it is real, HOW does it manifest? Flipping whole WINDOWS to losses is
     maximally clustered and maximally pessimistic. Flipping random ENTRIES is
     the neutral alternative. The truth is somewhere between, and the drawdown
     answer is very sensitive to which.

  Reporting a single drawdown number while both of these are open would be
  false precision. This bounds the answer instead.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260816)
D = pd.read_parquet(OUT / "grid.parquet")
QTY, MAX_GROSS, BANK = 30, 110.0, 367.31
NPATH, NDAY = 20000, 30
days = sorted(D.day.unique())


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


def stress(K, delta, mode):
    K = K.copy()
    if delta <= 0:
        return K
    need = int(round(delta * len(K)))
    wins = K.index[K.won == 1]
    if mode == "entry":                      # neutral: independent degradation
        K.loc[RNG.choice(wins, min(need, len(wins)), replace=False), "won"] = 0
    else:                                    # clustered: whole bad windows
        wk = K.loc[wins, "wkey"]
        flip, got = [], 0
        for w in RNG.permutation(wk.unique()):
            idx = list(wins[wk == w])
            flip += idx
            got += len(idx)
            if got >= need:
                break
        K.loc[flip, "won"] = 0
    return K


def sim(K):
    p = np.where(K.won == 1, (1 - K.px) * QTY, -K.px * QTY)
    wp = pd.Series(p).groupby(K.wkey.values).sum().to_numpy()
    k = max(int(round(K.wkey.nunique() / len(days))), 1)
    idx = RNG.integers(0, len(wp), size=(NPATH, NDAY, k))
    daily = wp[idx].sum(2)
    eq = BANK + np.cumsum(daily, axis=1)
    run = np.concatenate([np.full((NPATH, 1), BANK), eq], axis=1)
    dd = (np.maximum.accumulate(run, axis=1) - run).max(1)
    return (p.sum() / len(p) / QTY * 100 * QTY / QTY, daily.mean(),
            (daily < 0).mean(), np.percentile(dd, 95),
            (run.min(1) < 0.5 * BANK).mean())


BASE = entries()
BT = BASE.won.mean()
print(f"backtest win {BT:.4f}   live win 0.7788 (104 settlements, "
      f"edge CI [-4.24c,+15.03c])\n")
print("=== THE RANGE THE ANSWER LIVES IN ===")
print(f"{'assumed win rate':>18} {'stress mode':>12} {'edge':>7} {'$/day':>9} "
      f"{'P(day<0)':>9} {'95%DD':>9} {'P(ruin 30d)':>12}")
for win in (BT, 0.82, 0.7788):
    for mode in ("entry", "window"):
        if win >= BT and mode == "window":
            continue
        K = stress(BASE, BT - win, mode)
        e, d, pn, dd, ru = sim(K)
        tag = "(backtest)" if win >= BT else ""
        print(f"{win:>17.4f}{'':>1} {mode if win < BT else '-':>12} "
              f"{e:>+7.2f} {d:>+9.2f} {pn:>9.3f} {dd:>9.2f} {ru:>12.4f} {tag}")

print("\n  The 95% drawdown ranges from ~$25 to ~$460 on a $367 account")
print("  depending on assumptions that 104 settlements cannot resolve.")
print("  That spread IS the finding. No knob choice matters next to it.")

print("\n=== WHAT WOULD SETTLE IT ===")
sd = 41.23           # per-settlement sd, measured from live
for hw, lbl in ((4, "±4c"), (2, "±2c")):
    n = int((1.96 * sd / hw) ** 2)
    print(f"  {lbl} on the live edge: {n:,} settlements = "
          f"{n/121.6:.1f} days at the current rate")
print("\n  Until then the account is exposed to a drawdown profile we cannot")
print("  distinguish between 'trivial' and 'ruinous'. The protection that")
print("  does not depend on knowing which is the DRAWDOWN ENVELOPE, which is")
print("  already live and caps loss at 30% of peak equity regardless of regime.")

print("\n=== SO: WHAT ACTUALLY REDUCES RISK WITHOUT NEEDING THE ANSWER ===")
print(f"{'intervention':>28} {'$/day':>9} {'95%DD':>9} {'P(ruin)':>9} "
      f"{'DD cut per $/day given up':>26}")
opts = [("current  cap3 65-85c v2000", dict()),
        ("band 65-80c", dict(band=(0.65, 0.80))),
        ("vmin 4000", dict(vmin=4000)),
        ("cap 2", dict(cap=2)),
        ("band 65-80c + vmin 4000", dict(band=(0.65, 0.80), vmin=4000))]
out = []
for lbl, kw in opts:
    K = stress(entries(**kw), BT - 0.7788, "window")   # pessimistic regime
    e, d, pn, dd, ru = sim(K)
    out.append((lbl, d, dd, ru))
b = out[0]
for lbl, d, dd, ru in out:
    give, cut = b[1] - d, b[2] - dd
    r = f"{cut/give:>26.2f}" if give > 0.01 else f"{'(baseline)' if lbl==b[0] else 'free':>26}"
    print(f"{lbl:>28} {d:>+9.2f} {dd:>9.2f} {ru:>9.4f} {r}")
