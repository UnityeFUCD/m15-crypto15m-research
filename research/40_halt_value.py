"""Is halting after a bad window actually profitable?

THE QUESTION
  We just halted on a correlated burst loss (three positions, all lost, -$61.80,
  drawdown from peak $77.40 against a $75 rule). Restarting is mechanically
  instant, so "when do we trade again" is purely a decision - and the decision
  is only justified if a bad window PREDICTS more bad windows.

  If losses cluster in TIME, standing down is free money: we skip a bad patch.
  If windows are independent, halting is pure cost - we stop collecting a
  positive-expectation edge for emotional reasons, and we pay twice, because
  the entries we skip were as good as any other.

PRIOR EXPECTATION (which this test can overturn)
  Windows within a day already measured near-independent: ICC -0.002 within
  day, +0.04 within hour. That predicts halting is NOT profitable. But that ICC
  was computed on ALL windows; it says nothing about the conditional behaviour
  after an EXTREME window, which is exactly the case that triggers a halt.
  Tails can cluster even when the bulk does not, so it needs its own test.

WHAT IS MEASURED
  Condition on a window being a total wipeout (every position lost, 2+
  positions), then measure the edge of the next 1, 2, 4, 8 and 16 windows
  against the unconditional baseline. Also the reverse: does a GOOD window
  predict good ones? A significant result in either direction means time
  structure exists and halt/resume rules can be tuned. A null in both means
  every halt rule is a pure tax.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260906)
D = pd.read_parquet(OUT / "grid.parquet")
QTY = 30

q = D[(D.px >= 0.65) & (D.px < 0.85) & (D.minute >= 8) & (D.minute <= 14)
      & (D.vol >= 2000)]
E = (q.sort_values("minute", ascending=False)
       .groupby(["wkey", "coin"], as_index=False).first())
E = (E.sort_values(["wkey", "minute"], ascending=[True, False])
       .groupby("wkey", as_index=False).head(3)).copy()
E["pnl"] = np.where(E.won == 1, (1 - E.px) * QTY, -E.px * QTY)

# windows in true chronological order
W = E.groupby("wkey").agg(n=("pnl", "size"), pnl=("pnl", "sum"),
                          losses=("won", lambda s: int((s == 0).sum())),
                          edge=("edge", "mean"), day=("day", "first"))
W["close"] = pd.to_datetime(W.index, format="%y%b%d%H%M", utc=True,
                            errors="coerce")
W = W.dropna(subset=["close"]).sort_values("close").reset_index()
W["wipeout"] = (W.losses == W.n) & (W.n >= 2)
W["allwin"] = (W.losses == 0) & (W.n >= 2)
print(f"windows {len(W):,}   mean window P&L ${W.pnl.mean():+.2f}   "
      f"mean edge {W.edge.mean()*100:+.2f}c")
print(f"wipeout windows (2+ positions, all lost): {W.wipeout.sum()} "
      f"({W.wipeout.mean()*100:.1f}%)")
print(f"all-win windows (2+ positions):           {W.allwin.sum()} "
      f"({W.allwin.mean()*100:.1f}%)\n")

base_edge = W.edge.mean() * 100
base_pnl = W.pnl.mean()


def after(flag, k):
    """Edge of the k windows following each flagged window."""
    idx = np.where(W[flag].values)[0]
    take = []
    for i in idx:
        take += list(range(i + 1, min(i + 1 + k, len(W))))
    take = sorted(set(take))
    if len(take) < 15:
        return None
    sub = W.iloc[take]
    return sub.edge.mean() * 100, sub.pnl.mean(), len(sub)


print("=== DOES A WIPEOUT PREDICT MORE WIPEOUTS? ===")
print(f"  unconditional: edge {base_edge:+.2f}c   ${base_pnl:+.2f}/window")
print(f"{'next N windows':>16} {'n':>6} {'edge':>9} {'$/window':>10} "
      f"{'vs base':>9} {'p (day-clustered)':>19}")
for k in (1, 2, 4, 8, 16):
    r = after("wipeout", k)
    if r is None:
        print(f"{k:>16} -- too few --")
        continue
    e, pn, n = r
    idx = np.where(W["wipeout"].values)[0]
    bs = []
    for _ in range(4000):
        # resample DAYS to respect clustering, recompute the conditional mean
        days = W.day.unique()
        pick = RNG.choice(days, len(days), True)
        S = pd.concat([W[W.day == d] for d in pick], ignore_index=True)
        ii = np.where(S["wipeout"].values)[0]
        tk = []
        for i in ii:
            tk += list(range(i + 1, min(i + 1 + k, len(S))))
        tk = sorted(set(tk))
        if len(tk) > 15:
            bs.append(S.iloc[tk].edge.mean() * 100 - S.edge.mean() * 100)
    bs = np.sort(np.array(bs))
    pv = 2 * min((bs <= 0).mean(), (bs >= 0).mean()) if len(bs) else float("nan")
    print(f"{k:>16} {n:>6} {e:>+9.2f} {pn:>+10.2f} {e-base_edge:>+9.2f} "
          f"{pv:>19.4f}")

print("\n=== AND THE MIRROR: DOES A GOOD WINDOW PREDICT GOOD ONES? ===")
print(f"{'next N windows':>16} {'n':>6} {'edge':>9} {'vs base':>9}")
for k in (1, 2, 4, 8):
    r = after("allwin", k)
    if r is None:
        continue
    e, pn, n = r
    print(f"{k:>16} {n:>6} {e:>+9.2f} {e-base_edge:>+9.2f}")

print("\n=== WHAT WOULD A HALT RULE HAVE COST? ===")
print("  Simulate: after any wipeout, stand down for N windows, then resume.\n")
print(f"{'stand-down':>12} {'windows traded':>16} {'total $':>12} "
     f"{'vs always-on':>14} {'$/window':>10}")
tot_all = W.pnl.sum()
for k in (0, 1, 2, 4, 8, 16):
    skip = set()
    for i in np.where(W["wipeout"].values)[0]:
        skip |= set(range(i + 1, min(i + 1 + k, len(W))))
    keep = [i for i in range(len(W)) if i not in skip]
    tot = W.iloc[keep].pnl.sum()
    print(f"{k:>12} {len(keep):>16} {tot:>+12.2f} {tot-tot_all:>+14.2f} "
          f"{tot/max(len(keep),1):>+10.2f}")

print("\n=== AUTOCORRELATION OF WINDOW P&L ===")
v = W.pnl.values
for lag in (1, 2, 3, 4, 8):
    if len(v) > lag + 10:
        r = np.corrcoef(v[:-lag], v[lag:])[0, 1]
        nul = np.sort(np.array([
            np.corrcoef(p[:-lag], p[lag:])[0, 1]
            for p in (RNG.permutation(v) for _ in range(3000))]))
        pv = (np.abs(nul) >= abs(r)).mean()
        print(f"  lag {lag:>2}: r = {r:+.4f}   permutation p = {pv:.4f}")
print("\n  If every lag is null, window outcomes are serially independent and")
print("  EVERY halt-and-wait rule is a pure tax on a positive-expectation edge.")
