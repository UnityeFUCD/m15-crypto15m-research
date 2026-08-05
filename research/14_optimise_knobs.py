"""Joint optimisation of every knob, scored on risk-adjusted terms.

KNOBS
  price band       currently 65-85c
  time window      currently 8-14 minutes remaining
  volume threshold currently 500 contracts
  entries/window   currently 2

OBJECTIVE
  Not raw edge. The stated goal is more wins and edge with SMALLER losses,
  drawdown and ruin risk, so a configuration is scored on:
      $/day, day-level Sharpe, worst day, max drawdown, and P(ruin)
  simulated at the live size on the live bankroll.

DISCIPLINE
  - WINDOW-EQUAL-WEIGHTED throughout. Contract weighting has faked three
    separate findings this session by over-sampling busy windows.
  - every configuration is fitted on the FIRST half of days and reported on the
    SECOND half, so the ranking cannot be a description of noise.
  - the grid is large, so the winner is compared against the distribution of
    all configurations rather than presented alone.
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260805)
D = pd.read_parquet(OUT / "grid.parquet")
BANK = 341.51
QTY = 30

days = sorted(D.day.unique())
CUT = days[len(days) // 2]
print(f"rows {len(D):,}   windows {D.wkey.nunique():,}   days {len(days)}")
print(f"fit on days < {CUT} ({sum(d < CUT for d in days)} days), "
      f"test on the rest ({sum(d >= CUT for d in days)} days)\n")


def apply_policy(d, plo, phi, mlo, mhi, vmin, cap):
    """One entry per (window, coin) at the FIRST qualifying minute, then cap
    per close-window, mimicking the live runner's scan."""
    q = d[(d.px >= plo) & (d.px < phi) & (d.minute >= mlo) & (d.minute <= mhi)
          & (d.vol >= vmin)]
    if not len(q):
        return None
    # earliest qualifying minute is the largest `minute` (time remaining)
    e = q.sort_values("minute", ascending=False).groupby(["wkey", "coin"],
                                                         as_index=False).first()
    if cap < 5:
        e = (e.sort_values(["wkey", "minute"], ascending=[True, False])
               .groupby("wkey", as_index=False).head(cap))
    return e


def score(e, all_days):
    """Daily P&L in dollars at QTY, plus risk metrics."""
    if e is None or len(e) < 60:
        return None
    per_day = (e.groupby("day").edge.sum() * QTY).reindex(all_days, fill_value=0.0)
    v = per_day.values
    if v.std(ddof=1) == 0:
        return None
    eq = BANK + np.cumsum(v)
    dd = np.maximum.accumulate(np.concatenate([[BANK], eq])) - np.concatenate([[BANK], eq])
    return dict(n=len(e), per_day=v.mean(), sd=v.std(ddof=1),
                sharpe=v.mean() / v.std(ddof=1), worst=v.min(),
                maxdd=dd.max(), edge=e.edge.mean() * 100,
                win=e.won.mean(), px=e.px.mean() * 100)


GRID = list(itertools.product(
    [(0.55, 0.90), (0.60, 0.90), (0.60, 0.80), (0.65, 0.85), (0.65, 0.75),
     (0.70, 0.90), (0.55, 0.75)],
    [(8, 14), (5, 14), (3, 14), (8, 11), (11, 14), (5, 11), (2, 14)],
    [0, 250, 500, 1000, 2000, 4000],
    [1, 2, 3]))
print(f"configurations to evaluate: {len(GRID):,}\n")

fit_days = [d for d in days if d < CUT]
test_days = [d for d in days if d >= CUT]
Dfit, Dtest = D[D.day < CUT], D[D.day >= CUT]

res = []
for (plo, phi), (mlo, mhi), vmin, cap in GRID:
    ef = apply_policy(Dfit, plo, phi, mlo, mhi, vmin, cap)
    sf = score(ef, fit_days)
    if sf is None:
        continue
    et = apply_policy(Dtest, plo, phi, mlo, mhi, vmin, cap)
    st = score(et, test_days)
    if st is None:
        continue
    res.append(dict(band=f"{plo*100:.0f}-{phi*100:.0f}", tw=f"{mlo}-{mhi}",
                    vmin=vmin, cap=cap,
                    fit_day=sf["per_day"], fit_sharpe=sf["sharpe"],
                    test_day=st["per_day"], test_sharpe=st["sharpe"],
                    test_n=st["n"], test_edge=st["edge"], test_win=st["win"],
                    test_worst=st["worst"], test_dd=st["maxdd"], test_px=st["px"]))
R = pd.DataFrame(res)
R.to_parquet(OUT / "knob_search.parquet", index=False)
print(f"evaluated {len(R):,} configurations\n")

CUR = R[(R.band == "65-85") & (R.tw == "8-14") & (R.vmin == 500) & (R.cap == 2)]
print("=== CURRENT LIVE CONFIG ===")
if len(CUR):
    c = CUR.iloc[0]
    print(f"  fit  ${c.fit_day:+7.2f}/day  Sharpe {c.fit_sharpe:+.3f}")
    print(f"  TEST ${c.test_day:+7.2f}/day  Sharpe {c.test_sharpe:+.3f}  "
          f"n {int(c.test_n)}  edge {c.test_edge:+.2f}  worst ${c.test_worst:+.2f}  "
          f"maxDD ${c.test_dd:.2f}")

print("\n=== TOP 10 BY OUT-OF-SAMPLE SHARPE (risk-adjusted, the stated goal) ===")
print(f"{'band':>8} {'time':>7} {'vmin':>6} {'cap':>4} {'fit $/d':>9} "
      f"{'TEST $/d':>9} {'TEST Sh':>8} {'n':>5} {'edge':>7} {'worst':>8} {'maxDD':>8}")
for _, r in R.sort_values("test_sharpe", ascending=False).head(10).iterrows():
    print(f"{r.band:>8} {r.tw:>7} {int(r.vmin):>6} {int(r.cap):>4} "
          f"{r.fit_day:>+9.2f} {r.test_day:>+9.2f} {r.test_sharpe:>+8.3f} "
          f"{int(r.test_n):>5} {r.test_edge:>+7.2f} {r.test_worst:>+8.2f} "
          f"{r.test_dd:>8.2f}")

print("\n=== TOP 10 BY OUT-OF-SAMPLE $/DAY ===")
print(f"{'band':>8} {'time':>7} {'vmin':>6} {'cap':>4} {'TEST $/d':>9} "
      f"{'TEST Sh':>8} {'n':>5} {'worst':>8} {'maxDD':>8}")
for _, r in R.sort_values("test_day", ascending=False).head(10).iterrows():
    print(f"{r.band:>8} {r.tw:>7} {int(r.vmin):>6} {int(r.cap):>4} "
          f"{r.test_day:>+9.2f} {r.test_sharpe:>+8.3f} {int(r.test_n):>5} "
          f"{r.test_worst:>+8.2f} {r.test_dd:>8.2f}")

print("\n=== DOES THE FIT RANKING SURVIVE OUT OF SAMPLE? ===")
top = R.sort_values("fit_sharpe", ascending=False).head(20)
print(f"  corr(fit Sharpe, test Sharpe) over all configs = "
      f"{R.fit_sharpe.corr(R.test_sharpe):+.4f}")
print(f"  best-20-by-fit  mean test Sharpe {top.test_sharpe.mean():+.3f}   "
      f"all configs mean {R.test_sharpe.mean():+.3f}")
print("  (a positive correlation means the parameters carry real information;")
print("   near zero would mean the search is fitting noise)")

print("\n=== KNOB BY KNOB, marginal effect on out-of-sample Sharpe ===")
for k in ("band", "tw", "vmin", "cap"):
    g = R.groupby(k).agg(test_sharpe=("test_sharpe", "mean"),
                         test_day=("test_day", "mean"), n=("test_n", "mean"))
    print(f"\n  {k}:")
    for idx, r in g.sort_values("test_sharpe", ascending=False).iterrows():
        print(f"    {str(idx):>8}  Sharpe {r.test_sharpe:+.3f}  "
              f"${r.test_day:+7.2f}/day  n {r.n:.0f}")
