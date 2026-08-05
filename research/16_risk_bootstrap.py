"""Honest risk for each candidate knob setting: block bootstrap over WINDOWS.

WHY THE PREVIOUS SCRIPT'S RISK NUMBERS WERE VOID
  It computed max drawdown from 5 day-observations. All 5 happened to be
  positive, so it printed maxDD $0.00 for every configuration - reporting the
  absence of a bad day in a tiny sample as the absence of risk. Sharpe on n=5
  is noise with a decimal point.

WHAT THIS DOES
  Resamples whole WINDOWS with replacement. That is the right block: the five
  coins inside one 15-minute window settle against correlated price moves, so
  they win and lose together, and treating entries as independent understates
  daily variance by roughly the cluster size.

  Each synthetic day is built from the same number of windows the config
  actually trades per day, positions are capped by the same $110 gross limit,
  and 20,000 paths of 30 days are run at fixed live size (no compounding -
  that is what the runner actually does).

REPORTS, for each config, the things the goal actually named:
  edge, win rate, $/day, P(losing day), 5th-percentile day,
  median and 95th-percentile max drawdown over 30 days, and P(ruin).
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260806)
D = pd.read_parquet(OUT / "grid.parquet")
BANK = 341.51
MAX_GROSS = 110.0
NPATH, NDAY = 20_000, 30
RUIN = 0.50 * BANK          # ruin = equity halved

CANDS = [
    ("CURRENT  65-85 / 8-14 / v500  / cap2", 0.65, 0.85, 8, 14, 500, 2),
    ("         65-85 / 8-14 / v1000 / cap2", 0.65, 0.85, 8, 14, 1000, 2),
    ("         65-85 / 8-14 / v2000 / cap2", 0.65, 0.85, 8, 14, 2000, 2),
    ("         65-85 / 8-14 / v2000 / cap3", 0.65, 0.85, 8, 14, 2000, 3),
    ("         70-90 / 8-14 / v2000 / cap2", 0.70, 0.90, 8, 14, 2000, 2),
    ("         70-90 / 8-14 / v2000 / cap3", 0.70, 0.90, 8, 14, 2000, 3),
    ("         70-90 /11-14 / v2000 / cap2", 0.70, 0.90, 11, 14, 2000, 2),
    ("         55-90 / 8-14 / v2000 / cap3", 0.55, 0.90, 8, 14, 2000, 3),
    ("         70-90 / 8-14 / v4000 / cap3", 0.70, 0.90, 8, 14, 4000, 3),
]
NDAYS_DATA = D.day.nunique()


def entries(plo, phi, mlo, mhi, vmin, cap):
    """Window -> array of per-contract edges actually entered, live-runner order."""
    q = D[(D.px >= plo) & (D.px < phi) & (D.minute >= mlo) & (D.minute <= mhi)
          & (D.vol >= vmin)]
    if len(q) < 50:
        return None, None, None
    e = (q.sort_values("minute", ascending=False)
           .groupby(["wkey", "coin"], as_index=False).first())
    e = (e.sort_values(["wkey", "minute"], ascending=[True, False])
           .groupby("wkey", as_index=False).head(cap))
    groups, tot = [], 0
    for _, g in e.groupby("wkey"):
        px = g.px.to_numpy()
        ed = g.edge.to_numpy()
        keep_e, keep_p, gross = [], [], 0.0
        for p, x in zip(px, ed):
            c = QTY * p
            if gross + c > MAX_GROSS:
                continue
            gross += c
            keep_e.append(x * QTY)
            keep_p.append(c)
        if keep_e:
            groups.append(np.array(keep_e))
            tot += len(keep_e)
    return groups, tot, e


QTY = 30
print(f"bank ${BANK:.2f}   qty {QTY}   gross cap ${MAX_GROSS:.0f}   "
      f"{NPATH:,} paths x {NDAY} days   ruin = equity < ${RUIN:.0f}\n")
print(f"{'configuration':>38} {'ent/day':>8} {'edge':>7} {'win':>6} {'$/day':>8} "
      f"{'P(day<0)':>9} {'5% day':>8} {'medDD':>7} {'95%DD':>7} {'P(ruin)':>8}")

rows = []
for (name, plo, phi, mlo, mhi, vmin, cap) in CANDS:
    groups, tot, e = entries(plo, phi, mlo, mhi, vmin, cap)
    if groups is None:
        print(f"{name:>38}   -- too few entries --")
        continue
    nwin_day = len(groups) / NDAYS_DATA               # windows traded per day
    per_day_n = tot / NDAYS_DATA
    # pad the ragged window groups into a matrix for vectorised resampling
    W = len(groups)
    L = max(len(g) for g in groups)
    M = np.zeros((W, L))
    for i, g in enumerate(groups):
        M[i, :len(g)] = g
    wsum = M.sum(1)                                   # P&L of a whole window
    k = int(round(nwin_day))
    idx = RNG.integers(0, W, size=(NPATH, NDAY, k))
    daily = wsum[idx].sum(2)                          # (paths, days)
    eq = BANK + np.cumsum(daily, axis=1)
    run = np.concatenate([np.full((NPATH, 1), BANK), eq], axis=1)
    dd = np.maximum.accumulate(run, axis=1) - run
    mdd = dd.max(1)
    ruin = (run.min(1) < RUIN).mean()
    rows.append(dict(name=name.strip(), plo=plo, phi=phi, mlo=mlo, mhi=mhi,
                     vmin=vmin, cap=cap, ent=per_day_n,
                     edge=(e.edge.mean() * 100), win=e.won.mean(),
                     day=daily.mean(), pneg=(daily < 0).mean(),
                     p5=np.percentile(daily, 5), mdd=np.median(mdd),
                     dd95=np.percentile(mdd, 95), ruin=ruin))
    r = rows[-1]
    print(f"{name:>38} {r['ent']:>8.1f} {r['edge']:>+7.2f} {r['win']:>6.3f} "
          f"{r['day']:>+8.2f} {r['pneg']:>9.3f} {r['p5']:>+8.2f} "
          f"{r['mdd']:>7.2f} {r['dd95']:>7.2f} {r['ruin']:>8.4f}")

R = pd.DataFrame(rows)
R.to_parquet(OUT / "risk_bootstrap.parquet", index=False)

print("\n=== WHAT THE BOOTSTRAP SAYS THAT THE 5-DAY SAMPLE COULD NOT ===")
c = R.iloc[0]
print(f"  current config loses money on {c.pneg*100:.1f}% of days, and its 5th")
print(f"  percentile day is ${c.p5:+.2f}. The previous script reported")
print(f"  'worst $+234.30, maxDD $0.00' because 5 days is too few to contain one.")

print("\n=== RANKED BY THE STATED GOAL (drawdown and ruin, not raw $/day) ===")
for _, r in R.sort_values(["ruin", "dd95"]).iterrows():
    print(f"  {r['name']:<38} $/day {r.day:>+7.2f}  P(day<0) {r.pneg:.3f}  "
          f"95%DD ${r.dd95:>6.2f}  P(ruin) {r.ruin:.4f}")
