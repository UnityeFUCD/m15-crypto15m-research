"""Knob optimisation WITH the capital constraint actually enforced.

WHY THE FIRST PASS WAS VOID
  It scored configurations by summing every qualifying entry, which let the
  "optimum" take ~200 entries a day. At 30 contracts and ~67c that is roughly
  $4,000 of capital against a $341 account. The result was a worst day of
  +$447 and zero drawdown - a tell that the simulation had no constraint in
  it at all, because 200 diversified bets a day makes every day positive by
  the law of large numbers.

WHAT THIS DOES INSTEAD
  Walks the windows in chronological order and holds a live capital ledger:
    - a position ties up QTY * price from entry until its window settles
    - an entry is SKIPPED if it would breach MAX_GROSS
    - the pre-submit drawdown envelope is applied exactly as the live runner
      applies it
  So the reported $/day is money this account could actually have made.

Scored on the stated goal: more edge, smaller losses, smaller drawdown, lower
ruin - fitted on the first half of days, reported on the second.
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260805)
D = pd.read_parquet(OUT / "grid.parquet").sort_values(["wkey", "minute"],
                                                      ascending=[True, False])
BANK = 341.51
MAX_GROSS = 110.0
MAX_DD_FRAC = 0.30
days = sorted(D.day.unique())
CUT = days[len(days) // 2]
print(f"rows {len(D):,}  windows {D.wkey.nunique():,}  days {len(days)}  "
      f"bank ${BANK:.2f}  gross cap ${MAX_GROSS:.0f}\n")


def simulate(d, plo, phi, mlo, mhi, vmin, cap, qty):
    """Chronological walk with a real capital ledger. Returns per-day P&L."""
    q = d[(d.px >= plo) & (d.px < phi) & (d.minute >= mlo) & (d.minute <= mhi)
          & (d.vol >= vmin)]
    if len(q) < 40:
        return None, 0
    e = q.groupby(["wkey", "coin"], as_index=False).first()
    e = e.sort_values(["wkey", "minute"], ascending=[True, False])
    taken = []
    cash = BANK
    peak = BANK
    open_pos = []                      # (settle_wkey, cost, edge)
    for wk, g in e.groupby("wkey", sort=True):
        # settle everything from earlier windows
        still = []
        for (swk, cost, ed) in open_pos:
            if swk < wk:
                cash += cost + ed          # ed is ALREADY dollars (edge*qty)
            else:
                still.append((swk, cost, ed))
        open_pos = still
        eq = cash + sum(c for _, c, _ in open_pos)
        peak = max(peak, eq)
        n = 0
        for _, r in g.iterrows():
            if n >= cap:
                break
            cost = qty * r.px
            gross = sum(c for _, c, _ in open_pos)
            if gross + cost > MAX_GROSS:
                continue
            if cash - cost < (1 - MAX_DD_FRAC) * peak:      # live envelope
                continue
            cash -= cost
            open_pos.append((wk, cost, r.edge * qty))
            taken.append((r.day, r.edge * qty))
            n += 1
    if len(taken) < 30:
        return None, 0
    t = pd.DataFrame(taken, columns=["day", "pnl"])
    return t, len(taken)


def score(t, all_days, n):
    per = t.groupby("day").pnl.sum().reindex(all_days, fill_value=0.0)
    v = per.values
    if v.std(ddof=1) == 0:
        return None
    eq = BANK + np.cumsum(v)
    run = np.concatenate([[BANK], eq])
    dd = (np.maximum.accumulate(run) - run).max()
    return dict(n=n, per_day=v.mean(), sharpe=v.mean() / v.std(ddof=1),
                worst=v.min(), maxdd=dd, pos_days=(v > 0).mean())


GRID = list(itertools.product(
    [(0.55, 0.90), (0.60, 0.90), (0.65, 0.85), (0.60, 0.80), (0.70, 0.90)],
    [(8, 14), (5, 14), (3, 14), (11, 14), (8, 11)],
    [0, 500, 1000, 2000, 4000],
    [1, 2, 3],
    [15, 30]))
print(f"configurations: {len(GRID):,}\n")
fit_days = [x for x in days if x < CUT]
test_days = [x for x in days if x >= CUT]
Dfit, Dtest = D[D.day < CUT], D[D.day >= CUT]

res = []
for (plo, phi), (mlo, mhi), vmin, cap, qty in GRID:
    tf, nf = simulate(Dfit, plo, phi, mlo, mhi, vmin, cap, qty)
    if tf is None:
        continue
    sf = score(tf, fit_days, nf)
    tt, nt = simulate(Dtest, plo, phi, mlo, mhi, vmin, cap, qty)
    if tt is None:
        continue
    st = score(tt, test_days, nt)
    if sf is None or st is None:
        continue
    res.append(dict(band=f"{plo*100:.0f}-{phi*100:.0f}", tw=f"{mlo}-{mhi}",
                    vmin=vmin, cap=cap, qty=qty,
                    fit_day=sf["per_day"], fit_sharpe=sf["sharpe"],
                    day=st["per_day"], sharpe=st["sharpe"], n=st["n"],
                    worst=st["worst"], maxdd=st["maxdd"], pos=st["pos_days"]))
R = pd.DataFrame(res)
R.to_parquet(OUT / "knob_constrained.parquet", index=False)
print(f"evaluated {len(R):,}\n")

cur = R[(R.band == "65-85") & (R.tw == "8-14") & (R.vmin == 500)
        & (R.cap == 2) & (R.qty == 30)]
if len(cur):
    c = cur.iloc[0]
    print("=== CURRENT LIVE CONFIG (65-85, 8-14, vmin 500, cap 2, qty 30) ===")
    print(f"  fit  ${c.fit_day:+7.2f}/day   TEST ${c.day:+7.2f}/day  "
          f"Sharpe {c.sharpe:+.2f}  n {int(c.n)}  worst ${c.worst:+.2f}  "
          f"maxDD ${c.maxdd:.2f}  positive days {c.pos:.2f}")

print("\n=== TOP 12 BY OUT-OF-SAMPLE SHARPE ===")
print(f"{'band':>7} {'time':>6} {'vmin':>5} {'cap':>4} {'qty':>4} {'fit $/d':>8} "
      f"{'TEST $/d':>9} {'Sharpe':>7} {'n':>4} {'worst':>8} {'maxDD':>7} {'pos':>5}")
for _, r in R.sort_values("sharpe", ascending=False).head(12).iterrows():
    print(f"{r.band:>7} {r.tw:>6} {int(r.vmin):>5} {int(r.cap):>4} {int(r.qty):>4} "
          f"{r.fit_day:>+8.2f} {r.day:>+9.2f} {r.sharpe:>+7.2f} {int(r.n):>4} "
          f"{r.worst:>+8.2f} {r.maxdd:>7.2f} {r.pos:>5.2f}")

print("\n=== TOP 12 BY OUT-OF-SAMPLE $/DAY ===")
print(f"{'band':>7} {'time':>6} {'vmin':>5} {'cap':>4} {'qty':>4} {'TEST $/d':>9} "
      f"{'Sharpe':>7} {'n':>4} {'worst':>8} {'maxDD':>7}")
for _, r in R.sort_values("day", ascending=False).head(12).iterrows():
    print(f"{r.band:>7} {r.tw:>6} {int(r.vmin):>5} {int(r.cap):>4} {int(r.qty):>4} "
          f"{r.day:>+9.2f} {r.sharpe:>+7.2f} {int(r.n):>4} {r.worst:>+8.2f} "
          f"{r.maxdd:>7.2f}")

print(f"\n  corr(fit Sharpe, test Sharpe) = {R.fit_sharpe.corr(R.sharpe):+.4f}")
print(f"  corr(fit $/day,  test $/day)  = {R.fit_day.corr(R.day):+.4f}")

print("\n=== MARGINAL EFFECT OF EACH KNOB (out of sample) ===")
for k in ("vmin", "cap", "tw", "band", "qty"):
    g = R.groupby(k).agg(sharpe=("sharpe", "mean"), day=("day", "mean"),
                         worst=("worst", "mean"), dd=("maxdd", "mean"))
    print(f"\n  {k}:")
    for i, r in g.sort_values("sharpe", ascending=False).iterrows():
        print(f"    {str(i):>7}  Sharpe {r.sharpe:+.2f}  ${r.day:+7.2f}/day  "
              f"worst ${r.worst:+7.2f}  maxDD ${r.dd:6.2f}")
