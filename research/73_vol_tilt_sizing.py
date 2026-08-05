"""Size by PRIOR volatility instead of filtering on it.

WHY SIZING IS THE RIGHT RESPONSE HERE AND FILTERING IS NOT
  Prior-window volatility predicts edge - corr -0.0680, CI [-0.1086,-0.0194],
  P(>=0)=0.0045 - and it is strictly prior information, so unlike the
  contemporaneous measure it is not the payoff identity in disguise.

  But every filter built on it LOSES money:
      skip top 10% prior-vol   -$22.96/day   discarded cohort +10.83c
      skip top 20%             -$37.36/day   discarded cohort  +8.85c
      skip top 33%             -$58.82/day   discarded cohort  +8.43c

  The discards are POSITIVE. High-volatility entries are less profitable, not
  unprofitable, so removing them removes profit. That is the identical failure
  that killed the intraday hours filter yesterday.

  Tilting has no such defect. It keeps every entry and only moves contracts
  from lower-edge conditions to higher-edge ones, holding the TOTAL contract
  count fixed. Nothing profitable is discarded and gross exposure is unchanged.

WHY THIS IS A BETTER CANDIDATE THAN THE HOUR TILT
  The hour tilt was worth +$25.57/day pre-shift and was NOT deployed, because
  its live confirmation rested on a single night and the curve was fitted on
  coins whose edge has since gone negative. It is a calendar with no mechanism.

  Prior volatility is different:
    - it has a mechanism. Settlement is A1 >= A0; the favourite's standing lead
      survives or does not depending on how much the underlying still moves.
    - it is a continuous market observable, not a fitted 24-point curve, so
      there is far less to overfit - one number per entry.
    - it does not depend on the coin mix, which is what broke the rank result.

WHAT WOULD STILL KILL IT
  1. contract drift. The last sizing model deployed was reversed partly
     because clipping broke neutrality once distributions moved. Neutrality is
     verified numerically here, not assumed.
  2. the signal is WEAK (-0.068). A weak signal tilted aggressively is mostly
     noise amplification. The strength sweep will show where that starts.
  3. walk-forward only. Percentiles come from prior days, never the day scored.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261052)
QTY, QMAX, QMIN = 20, 30, 10

Y = pd.read_parquet(OUT / "yesno.parquet")
Y["yes_px"] = np.where(Y.maker_yes, Y.px, 1.0 - Y.px)
V = (Y.groupby(["wkey", "coin"])
       .agg(sd=("yes_px", "std"), k=("yes_px", "size"), day=("day", "first"))
       .reset_index())
V = V[V.k >= 4].dropna(subset=["sd"]).sort_values(["coin", "wkey"])
for lag in (1, 2, 3):
    V[f"l{lag}"] = V.groupby("coin").sd.shift(lag)
V["prior"] = V[["l1", "l2", "l3"]].mean(axis=1)

q = Y[(Y.px >= 0.65) & (Y.px < 0.80) & (Y.minute >= 8) & (Y.minute <= 14)
      & (Y.vol >= 2000)]
E = (q.sort_values("minute", ascending=False)
       .groupby(["wkey", "coin"], as_index=False).first())
E = E.merge(V[["wkey", "coin", "prior"]], on=["wkey", "coin"], how="inner")
E = (E.sort_values(["wkey", "minute"], ascending=[True, False])
       .groupby("wkey", as_index=False).head(3))
E = E.dropna(subset=["prior"]).copy()
days = sorted(E.day.unique())
print(f"entries {len(E):,}   days {len(days)}   "
      f"edge {E.edge.mean()*100:+.2f}c\n")


def tilt(prior, mu, sd, strength):
    """Mean-preserving tilt: LOW prior vol gets MORE size."""
    z = np.clip((prior - mu) / (sd + 1e-9), -2.5, 2.5)
    return np.clip(QTY * (1.0 - strength * z), QMIN, QMAX)


print("=" * 78)
print("WALK-FORWARD: mean and sd of prior-vol taken from PRIOR DAYS ONLY")
print("=" * 78)
flat = E.groupby("day").edge.sum().reindex(days, fill_value=0) * QTY
print(f"{'rule':>26} {'$/day':>10} {'contracts/day':>15} {'drift':>8} "
      f"{'vs flat qty20':>22}")
print(f"{'flat qty 20':>26} {flat.mean():>10.2f} "
      f"{len(E)*QTY/len(days):>15.0f} {'-':>8} {'-':>22}")
best = None
for strength in (0.10, 0.15, 0.20, 0.30, 0.40):
    pnl, ct, cl = [], [], 0
    for i in range(2, len(days)):
        tr = E[E.day.isin(days[:i])]
        te = E[E.day == days[i]]
        if len(tr) < 60 or not len(te):
            continue
        qv = tilt(te.prior.to_numpy(), tr.prior.mean(), tr.prior.std(), strength)
        cl += int(((qv >= QMAX - 1e-9) | (qv <= QMIN + 1e-9)).sum())
        pnl.append(float((te.edge.to_numpy() * qv).sum()))
        ct.append(float(qv.sum()))
    if not pnl:
        continue
    base = flat.reindex(days[2:]).values[:len(pnl)]
    bct = np.array([len(E[E.day == d]) * QTY for d in days[2:]][:len(pnl)])
    d = np.array(pnl) - base
    bs = np.sort(np.array([RNG.choice(d, len(d), True).mean()
                           for _ in range(8000)]))
    drift = (np.mean(ct) / np.mean(bct) - 1) * 100
    print(f"{'vol tilt ' + f'{strength:.2f}':>26} {np.mean(pnl):>10.2f} "
          f"{np.mean(ct):>15.0f} {drift:>+7.1f}% "
          f"{d.mean():>+7.2f} [{bs[200]:+.0f},{bs[7799]:+.0f}] "
          f"P<=0 {(bs <= 0).mean():.3f}")
    if cl:
        print(f"{'':>26}  clipped {cl} entries")
    if best is None or d.mean() > best[1]:
        best = (strength, d.mean(), (bs <= 0).mean())

print("\n" + "=" * 78)
print("SANITY - IS THE TILT DOING WHAT IT SHOULD?")
print("=" * 78)
mu, sd = E.prior.mean(), E.prior.std()
for s in (0.15, 0.30):
    qv = tilt(E.prior.to_numpy(), mu, sd, s)
    print(f"  strength {s:.2f}: mean qty {qv.mean():>6.2f}  "
          f"min {qv.min():>5.1f}  max {qv.max():>5.1f}  "
          f"clipped {int(((qv>=QMAX-1e-9)|(qv<=QMIN+1e-9)).sum())}")
    lo = E.prior <= E.prior.quantile(0.2)
    hi = E.prior >= E.prior.quantile(0.8)
    print(f"{'':>17} avg qty on LOW-vol entries  {qv[lo.to_numpy()].mean():.2f} "
          f"(edge {E[lo].edge.mean()*100:+.2f}c)")
    print(f"{'':>17} avg qty on HIGH-vol entries {qv[hi.to_numpy()].mean():.2f} "
          f"(edge {E[hi].edge.mean()*100:+.2f}c)")

print("\n" + "=" * 78)
print("PLACEBO - tilt on a SHUFFLED signal, which should be worth nothing")
print("=" * 78)
res = []
for _ in range(200):
    sh = E.copy()
    sh["prior"] = RNG.permutation(sh.prior.values)
    pnl = []
    for i in range(2, len(days)):
        tr = sh[sh.day.isin(days[:i])]
        te = sh[sh.day == days[i]]
        if len(tr) < 60 or not len(te):
            continue
        qv = tilt(te.prior.to_numpy(), tr.prior.mean(), tr.prior.std(), 0.20)
        pnl.append(float((te.edge.to_numpy() * qv).sum()))
    if pnl:
        base = flat.reindex(days[2:]).values[:len(pnl)]
        res.append(np.mean(np.array(pnl) - base))
res = np.sort(np.array(res))
print(f"  placebo tilt at strength 0.20: mean ${res.mean():+.2f}/day   "
      f"95% range [{res[5]:+.2f}, {res[194]:+.2f}]")
print("  the real signal must beat this range to mean anything")

print("\n" + "=" * 78)
print("VERDICT")
print("=" * 78)
if best:
    print(f"  best strength {best[0]:.2f}: ${best[1]:+.2f}/day, P(<=0) {best[2]:.3f}")
    print(f"  placebo 95th percentile: ${res[194]:+.2f}/day")
    if best[1] > res[194] and best[2] < 0.05:
        print("\n  Beats the placebo and clears 95%. This is a genuine candidate -")
        print("  mechanism, prior information, contract-neutral, walk-forward.")
        print("  Still needs post-shift verification before deployment.")
    else:
        print("\n  Does NOT clear both bars. A -0.068 signal is too weak to tilt")
        print("  on: the gain is inside what a shuffled signal produces.")
