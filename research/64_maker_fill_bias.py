"""The backtest assumes 100% fill. Real maker fill is biased against winners.

THE DEFECT THIS EXPOSES
  Every backtest in this project prices an entry as if posting the order got
  it. Live, 2020 of 2465 contracts filled - and the shortfall is not random.

  Measured on the live book (120 orders, all resolved):
    markets whose price came back to our bid    filled 51/53 = 0.962
    markets whose price ran away from our bid   filled 53/67 = 0.791
  and the first cohort won 0.5472 while the second won 1.0000.

  Under binary settlement a losing favourite MUST pass back down through our
  price on its way to zero, so 'came back' is near-synonymous with 'lost'. That
  gives a fill model in terms of the outcome directly:

    P(reachable | lost) = 1.000       every loser returns to our price
    P(reachable | won)  = 0.302       53*0.5472 / (53*0.5472 + 67*1.0)

    P(fill | lost) = 0.962
    P(fill | won)  = 0.302*0.962 + 0.698*0.791 = 0.843

  We fill 96.2% of the losers and 84.3% of the winners. The backtest fills
  100% of both. So every historical edge number in this project is optimistic,
  and the question is by how much.

WHY IT MATTERS BEYOND BOOKKEEPING
  If the fill bias is large, the maker edge is smaller than believed, and the
  taker alternative - which fills 100% of both cohorts but pays the spread and
  real fees - stops being obviously worse. The live 2-day taker comparison came
  out +$25.48/day with a CI of [-9.85, +60.82], which settles nothing. Applying
  the measured fill model to 13 days of history is the powered version.

WHAT IS ASSUMED, PLAINLY
  The fill model is estimated from 120 live orders and is held FIXED across
  history. It is two numbers, not a fitted curve, and both come from a regime
  the historical data mostly predates. Sensitivity to both is reported rather
  than hidden.

NOT FOR DEPLOYMENT
  Switching to taker is a strategy change and the standing constraints forbid
  it. This prices the alternative; it does not propose it.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261032)
QTY = 20

P_FILL_LOSE = 0.962
P_FILL_WIN = 0.843
SPREAD_MEAN = 0.0121      # measured live: mean 1.21c, median 1.00c, 80.8% 1-tick
SPREAD_MED = 0.0100


def taker_fee(c, p):
    return math.ceil(0.07 * c * p * (1.0 - p) * 1e4) / 1e4


def load(which):
    if which == "pre":
        D = pd.read_parquet(OUT / "grid.parquet")
        q = D[(D.px >= 0.65) & (D.px < 0.80) & (D.minute >= 8) & (D.minute <= 14)
              & (D.vol >= 2000)]
        e = (q.sort_values("minute", ascending=False)
               .groupby(["wkey", "coin"], as_index=False).first())
        e = (e.sort_values(["wkey", "minute"], ascending=[True, False])
               .groupby("wkey", as_index=False).head(3))
    else:
        D = pd.read_parquet(OUT / "postshift_nofilter.parquet")
        e = D[(D.vol >= 2000) & (D.px >= 0.65) & (D.px < 0.80)].copy()
        e = (e.sort_values(["wkey", "px"], ascending=[True, False])
               .groupby("wkey", as_index=False).head(3))
    return e.copy()


def simulate(E, spread, n_iter=4000):
    """Maker with the measured fill bias vs taker at bid+spread, per day."""
    days = sorted(E.day.unique())
    won = E.won.to_numpy().astype(bool)
    px = E.px.to_numpy()
    dayix = pd.Categorical(E.day, categories=days).codes
    pf = np.where(won, P_FILL_WIN, P_FILL_LOSE)

    # taker is deterministic: fills everything at px+spread, pays fees
    ask = np.minimum(px + spread, 0.99)
    tk_ct = np.full(len(E), QTY)
    tk_pnl = (np.where(won, 1.0, 0.0) - ask) * tk_ct
    tk_fee = np.array([taker_fee(QTY, a) for a in ask])
    tk_day = np.zeros(len(days))
    np.add.at(tk_day, dayix, tk_pnl - tk_fee)

    # maker is stochastic in WHICH orders fill
    mk_naive_day = np.zeros(len(days))
    np.add.at(mk_naive_day, dayix, (np.where(won, 1.0, 0.0) - px) * QTY)

    mk_days = np.zeros((n_iter, len(days)))
    for i in range(n_iter):
        hit = RNG.random(len(E)) < pf
        v = np.where(hit, (np.where(won, 1.0, 0.0) - px) * QTY, 0.0)
        acc = np.zeros(len(days))
        np.add.at(acc, dayix, v)
        mk_days[i] = acc
    return days, mk_naive_day, mk_days, tk_day


for lbl, which in (("PRE-SHIFT", "pre"), ("POST-SHIFT", "post")):
    E = load(which)
    if len(E) < 50:
        print(f"{lbl}: too few entries")
        continue
    days, naive, mk, tk = simulate(E, SPREAD_MEAN)
    nd = len(days)
    print("=" * 78)
    print(f"{lbl}   entries {len(E):,}   days {nd}   "
          f"population edge {E.edge.mean()*100:+.2f}c")
    print("=" * 78)
    mk_mean = mk.mean(axis=0)
    print(f"{'accounting':>36} {'$/day':>10} {'per-contract':>14}")
    print(f"{'BACKTEST (100% fill, the old number)':>36} "
          f"{naive.mean():>10.2f} {naive.sum()/(len(E)*QTY)*100:>+13.2f}c")
    fill_ct = (np.where(E.won.to_numpy().astype(bool), P_FILL_WIN,
                        P_FILL_LOSE)).sum() * QTY
    print(f"{'MAKER with measured fill bias':>36} "
          f"{mk_mean.mean():>10.2f} {mk_mean.sum()/fill_ct*100:>+13.2f}c")
    print(f"{'TAKER (100% fill, pays spread+fees)':>36} "
          f"{tk.mean():>10.2f} {tk.sum()/(len(E)*QTY)*100:>+13.2f}c")
    over = naive.mean() - mk_mean.mean()
    print(f"\n  the 100%-fill assumption OVERSTATES maker by "
          f"${over:+.2f}/day ({over/max(naive.mean(),1e-9)*100:.1f}%)")

    d = tk - mk_mean
    if nd >= 3:
        bs = np.sort(np.array([RNG.choice(d, len(d), True).mean()
                               for _ in range(8000)]))
        print(f"  TAKER minus MAKER: ${d.mean():+.2f}/day   "
              f"95% CI [{bs[200]:+.2f}, {bs[7799]:+.2f}]   "
              f"better {(tk > mk_mean).sum()}/{nd} days   "
              f"P(taker worse) {(bs <= 0).mean():.4f}")
    else:
        print(f"  TAKER minus MAKER: ${d.mean():+.2f}/day  "
              f"better {(tk > mk_mean).sum()}/{nd} days (too few days for a CI)")
    print()

print("=" * 78)
print("SENSITIVITY - the two fill numbers and the spread are estimates")
print("=" * 78)
E = load("pre")
print(f"{'P(fill|win)':>12} {'P(fill|lose)':>13} {'spread':>8} "
      f"{'maker $/day':>13} {'taker $/day':>13} {'taker-maker':>13}")
for pw, pl in ((0.843, 0.962), (0.90, 0.962), (0.95, 0.962),
               (1.00, 1.000), (0.80, 0.98)):
    for sp in (SPREAD_MED, SPREAD_MEAN, 0.02):
        P_FILL_WIN, P_FILL_LOSE = pw, pl
        days, naive, mk, tk = simulate(E, sp, n_iter=1200)
        m, t = mk.mean(axis=0).mean(), tk.mean()
        print(f"{pw:>12.3f} {pl:>13.3f} {sp*100:>7.2f}c "
              f"{m:>13.2f} {t:>13.2f} {t-m:>+13.2f}")
P_FILL_WIN, P_FILL_LOSE = 0.843, 0.962

print("\n" + "=" * 78)
print("READING THIS HONESTLY")
print("=" * 78)
print("  The fill-bias correction is the solid part: it comes from 120 live")
print("  orders and it makes every historical edge number in this project")
print("  smaller. That is worth knowing regardless of what is done about it.")
print()
print("  The taker comparison is the speculative part. It assumes the quoted")
print("  ask has size for a full 20-lot, ignores that taking removes us from")
print("  the liquidity-provision role the longshot premium may be paying for,")
print("  and applies a fill model measured in the CURRENT regime to mostly")
print("  PRE-shift data. It is a reason to keep measuring, not to switch.")
