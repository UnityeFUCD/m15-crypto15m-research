"""Backtest the claim: "these signals become valuable the moment size can vary."

THE CLAIM BEING TESTED (mine, from the previous turn)
  Best cell (others_px HI + frac_long HI): win 0.900, +16.1c -> deserves >30.
  Worst cell (both LO):                    win 0.789,  +4.9c -> deserves <30.
  I asserted this converts three otherwise-unusable signals into money. It is
  an assertion until it survives a walk-forward test with the real capital
  constraint and an honest drawdown measure, so that is what this does.

DESIGN
  - thresholds for BOTH signals are learned on the first half of days only and
    applied unseen to the second half. No cell is sized using its own outcome.
  - the $110 gross cap is enforced chronologically. This matters more as size
    rises: at qty 50 and 73c a single position is $36.50, so three of them no
    longer fit and the cap starts refusing exactly the entries we sized up.
  - the cross-coin signal needs 3+ coins quoting simultaneously and covers 826
    of 1,216 entries. Entries without it keep the default 30 rather than being
    dropped, so coverage is not silently confused with selection.
  - drawdown comes from resampling whole WINDOWS (positions inside a window
    fail together at 6.2x independence) and is reported at BOTH the backtest
    win rate and the stressed live win rate, because the two disagree by 24x
    and the answer is meaningless without saying which regime is assumed.

THE RISK THAT MOTIVATES THIS SCRIPT
  The best cell is defined partly by the whole market being decisive - which is
  also when several coins qualify at once. So sizing up there may concentrate
  capital exactly where simultaneous positions cluster. That is measured, not
  assumed.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260822)
D = pd.read_parquet(OUT / "grid.parquet")
MAX_GROSS, BANK = 110.0, 394.91
NPATH, NDAY, NSEED = 4000, 30, 8
BT_WIN, LIVE_WIN = 0.8462, 0.7788
DELTA = BT_WIN - LIVE_WIN

G = D.groupby(["wkey", "minute"]).agg(s_px=("px", "sum"),
                                      n=("px", "size")).reset_index()
D2 = D.merge(G, on=["wkey", "minute"], how="left")
D2["others_px"] = np.where(D2.n >= 3, (D2.s_px - D2.px) / (D2.n - 1), np.nan)
q = D2[(D2.px >= 0.65) & (D2.px < 0.85) & (D2.minute >= 8) & (D2.minute <= 14)
       & (D2.vol >= 2000)]
E = (q.sort_values("minute", ascending=False)
       .groupby(["wkey", "coin"], as_index=False).first()).copy()
E = (E.sort_values(["wkey", "minute"], ascending=[True, False])
       .groupby("wkey", as_index=False).head(3)).copy()
E["pb"] = (E.px * 20).astype(int)
days = sorted(E.day.unique())
CUT = days[len(days) // 2]
fit, test = E[E.day < CUT], E[E.day >= CUT].copy()
print(f"entries {len(E):,}  fit days {fit.day.nunique()}  "
      f"test days {test.day.nunique()}")
print(f"cross-coin signal available on {E.others_px.notna().mean()*100:.1f}% "
      f"of entries\n")

# thresholds from the FIT half only
to = fit.groupby(["coin", "pb"]).others_px.median()
tf = fit.groupby(["coin", "pb"]).frac_long.median()
test["to"] = [to.get(k, np.nan) for k in zip(test.coin, test.pb)]
test["tf"] = [tf.get(k, np.nan) for k in zip(test.coin, test.pb)]
hi_o = test.others_px > test.to
hi_f = test.frac_long > test.tf
known = test.others_px.notna() & test.to.notna() & test.tf.notna()
test["cell"] = np.where(~known, "unknown",
                np.where(hi_o & hi_f, "best",
                np.where(~hi_o & ~hi_f, "worst", "mid")))
print("=== OUT-OF-SAMPLE CELL BEHAVIOUR (thresholds never saw these days) ===")
print(f"{'cell':>9} {'n':>5} {'share':>7} {'win':>8} {'edge/ct':>9} "
      f"{'pos/window':>11}")
for c in ("best", "mid", "worst", "unknown"):
    g = test[test.cell == c]
    if not len(g):
        continue
    ppw = g.groupby("wkey").size().mean()
    print(f"{c:>9} {len(g):>5} {len(g)/len(test):>7.3f} {g.won.mean():>8.4f} "
          f"{g.edge.mean()*100:>+9.2f} {ppw:>11.2f}")
print("\n  'pos/window' is the concentration check: if the best cell also has")
print("  the most simultaneous positions, sizing it up concentrates risk.")


def simulate(qhi, qlo, qmid=30, qunk=30, stress=0.0, seed=0):
    rng = np.random.default_rng(9000 + seed)
    t = test.copy()
    t["qty"] = t.cell.map({"best": qhi, "worst": qlo, "mid": qmid,
                           "unknown": qunk}).astype(float)
    keep = []
    for _, g in t.groupby("wkey"):
        gross = 0.0
        for r in g.itertuples():
            c = r.qty * r.px
            if r.qty <= 0 or gross + c > MAX_GROSS:
                continue
            gross += c
            keep.append((r.wkey, r.day, r.px, r.won, r.qty))
    K = pd.DataFrame(keep, columns=["wkey", "day", "px", "won", "qty"])
    if len(K) < 40:
        return None
    if stress > 0:                       # flip whole windows, clustered
        need = int(round(stress * len(K)))
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
    K["pnl"] = np.where(K.won == 1, (1 - K.px) * K.qty, -K.px * K.qty)
    wp = K.groupby("wkey").pnl.sum().to_numpy()
    k = max(int(round(K.wkey.nunique() / test.day.nunique())), 1)
    idx = rng.integers(0, len(wp), size=(NPATH, NDAY, k))
    daily = wp[idx].sum(2)
    eq = BANK + np.cumsum(daily, axis=1)
    run = np.concatenate([np.full((NPATH, 1), BANK), eq], axis=1)
    dd = (np.maximum.accumulate(run, axis=1) - run).max(1)
    return dict(day=daily.mean(), sd=daily.std(),
                edge=K.pnl.sum() / K.qty.sum() * 100,
                contracts=K.qty.sum(), n=len(K),
                maxpos=(K.qty * K.px).max(),
                dd95=np.percentile(dd, 95),
                ruin=(run.min(1) < 0.5 * BANK).mean())


def avg(qhi, qlo, stress):
    rs = [simulate(qhi, qlo, stress=stress, seed=s) for s in range(NSEED)]
    rs = [r for r in rs if r]
    if not rs:
        return None
    return {k: np.mean([r[k] for r in rs]) for k in rs[0]}


CFG = [(30, 30, "FLAT 30 (current)"), (40, 30, "40 best / 30"),
       (40, 20, "40 best / 20 worst"), (40, 0, "40 best / skip worst"),
       (50, 20, "50 best / 20 worst"), (50, 0, "50 best / skip worst"),
       (30, 15, "30 / 15 worst  (no size-up)"),
       (30, 0, "30 / skip worst (no size-up)")]

for lbl, stress in (("BACKTEST REGIME (win 0.846)", 0.0),
                    ("STRESSED LIVE REGIME (win 0.779)", DELTA)):
    print(f"\n=== {lbl} ===")
    print(f"{'sizing':>28} {'$/day':>9} {'edge':>7} {'sd':>8} {'Sharpe':>7} "
          f"{'95%DD':>8} {'P(ruin)':>9} {'max pos':>8}")
    base = None
    for qhi, qlo, name in CFG:
        r = avg(qhi, qlo, stress)
        if r is None:
            continue
        if base is None:
            base = r
        print(f"{name:>28} {r['day']:>+9.2f} {r['edge']:>+7.2f} {r['sd']:>8.2f} "
              f"{r['day']/r['sd']:>7.2f} {r['dd95']:>8.2f} {r['ruin']:>9.4f} "
              f"${r['maxpos']:>7.2f}")

print("\n=== DOES SIZING UP BEAT SIMPLY RAISING FLAT SIZE? ===")
print("   the fair comparison: a signal is only worth having if targeted size")
print("   beats untargeted size at the SAME total contract count.\n")
for stress, lbl in ((0.0, "backtest"), (DELTA, "stressed")):
    a = avg(40, 30, stress)
    ref = simulate(30, 30, stress=stress, seed=0)
    flat_equiv = a["contracts"] / ref["contracts"] * 30
    b = avg(int(round(flat_equiv)), int(round(flat_equiv)), stress)
    print(f"  {lbl:>9}: targeted 40/30 -> ${a['day']:+.2f}/day, "
          f"Sharpe {a['day']/a['sd']:.2f}, {a['contracts']:,.0f} contracts")
    print(f"  {'':>9}  flat {flat_equiv:.0f}    -> ${b['day']:+.2f}/day, "
          f"Sharpe {b['day']/b['sd']:.2f}, {b['contracts']:,.0f} contracts")
    print(f"  {'':>9}  targeting is worth "
          f"${a['day']-b['day']:+.2f}/day and "
          f"{a['day']/a['sd']-b['day']/b['sd']:+.3f} Sharpe\n")
