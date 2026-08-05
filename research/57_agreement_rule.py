"""THE AGREEMENT RULE: only add positions that agree in direction.

THE FINDING THAT PROMPTED THIS
  Post-shift, direction-MIXED windows earn -11.08c while direction-ALIGNED
  windows earn +6.51c - a 17.58c gap with a 95% CI of [+16.06, +19.22].

  That inverts what I assumed. I had been treating mixed windows as the safe
  ones because they wipe out less often (1.52% vs 8.44%). They do. They are
  also unprofitable, which matters more: a window that rarely loses everything
  but loses a little constantly is worse than one that occasionally loses
  everything and wins the rest of the time.

WHY IT MIGHT BE REAL
  When every coin's favourite points the same way, the whole complex is
  TRENDING and favourites tend to hold. When they point different ways, the
  complex is CHOPPING - and a choppy market is exactly where a favourite gets
  overturned in the last eight minutes. The Kalshi price of each coin cannot
  see the cross-sectional disagreement; it only sees its own book.

WHY IT IS IMPLEMENTABLE, unlike everything else that worked statistically
  The runner enters sequentially. It cannot know at the first entry whether the
  window will end up mixed. But it CAN apply the rule incrementally:

      take the first qualifying candidate in a window,
      then only add further candidates whose favourite points the SAME way.

  That needs no forecast - only the direction of a position we already hold.
  It costs some entries (the disagreeing ones), so it must clear the usual bar:
  the discarded entries have to be worth approximately nothing.

TESTED ON BOTH REGIMES, because a rule that only works post-shift on three
days of data is a rule fitted to three days of data.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261010)
QTY = 15


def load_pre():
    Y = pd.read_parquet(OUT / "yesno.parquet")
    q = Y[(Y.px >= 0.65) & (Y.px < 0.85) & (Y.minute >= 8) & (Y.minute <= 14)
          & (Y.vol >= 2000)]
    C = (q.sort_values("minute", ascending=False)
           .groupby(["wkey", "coin"], as_index=False).first()).copy()
    C["ord"] = C.coin.map({"BTC": 0, "ETH": 1, "SOL": 2, "XRP": 3,
                           "DOGE": 4}).fillna(9)
    return C.sort_values(["wkey", "ord", "minute"],
                         ascending=[True, True, False])


def load_post():
    D = pd.read_parquet(OUT / "postshift_windows.parquet")
    D["day"] = pd.to_datetime(D.close).dt.strftime("%Y-%m-%d")
    D["minute"] = 0
    D["ord"] = D.coin.map({"BTC": 0, "ETH": 1, "SOL": 2, "XRP": 3,
                           "DOGE": 4, "HYPE": 5}).fillna(9)
    return D.sort_values(["wkey", "ord"])


def apply_rule(C, agree_only, cap=3):
    """Walk each window in the runner's own order, applying the cap and,
    optionally, the agreement rule."""
    keep = []
    for wk, g in C.groupby("wkey", sort=True):
        first_dir = None
        n = 0
        for r in g.itertuples():
            if n >= cap:
                break
            if first_dir is None:
                first_dir = bool(r.maker_yes)
            elif agree_only and bool(r.maker_yes) != first_dir:
                continue                       # disagrees - skip, keep looking
            keep.append(r)
            n += 1
    if not keep:
        return pd.DataFrame()
    return pd.DataFrame([r._asdict() for r in keep])


for lbl, C in (("PRE-SHIFT (Jul 23 - Aug 1)", load_pre()),
               ("POST-SHIFT (Aug 3 - Aug 5)", load_post())):
    print("=" * 76)
    print(lbl)
    print("=" * 76)
    days = sorted(C.day.unique())
    base = apply_rule(C, False)
    agree = apply_rule(C, True)
    for nm, K in (("all (current)", base), ("agreement rule", agree)):
        if not len(K):
            continue
        W = K.groupby("wkey").agg(n=("won", "size"),
                                  k=("won", "sum")).reset_index()
        M = W[W.n >= 2]
        wipe = (M.k == 0).mean() if len(M) else float("nan")
        print(f"  {nm:>16}: entries {len(K):>5} ({len(K)/len(days):>5.1f}/day)  "
              f"win {K.won.mean():.4f}  edge {K.edge.mean()*100:>+6.2f}c  "
              f"wipeout {wipe:.4f}  ${K.edge.sum()*QTY/len(days):>+8.2f}/day")
    # what exactly is being discarded
    kb = set(zip(base.wkey, base.coin))
    ka = set(zip(agree.wkey, agree.coin))
    drop = kb - ka
    if drop:
        dd = base[[(w, c) in drop for w, c in zip(base.wkey, base.coin)]]
        print(f"\n  DISCARDED: {len(dd)} entries ({len(dd)/len(base)*100:.1f}%)  "
              f"win {dd.won.mean():.4f}  edge {dd.edge.mean()*100:+.2f}c")
        print(f"  -> the rule only pays if THIS number is near zero or negative")
        # day-clustered CI on the discarded cohort's edge
        g = {d: gg for d, gg in dd.groupby("day")}
        ds = sorted(g)
        if len(ds) >= 3:
            bs = np.sort(np.array([
                np.concatenate([g[ds[i]].edge.values
                                for i in RNG.integers(0, len(ds), len(ds))]).mean() * 100
                for _ in range(6000)]))
            print(f"     discarded-cohort edge 95% CI "
                  f"[{bs[150]:+.2f}, {bs[5849]:+.2f}]")
    print()

print("=" * 76)
print("WALK-FORWARD DOLLARS - pre-shift, where there are enough days")
print("=" * 76)
C = load_pre()
days = sorted(C.day.unique())
base = apply_rule(C, False)
agree = apply_rule(C, True)
rows = []
for d in days:
    a = base[base.day == d].edge.sum() * QTY
    b = agree[agree.day == d].edge.sum() * QTY
    rows.append((a, b))
    print(f"  {d}  all ${a:>+8.2f}   agreement ${b:>+8.2f}   diff ${b-a:>+7.2f}")
A = np.array([r[0] for r in rows])
B = np.array([r[1] for r in rows])
d_ = B - A
bs = np.sort(np.array([RNG.choice(d_, len(d_), True).mean() for _ in range(8000)]))
print(f"\n  all       ${A.mean():+.2f}/day")
print(f"  agreement ${B.mean():+.2f}/day")
print(f"  difference ${d_.mean():+.2f}/day   better on {(B > A).sum()}/{len(A)} days")
print(f"  95% CI [{bs[200]:+.2f}, {bs[7799]:+.2f}]   "
      f"P(agreement worse) = {(bs < 0).mean():.4f}")
