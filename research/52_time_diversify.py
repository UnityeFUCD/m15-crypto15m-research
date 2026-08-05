"""MATCHED-CONTRACT TIME DIVERSIFICATION: the last structural lever.

THE IDEA
  Within-window correlation is real and unavoidable: P(all 3 lose) runs 6.2x
  independence, direction mixing is available in only 16.5% of windows, and no
  coin pair is safer than another (joint-loss ratios all 2.75-3.14).

  BUT windows are SERIALLY INDEPENDENT - autocorrelation of window P&L is null
  at every lag, and the wipeout event itself shows no time clustering. So
  correlation lives ENTIRELY inside a window and vanishes between them.

  That means the same total exposure can be arranged two ways:

     A  cap 3, qty 15  ->  121.6 entries/day x 15 =  1,824 contracts/day
                            stacked 3-deep into correlated baskets
     B  cap 1, qty 27  ->   67.6 entries/day x 27 =  1,825 contracts/day
                            one position per window, ZERO within-window
                            correlation

  Same contracts. Same capital. Same edge mechanism. Completely different
  correlation structure. If B has materially lower variance for the same mean,
  the correlated-loss problem is solved without giving up profit - which is
  exactly the thing every other approach failed to do.

THE COST THAT MIGHT KILL IT
  Rank-1 entries are not the best ones. Measured: rank 1 earns +10.20c, rank 2
  +11.28c, rank 3 +16.65c - the LATER entries in a window are better, because a
  3rd entry only exists when three coins independently clear the volume floor,
  which selects high-agreement windows. Dropping to cap 1 throws those away.

  So B trades a worse average edge for a better correlation structure. Whether
  that is a good deal is an empirical question, and this measures it at
  MATCHED contracts per day rather than at matched entry counts.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260926)
BANK = 450.53
NPATH, NDAY = 20000, 30
STOP_FRAC = 0.20
D = pd.read_parquet(OUT / "grid.parquet")
days = sorted(D.day.unique())

q = D[(D.px >= 0.65) & (D.px < 0.80) & (D.minute >= 8) & (D.minute <= 14)
      & (D.vol >= 2000)]                     # 65-80c, the live band now
E = (q.sort_values("minute", ascending=False)
       .groupby(["wkey", "coin"], as_index=False).first())
E = (E.sort_values(["wkey", "minute"], ascending=[True, False])).copy()
E["rank"] = E.groupby("wkey").cumcount() + 1


def cfg(cap, qty):
    """Window-level P&L in dollars for a (cap, qty) configuration."""
    K = E[E["rank"] <= cap].copy()
    K["ec"] = np.where(K.won == 1, (1 - K.px), -K.px)      # per contract
    W = K.groupby("wkey").ec.sum() * qty
    return W.to_numpy(), len(K), K.ec.mean() * 100


print(f"{'config':>18} {'entries/day':>12} {'contracts/day':>14} "
      f"{'edge/ct':>9} {'mean $/day':>11} {'sd $/day':>10} {'worst window':>13}")
CFGS = [(3, 15, "cap3 x qty15"), (2, 22, "cap2 x qty22"),
        (1, 27, "cap1 x qty27"), (1, 45, "cap1 x qty45"),
        (3, 30, "cap3 x qty30 (old)")]
store = {}
for cap, qty, lbl in CFGS:
    wp, n, ec = cfg(cap, qty)
    k = max(int(round(len(wp) / len(days))), 1)
    daily = wp.sum() / len(days)
    sd = np.std(wp) * np.sqrt(k)
    store[lbl] = (wp, k)
    print(f"{lbl:>18} {n/len(days):>12.1f} {n/len(days)*qty:>14,.0f} "
          f"{ec:>+9.2f} {daily:>+11.2f} {sd:>10.2f} {wp.min():>+13.2f}")

print("\n=== MATCHED CONTRACTS: cap3xqty15 vs cap1xqty27 (both ~1,825/day) ===")
a_wp, a_k = store["cap3 x qty15"]
b_wp, b_k = store["cap1 x qty27"]
print(f"  A cap3 x qty15 : mean ${a_wp.sum()/len(days):+.2f}/day  "
      f"window sd ${np.std(a_wp):.2f}  worst ${a_wp.min():+.2f}")
print(f"  B cap1 x qty27 : mean ${b_wp.sum()/len(days):+.2f}/day  "
      f"window sd ${np.std(b_wp):.2f}  worst ${b_wp.min():+.2f}")
print(f"  daily sd:  A ${np.std(a_wp)*np.sqrt(a_k):.2f}   "
      f"B ${np.std(b_wp)*np.sqrt(b_k):.2f}")


def sim(wp, k, qty_note=""):
    idx = RNG.integers(0, len(wp), size=(NPATH, NDAY * k))
    draws = wp[idx]
    eq = np.full(NPATH, BANK)
    peak = np.full(NPATH, BANK)
    dead = np.zeros(NPATH, dtype=bool)
    for t in range(draws.shape[1]):
        live = ~dead
        eq[live] += draws[live, t]
        peak = np.maximum(peak, eq)
        dead |= (eq < (1 - STOP_FRAC) * peak)
    return eq, dead


print(f"\n=== 30-DAY OUTCOME WITH THE LIVE 20% STOP ACTIVE ===")
print(f"{'config':>18} {'mean':>10} {'median':>10} {'5th pct':>10} "
      f"{'P(stopped)':>11} {'mean/sd':>9}")
for lbl in ("cap3 x qty15", "cap2 x qty22", "cap1 x qty27", "cap1 x qty45",
            "cap3 x qty30 (old)"):
    wp, k = store[lbl]
    eq, dead = sim(wp, k)
    daily_mu = wp.sum() / len(days)
    daily_sd = np.std(wp) * np.sqrt(k)
    print(f"{lbl:>18} {eq.mean():>10,.0f} {np.median(eq):>10,.0f} "
          f"{np.percentile(eq, 5):>10,.0f} {dead.mean():>11.4f} "
          f"{daily_mu/daily_sd:>9.3f}")

print("\n=== THE THREE CONDITIONS, A vs B ===")
ea, da = sim(a_wp, a_k)
eb, db = sim(b_wp, b_k)
print(f"  contracts/day     : matched by construction (~1,825)")
print(f"  mean terminal     : A ${ea.mean():,.0f}   B ${eb.mean():,.0f}   "
      f"({(eb.mean()/ea.mean()-1)*100:+.1f}%)")
print(f"  5th percentile    : A ${np.percentile(ea,5):,.0f}   "
      f"B ${np.percentile(eb,5):,.0f}")
print(f"  P(stopped)        : A {da.mean():.4f}   B {db.mean():.4f}")
print(f"  worst window      : A ${a_wp.min():+.2f}   B ${b_wp.min():+.2f}   "
      f"({(1-abs(b_wp.min())/abs(a_wp.min()))*100:+.0f}% smaller)")
print(f"  daily Sharpe      : A {(a_wp.sum()/len(days))/(np.std(a_wp)*np.sqrt(a_k)):.3f}"
      f"   B {(b_wp.sum()/len(days))/(np.std(b_wp)*np.sqrt(b_k)):.3f}")

print("\n=== WHERE DOES THE WITHIN-WINDOW CORRELATION ACTUALLY COST? ===")
K3 = E[E["rank"] <= 3].copy()
K3["ec"] = np.where(K3.won == 1, (1 - K3.px), -K3.px)
per_w = K3.groupby("wkey").agg(n=("ec", "size"), s=("ec", "sum"))
for n_ in (1, 2, 3):
    g = per_w[per_w.n == n_]
    if len(g) < 20:
        continue
    indep_sd = np.sqrt(n_) * K3.ec.std()
    print(f"  windows with {n_} position(s): n={len(g):>3}  "
          f"actual sd {g.s.std():.4f}   if independent {indep_sd:.4f}   "
          f"ratio {g.s.std()/indep_sd:.2f}x")
print("\n  A ratio above 1 is the correlation penalty - the extra variance we")
print("  carry purely because positions share a window.")
