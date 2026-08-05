"""Where else is the edge, and what would raising throughput actually earn?

The live book is running well above the historical prediction (+15.4c/contract
against +9 to +13c forecast) on 56 settlements over 8 hours. That gap is almost
certainly a hot streak, so the right question is NOT "how do we extend the
current rate" but "where does the HISTORICAL edge say we are leaving money on
the table."

Three candidate levers, costed against 4,480 windows / 6.2M trades:
  A  price band     currently 65-85c
  B  time window    currently 8-14 minutes remaining
  C  throughput     currently 2 entries per 15-minute close window

Everything is window-equal-weighted, because contract-weighting over-samples
the high-volume windows and those skew toward maker losses in thin coins.
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent
D = pd.read_parquet(OUT / "discovery_cells.parquet")
D["contracts"] = D.cnt_h / 100.0
D["net_c"] = D.maker_net_micros / 10000.0
D["min_left"] = (D.tbin * 5) / 60.0
D["won_ct"] = D.maker_win_cnt_h / 100.0


def eqw(d):
    """window-equal-weighted cents per contract, plus support."""
    if not len(d):
        return np.nan, 0, 0.0
    per = d.groupby(["coin", "ticker"]).agg(n=("net_c", "sum"), k=("contracts", "sum"))
    per = per[per.k > 0]
    if not len(per):
        return np.nan, 0, 0.0
    return float((per.n / per.k).mean()), len(per), float(d.contracts.sum())


print("=== A. THE GRID: c/contract by price and minutes remaining ===")
print("our live config sits at 65-85c x 8-14min\n")
TB = [(0, 3), (3, 6), (6, 8), (8, 11), (11, 15)]
hdr = "".join(f"{a}-{b}m".rjust(10) for a, b in TB)
print(f"{'price':>9}{hdr}")
for pb in range(10, 20):
    row = ""
    for a, b in TB:
        d = D[(D.pbin == pb) & (D.min_left >= a) & (D.min_left < b)]
        v, nw, ct = eqw(d)
        row += (f"{v:>10.1f}" if np.isfinite(v) and ct > 3000 else f"{'-':>10}")
    print(f"{pb*5:>4}-{pb*5+5:<4}{row}")

print("\n=== B. WHAT OUR CURRENT BOX EARNS vs ALTERNATIVES ===")
print("'windows' = distinct market-windows offering at least one such trade,")
print("which is the real throughput limit.\n")
BOXES = {
    "LIVE NOW  65-85c, 8-14m": (13, 16, 8, 14),
    "wider band 60-90c, 8-14m": (12, 17, 8, 14),
    "tight band 70-80c, 8-14m": (14, 15, 8, 14),
    "wider time 65-85c, 5-15m": (13, 16, 5, 15),
    "wider time 65-85c, 3-15m": (13, 16, 3, 15),
    "both wide  60-90c, 3-15m": (12, 17, 3, 15),
    "early only 65-90c, 11-15m": (13, 17, 11, 15),
}
print(f"{'configuration':28} {'c/contract':>11} {'windows':>9} {'contracts':>11} "
      f"{'rel. throughput':>16}")
base = None
for lab, (p0, p1, t0, t1) in BOXES.items():
    d = D[(D.pbin.between(p0, p1)) & (D.min_left >= t0) & (D.min_left <= t1)]
    v, nw, ct = eqw(d)
    if base is None:
        base = nw
    print(f"{lab:28} {v:>11.2f} {nw:>9,} {ct:>11,.0f} {nw/base:>15.2f}x")

print("\n=== C. THROUGHPUT: how many coins per window actually offer a trade? ===")
cur = D[(D.pbin.between(13, 16)) & (D.min_left >= 8) & (D.min_left <= 14)]
per_win = cur.groupby("ticker").coin.nunique()
print(f"  windows with >=1 eligible coin : {len(per_win):,}")
for k in range(1, 6):
    print(f"  windows with >={k} coins        : {int((per_win >= k).sum()):,}"
          f"   ({(per_win >= k).mean():.3f})")
print(f"\n  mean eligible coins per window  : {per_win.mean():.2f}")
print(f"  we currently cap at 2 per window -> capturing "
      f"{min(per_win.mean(),2)/per_win.mean():.1%} of available flow")

print("\n=== D. DOES EDGE DIFFER BY COIN IN OUR BOX? ===")
print(f"{'coin':>6} {'windows':>9} {'contracts':>11} {'c/contract':>11}")
for c, d in cur.groupby("coin"):
    v, nw, ct = eqw(d)
    print(f"{c:>6} {nw:>9,} {ct:>11,.0f} {v:>11.2f}")

print("\n=== E. THE CORRELATION COST OF MORE COINS PER WINDOW ===")
print("taking N coins in one window is NOT N independent bets.\n")
w = cur.groupby(["ticker", "coin"]).agg(n=("net_c", "sum"), k=("contracts", "sum"))
w = (w.n / w.k).reset_index()
piv = w.pivot(index="ticker", columns="coin", values=0)
cc = piv.corr().values
off = cc[np.triu_indices_from(cc, 1)]
off = off[np.isfinite(off)]
print(f"  mean pairwise correlation of per-window edge across coins: {off.mean():+.3f}")
n_eff = lambda n, r: n / (1 + (n - 1) * max(r, 0))
for n in (2, 3, 4, 5):
    print(f"  taking {n} coins  ->  effective independent bets "
          f"{n_eff(n, off.mean()):.2f}   variance multiple {n/n_eff(n,off.mean()):.2f}x")
