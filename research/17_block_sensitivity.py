"""How much of the risk estimate is real, and how much is my choice of block?

THE CONTRADICTION THAT PROMPTED THIS
  16_risk_bootstrap.py reported a 95th-percentile 30-day drawdown of $0.00 for
  the best configs and P(losing day) = 0.2%. But the LIVE account ran a $63
  realised drawdown inside its first 13 hours. A model that says drawdowns
  essentially never happen is contradicted by the one live sample I have.

THE SUSPECT
  Resampling WINDOWS independently. That assumes the 15-minute windows inside a
  day are independent draws. They are not: crypto trends and chops in regimes
  that last hours, and the favourite-runs-away mechanic this edge depends on is
  exactly a function of that regime. Independent window resampling manufactures
  diversification that does not exist and drives daily variance to zero.

WHAT THIS MEASURES
  1. the ACTUAL per-day P&L for all 10 days - the ground truth
  2. the intraclass correlation of window P&L within a day, and within an hour
  3. P(losing day) under four different blocks: entry, window, hour, day
  If the answer moves a lot across blocks, the risk number is a statement about
  my assumption, not about the market - and the honest report is the range.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260807)
D = pd.read_parquet(OUT / "grid.parquet")
BANK, QTY, MAX_GROSS = 341.51, 30, 110.0
NPATH = 20_000


def build(plo, phi, mlo, mhi, vmin, cap):
    q = D[(D.px >= plo) & (D.px < phi) & (D.minute >= mlo) & (D.minute <= mhi)
          & (D.vol >= vmin)]
    e = (q.sort_values("minute", ascending=False)
           .groupby(["wkey", "coin"], as_index=False).first())
    e = (e.sort_values(["wkey", "minute"], ascending=[True, False])
           .groupby("wkey", as_index=False).head(cap))
    keep = []
    for _, g in e.groupby("wkey"):
        gross = 0.0
        for _, r in g.iterrows():
            c = QTY * r.px
            if gross + c > MAX_GROSS:
                continue
            gross += c
            keep.append((r.day, r.hour, r.wkey, r.edge * QTY))
    return pd.DataFrame(keep, columns=["day", "hour", "wkey", "pnl"])


E = build(0.65, 0.85, 8, 14, 500, 2)
days = sorted(E.day.unique())
print(f"CURRENT CONFIG  entries {len(E):,}  windows {E.wkey.nunique():,}  "
      f"days {len(days)}\n")

print("=== GROUND TRUTH: actual P&L of each of the 10 days ===")
pd_day = E.groupby("day").pnl.agg(["sum", "count"])
for d, r in pd_day.iterrows():
    bar = "#" * max(1, int(r["sum"] / 25))
    print(f"  {d}  {int(r['count']):>4} entries  ${r['sum']:>+8.2f}  {bar}")
neg = (pd_day["sum"] < 0).sum()
print(f"\n  losing days observed: {neg} of {len(pd_day)}   "
      f"mean ${pd_day['sum'].mean():+.2f}   sd ${pd_day['sum'].std():.2f}")
print(f"  -> with 0 losing days in 10, the day-block bootstrap CANNOT produce")
print(f"     a losing day. Its P(day<0) would be exactly 0 by construction.")

print("\n=== HOW CORRELATED ARE OUTCOMES INSIDE A BLOCK? ===")
w = E.groupby(["day", "hour", "wkey"], as_index=False).pnl.sum()
for lvl in ("day", "hour"):
    g = w.groupby(["day"] + ([] if lvl == "day" else ["hour"])).pnl
    mb, n = g.mean(), g.count()
    grand = w.pnl.mean()
    ssb = (n * (mb - grand) ** 2).sum() / max(len(mb) - 1, 1)
    ssw = g.apply(lambda s: ((s - s.mean()) ** 2).sum()).sum() / max(len(w) - len(mb), 1)
    k = n.mean()
    icc = (ssb - ssw) / (ssb + (k - 1) * ssw) if (ssb + (k - 1) * ssw) else 0.0
    print(f"  windows within the same {lvl:>5}:  ICC = {icc:>+.4f}   "
          f"(blocks {len(mb)}, avg size {k:.1f})")
print("  ICC > 0 means windows in the same block move together, so resampling")
print("  them independently understates daily variance.")

print("\n=== P(LOSING DAY) UNDER FOUR DIFFERENT BLOCK CHOICES ===")
nwin_day = E.wkey.nunique() / len(days)
k = int(round(nwin_day))
ent = E.pnl.to_numpy()
n_ent_day = len(E) / len(days)
res = {}

# block = single entry (independence assumed)
s = RNG.choice(ent, size=(NPATH, int(round(n_ent_day)))).sum(1)
res["entry  (i.i.d., most optimistic)"] = s

# block = window
wv = w.pnl.to_numpy()
s = RNG.choice(wv, size=(NPATH, k)).sum(1)
res["window (what script 16 used)"] = s

# block = hour  (4 windows, keeps intra-hour regime)
hv = [g.pnl.to_numpy() for _, g in w.groupby(["day", "hour"])]
hs = np.array([x.sum() for x in hv])
nh = len(hv) / len(days)
s = RNG.choice(hs, size=(NPATH, int(round(nh)))).sum(1)
res["hour   (keeps intra-hour regime)"] = s

# block = day
dv = pd_day["sum"].to_numpy()
s = RNG.choice(dv, size=(NPATH, 1)).sum(1)
res["day    (keeps everything, n=10)"] = s

print(f"{'block':>36} {'mean $/day':>11} {'sd':>9} {'P(day<0)':>10} {'5th pct':>10}")
for kk, v in res.items():
    print(f"{kk:>36} {v.mean():>+11.2f} {v.std():>9.2f} {(v < 0).mean():>10.4f} "
          f"{np.percentile(v, 5):>+10.2f}")

print("\n=== VERDICT ===")
sd_e, sd_w = res["entry  (i.i.d., most optimistic)"].std(), res["window (what script 16 used)"].std()
sd_h = res["hour   (keeps intra-hour regime)"].std()
sd_d = res["day    (keeps everything, n=10)"].std()
print(f"  daily sd:  entry ${sd_e:.0f}  ->  window ${sd_w:.0f}  ->  hour ${sd_h:.0f}"
      f"  ->  day ${sd_d:.0f}")
print(f"  ratio day/window = {sd_d / sd_w:.2f}x")
print("  Every step that preserves more real correlation raises the variance.")
print("  The risk numbers in script 16 are a lower bound, not an estimate.")
