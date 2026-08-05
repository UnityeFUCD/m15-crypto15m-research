"""Do the OTHER coins in the same window tell us this window is going bad?

WHY THIS IS THE RIGHT PLACE TO LOOK
  The single hardest fact in this project is that positions inside one window
  fail TOGETHER: P(all 3 lose) runs 6.2x independence, and the worst 5% of
  windows average 2.30 positions and 2.30 losses - every position down. That
  clustering is the entire left tail and therefore the entire drawdown.

  Every signal tested so far has been WITHIN a coin: its own price, its own
  volume, its own flow. But if losses are market-wide, the information about a
  bad window lives in the OTHER coins, and a within-coin test is structurally
  blind to it.

  This asks: at the moment we enter coin C, does the state of the other four
  coins in the same 15-minute window predict whether C settles against us -
  over and above C's own price?

  A hit here is worth more than any edge signal, because it would let us skip
  the correlated windows that produce the drawdown rather than just sizing
  around them.

FEATURES (all observable at entry, no lookahead)
  others_px      mean favourite price of the other coins, same window/minute
  others_weak    how many other favourites are priced below 65c (shaky)
  others_disp    dispersion of the other favourites' prices
  others_flong   mean longshot share of the other coins
  others_n       how many other coins are even quoting in band
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260820)
D = pd.read_parquet(OUT / "grid.parquet")
days = sorted(D.day.unique())

# per (window, minute) aggregates across ALL coins, then subtract self to get
# a leave-one-out view of the rest of the market at that instant
G = D.groupby(["wkey", "minute"]).agg(
    s_px=("px", "sum"), s_fl=("frac_long", "sum"), n=("px", "size"),
    s_weak=("px", lambda s: (s < 0.65).sum()),
    s_sq=("px", lambda s: (s ** 2).sum())).reset_index()
D2 = D.merge(G, on=["wkey", "minute"], how="left")
D2 = D2[D2.n >= 3].copy()                       # need at least 2 others
k = D2.n - 1
D2["others_px"] = (D2.s_px - D2.px) / k
D2["others_flong"] = (D2.s_fl - D2.frac_long) / k
D2["others_weak"] = (D2.s_weak - (D2.px < 0.65).astype(int)) / k
var = (D2.s_sq - D2.px ** 2) / k - D2.others_px ** 2
D2["others_disp"] = np.sqrt(var.clip(lower=0))
D2["others_n"] = k

q = D2[(D2.px >= 0.65) & (D2.px < 0.85) & (D2.minute >= 8) & (D2.minute <= 14)
       & (D2.vol >= 2000)]
E = (q.sort_values("minute", ascending=False)
       .groupby(["wkey", "coin"], as_index=False).first()).copy()
E = (E.sort_values(["wkey", "minute"], ascending=[True, False])
       .groupby("wkey", as_index=False).head(3))
E["pb"] = (E.px * 20).astype(int)
print(f"entries {len(E):,}  windows {E.wkey.nunique():,}  days {len(days)}")
print(f"base win {E.won.mean():.4f}  edge {E.edge.mean()*100:+.2f}c  "
      f"median others quoting {E.others_n.median():.0f}\n")

FE = ["others_px", "others_flong", "others_weak", "others_disp"]


def dayboot(d, col, n=5000):
    ds = sorted(d.day.unique())
    g = {x: gg for x, gg in d.groupby("day")}
    out = []
    for _ in range(n):
        s = pd.concat([g[ds[i]] for i in RNG.choice(len(ds), len(ds), True)])
        a, b = s[~s[col]], s[s[col]]
        if len(a) > 25 and len(b) > 25:
            out.append((b.edge.mean() - a.edge.mean()) * 100)
    return np.sort(np.array(out))


print("=== DOES THE REST OF THE MARKET PREDICT OUR OUTCOME? ===")
print("   split at the within-(coin, price-bucket) median, so our own price")
print("   and coin are held fixed\n")
print(f"{'feature':>14} {'lo edge':>9} {'hi edge':>9} {'DIFF':>8} "
      f"{'95% CI':>20} {'p':>7} {'lo win':>8} {'hi win':>8}")
hits = []
for f in FE:
    d = E.dropna(subset=[f]).copy()
    d["hi"] = d.groupby(["coin", "pb"])[f].transform(lambda s: s > s.median())
    a, b = d[~d.hi], d[d.hi]
    if len(a) < 60 or len(b) < 60:
        continue
    diff = (b.edge.mean() - a.edge.mean()) * 100
    bs = dayboot(d, "hi")
    lo, hi = bs[int(.025 * len(bs))], bs[int(.975 * len(bs))]
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    print(f"{f:>14} {a.edge.mean()*100:>+9.2f} {b.edge.mean()*100:>+9.2f} "
          f"{diff:>+8.2f} [{lo:>+8.2f},{hi:>+8.2f}] {p:>7.4f} "
          f"{a.won.mean():>8.4f} {b.won.mean():>8.4f}")
    if p < 0.05:
        hits.append((f, diff))

print("\n=== THE DIRECT VERSION: does a WEAK rest-of-market predict our loss? ===")
print("  quartiles of the other coins' mean favourite price:\n")
E["oq"] = E.groupby(["coin", "pb"]).others_px.transform(
    lambda s: pd.qcut(s.rank(method="first"), 4, labels=False, duplicates="drop"))
print(f"{'q':>3} {'n':>6} {'others px':>11} {'our px':>8} {'our win':>9} "
      f"{'excess':>9} {'edge':>9}")
for qq, g in E.dropna(subset=["oq"]).groupby("oq"):
    print(f"{int(qq):>3} {len(g):>6,} {g.others_px.mean()*100:>10.1f}c "
          f"{g.px.mean()*100:>7.1f}c {g.won.mean():>9.4f} "
          f"{(g.won.mean()-g.px.mean())*100:>+8.2f}c {g.edge.mean()*100:>+9.2f}")

print("\n=== AND THE THING THAT ACTUALLY MATTERS: does it predict ALL-LOSE? ===")
W = E.groupby("wkey").agg(n=("won", "size"), k=("won", "sum"),
                          opx=("others_px", "mean"),
                          odisp=("others_disp", "mean"))
W = W[W.n >= 2]
W["allbad"] = (W.k == 0).astype(int)
print(f"  windows with >=2 positions: {len(W):,}   all-lose rate "
      f"{W.allbad.mean():.4f}")
for f in ("opx", "odisp"):
    W["hi"] = W[f] > W[f].median()
    a, b = W[~W.hi], W[W.hi]
    print(f"  {f:>6}: low half all-lose {a.allbad.mean():.4f}   "
          f"high half {b.allbad.mean():.4f}   "
          f"ratio {b.allbad.mean()/max(a.allbad.mean(),1e-9):.2f}x")

if hits:
    print("\n=== WALK-FORWARD ON SURVIVORS ===")
    CUT = days[len(days) // 2]
    for f, diff in hits:
        d = E.dropna(subset=[f]).copy()
        thr = d[d.day < CUT].groupby(["coin", "pb"])[f].median()
        te = d[d.day >= CUT].copy()
        te["thr"] = [thr.get(kk, np.nan) for kk in zip(te.coin, te.pb)]
        te = te.dropna(subset=["thr"])
        te["hi"] = te[f] > te.thr
        a, b = te[~te.hi], te[te.hi]
        if len(a) < 25 or len(b) < 25:
            print(f"  {f:>14}  too few out of sample")
            continue
        keep = b if diff > 0 else a
        nd = te.day.nunique()
        print(f"  {f:>14}  OOS diff {(b.edge.mean()-a.edge.mean())*100:>+6.2f}c   "
              f"good half {keep.edge.mean()*100:+.2f}c "
              f"(${keep.edge.sum()*30/nd:+.0f}/day) vs all "
              f"{te.edge.mean()*100:+.2f}c (${te.edge.sum()*30/nd:+.0f}/day)")
else:
    print("\n=== NOTHING SURVIVED ===")
    print("  The rest of the market carries no usable information about our")
    print("  outcome once our own price is fixed. The correlation is real but")
    print("  it is not FORECASTABLE from the cross-section at entry time.")
