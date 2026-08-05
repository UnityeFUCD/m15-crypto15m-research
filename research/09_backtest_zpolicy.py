"""Backtest the z-score as a POLICY, not as a signal.

08_edge_and_mechanism showed z separates winners from losers inside price
buckets. That is a statement about a signal. It is NOT a statement that
swapping our selection rule for a z-ranked one earns more money, because our
policy is capped at 2 entries per 15-minute window and currently picks them in
SCAN ORDER (BTC, ETH, SOL, XRP, DOGE). The question is whether ranking the same
candidate set by z beats ranking it by an accident of iteration order.

HONEST CONSTRUCTION
  - z is computed AT ENTRY, defined as the earliest trade for that coin inside
    our band and time window. That is the moment the live runner would see the
    market as eligible and decide. Using z at fill time would leak information
    we would not have had.
  - both policies draw from exactly the same candidate set, so no policy can
    invent opportunities the other could not have taken.
  - fills are not simulated. Historical book data does not exist for this
    period, so the comparison is over trades that DID occur - it measures
    SELECTION, not execution. Stated plainly rather than hidden.

GUARDS: day-clustered bootstrap, chronological out-of-sample split, per-coin
breakdown, and a random-selection control so "ranked by z" is compared against
"ranked by nothing" rather than only against scan order.
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260805)
D = pd.read_parquet(OUT / "mechanism.parquet").rename(columns={"win": "ticker"})
D["wkey"] = D.ticker.str.split("-").str[1]
SCAN = {"BTC": 0, "ETH": 1, "SOL": 2, "XRP": 3, "DOGE": 4}

# ENTRY = earliest trade for that coin in the band/time box (secs = time
# remaining, so the largest secs is the earliest moment).
E = (D.sort_values("secs", ascending=False)
       .groupby(["wkey", "coin"], as_index=False).first())
E["scan"] = E.coin.map(SCAN)
E["edge"] = (E.won - E.px)          # realised P&L per contract, in dollars
print(f"candidate entries {len(E):,}   windows {E.wkey.nunique():,}   "
      f"days {E.day.nunique()}   coins {E.coin.nunique()}")
print(f"mean candidates per window {len(E)/E.wkey.nunique():.2f}")

CAP = 2


def run(rule, seed=None):
    """Pick up to CAP entries per window under `rule`; return per-window P&L."""
    r = RNG if seed is None else np.random.default_rng(seed)
    out = []
    for wk, g in E.groupby("wkey", sort=False):
        if rule == "scan":
            g2 = g.sort_values("scan")
        elif rule == "z":
            g2 = g.sort_values("z", ascending=False)
        elif rule == "z_lowfirst":
            g2 = g.sort_values("z")
        elif rule == "random":
            g2 = g.iloc[r.permutation(len(g))]
        elif rule == "price":
            g2 = g.iloc[(g.px - 0.775).abs().values.argsort()]   # nearest 77.5c
        else:
            g2 = g
        k = g2.head(CAP)
        out.append((wk, g.day.iloc[0], k.edge.mean(), k.px.mean(),
                    k.won.mean(), len(k)))
    return pd.DataFrame(out, columns=["wkey", "day", "edge", "px", "won", "n"])


def dayci(df, B=6000):
    g = df.groupby("day").edge.mean()
    v = g.values
    idx = RNG.integers(0, len(v), size=(B, len(v)))
    b = np.sort(v[idx].mean(1))
    return v.mean(), b[int(.025 * B)], b[int(.975 * B)]


print(f"\n=== POLICY COMPARISON, cap {CAP} per window ===")
print(f"{'rule':>14} {'entries':>8} {'avg px':>8} {'win':>8} {'edge c/ct':>10} "
      f"{'day-clustered 95% CI':>24}")
res = {}
for rule in ("scan", "random", "price", "z", "z_lowfirst"):
    df = run(rule, seed=7 if rule == "random" else None)
    m, lo, hi = dayci(df)
    res[rule] = df
    print(f"{rule:>14} {int(df.n.sum()):>8,} {df.px.mean()*100:>7.1f}c "
          f"{df.won.mean():>8.4f} {m*100:>+10.2f} "
          f"[{lo*100:>+9.2f},{hi*100:>+9.2f}]")

print("\n=== PAIRED: z-ranked minus scan-order, same windows, same days ===")
a, b = res["z"], res["scan"]
mg = a.merge(b, on=["wkey", "day"], suffixes=("_z", "_s"))
mg["d"] = mg.edge_z - mg.edge_s
g = mg.groupby("day").d.mean().values
idx = RNG.integers(0, len(g), size=(6000, len(g)))
bs = np.sort(g[idx].mean(1))
print(f"  improvement {mg.d.mean()*100:+.2f} c/contract   "
      f"day-clustered CI [{bs[150]*100:+.2f},{bs[5850]*100:+.2f}]   "
      f"{'CLEARS 0' if bs[150] > 0 else 'spans 0'}")
print(f"  windows where z-rank differs from scan-order: "
      f"{(mg.d.abs() > 1e-9).mean():.3f}")

print("\n=== OUT OF SAMPLE (chronological split) ===")
days = sorted(E.day.unique())
cut = days[len(days) // 2]
for lab, sel in (("first half", mg.day < cut), ("SECOND HALF", mg.day >= cut)):
    q = mg[sel]
    print(f"  {lab:12s} windows {len(q):>5,}  z {q.edge_z.mean()*100:>+7.2f}  "
          f"scan {q.edge_s.mean()*100:>+7.2f}  improvement {q.d.mean()*100:>+6.2f}")

print("\n=== BY COIN: which coins does z reorder, and does it help? ===")
zsel = []
for wk, g2 in E.groupby("wkey", sort=False):
    for _, r_ in g2.sort_values("z", ascending=False).head(CAP).iterrows():
        zsel.append(r_.coin)
ssel = []
for wk, g2 in E.groupby("wkey", sort=False):
    for _, r_ in g2.sort_values("scan").head(CAP).iterrows():
        ssel.append(r_.coin)
zc = pd.Series(zsel).value_counts()
sc = pd.Series(ssel).value_counts()
print(f"{'coin':>6} {'scan picks':>11} {'z picks':>9} {'change':>9}")
for c in SCAN:
    print(f"{c:>6} {sc.get(c,0):>11,} {zc.get(c,0):>9,} "
          f"{zc.get(c,0)-sc.get(c,0):>+9,}")

print("\n=== SANITY: is z just picking the cheapest / most expensive? ===")
print(f"  corr(z, price) among candidates = {E.z.corr(E.px):+.4f}")
print(f"  scan-order avg price {res['scan'].px.mean()*100:.2f}c   "
      f"z-ranked avg price {res['z'].px.mean()*100:.2f}c")
