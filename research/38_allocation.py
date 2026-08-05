"""When capital is scarce, does RANKING candidates beat taking them first-come?

THE GAP THIS TESTS
  Every previous test asked which entries to REMOVE. Under a binding capital
  constraint the sharper question is which entries get the scarce capital when
  several compete at once. The live runner answers that by accident: it loops
  SERIES = [BTC, ETH, SOL, XRP, DOGE] in list order, so BTC systematically wins
  a contested window over DOGE for no reason but list position.

  Meanwhile others_px and frac_long demonstrably separate +16.1c entries from
  +2.0c entries, and are used only to reject, never to prioritise. If capital
  is the binding constraint - 177 refusals per 36 fills live - then allocating
  it to the best available candidate rather than the first one is a pure
  efficiency gain: no extra capital, no size change, no band change.

DESIGN
  - the ranking model is fitted on PRIOR DAYS ONLY and applied unseen, so this
    is not hindsight ranking
  - the model is deliberately crude (cell means over others_px x frac_long x
    price bucket). A complex model would fit noise on 10 days
  - the capital cap is swept from loose to tight, because the backtest's own
    cap does not bind (blocked 0.000 at $110) while live it binds constantly.
    If ranking only pays when capital is tight, that is precisely the live
    regime and the loose-cap result understates it.
  - first-come is simulated in the runner's ACTUAL order (series list order),
    not an idealised one
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260830)
D = pd.read_parquet(OUT / "grid.parquet")
QTY, LAG = 30, 45.0
SERIES_ORDER = {"BTC": 0, "ETH": 1, "SOL": 2, "XRP": 3, "DOGE": 4}
days = sorted(D.day.unique())

G = D.groupby(["wkey", "minute"]).agg(s_px=("px", "sum"),
                                      n=("px", "size")).reset_index()
W = D.merge(G, on=["wkey", "minute"], how="left")
W["others_px"] = np.where(W.n >= 3, (W.s_px - W.px) / (W.n - 1), np.nan)
W["close"] = pd.to_datetime(W.wkey, format="%y%b%d%H%M", utc=True, errors="coerce")
W = W.dropna(subset=["close"]).copy()
W["t_entry"] = W.close - pd.to_timedelta(W.secs, unit="s")
W["t_free"] = W.close + pd.to_timedelta(LAG, unit="s")

q = W[(W.px >= 0.65) & (W.px < 0.85) & (W.minute >= 8) & (W.minute <= 14)
      & (W.vol >= 2000)]
E = (q.sort_values("minute", ascending=False)
       .groupby(["wkey", "coin"], as_index=False).first()).copy()
E["pb"] = (E.px * 20).astype(int)
E["ord"] = E.coin.map(SERIES_ORDER).fillna(9)
print(f"candidates {len(E):,} over {len(days)} days   "
      f"windows {E.wkey.nunique():,}")
print(f"candidates per window: mean {len(E)/E.wkey.nunique():.2f}  "
      f"max {E.groupby('wkey').size().max()}\n")


def fit_model(hist):
    """Crude, deliberately: mean edge by (others_px tercile, frac_long tercile,
    price bucket), with fallbacks. Complex models fit noise on 10 days."""
    h = hist.dropna(subset=["others_px"]).copy()
    if len(h) < 60:
        return None
    oq = h.others_px.quantile([1 / 3, 2 / 3]).values
    fq = h.frac_long.quantile([1 / 3, 2 / 3]).values
    h["oa"] = np.digitize(h.others_px, oq)
    h["fa"] = np.digitize(h.frac_long, fq)
    cell = h.groupby(["oa", "fa"]).edge.mean()
    return dict(oq=oq, fq=fq, cell=cell, base=h.edge.mean())


def score(m, row):
    if m is None or not np.isfinite(row.others_px):
        return m["base"] if m else 0.0
    oa = int(np.digitize([row.others_px], m["oq"])[0])
    fa = int(np.digitize([row.frac_long], m["fq"])[0])
    return float(m["cell"].get((oa, fa), m["base"]))


def simulate(policy, gross_cap):
    """Chronological ledger; policy decides ORDER of consideration only."""
    taken, blocked = [], 0
    for i, d in enumerate(days):
        if i < 3:
            continue
        m = fit_model(E[E.day.isin(days[:i])]) if policy == "rank" else None
        day = E[E.day == d].copy()
        if policy == "rank":
            day["s"] = [score(m, r) for r in day.itertuples()]
        openp = []
        for wk, g in day.groupby("wkey", sort=True):
            t0 = g.t_entry.min()
            openp = [p for p in openp if p[0] > t0]
            if policy == "rank":
                g = g.sort_values("s", ascending=False)
            else:                       # the runner's real order: series list
                g = g.sort_values(["ord", "minute"], ascending=[True, False])
            for r in g.itertuples():
                gr = sum(c for _, c in openp)
                cost = QTY * r.px
                if gr + cost > gross_cap:
                    blocked += 1
                    continue
                openp.append((r.t_free, cost))
                taken.append((r.day, r.edge * QTY, cost))
    if len(taken) < 30:
        return None
    t = pd.DataFrame(taken, columns=["day", "pnl", "cost"])
    nd = t.day.nunique()
    return dict(n=len(t), day=t.pnl.sum() / nd, blocked=blocked,
                edge=t.pnl.sum() / (len(t) * QTY) * 100,
                per_dollar=t.pnl.sum() / t.cost.sum() * 100)


print("=== FIRST-COME (runner's real series order) vs RANKED ===")
print("   ranking model fitted on prior days only, applied unseen\n")
print(f"{'gross cap':>10} {'policy':>12} {'entries':>8} {'blocked':>8} "
      f"{'edge/ct':>9} {'$/day':>9} {'ret/$ deployed':>16}")
for gc in (110, 80, 60, 45, 35, 25):
    out = {}
    for pol in ("first", "rank"):
        r = simulate(pol, gc)
        if r:
            out[pol] = r
            print(f"{gc:>10} {pol:>12} {r['n']:>8} {r['blocked']:>8} "
                  f"{r['edge']:>+9.2f} {r['day']:>+9.2f} "
                  f"{r['per_dollar']:>15.2f}%")
    if len(out) == 2:
        d = out["rank"]["day"] - out["first"]["day"]
        print(f"{'':>10} {'GAIN':>12} {'':>8} {'':>8} "
              f"{out['rank']['edge']-out['first']['edge']:>+9.2f} "
              f"{d:>+9.2f} "
              f"{out['rank']['per_dollar']-out['first']['per_dollar']:>+15.2f}%")
    print()

print("=== HOW OFTEN DOES ALLOCATION EVEN MATTER? ===")
cw = E.groupby("wkey").size()
print(f"  windows with 1 candidate : {(cw == 1).sum():>5}  "
      f"({(cw == 1).mean()*100:.1f}%)  - order is irrelevant")
print(f"  windows with 2+          : {(cw >= 2).sum():>5}  "
      f"({(cw >= 2).mean()*100:.1f}%)  - order can matter")
print(f"  windows with 3+          : {(cw >= 3).sum():>5}  "
      f"({(cw >= 3).mean()*100:.1f}%)")
print("\n  Ranking can only help where candidates compete AND capital is")
print("  short. If most windows have one candidate, the ceiling is low.")
