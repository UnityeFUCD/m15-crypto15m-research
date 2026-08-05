"""STAGE 4+5 - honest denominators, then clustered / post-selection uncertainty.

The headline discovery was CONTRACT-weighted. Contract count is not sample
size: 311.5M contracts arise from 4,475 windows on TEN days. This recomputes
the same cells under every denominator the protocol demands and then asks how
much of the result survives clustering, leave-one-out, concentration, and the
fact that ~165 cells were searched before this one was reported.

DISCOVERY CELL (frozen from DISCOVERY_HISTORY.md, not re-optimised here):
    maker price 65-90c   (pbin 13..17)
    final 120 seconds    (tbin 0..23, 5s bins)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("C:/Users/fycin/m15-claude-independent/research/final_minute_favorite_maker_validation")
RNG = np.random.default_rng(20260804)
D = pd.read_parquet(OUT / "discovery_cells.parquet")
D["contracts"] = D.cnt_h / 100.0
D["net_c"] = D.maker_net_micros / 10_000.0        # micros -> cents
CELL = (D.pbin.between(13, 17)) & (D.tbin < 24)

print(f"cells {len(D):,}   windows {D.ticker.nunique():,}   days {D.day.nunique()}   "
      f"coins {D.coin.nunique()}")


def summarise(d, label):
    """Every denominator the protocol asks for."""
    if not len(d):
        return None
    ct = d.contracts.sum()
    net = d.net_c.sum()
    per_win = d.groupby("ticker").agg(n=("net_c", "sum"), c=("contracts", "sum"))
    per_day = d.groupby("day").agg(n=("net_c", "sum"), c=("contracts", "sum"))
    per_trade_w = (d.net_c / d.trades.clip(lower=1)).mean()
    return dict(
        label=label, markets=d.ticker.nunique(), windows=d.ticker.nunique(),
        days=d.day.nunique(), trades=int(d.trades.sum()), contracts=float(ct),
        net_total_c=float(net),
        per_contract_c=float(net / ct) if ct else np.nan,
        per_trade_c=float(net / d.trades.sum()) if d.trades.sum() else np.nan,
        eqw_per_window_c=float(per_win.n.mean()),
        eqw_per_window_per_contract_c=float((per_win.n / per_win.c).mean()),
        median_window_c=float(per_win.n.median()),
        eqw_per_day_c=float(per_day.n.mean()),
        win_rate=float(d.maker_win_cnt_h.sum() / d.cnt_h.sum()) if d.cnt_h.sum() else np.nan)


print("\n=== STAGE 4: THE DISCOVERY CELL UNDER EVERY DENOMINATOR ===")
S = summarise(D[CELL], "65-90c, final 120s, all coins")
Sb = summarise(D[CELL & (D.coin == "BTC")], "65-90c, final 120s, BTC only")
for s in (Sb, S):
    print(f"\n  {s['label']}")
    print(f"    windows {s['windows']:,}  days {s['days']}  trades {s['trades']:,}  "
          f"contracts {s['contracts']:,.0f}")
    print(f"    maker win rate                {s['win_rate']:.4f}")
    print(f"    CONTRACT-weighted   per ct    {s['per_contract_c']:+.3f} c   <- the headline")
    print(f"    trade-weighted      per trade {s['per_trade_c']:+.3f} c")
    print(f"    equal-weight window per ct    {s['eqw_per_window_per_contract_c']:+.3f} c")
    print(f"    equal-weight window total     {s['eqw_per_window_c']:+.3f} c/window")
    print(f"    MEDIAN window                 {s['median_window_c']:+.3f} c/window")
    print(f"    equal-weight day              {s['eqw_per_day_c']:+.3f} c/day")

print("\n=== STAGE 4: PRICE x TIME MATRIX (contract-weighted c/contract, BTC) ===")
b = D[D.coin == "BTC"]
piv = b.pivot_table(index="pbin", columns=pd.cut(b.tbin * 5, [0, 10, 30, 60, 120, 180],
                                                 labels=["0-10s", "10-30s", "30-60s",
                                                         "60-120s", "120-180s"]),
                    values="maker_net_micros", aggfunc="sum")
cts = b.pivot_table(index="pbin", columns=pd.cut(b.tbin * 5, [0, 10, 30, 60, 120, 180],
                                                 labels=["0-10s", "10-30s", "30-60s",
                                                         "60-120s", "120-180s"]),
                    values="cnt_h", aggfunc="sum")
M = (piv / 10_000.0) / (cts / 100.0)
M.index = [f"{int(i)*5}-{int(i)*5+5}c" for i in M.index]
print(M.round(3).to_string())

print("\n=== STAGE 5: CLUSTERED UNCERTAINTY ON THE DISCOVERY CELL ===")


def boot(d, by, B=8000, stat="per_contract"):
    g = d.groupby(by).agg(n=("net_c", "sum"), c=("contracts", "sum"))
    n, c = g.n.values, g.c.values
    K = len(n)
    idx = RNG.integers(0, K, size=(B, K))
    if stat == "per_contract":
        v = n[idx].sum(1) / np.maximum(c[idx].sum(1), 1e-9)
        obs = n.sum() / c.sum()
    else:
        v = n[idx].mean(1)
        obs = n.mean()
    v = np.sort(v)
    return obs, v[int(.025 * B)], v[int(.975 * B)], K


for coinlab, sub in (("BTC", D[CELL & (D.coin == "BTC")]), ("all coins", D[CELL])):
    print(f"\n  --- {coinlab}")
    for by, nm in (("ticker", "window-clustered"), ("day", "DAY-clustered")):
        o, lo, hi, K = boot(sub, by)
        print(f"    {nm:18s} per contract {o:+.3f}c  CI[{lo:+.3f},{hi:+.3f}]  "
              f"clusters={K}  {'clears 0' if lo > 0 else 'SPANS 0'}")
    o, lo, hi, K = boot(sub, "day", stat="mean")
    print(f"    {'DAY-clustered':18s} per DAY      {o:+.1f}c  CI[{lo:+.1f},{hi:+.1f}]")

print("\n=== STAGE 5: CONCENTRATION ===")
for coinlab, sub in (("BTC", D[CELL & (D.coin == "BTC")]), ("all coins", D[CELL])):
    pw = sub.groupby("ticker").net_c.sum().sort_values(ascending=False)
    pd_ = sub.groupby("day").net_c.sum().sort_values(ascending=False)
    cw = sub.groupby("ticker").contracts.sum()
    tot = pw.sum()
    hh = float(((cw / cw.sum()) ** 2).sum())
    print(f"\n  --- {coinlab}   total {tot:+,.0f}c over {len(pw):,} windows")
    for k in (1, 5, 10):
        print(f"    top {k:>2} windows share of P&L  {pw.head(k).sum()/tot:>7.3f}"
              f"     top {k:>2} days share  {pd_.head(k).sum()/tot:>7.3f}")
    print(f"    volume Herfindahl {hh:.5f}  -> effective n_windows "
          f"{1/hh:,.1f} of {len(pw):,}")
    print(f"    windows with POSITIVE P&L {float((pw>0).mean()):.4f}")
    print(f"    excluding best day   {(tot-pd_.iloc[0])/max(sub[sub.day!=pd_.index[0]].contracts.sum(),1e-9):+.3f} c/contract")
    print(f"    excluding worst day  {(tot-pd_.iloc[-1])/max(sub[sub.day!=pd_.index[-1]].contracts.sum(),1e-9):+.3f} c/contract")
    print(f"    excluding biggest window {(tot-pw.iloc[0])/max(sub[sub.ticker!=pw.index[0]].contracts.sum(),1e-9):+.3f} c/contract")

print("\n=== STAGE 5: LEAVE-ONE-DAY-OUT (all coins) ===")
sub = D[CELL]
tot_c = sub.contracts.sum()
print(f"{'day held out':>14} {'per-contract c':>15} {'that day alone':>16} {'its share of ct':>16}")
for dy in sorted(sub.day.unique()):
    r = sub[sub.day != dy]
    o = sub[sub.day == dy]
    print(f"{dy:>14} {r.net_c.sum()/r.contracts.sum():>+15.3f} "
          f"{o.net_c.sum()/max(o.contracts.sum(),1e-9):>+16.3f} "
          f"{o.contracts.sum()/tot_c:>16.4f}")

print("\n=== STAGE 5: POST-SELECTION - was this cell special among the ones searched? ===")
print("  distribution of per-contract net across ALL (5c price x 4 time-region) cells,")
print("  BTC, which is the family the reported cell was chosen from\n")
b = D[D.coin == "BTC"].copy()
b["treg"] = pd.cut(b.tbin * 5, [0, 10, 30, 60, 120, 180], labels=False)
g = b.groupby(["pbin", "treg"]).agg(n=("net_c", "sum"), c=("contracts", "sum"))
g = g[g.c > 10_000]
g["pc"] = g.n / g.c
q = g.pc.describe(percentiles=[.05, .25, .5, .75, .95])
print("   " + q.round(3).to_string().replace("\n", "\n   "))
cellv = D[CELL & (D.coin == "BTC")]
cv = cellv.net_c.sum() / cellv.contracts.sum()
print(f"\n   reported cell {cv:+.3f}c is the {float((g.pc < cv).mean()):.3f} quantile "
      f"of {len(g)} searched cells")

json.dump({"discovery_cell_all_coins": S, "discovery_cell_btc": Sb},
          open(OUT / "stage45_summary.json", "w"), indent=1, default=float)
print(f"\nwrote {OUT/'stage45_summary.json'}")
