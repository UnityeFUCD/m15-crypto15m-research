"""BLOCKING AUDIT: why does BTC produce zero historical entries in a box where
it fills live?

Live, our first fills were BTC at 72c inside 65-85c x 8-14 minutes. The
historical cell table showed DOGE/ETH/SOL/XRP in that box and NO BTC at all.
Until that is explained, the historical policy and the live policy are not
demonstrably the same policy, and no widening decision may rest on the
backtest.

Gate-by-gate count for every coin, so the exact gate where BTC goes to zero is
visible rather than inferred.
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent
D = pd.read_parquet(OUT / "discovery_cells.parquet")
D["contracts"] = D.cnt_h / 100.0
D["min_left"] = (D.tbin * 5) / 60.0
D["win"] = D.ticker.str.split("-").str[1]

COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE"]
print("=== GATE-BY-GATE FUNNEL, distinct market-windows ===")
print("each gate is a strict subset of the one above it\n")
print(f"{'gate':44} " + " ".join(f"{c:>8}" for c in COINS))


def row(lab, d):
    cells = []
    for c in COINS:
        cells.append(f"{d[d.coin == c].ticker.nunique():>8,}")
    print(f"{lab:44} " + " ".join(cells))


row("1. markets present in cell table", D)
row("2. any trade in 8-14 min region", D[(D.min_left >= 8) & (D.min_left <= 14)])
t = D[(D.min_left >= 8) & (D.min_left <= 14)]
row("3.   ... AND maker price 65-85c", t[t.pbin.between(13, 16)])
row("4.   ... AND >=10 contracts traded",
    t[t.pbin.between(13, 16)].groupby(["coin", "ticker"]).filter(
        lambda q: q.contracts.sum() >= 10))

print("\n=== WHERE DOES BTC ACTUALLY TRADE? price x time, contracts ===")
b = D[D.coin == "BTC"]
print(f"BTC total contracts in cell table: {b.contracts.sum():,.0f}")
print(f"BTC distinct windows: {b.ticker.nunique():,}\n")
TB = [(0, 3), (3, 6), (6, 8), (8, 11), (11, 15)]
print(f"{'maker price':>12} " + " ".join(f"{a}-{b_}m".rjust(11) for a, b_ in TB))
for pb in range(0, 20):
    cells, tot = [], 0
    for a, b_ in TB:
        v = b[(b.pbin == pb) & (b.min_left >= a) & (b.min_left < b_)].contracts.sum()
        tot += v
        cells.append(f"{v:>11,.0f}" if v else f"{'-':>11}")
    if tot > 0:
        print(f"{pb*5:>5}-{pb*5+5:<6} " + " ".join(cells))

print("\n=== THE SAME FOR ETH, AS A CONTROL ===")
e = D[D.coin == "ETH"]
print(f"{'maker price':>12} " + " ".join(f"{a}-{b_}m".rjust(11) for a, b_ in TB))
for pb in range(0, 20):
    cells, tot = [], 0
    for a, b_ in TB:
        v = e[(e.pbin == pb) & (e.min_left >= a) & (e.min_left < b_)].contracts.sum()
        tot += v
        cells.append(f"{v:>11,.0f}" if v else f"{'-':>11}")
    if tot > 0:
        print(f"{pb*5:>5}-{pb*5+5:<6} " + " ".join(cells))

print("\n=== PRICE DISTRIBUTION OF ALL MAKER FLOW, BY COIN ===")
print("share of each coin's contracts landing in each 5c maker-price bucket\n")
print(f"{'price':>10} " + " ".join(f"{c:>9}" for c in COINS))
for pb in range(8, 20):
    cells = []
    for c in COINS:
        d = D[D.coin == c]
        s = d[d.pbin == pb].contracts.sum() / max(d.contracts.sum(), 1)
        cells.append(f"{s:>9.4f}")
    print(f"{pb*5:>4}-{pb*5+5:<5} " + " ".join(cells))
