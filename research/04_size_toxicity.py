"""STAGE 6 (the decisive part) - is a BACK-OF-QUEUE fill the same fill?

WHY THIS IS THE WHOLE QUESTION
  Stage 4/5 established the historical effect survives honest denominators and
  day-clustering: +5.3c/contract all-coins, day-clustered CI [+2.63, +7.32].
  But that is the average over ALL maker fills. A newly posted order joins the
  BACK of the queue, and prior work in this repo (CLAUDE_QUEUE_RESOLUTION.md)
  measured the two distributions that decide what reaches it:

      queue ahead at best level  median 85 contracts, p75 520, p90 1,431
      arriving sell prints       median 21.3, p90 180

  A back-of-queue order is therefore only reached by LARGE prints. If large
  prints are more toxic to the maker than small ones, the average maker fill
  overstates what we would earn - possibly by enough to flip the sign.

  This measures maker P&L as a function of PRINT SIZE directly. It requires no
  book data and no queue model, so it is answerable on the corpus we hold.

  A "queue-reachable" fill is approximated by print size: to reach an order
  sitting behind Q contracts, a print must exceed Q.
"""
import calendar
import glob
import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import importlib.util
_s = importlib.util.spec_from_file_location("ta", HERE / "01_trade_accounting.py")
ta = importlib.util.module_from_spec(_s)
_s.loader.exec_module(ta)
_b = importlib.util.spec_from_file_location("bc", HERE / "02_build_cells.py")

ROOT = Path("C:/Users/fycin/m15-claude-independent")
OUT = HERE
H, W = 60, 900
LOOK = 180
MON = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
# queue-ahead quantiles measured in this repo (A-015, 8,962 observations)
QUEUE_Q = {"median": 85, "p75": 520, "p90": 1431}
SIZE_EDGES = [0, 5, 21, 50, 85, 180, 520, 1431, 10**9]
SIZE_LAB = ["<5", "5-21", "21-50", "50-85", "85-180", "180-520",
            "520-1431", ">1431"]


def parse_ts(ct):
    if not ct or len(ct) < 19:
        return None
    try:
        base = calendar.timegm(time.strptime(ct[:19], "%Y-%m-%dT%H:%M:%S"))
    except Exception:
        return None
    rest = ct[19:].rstrip("Zz")
    if not rest:
        return float(base)
    if not rest.startswith(".") or not rest[1:].isdigit():
        return None
    return base + int(rest[1:]) / (10.0 ** len(rest[1:]))


def close_utc(tk):
    p = tk.split("-")[1]
    return calendar.timegm((2000 + int(p[:2]), MON[p[2:5]], int(p[5:7]),
                            int(p[7:9]), int(p[9:11]), 0, 0, 0, 0)) + 4 * 3600


def second_grid(coin):
    fs = sorted(glob.glob(str(ROOT / f"data/sare/cf/{coin}_*.npz")),
                key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
    if not fs:
        return None, None
    h0 = int(os.path.basename(fs[0]).split("_")[1].split(".")[0])
    h1 = int(os.path.basename(fs[-1]).split("_")[1].split(".")[0]) + 3600
    g = np.full(h1 - h0, np.nan)
    for f in fs:
        z = np.load(f)
        t, v = z["t"], z["v"]
        s = (t // 1000).astype(np.int64) - h0
        ok = (s >= 0) & (s < len(g))
        if ok.any():
            g[s[ok]] = v[ok]
        z.close()
        del z, t, v, s, ok
    gc.collect()
    return g, h0


rows = []
for coin in ["BTC", "ETH", "SOL", "XRP", "DOGE"]:
    grid, h0 = second_grid(coin)
    if grid is None:
        continue
    files = sorted(glob.glob(str(ROOT / f"data/sare/trades/KX{coin}15M-*.json")))
    print(f"{coin}: {len(files)} windows", flush=True)
    for fi, fp in enumerate(files):
        tk = os.path.basename(fp)[:-5]
        T = close_utc(tk)
        ci, oi = T - h0, T - W - h0
        if oi - H + 1 < 0 or ci >= len(grid):
            continue
        a1, a0 = grid[ci - H + 1:ci + 1], grid[oi - H + 1:oi + 1]
        if not (np.isfinite(a1).all() and np.isfinite(a0).all()):
            continue
        y = 1 if a1.mean() >= a0.mean() else 0
        day = time.strftime("%Y-%m-%d", time.gmtime(T))
        agg = defaultdict(lambda: [0, 0, 0, 0])
        for t in json.loads(Path(fp).read_text()):
            ts = parse_ts(t.get("created_time", ""))
            if ts is None:
                continue
            s = T - ts
            if not (0 < s <= LOOK):
                continue
            rec, why = ta.reconstruct(t)
            if rec is None:
                continue
            # discovery cell: maker pays 65-90c
            if not (650_000 <= rec["maker_px"] < 900_000):
                continue
            _, _, _, mn = ta.pnl(rec, y)
            size = rec["cnt"] / 100.0
            sb = int(np.searchsorted(SIZE_EDGES, size, side="right") - 1)
            treg = 0 if s <= 10 else 1 if s <= 30 else 2 if s <= 60 else 3 if s <= 120 else 4
            a = agg[(sb, treg)]
            a[0] += 1
            a[1] += rec["cnt"]
            a[2] += mn
            a[3] += rec["cnt"] if (rec["maker_side"] == "yes") == bool(y) else 0
        for (sb, treg), a in agg.items():
            rows.append((coin, tk, day, sb, treg, a[0], a[1], a[2], a[3]))
        if (fi + 1) % 400 == 0:
            print(f"   {fi+1}/{len(files)}", flush=True)
    del grid
    gc.collect()

S = pd.DataFrame(rows, columns=["coin", "ticker", "day", "sbin", "treg",
                                "trades", "cnt_h", "net_micros", "win_h"])
S["contracts"] = S.cnt_h / 100.0
S["net_c"] = S.net_micros / 10_000.0
S.to_parquet(OUT / "size_toxicity.parquet", index=False)
RNG = np.random.default_rng(20260804)


def dayci(d, B=8000):
    g = d.groupby("day").agg(n=("net_c", "sum"), c=("contracts", "sum"))
    if len(g) < 2 or g.c.sum() <= 0:
        return np.nan, np.nan, np.nan
    n, c = g.n.values, g.c.values
    idx = RNG.integers(0, len(n), size=(B, len(n)))
    v = np.sort(n[idx].sum(1) / np.maximum(c[idx].sum(1), 1e-9))
    return n.sum() / c.sum(), v[int(.025 * B)], v[int(.975 * B)]


print("\n=== MAKER P&L BY PRINT SIZE, 65-90c band, final 120s, all coins ===")
print("a back-of-queue order behind Q contracts is only reached by a print > Q\n")
print(f"{'print size':>10} {'trades':>9} {'contracts':>12} {'win rate':>9} "
      f"{'c/contract':>11} {'day-clustered 95% CI':>24}")
sub = S[S.treg <= 3]
for sb in sorted(sub.sbin.unique()):
    d = sub[sub.sbin == sb]
    if d.contracts.sum() < 5000:
        continue
    o, lo, hi = dayci(d)
    wr = d.win_h.sum() / d.cnt_h.sum()
    print(f"{SIZE_LAB[int(sb)]:>10} {int(d.trades.sum()):>9,} {d.contracts.sum():>12,.0f} "
          f"{wr:>9.4f} {o:>+11.3f} [{lo:>+9.3f},{hi:>+9.3f}]")

print("\n=== THE QUEUE-REACHABILITY TEST ===")
print("restrict to prints big enough to clear a given queue-ahead depth:\n")
print(f"{'must exceed':>12} {'trades':>9} {'contracts':>12} {'c/contract':>11} "
      f"{'day-clustered 95% CI':>24} {'verdict':>10}")
for lab, q in [("0 (all)", 0)] + list(QUEUE_Q.items()):
    keep = sub[sub.sbin >= int(np.searchsorted(SIZE_EDGES, q, side="right") - 1)]
    if not len(keep):
        continue
    o, lo, hi = dayci(keep)
    print(f"{lab:>12} {int(keep.trades.sum()):>9,} {keep.contracts.sum():>12,.0f} "
          f"{o:>+11.3f} [{lo:>+9.3f},{hi:>+9.3f}] "
          f"{'clears 0' if lo > 0 else 'SPANS 0':>10}")

print("\n=== BY TIME REGION x QUEUE REACHABILITY (c/contract) ===")
TLAB = ["0-10s", "10-30s", "30-60s", "60-120s", "120-180s"]
print(f"{'region':>10} " + " ".join(f"{l:>12}" for l in
                                    ["all prints", ">85", ">520", ">1431"]))
for tr in range(5):
    d = S[S.treg == tr]
    cells = []
    for q in (0, 85, 520, 1431):
        k = d[d.sbin >= int(np.searchsorted(SIZE_EDGES, q, side="right") - 1)]
        cells.append(f"{k.net_c.sum()/k.contracts.sum():+12.3f}"
                     if k.contracts.sum() > 2000 else f"{'-':>12}")
    print(f"{TLAB[tr]:>10} " + " ".join(cells))
print(f"\nwrote {OUT/'size_toxicity.parquet'}")
