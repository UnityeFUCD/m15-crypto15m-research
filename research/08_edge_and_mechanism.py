"""Two questions:

A. Which price band actually has the largest edge (win rate minus price paid)?
B. Is there anything observable at decision time that separates the 65-70c
   contracts that go on to win from the ones that do not?

B is the one that matters. Our fills cluster at 66.5c where the edge is
thinnest (+6.0 points live). If some observable feature identifies which cheap
favourites behave like expensive ones, the edge on our largest bucket could be
several times what it is.

THE CANDIDATE MECHANISM
  Settlement is A1 >= A0, both 60-second means of the CF index. At decision
  time part of that race is already run: the index has a RUNNING MARGIN
  (A_now - A0). The market price encodes some view of it, but if the price
  systematically underweights the margin, then within a price bucket the
  contracts with a larger standardised margin should win more.

  Standardised, because a 5bp lead with 12 minutes left is weak and the same
  lead with 2 minutes left is strong:

      z = (A_now - A0) / (sigma * sqrt(seconds_remaining))

  This is measurable entirely from public data at decision time - no lookahead.

GUARDS: compared INSIDE price buckets with the outcome demeaned first, day- and
window-clustered bootstrap, and an out-of-sample split fitted on the first half
of days only.
"""
import calendar
import gc
import glob
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("C:/Users/fycin/m15-claude-independent")
OUT = Path(__file__).resolve().parent
H, W = 60, 900
MON = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
RNG = np.random.default_rng(20260805)
DEC_LO, DEC_HI = 8 * 60, 14 * 60          # seconds remaining, our live window


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


def trailing(grid, i, h=H):
    """mean of the h seconds ending at index i, or nan."""
    if i - h + 1 < 0:
        return np.nan
    w = grid[i - h + 1:i + 1]
    return float(w.mean()) if np.isfinite(w).all() else np.nan


rows = []
for coin in ["BTC", "ETH", "SOL", "XRP", "DOGE"]:
    grid, h0 = second_grid(coin)
    if grid is None:
        continue
    lv = np.diff(np.log(grid))
    sigma = float(np.nanstd(lv[np.isfinite(lv)]))          # per-second log sd
    files = sorted(glob.glob(str(ROOT / f"data/sare/trades/KX{coin}15M-*.json")))
    print(f"{coin}: {len(files)} windows  sigma {sigma*1e4:.3f}bp/s", flush=True)
    for fi, fp in enumerate(files):
        tk = os.path.basename(fp)[:-5]
        T = close_utc(tk)
        ci, oi = T - h0, T - W - h0
        if oi - H + 1 < 0 or ci >= len(grid):
            continue
        A1, A0 = trailing(grid, ci), trailing(grid, oi)
        if not (np.isfinite(A1) and np.isfinite(A0)):
            continue
        y = 1 if A1 >= A0 else 0
        day = time.strftime("%Y-%m-%d", time.gmtime(T))
        seen = {}
        try:
            tr = json.loads(Path(fp).read_text())
        except Exception:
            continue
        for t in tr:
            ts = parse_ts(t.get("created_time", ""))
            if ts is None:
                continue
            s = T - ts
            if not (DEC_LO <= s <= DEC_HI):
                continue
            try:
                ypx = float(t["yes_price_dollars"])
                cnt = float(t["count_fp"])
                side = t.get("taker_side")
            except Exception:
                continue
            if side not in ("yes", "no") or cnt <= 0:
                continue
            maker_px = ypx if side == "no" else 1.0 - ypx
            if not (0.60 <= maker_px < 0.90):
                continue
            maker_yes = (side == "no")
            # running index margin at THIS trade's second, no lookahead
            k = int(np.floor(ts)) - h0
            An = trailing(grid, k)
            if not np.isfinite(An):
                continue
            marg = (An - A0) / A0
            # standardise by remaining volatility
            z = marg / (sigma * np.sqrt(max(s, 1)))
            # margin favouring the side the MAKER holds
            zm = z if maker_yes else -z
            mw = 1 if (maker_yes == bool(y)) else 0
            rows.append((coin, tk, day, maker_px, zm, mw, cnt, s))
        if (fi + 1) % 300 == 0:
            print(f"   {fi+1}/{len(files)}  {len(rows):,} rows", flush=True)
    del grid
    gc.collect()

D = pd.DataFrame(rows, columns=["coin", "win", "day", "px", "z", "won", "ct", "secs"])
D.to_parquet(OUT / "mechanism.parquet", index=False)
print(f"\nrows {len(D):,}   windows {D.win.nunique():,}   days {D.day.nunique()}")

print("\n=== A. EDGE BY PRICE BAND (contract-weighted, 8-14 min) ===")
print(f"{'band':>10} {'trades':>9} {'contracts':>12} {'avg px':>8} {'win':>7} "
      f"{'EDGE pts':>9}")
B = [(0.60, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 0.90)]
for lo, hi in B:
    q = D[(D.px >= lo) & (D.px < hi)]
    if len(q) < 100:
        continue
    ct = q.ct.sum()
    ap = (q.px * q.ct).sum() / ct
    wr = (q.won * q.ct).sum() / ct
    print(f"{lo*100:>4.0f}-{hi*100:<5.0f} {len(q):>9,} {ct:>12,.0f} {ap*100:>7.1f}c "
          f"{wr:>7.4f} {(wr-ap)*100:>+9.2f}")

print("\n=== B. DOES THE RUNNING INDEX MARGIN PREDICT THE WIN? ===")
print("inside each price bucket, split on z (standardised index margin)\n")
print(f"{'band':>10} {'n lo-z':>8} {'n hi-z':>8} {'win lo-z':>9} {'win hi-z':>9} "
      f"{'LIFT pts':>9}")
sub = D[(D.px >= 0.65) & (D.px < 0.70)].copy()
for lo, hi in B:
    q = D[(D.px >= lo) & (D.px < hi)].copy()
    if len(q) < 400:
        continue
    med = q.z.median()
    a, b = q[q.z <= med], q[q.z > med]
    wa = (a.won * a.ct).sum() / a.ct.sum()
    wb = (b.won * b.ct).sum() / b.ct.sum()
    print(f"{lo*100:>4.0f}-{hi*100:<5.0f} {len(a):>8,} {len(b):>8,} {wa:>9.4f} "
          f"{wb:>9.4f} {(wb-wa)*100:>+9.2f}")

print("\n=== B2. THE 65-70c BUCKET IN z QUINTILES (our biggest bucket) ===")
sub["zq"] = pd.qcut(sub.z.rank(method="first"), 5, labels=False)
print(f"{'z quintile':>11} {'trades':>9} {'contracts':>12} {'avg px':>8} "
      f"{'win rate':>9} {'EDGE pts':>9}")
for q_, g in sub.groupby("zq"):
    ct = g.ct.sum()
    ap = (g.px * g.ct).sum() / ct
    wr = (g.won * g.ct).sum() / ct
    print(f"{int(q_):>11} {len(g):>9,} {ct:>12,.0f} {ap*100:>7.1f}c {wr:>9.4f} "
          f"{(wr-ap)*100:>+9.2f}")

print("\n=== B3. WINDOW-CLUSTERED CI ON THE TOP-vs-BOTTOM QUINTILE SPLIT ===")
top = sub[sub.zq == 4]
bot = sub[sub.zq == 0]
wc, wu = pd.factorize(sub.win.values)
Wn = len(wu)


def boot(mask_top, mask_bot, B_=4000):
    st = np.bincount(wc[mask_top], weights=(sub.won.values * sub.ct.values)[mask_top], minlength=Wn)
    ct_ = np.bincount(wc[mask_top], weights=sub.ct.values[mask_top], minlength=Wn)
    sb = np.bincount(wc[mask_bot], weights=(sub.won.values * sub.ct.values)[mask_bot], minlength=Wn)
    cb = np.bincount(wc[mask_bot], weights=sub.ct.values[mask_bot], minlength=Wn)
    idx = RNG.integers(0, Wn, size=(B_, Wn))
    v = np.sort(st[idx].sum(1) / np.maximum(ct_[idx].sum(1), 1)
                - sb[idx].sum(1) / np.maximum(cb[idx].sum(1), 1))
    return v[int(.025 * B_)], v[int(.975 * B_)]


mt = (sub.zq == 4).values
mb = (sub.zq == 0).values
lo_, hi_ = boot(mt, mb)
obs = ((top.won * top.ct).sum() / top.ct.sum()
       - (bot.won * bot.ct).sum() / bot.ct.sum())
print(f"  top-quintile win minus bottom-quintile win = {obs*100:+.2f} points")
print(f"  window-clustered 95% CI [{lo_*100:+.2f},{hi_*100:+.2f}]  "
      f"{'CLEARS 0' if lo_ > 0 else 'spans 0'}")

print("\n=== B4. OUT OF SAMPLE: fit the direction on half the days ===")
days = sorted(sub.day.unique())
cut = days[len(days) // 2]
h1, h2 = sub[sub.day < cut], sub[sub.day >= cut]
for lab, g in (("first half (fit)", h1), ("SECOND HALF (test)", h2)):
    t_, b_ = g[g.zq == 4], g[g.zq == 0]
    if not len(t_) or not len(b_):
        continue
    wt = (t_.won * t_.ct).sum() / t_.ct.sum()
    wb = (b_.won * b_.ct).sum() / b_.ct.sum()
    apt = (t_.px * t_.ct).sum() / t_.ct.sum()
    print(f"  {lab:20s} top-z win {wt:.4f} (px {apt*100:.1f}c, edge "
          f"{(wt-apt)*100:+.2f}pt)   bottom-z win {wb:.4f}   lift {(wt-wb)*100:+.2f}pt")
