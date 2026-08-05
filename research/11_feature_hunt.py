"""Hunt for conditioning variables that raise the edge above its ~+9.6 baseline.

WHAT IS ALREADY RULED OUT
  price band   edge declines monotonically with price; we already trade the best
  z / index margin   is ~0.65 correlated with price and adds nothing beyond it
  take profit  strictly worse at every exit level

WHAT IS TESTED HERE, all computable at decision time from public data
  hour_utc      session effects - liquidity and participant mix vary by hour
  vol_regime    realised index volatility so far in the window
  activity      trades in the window before our entry (a liquidity proxy)
  persistence   how long this side has already been the favourite
  n_eligible    how many coins qualify in the same window (crowding)
  secs          exact time remaining inside the 8-14 minute band

METHOD, with every guard this session has learned the hard way
  - WINDOW-EQUAL-WEIGHTED. Contract weighting has now faked three separate
    findings (thin-coin P&L, the z signal, the band ranking) by over-sampling
    busy windows in which the favourite had already run away.
  - one entry per (window, coin), taken at the first eligible moment, which is
    what the live runner actually does
  - split at each feature's median INSIDE price buckets, so nothing can win by
    being a proxy for price
  - day-clustered bootstrap CI
  - MAX-STATISTIC null across every feature tested, because scanning six
    features and reporting the best is how noise gets promoted
  - chronological out-of-sample split
"""
import calendar
import gc
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("C:/Users/fycin/m15-claude-independent")
OUT = Path(__file__).resolve().parent
H, W = 60, 900
MON = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
RNG = np.random.default_rng(20260805)


def parse_ts(ct):
    if not ct or len(ct) < 19:
        return None
    try:
        base = calendar.timegm(time.strptime(ct[:19], "%Y-%m-%dT%H:%M:%S"))
    except Exception:
        return None
    r = ct[19:].rstrip("Zz")
    if not r:
        return float(base)
    if not r.startswith(".") or not r[1:].isdigit():
        return None
    return base + int(r[1:]) / (10.0 ** len(r[1:]))


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


sys.path.insert(0, str(ROOT))
from claude_frame_ext import load_wide
fr = load_wide(blocks=("A013", "E", "H"))
fr = fr[fr.source != "A013"]
fr = fr[np.isin(fr["y"].to_numpy(), (0, 1))]
Y = dict(zip(fr.ticker.values, fr["y"].to_numpy().astype(int)))

rows = []
for coin in ["ETH", "SOL", "XRP", "DOGE"]:
    grid, h0 = second_grid(coin)
    if grid is None:
        continue
    files = sorted(glob.glob(str(ROOT / f"data/sare/trades/KX{coin}15M-*.json")))
    for fp in files:
        tk = os.path.basename(fp)[:-5]
        y = Y.get(tk)
        if y is None:
            continue
        T = close_utc(tk)
        oi = T - W - h0
        if oi - H + 1 < 0 or T - h0 >= len(grid):
            continue
        try:
            tr = json.loads(Path(fp).read_text())
        except Exception:
            continue
        seq = []
        for t in tr:
            ts = parse_ts(t.get("created_time", ""))
            if ts is None:
                continue
            s = T - ts
            if s <= 0 or s > W:
                continue
            try:
                yp = float(t["yes_price_dollars"])
                side = t.get("taker_side")
            except Exception:
                continue
            if side not in ("yes", "no"):
                continue
            seq.append((s, yp if side == "no" else 1.0 - yp, side == "no"))
        if len(seq) < 20:
            continue
        seq.sort(key=lambda x: -x[0])
        ent = None
        for i, (s, mp, myes) in enumerate(seq):
            if 480 <= s <= 840 and 0.60 <= mp < 0.90:
                ent = (i, s, mp, myes)
                break
        if ent is None:
            continue
        i0, s0, px0, myes = ent
        # persistence: of trades before entry, how many had the same favourite
        prior = seq[:i0]
        persist = (sum(1 for _, _, mo in prior if mo == myes) / len(prior)) if prior else 0.5
        # realised index volatility over the window so far, in bp/s
        k0 = int(np.floor(T - s0)) - h0
        seg = grid[max(oi, k0 - 300):k0 + 1]
        seg = seg[np.isfinite(seg)]
        vol = float(np.std(np.diff(np.log(seg))) * 1e4) if len(seg) > 30 else np.nan
        rows.append(dict(coin=coin, ticker=tk, wkey=tk.split("-")[1],
                         day=time.strftime("%Y-%m-%d", time.gmtime(T)),
                         hour=time.gmtime(T).tm_hour, px=px0, secs=s0,
                         won=1 if (myes == bool(y)) else 0,
                         activity=len(prior), persist=persist, vol=vol,
                         edge=(1 - px0 if (myes == bool(y)) else -px0)))
    del grid
    gc.collect()
    print(f"  {coin}: {len(rows):,} entries", flush=True)

D = pd.DataFrame(rows)
D["n_elig"] = D.groupby("wkey").coin.transform("size")
D["pb"] = (D.px * 20).astype(int)
D.to_parquet(OUT / "features.parquet", index=False)
print(f"\nentries {len(D):,}  windows {D.wkey.nunique():,}  days {D.day.nunique()}")
print(f"baseline edge {D.edge.mean()*100:+.2f} pts   entry px {D.px.mean()*100:.1f}c")

FEATS = ["hour", "vol", "activity", "persist", "n_elig", "secs"]


def lift(d, f):
    """within price bucket, median split on f, edge difference in points."""
    d = d.dropna(subset=[f])
    if len(d) < 200:
        return np.nan
    d = d.copy()
    d["hi"] = d.groupby("pb")[f].transform(lambda s: s > s.median())
    a, b = d[~d.hi], d[d.hi]
    if len(a) < 50 or len(b) < 50:
        return np.nan
    return (b.edge.mean() - a.edge.mean()) * 100


print(f"\n=== FEATURE LIFTS (edge points, within price bucket) ===")
print(f"{'feature':>10} {'lift':>8} {'day-clustered 95% CI':>24}")
obs = {}
for f in FEATS:
    L = lift(D, f)
    obs[f] = L
    days = sorted(D.day.unique())
    bs = []
    for _ in range(3000):
        pick = RNG.choice(days, len(days), True)
        s = pd.concat([D[D.day == p] for p in pick])
        v = lift(s, f)
        if np.isfinite(v):
            bs.append(v)
    bs = np.sort(np.array(bs))
    print(f"{f:>10} {L:>+8.2f} [{bs[int(.025*len(bs))]:>+9.2f},"
          f"{bs[int(.975*len(bs))]:>+9.2f}]")

print(f"\n=== MAX-STATISTIC NULL over {len(FEATS)} features ===")
mx = []
for _ in range(600):
    s = D.copy()
    s["edge"] = s.groupby("pb").edge.transform(lambda x: RNG.permutation(x.values))
    v = [abs(lift(s, f)) for f in FEATS]
    v = [x for x in v if np.isfinite(x)]
    if v:
        mx.append(max(v))
mx = np.sort(np.array(mx))
best = max(obs, key=lambda k: abs(obs[k]) if np.isfinite(obs[k]) else 0)
print(f"  null max-|lift|: 95th {mx[int(.95*len(mx))]:.2f}  99th {mx[int(.99*len(mx))]:.2f}")
print(f"  best observed: {best} at {obs[best]:+.2f}  ->  "
      f"p = {(mx >= abs(obs[best])).mean():.4f}  "
      f"{'SURVIVES' if (mx >= abs(obs[best])).mean() < 0.05 else 'DIES'}")

print(f"\n=== OUT OF SAMPLE (chronological) ===")
days = sorted(D.day.unique())
cut = days[len(days) // 2]
print(f"{'feature':>10} {'H1 lift':>9} {'H2 lift':>9}")
for f in FEATS:
    print(f"{f:>10} {lift(D[D.day < cut], f):>+9.2f} {lift(D[D.day >= cut], f):>+9.2f}")
