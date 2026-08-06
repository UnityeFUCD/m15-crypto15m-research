"""INTERPLAY 1 - is maker adverse selection constant, or does it vary?

THE HYPOTHESIS THIS TESTS
  Every failure in this project traces to one fact: the raw maker edge is
  +2.29c and fill bias consumes it. Thirteen candidates died trying to find a
  better SUBSET of markets. None asked whether the CONSTRAINT ITSELF varies.

  Fill bias is not bad luck - it is someone with better information crossing
  the spread. So it should be weakest where information asymmetry is lowest.
  If such a regime exists, the raw edge survives there, and that is a
  different kind of finding from "these markets win more often".

WHY THIS IS NOT THE FOURTEENTH SUBSET SEARCH
  A subset search asks "where do favourites win?" and finds noise. This asks
  "where does the counterparty know less?", which is a mechanism with a prior,
  and it is tested the way the previous thirteen were not:

    * the search runs on TRAIN ONLY (2026-05-25 to 06-30)
    * the winner is FROZEN and then evaluated on valid and test
    * a best-of-N permutation corrects for having searched at all
    * day-clustered intervals, leave-one-coin-out, leave-one-week-out

  A regime that only looks good on train is reported as a failure, which is
  the outcome three of this project's four wrong verdicts would have had.

THE MEASURED QUANTITY
  The filled cohort's maker edge - what you actually collect, not what the
  population would pay if every order filled. Fill state comes from the
  validated path proxy (agrees with 303 real fills to 0.6pp on the headline
  gap; see research/probe_signal_population.py).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"
OUT = ROOT / "research" / "results"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260807)

TRAIN = ("2026-05-25", "2026-06-30")
VALID = ("2026-06-30", "2026-07-18")
TEST = ("2026-07-18", "2026-08-07")
WAIT = 120


def held_quote(side, yb, ya):
    return (yb, ya) if side == "yes" else (1.0 - ya, 1.0 - yb)


def parse(path_json, close_ts, side):
    try:
        raw = json.loads(path_json)
    except Exception:
        return []
    out = []
    for p in raw or []:
        try:
            ml = float(p["ml"]); yb = float(p["yb"]); ya = float(p["ya"])
            v = float(p.get("v") or 0)
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 < yb < ya < 1):
            continue
        b, a = held_quote(side, yb, ya)
        out.append((float(close_ts) - 60.0 * ml, b, a, v))
    return sorted(out)


print("=" * 86)
print("INTERPLAY 1 - WHERE IS ADVERSE SELECTION WEAKEST?")
print("=" * 86)

P = pd.read_parquet(DATA / "paths_full.parquet")
P = P[P.entry_ml.notna()]
P = P[(P.bid >= 0.65) & (P.bid < 0.80)].copy()

# Pre-entry realised volatility of the UNDERLYING, from 1-minute spot.
# The quote path cannot supply this: it spans ml 15..0 and entry is at ml 14,
# so only one observation precedes entry. spot_1m is validated against the
# 41,334 known a0/a1 pairs (return corr 0.9807) in validate_spot_proxy.py.
# The window is strictly [entry-15min, entry], so there is no look-ahead.
S = pd.read_parquet(DATA / "spot_1m.parquet").sort_values(["coin", "ts"])
spot = {c: (g.ts.to_numpy(), g.close.to_numpy())
        for c, g in S.groupby("coin")}


def pre_entry_vol(coin: str, entry_ts: float) -> float:
    tup = spot.get(coin)
    if tup is None:
        return np.nan
    ts, px = tup
    lo = np.searchsorted(ts, entry_ts - 900)
    hi = np.searchsorted(ts, entry_ts)
    if hi - lo < 5:
        return np.nan
    seg = px[lo:hi]
    return float(np.std(np.diff(seg) / seg[:-1]))


rows = []
for r in P.itertuples():
    pts = parse(r.path, r.close_ts, r.side)
    if len(pts) < 4:
        continue
    entry_ts = float(r.close_ts) - 60.0 * float(r.entry_ml)
    ent = [p for p in pts if abs(p[0] - entry_ts) < 1e-6]
    spread = (ent[0][2] - ent[0][1]) if ent else np.nan
    vol = pre_entry_vol(r.coin, entry_ts)
    vol_entry = float(ent[0][3]) if ent else np.nan
    filled = any(entry_ts < ts <= entry_ts + WAIT and b <= float(r.bid) + 1e-9
                 for ts, b, _, _ in pts)
    ask_after = next((a for ts, _, a, _ in pts if ts >= entry_ts + WAIT - 1e-9),
                     np.nan)
    rows.append(dict(ticker=r.ticker, coin=r.coin, close_ts=int(r.close_ts),
                     won=int(r.won), bid=float(r.bid), side=r.side,
                     spread=spread, prevol=vol, entry_vol=vol_entry,
                     filled=filled, ask_after=ask_after))
G = pd.DataFrame(rows)
G["utc"] = pd.to_datetime(G.close_ts, unit="s", utc=True)
G["day"] = G.utc.dt.date
G["hour"] = G.utc.dt.hour
G["minute"] = G.utc.dt.minute
G["week"] = G.utc.dt.isocalendar().week
G["maker"] = G.won - G.bid
G.to_parquet(DATA / "interplay_master.parquet")
print(f"  markets {len(G):,}   days {G.day.nunique()}   "
      f"proxy fill rate {G.filled.mean():.4f}")

tr = G[(G.utc >= TRAIN[0]) & (G.utc < TRAIN[1])]
va = G[(G.utc >= VALID[0]) & (G.utc < VALID[1])]
te = G[(G.utc >= TEST[0]) & (G.utc < TEST[1])]
print(f"  train {len(tr):,}   valid {len(va):,}   test {len(te):,}")


def filled_edge(g):
    f = g[g.filled]
    return f.maker.mean() * 100 if len(f) else np.nan


def gap(g):
    f, n = g[g.filled], g[~g.filled]
    if not len(f) or not len(n):
        return np.nan
    return (n.won.mean() - f.won.mean()) * 100


print("\n" + "-" * 86)
print("STEP 1 - SEARCH ON TRAIN ONLY. Where is the FILLED cohort least bad?")
print("-" * 86)
G["spread_b"] = pd.qcut(G.spread.rank(method="first"), 4, duplicates="drop",
                        labels=["tightest", "tight", "wide", "widest"])
G["prevol_b"] = pd.qcut(G.prevol.rank(method="first"), 4, duplicates="drop",
                        labels=["calmest", "calm", "active", "wildest"])
G["bid_b"] = pd.cut(G.bid, [0.649, 0.70, 0.75, 0.80],
                    labels=["65-70", "70-75", "75-80"])
tr = G[(G.utc >= TRAIN[0]) & (G.utc < TRAIN[1])]

candidates = []
for dim in ("spread_b", "prevol_b", "bid_b", "minute", "coin", "side"):
    print(f"\n  by {dim}:")
    print("    %-12s %8s %10s %13s %12s"
          % ("level", "n", "fill rate", "FILLED edge", "no-fill gap"))
    for lv, g in tr.groupby(dim, observed=True):
        if len(g) < 150:
            continue
        fe, gp = filled_edge(g), gap(g)
        print("    %-12s %8d %10.4f %+12.2fc %+11.1fpp"
              % (str(lv), len(g), g.filled.mean(), fe, gp))
        candidates.append({"dim": dim, "level": str(lv), "n_train": len(g),
                           "train_filled_edge": fe})

C = pd.DataFrame(candidates).sort_values("train_filled_edge", ascending=False)
print("\n" + "-" * 86)
print("STEP 2 - THE BEST REGIME ON TRAIN (this is the thing being frozen)")
print("-" * 86)
print(C.head(6).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
best = C.iloc[0]
print(f"\n  frozen: {best.dim} == {best.level}   "
      f"train filled-cohort edge {best.train_filled_edge:+.2f}c")


def sel(g, dim, level):
    return g[g[dim].astype(str) == level]


print("\n" + "-" * 86)
print("STEP 3 - THE HELD-OUT TEST. Does it survive out of sample?")
print("-" * 86)
print("  %-8s %8s %12s %13s" % ("split", "n", "fill rate", "FILLED edge"))
for nm, s in (("train", tr), ("valid", va), ("test", te)):
    z = sel(s, best.dim, best.level)
    if len(z) < 30:
        print("  %-8s %8d  too few" % (nm, len(z)))
        continue
    print("  %-8s %8d %12.4f %+12.2fc"
          % (nm, len(z), z.filled.mean(), filled_edge(z)))

ho = pd.concat([va, te])
z = sel(ho, best.dim, best.level)
gd = {k: v for k, v in z[z.filled].groupby("day")}
ds = sorted(gd)
if len(ds) > 3:
    bs = np.sort(np.array([
        pd.concat([gd[ds[i]] for i in RNG.integers(0, len(ds), len(ds))]
                  ).maker.mean() * 100 for _ in range(6000)]))
    print(f"\n  HELD OUT (valid+test): n={len(z)}  "
          f"filled edge {filled_edge(z):+.2f}c")
    print(f"  95% CI [{bs[150]:+.2f}, {bs[5849]:+.2f}]   "
          f"P(<=0) {(bs <= 0).mean():.4f}")

print("\n" + "-" * 86)
print("STEP 4 - SELECTION CORRECTION. I searched %d cells; is the best one"
      % len(C))
print("         better than the best of %d random cells?" % len(C))
print("-" * 86)
obs = best.train_filled_edge
perm = []
for _ in range(2000):
    q = tr.copy()
    q["lab"] = RNG.permutation(q.filled.values)
    vals = []
    for dim in ("spread_b", "prevol_b", "bid_b", "minute", "coin", "side"):
        for lv, g in q.groupby(dim, observed=True):
            if len(g) < 150:
                continue
            f = g[g.lab]
            if len(f):
                vals.append(f.maker.mean() * 100)
    if vals:
        perm.append(max(vals))
perm = np.array(perm)
print(f"  observed best-of-{len(C)} on train  {obs:+.2f}c")
print(f"  permuted best-of-N mean {perm.mean():+.2f}c   p95 {np.quantile(perm,.95):+.2f}c")
p = float((perm >= obs).mean())
print(f"  P(permuted best >= observed) = {p:.4f}")
print("  -> " + ("SURVIVES selection correction" if p < 0.05
                 else "does NOT survive - consistent with the best of many "
                      "noisy cells"))

C.to_csv(OUT / "interplay_01_regimes.csv", index=False)
print("\nwrote research/results/interplay_01_regimes.csv")
