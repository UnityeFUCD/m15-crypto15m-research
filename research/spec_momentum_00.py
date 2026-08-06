"""User-specified strategy, implemented exactly as written, then stress-tested.

SPEC (frozen verbatim, no reinterpretation)
  universe      BTC ETH SOL XRP DOGE HYPE, 15-minute contracts
  close window  true close minute == :00 UTC, DERIVED FROM THE TICKER
  entry obs     first complete valid quote with 8-14 minutes remaining
  entry cond    original favourite bid >= 0.65 and < 0.80
                favourite side frozen at entry, never redefined later
  wait          exactly two complete minutes
  delayed cond  favourite bid increased by >= 2c
                favourite current ask in [0.60, 0.85]
                spread widened by no more than 1c vs the entry observation
                observed volume >= 2000
  portfolio     at most one market per shared close window
  ranking       1 largest bid increase, 2 narrowest spread, 3 lowest ask,
                4 highest volume, 5 fixed coin order
  execution     IOC buy of the ORIGINAL favourite at the displayed ask,
                no retry, no chase, no probe, hold to settlement
  quantity      q15

WHAT THIS IS, MECHANICALLY
  A momentum filter. Everything previously tested here selected on state
  (price level, minute, calm, fill). This selects on CHANGE - the favourite
  must strengthen by 2c while the book stays orderly. That is a different
  hypothesis and it deserves a clean test.

HOW IT IS JUDGED
  The spec carries roughly five free parameters (2c move, 2-minute wait, 85c
  ceiling, 1c spread widening, 2000 volume). None were fitted here, but none
  were pre-registered either, so the report includes:
    * chronological train / valid / test, with train NOT used to choose
    * day-clustered intervals and leave-one-out on coin and week
    * concentration: what survives dropping the best trades
    * a parameter perturbation grid - a real effect degrades smoothly
    * baselines, so any gain over plain ':00 at the ask' is visible
  Volume is per-candle, not cumulative (median 727 at ml=12), so the 2000
  threshold admits roughly the top quartile of activity.
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
RNG = np.random.default_rng(20260807)

COIN_ORDER = {"BTC": 0, "ETH": 1, "SOL": 2, "XRP": 3, "DOGE": 4, "HYPE": 5}
TRAIN, VALID, TEST = "2026-06-30", "2026-07-18", "2026-08-07"

SPEC = dict(bid_min=0.65, bid_max=0.80, wait_min=2, rise_min=0.02,
            ask_lo=0.60, ask_hi=0.85, spread_widen_max=0.01, vol_min=2000.0,
            qty=15)


def fee_total(qty, price):
    return math.ceil(0.07 * qty * price * (1 - price) * 10_000 - 1e-12) / 10_000


def held(side, yb, ya):
    return (yb, ya) if side == "yes" else (1.0 - ya, 1.0 - yb)


def build(p=SPEC):
    P = pd.read_parquet(DATA / "paths_full.parquet")
    P = P[(P.minute == 0) & P.entry_ml.notna()]        # :00 from the ticker
    rows = []
    for r in P.itertuples():
        try:
            raw = json.loads(r.path)
        except Exception:
            continue
        pts = {}
        for k in raw or []:
            try:
                ml = float(k["ml"]); yb = float(k["yb"]); ya = float(k["ya"])
                v = float(k.get("v") or 0.0)
            except (KeyError, TypeError, ValueError):
                continue
            if 0 < yb < ya < 1:
                pts[round(ml, 3)] = (yb, ya, v)
        # entry: FIRST complete valid quote with 8-14 minutes remaining
        cand = sorted([m for m in pts if 8 <= m <= 14], reverse=True)
        if not cand:
            continue
        e_ml = cand[0]
        eyb, eya, _ = pts[e_ml]
        fav_yes = eyb >= 0.5
        side = "yes" if fav_yes else "no"
        ebid, eask = held(side, eyb, eya)
        if not (p["bid_min"] <= ebid < p["bid_max"]):
            continue
        d_ml = round(e_ml - p["wait_min"], 3)          # exactly two minutes
        if d_ml not in pts:
            continue
        dyb, dya, dv = pts[d_ml]
        dbid, dask = held(side, dyb, dya)              # side FROZEN at entry
        rows.append(dict(
            ticker=r.ticker, coin=r.coin, close_ts=int(r.close_ts),
            side=side, won=int(side == ("yes" if r.won and r.side == "yes"
                                        else r.side if r.won else
                                        ("no" if r.side == "yes" else "yes"))),
            entry_bid=ebid, entry_ask=eask, entry_spread=eask - ebid,
            bid=dbid, ask=dask, spread=dask - dbid, vol=dv,
            rise=dbid - ebid, widen=(dask - dbid) - (eask - ebid),
            coin_order=COIN_ORDER.get(r.coin, 9)))
    D = pd.DataFrame(rows)
    # `won` in paths_full is already relative to its own stored side; recompute
    W = pd.read_parquet(DATA / "paths_full.parquet")[["ticker", "side", "won"]]
    D = D.drop(columns=["won"]).merge(W, on="ticker", how="left",
                                      suffixes=("", "_stored"))
    D["won"] = np.where(D.side == D.side_stored, D.won, 1 - D.won)
    D = D.drop(columns=["side_stored"])
    D["utc"] = pd.to_datetime(D.close_ts, unit="s", utc=True)
    D["day"] = D.utc.dt.date
    D["week"] = D.utc.dt.isocalendar().week
    return D


def apply_spec(D, p=SPEC):
    q = D[(D.rise >= p["rise_min"])
          & D.ask.between(p["ask_lo"], p["ask_hi"])
          & (D.widen <= p["spread_widen_max"])
          & (D.vol >= p["vol_min"])].copy()
    if not len(q):
        return q
    q = q.sort_values(["close_ts", "rise", "spread", "ask", "vol",
                       "coin_order"],
                      ascending=[True, False, True, True, False, True])
    q = q.groupby("close_ts", as_index=False).head(1).copy()
    q["fee"] = [fee_total(p["qty"], a) for a in q.ask]
    q["pnl"] = p["qty"] * (q.won - q.ask) - q.fee
    q["edge_c"] = (q.pnl / p["qty"]) * 100
    return q


def dayboot(g, col="edge_c", n=8000):
    gd = {k: v for k, v in g.groupby("day")}
    ds = sorted(gd)
    if len(ds) < 4:
        return np.nan, np.nan, np.nan
    b = np.sort(np.array([
        pd.concat([gd[ds[i]] for i in RNG.integers(0, len(ds), len(ds))]
                  )[col].mean() for _ in range(n)]))
    return b[int(n * .025)], b[int(n * .975)], float((b <= 0).mean())


D = build()
print("=" * 88)
print("SPEC AS WRITTEN - minute :00 momentum filter")
print("=" * 88)
print(f"  :00 markets with entry in band and a +2min observation: {len(D):,} "
      f"over {D.day.nunique()} days")
print(f"  of these, bid rose >= 2c        : {(D.rise >= .02).sum():,}")
print(f"           ask in [60c, 85c]      : {D.ask.between(.60, .85).sum():,}")
print(f"           spread widened <= 1c   : {(D.widen <= .01).sum():,}")
print(f"           volume >= 2000         : {(D.vol >= 2000).sum():,}")

S = apply_spec(D)
print(f"\n  QUALIFYING TRADES (one per close window): {len(S):,} over "
      f"{S.day.nunique()} days = {len(S)/S.day.nunique():.2f}/day")
if not len(S):
    sys.exit("no qualifying trades")
lo, hi, p0 = dayboot(S)
print(f"\n  win rate      {S.won.mean():.4f}")
print(f"  mean ask      {S.ask.mean()*100:.2f}c")
print(f"  mean rise     {S.rise.mean()*100:.2f}c")
print(f"  fees paid     ${S.fee.sum():.2f}")
print(f"  TOTAL P&L     ${S.pnl.sum():+.2f}  at q{SPEC['qty']}")
print(f"  per contract  {S.edge_c.mean():+.2f}c   95% CI [{lo:+.2f}, {hi:+.2f}]"
      f"   P(<=0) {p0:.4f}")
print(f"  $/day         {S.pnl.sum()/S.day.nunique():+.2f}")

print("\n" + "-" * 88)
print("CHRONOLOGICAL - train was NOT used to choose anything here")
print("-" * 88)
print("  %-8s %7s %9s %12s %14s" % ("split", "n", "win", "per contract",
                                    "$ total"))
for nm, a, b in (("train", "2026-05-25", TRAIN), ("valid", TRAIN, VALID),
                 ("test", VALID, TEST)):
    z = S[(S.utc >= a) & (S.utc < b)]
    if not len(z):
        continue
    print("  %-8s %7d %9.4f %+11.2fc %+13.2f"
          % (nm, len(z), z.won.mean(), z.edge_c.mean(), z.pnl.sum()))

print("\n" + "-" * 88)
print("CONCENTRATION")
print("-" * 88)
s = S.sort_values("pnl")
print(f"  worst trade {s.pnl.iloc[0]:+.2f}   best {s.pnl.iloc[-1]:+.2f}")
for k in (1, 3, 5):
    print(f"  drop best {k}: {s.iloc[:-k].edge_c.mean():+.2f}c   "
          f"drop worst {k}: {s.iloc[k:].edge_c.mean():+.2f}c")
vals = [S[S.coin != c].edge_c.mean() for c in sorted(S.coin.unique())]
print(f"  leave-one-coin-out: min {min(vals):+.2f}c max {max(vals):+.2f}c  "
      f"all>0 {'YES' if min(vals) > 0 else 'NO'}")
wv = [S[S.week != w].edge_c.mean() for w in sorted(S.week.unique())
      if len(S[S.week != w]) > 20]
print(f"  leave-one-week-out: min {min(wv):+.2f}c max {max(wv):+.2f}c  "
      f"all>0 {'YES' if min(wv) > 0 else 'NO'}")

print("\n" + "-" * 88)
print("BASELINES - is the filter earning its keep?")
print("-" * 88)
base = D.copy()
base = base[base.ask.between(0.60, 0.85)]
base = base.sort_values(["close_ts", "coin_order"]).groupby(
    "close_ts", as_index=False).head(1).copy()
base["fee"] = [fee_total(15, a) for a in base.ask]
base["pnl"] = 15 * (base.won - base.ask) - base.fee
base["edge_c"] = base.pnl / 15 * 100
bl, bh, bp = dayboot(base)
print(f"  :00, ask 60-85c, NO momentum filter   n={len(base):4d}  "
       f"{base.edge_c.mean():+.2f}c  CI [{bl:+.2f}, {bh:+.2f}]  P {bp:.4f}")
print(f"  SPEC (with momentum filter)           n={len(S):4d}  "
      f"{S.edge_c.mean():+.2f}c  CI [{lo:+.2f}, {hi:+.2f}]  P {p0:.4f}")
print(f"  filter adds {S.edge_c.mean() - base.edge_c.mean():+.2f}c per contract")

print("\n" + "-" * 88)
print("PARAMETER PERTURBATION - a real effect degrades smoothly")
print("-" * 88)
print("  %-24s %7s %12s %10s" % ("variation", "n", "per contract", "$/day"))
for label, over in (
        ("as specified", {}),
        ("rise >= 1c", {"rise_min": 0.01}),
        ("rise >= 3c", {"rise_min": 0.03}),
        ("rise >= 0c (none)", {"rise_min": 0.0}),
        ("wait 1 min", {"wait_min": 1}),
        ("wait 3 min", {"wait_min": 3}),
        ("volume >= 0", {"vol_min": 0.0}),
        ("volume >= 1000", {"vol_min": 1000.0}),
        ("volume >= 4000", {"vol_min": 4000.0}),
        ("widen <= 0c", {"spread_widen_max": 0.0}),
        ("widen <= 2c", {"spread_widen_max": 0.02}),
        ("ask <= 80c", {"ask_hi": 0.80}),
        ("ask <= 90c", {"ask_hi": 0.90}),
):
    pp = dict(SPEC); pp.update(over)
    dd = build(pp) if "wait_min" in over else D
    z = apply_spec(dd, pp)
    if not len(z):
        print("  %-24s %7d %12s %10s" % (label, 0, "-", "-"))
        continue
    print("  %-24s %7d %+11.2fc %+9.2f"
          % (label, len(z), z.edge_c.mean(),
             z.pnl.sum() / z.day.nunique()))

S.to_csv(OUT / "spec_momentum_00_trades.csv", index=False)
print("\nwrote research/results/spec_momentum_00_trades.csv")
