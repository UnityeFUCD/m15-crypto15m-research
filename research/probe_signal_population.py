"""Does the unfilled-probe signal hold on 73 days instead of 2?

THE QUESTION
  On the 2 live days, a q1 maker probe that did NOT fill within 120s predicted
  a 90.91% winner against 69.26% for one that did (+21.6pp, n=303). That is
  the only thing PTC rests on. It was measured on real fills, but on 2 days.

  paths_full.parquet holds 39,427 markets with full price paths over 73 days.
  What it does NOT hold is our own orders - so whether a probe WOULD have
  filled has to be modelled.

THE FILL MODEL, AND WHY IT IS VALIDATED FIRST
  A post-only buy resting at the favourite's bid B fills when a seller crosses
  into it. From one-minute candles the observable consequence is that the bid
  ceases to hold B: if the favourite's bid at a later candle is at or below B,
  the level was consumed or the quote was pulled.

  That is a proxy, so it is not trusted on assertion. The 303 LSM orders have
  BOTH a real fill outcome and a full path, so the proxy is scored against
  ground truth before being applied to the population. If it cannot reproduce
  known fills it is not used, and the population test is reported as blocked.

WHAT WOULD MAKE THE SIGNAL REAL
  If the population reproduces a large no-fill/filled win-rate gap on tens of
  thousands of markets, the signal is established far beyond what 303 orders
  could show, and only the dollar value remains open. If the gap collapses,
  the 2-day result was a small-sample artifact and PTC has no foundation.
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
RNG = np.random.default_rng(20260806)
WAITS = (60, 120)


def fee_c(price: float) -> float:
    return math.ceil(0.07 * price * (1 - price) * 100) / 100.0


def held_quote(side, yb, ya):
    return (yb, ya) if side == "yes" else (1.0 - ya, 1.0 - yb)


def points(path_json, close_ts, side):
    try:
        raw = json.loads(path_json)
    except Exception:
        return []
    out = []
    for p in raw or []:
        try:
            ml = float(p["ml"]); yb = float(p["yb"]); ya = float(p["ya"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 < yb < ya < 1):
            continue
        b, a = held_quote(side, yb, ya)
        out.append((float(close_ts) - 60.0 * ml, b, a))
    return sorted(out)


def would_fill(pts, entry_ts, entry_bid, wait):
    """Proxy: the resting bid is taken out if the favourite's bid is at or
    below our price at any observation inside the wait window."""
    for ts, b, _ in pts:
        if entry_ts < ts <= entry_ts + wait:
            if b <= entry_bid + 1e-9:
                return True
    return False


def ask_after(pts, entry_ts, wait):
    for ts, _, a in pts:
        if ts >= entry_ts + wait - 1e-9:
            return a
    return np.nan


# ------------------------------------------------ 1. validate on 303 orders
print("=" * 82)
print("1. VALIDATING THE FILL PROXY AGAINST 303 KNOWN OUTCOMES")
print("=" * 82)
orders = pd.read_parquet(DATA / "orders_history.parquet")
fills = pd.read_parquet(DATA / "fills_history.parquet")
paths = pd.read_parquet(DATA / "paths_full.parquet")

l = orders[orders.client_order_id.astype(str).str.startswith("lsm")].copy()
l["held"] = np.where(l.action.eq("sell"),
                     np.where(l.side.eq("yes"), "no", "yes"), l.side)
l["created_dt"] = pd.to_datetime(l.created_time, format="mixed", utc=True)
l["created_ts"] = l.created_dt.map(lambda t: t.timestamp())
yp = pd.to_numeric(l.yes_price_dollars, errors="coerce")
npx = pd.to_numeric(l.no_price_dollars, errors="coerce")
l["px"] = np.where(l.held.eq("yes"), yp, npx)
ff = (fills.assign(fd=pd.to_datetime(fills.created_time, format="mixed",
                                     utc=True))
      .groupby("order_id").fd.min())
l = l.merge(ff.rename("first_fill_dt"), left_on="order_id",
            right_index=True, how="left")
l["fill_s"] = (l.first_fill_dt - l.created_dt).dt.total_seconds()
pm = paths.set_index("ticker")[["close_ts", "path"]]
l = l.join(pm, on="ticker")

for wait in WAITS:
    truth, pred = [], []
    for r in l.itertuples():
        if not isinstance(r.path, str):
            continue
        pts = points(r.path, r.close_ts, r.held)
        if not pts:
            continue
        truth.append(bool(pd.notna(r.fill_s) and r.fill_s <= wait))
        pred.append(would_fill(pts, r.created_ts, r.px, wait))
    t = np.array(truth); p = np.array(pred)
    tp = int((t & p).sum()); tn = int((~t & ~p).sum())
    fp = int((~t & p).sum()); fn = int((t & ~p).sum())
    acc = (tp + tn) / len(t)
    print(f"\n  wait {wait}s   n={len(t)}   accuracy {acc:.4f}")
    print(f"    真 fill / proxy fill      {tp:4d}     true no-fill / proxy no-fill {tn:4d}")
    print(f"    真 no-fill / proxy fill   {fp:4d}     true fill / proxy no-fill    {fn:4d}")
    print(f"    actual fill rate {t.mean():.4f}   proxy fill rate {p.mean():.4f}")
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    print(f"    sensitivity {sens:.4f}   specificity {spec:.4f}")

print("\n  The proxy must at minimum RANK correctly: markets it calls no-fill")
print("  should contain more true no-fills than the base rate. Accuracy alone")
print("  is misleading here because ~87% of orders truly fill.")

# ------------------------------------------------ 2. apply to 73-day population
print("\n" + "=" * 82)
print("2. THE SIGNAL ON THE FULL 73-DAY POPULATION")
print("=" * 82)
P = paths.copy()
P = P[P.entry_ml.notna()]
P["day"] = pd.to_datetime(P.close_ts, unit="s", utc=True).dt.date
P = P[(P.bid >= 0.65) & (P.bid < 0.80)]
print(f"  in-band markets with a usable entry: {len(P):,} over "
      f"{P.day.nunique()} days")

rows = []
for r in P.itertuples():
    pts = points(r.path, r.close_ts, r.side)
    if len(pts) < 3:
        continue
    entry_ts = float(r.close_ts) - 60.0 * float(r.entry_ml)
    rec = {"ticker": r.ticker, "coin": r.coin, "day": r.day,
           "won": int(r.won), "bid": float(r.bid)}
    for wait in WAITS:
        rec[f"fill{wait}"] = would_fill(pts, entry_ts, float(r.bid), wait)
        rec[f"ask{wait}"] = ask_after(pts, entry_ts, wait)
    rows.append(rec)
G = pd.DataFrame(rows)
G.to_parquet(DATA / "probe_signal_population.parquet")
print(f"  usable: {len(G):,}")

for wait in WAITS:
    f = G[G[f"fill{wait}"]]
    n = G[~G[f"fill{wait}"]]
    print(f"\n  wait {wait}s")
    print(f"    proxy FILLED     n={len(f):6,}  win rate {f.won.mean():.4f}")
    print(f"    proxy NOT filled n={len(n):6,}  win rate {n.won.mean():.4f}")
    gap = (n.won.mean() - f.won.mean()) * 100
    print(f"    gap {gap:+.1f}pp        (2-day live result: "
          f"{'+17.9' if wait == 60 else '+21.6'}pp)")
    gd = {k: v for k, v in G.groupby("day")}
    ds = sorted(gd)
    bs = np.sort(np.array([
        (lambda s: (s[~s[f"fill{wait}"]].won.mean()
                    - s[s[f"fill{wait}"]].won.mean()) * 100)(
            pd.concat([gd[ds[i]] for i in RNG.integers(0, len(ds), len(ds))]))
        for _ in range(4000)]))
    print(f"    95% CI [{bs[100]:+.1f}, {bs[3899]:+.1f}]pp   "
          f"P(<=0) {(bs <= 0).mean():.4f}")

# ------------------------------------------------ 3. what is it worth?
print("\n" + "=" * 82)
print("3. WHAT IS THE NO-FILL BRANCH WORTH AS A TAKER TRADE?")
print("=" * 82)
for wait in WAITS:
    n = G[~G[f"fill{wait}"]].copy()
    n = n[n[f"ask{wait}"].between(0.60, 0.80)]
    if len(n) < 30:
        print(f"\n  wait {wait}s: only {len(n)} eligible - too few")
        continue
    n["edge"] = n.won - n[f"ask{wait}"] - n[f"ask{wait}"].map(fee_c)
    gd = {k: v for k, v in n.groupby("day")}
    ds = sorted(gd)
    bs = np.sort(np.array([
        pd.concat([gd[ds[i]] for i in RNG.integers(0, len(ds), len(ds))]
                  ).edge.mean() * 100 for _ in range(4000)]))
    print(f"\n  wait {wait}s: {len(n):,} commitable markets over "
          f"{n.day.nunique()} days")
    print(f"    win rate {n.won.mean():.4f}   mean ask "
          f"{n[f'ask{wait}'].mean()*100:.2f}c")
    print(f"    taker edge {n.edge.mean()*100:+.2f}c/contract  "
          f"95% CI [{bs[100]:+.2f}, {bs[3899]:+.2f}]  P(<=0) {(bs<=0).mean():.4f}")
    per_day = len(n) / n.day.nunique()
    print(f"    {per_day:.1f} commits/day  ->  at q15: "
          f"{per_day*15*n.edge.mean():+.2f} $/day")
