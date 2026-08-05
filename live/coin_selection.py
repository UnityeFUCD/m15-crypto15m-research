"""Should DOGE and SOL be dropped from the series list?

WHY THIS IS THE STRONGEST REMAINING CANDIDATE
  Post-shift edge by coin, in the live band, is not subtle:
      ETH   +12.97c  (n=100)
      XRP    +5.42c  (n=101)
      HYPE   +4.53c  (n= 73)
      SOL    -1.83c  (n= 47)
      DOGE   -2.32c  (n=105)

  Unlike every other lever tested today, this one does not require predicting
  anything. Coin identity is known before the order is placed, it is stable, and
  dropping a series is a configuration change rather than a strategy change.

  It also differs from the intraday result in the way that matters: the
  intraday tilt needs the pre-shift CURVE, fitted on coins whose edge has since
  moved. Coin identity carries no such fitted object.

WHY IT IS NOT OBVIOUS
  An earlier pass found DOGE's interval running [-4.21, +35.00] - wide and
  mostly positive. Three days is not much, and per-coin slices are thin. A coin
  can look negative for a week on ordinary variance.

  There is also a portfolio argument against dropping: entries are capped at 3
  per window, and fewer eligible series means fewer windows reach the cap, so
  removing a coin removes entries rather than replacing them. That is exactly
  how the hours filter destroyed $34-96/day today.

TWO INDEPENDENT DATASETS, both post-shift
  1. the market-data panel (postshift_nofilter) - all candidate entries
  2. the live settled book - our actual fills at our actual prices
  They overlap in time but not in construction. Agreement between them is
  worth more than either alone.

THE BAR
  Dropping a coin means discarding every one of its entries, so the discarded
  cohort must be verifiably NEGATIVE - a wide interval straddling zero is a
  reason to keep trading it, not to cut it. And the comparison must be in
  DOLLARS against keep-all, accounting for the entries not replaced.
"""
import glob
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from acct import api

RNG = np.random.default_rng(20261041)
RES = Path("../research/final_minute_favorite_maker_validation")
QTY = 20


def ts(s):
    try:
        return datetime.fromisoformat((s or "").replace("Z", "+00:00"))
    except Exception:
        return None


# ---------- dataset 1: live settled fills ----------
PX = {}
for fp in sorted(glob.glob("lsm_*.jsonl")) + sorted(glob.glob("lsm_*.log")):
    try:
        for ln in open(fp, errors="ignore"):
            ln = ln.strip()
            if not ln.startswith("{"):
                continue
            try:
                d = json.loads(ln)
            except Exception:
                continue
            if d.get("ev") == "post" and d.get("ticker") and d.get("price"):
                PX[d["ticker"]] = float(d["price"])
    except Exception:
        pass
rows, cur, seen = [], None, set()
for _ in range(12):
    p = {"limit": 200, "settlement_status": "settled"}
    if cur:
        p["cursor"] = cur
    c, j = api("/portfolio/settlements", p)
    if c != 200 or not j:
        break
    b = j.get("settlements") or []
    if not b:
        break
    for s in b:
        if s.get("ticker") and s["ticker"] not in seen:
            seen.add(s["ticker"])
            rows.append(s)
    cur = j.get("cursor")
    if not cur:
        break
out = []
for s in rows:
    t = s.get("ticker")
    if t not in PX:
        continue
    n = float(s.get("yes_count_fp") or 0) + float(s.get("no_count_fp") or 0)
    cost = float(s.get("yes_total_cost_dollars") or 0) + float(
        s.get("no_total_cost_dollars") or 0)
    if n <= 0 or cost <= 0:
        continue
    rev = float(s.get("revenue") or 0) / 100.0
    out.append(dict(coin=t.split("-")[0].replace("KX", "").replace("15M", ""),
                    n=n, px=cost / n, window=t.split("-")[1],
                    pnl=rev - cost - float(s.get("fee_cost") or 0)))
L = pd.DataFrame(out)
print(f"LIVE  fills {len(L)}  contracts {L.n.sum():.0f}  "
      f"net ${L.pnl.sum():+.2f} ({L.pnl.sum()/L.n.sum()*100:+.2f}c/ct)")

# ---------- dataset 2: post-shift market panel ----------
P = pd.read_parquet(RES / "postshift_nofilter.parquet")
P = P[(P.vol >= 2000) & (P.px >= 0.65) & (P.px < 0.80)].copy()
P = (P.sort_values(["wkey", "px"], ascending=[True, False])
       .groupby("wkey", as_index=False).head(3)).copy()
print(f"PANEL entries {len(P)}  days {P.day.nunique()}  "
      f"edge {P.edge.mean()*100:+.2f}c\n")

print("=" * 78)
print("PER-COIN EDGE, BOTH DATASETS, WITH INTERVALS")
print("=" * 78)
print(f"{'coin':>6} | {'PANEL n':>8} {'edge':>8} {'95% CI':>18} {'P(>=0)':>8} "
      f"| {'LIVE n':>7} {'edge':>8} {'95% CI':>18}")
coins = sorted(set(P.coin) | set(L.coin))
verdict = {}
for c in coins:
    g = P[P.coin == c]
    if len(g) >= 20:
        gd = {d: x for d, x in g.groupby("day")}
        ds = sorted(gd)
        bs = np.sort(np.array([
            np.concatenate([gd[ds[i]].edge.values
                            for i in RNG.integers(0, len(ds), len(ds))]).mean() * 100
            for _ in range(6000)]))
        pe, plo, phi = g.edge.mean() * 100, bs[150], bs[5849]
        pge = (bs >= 0).mean()
    else:
        pe = plo = phi = pge = float("nan")
    gl = L[L.coin == c]
    if len(gl) >= 10:
        wl = sorted(gl.window.unique())
        gwl = {w: x for w, x in gl.groupby("window")}
        bs2 = np.sort(np.array([
            (lambda s: s.pnl.sum() / max(s.n.sum(), 1) * 100)(
                pd.concat([gwl[wl[i]] for i in RNG.integers(0, len(wl), len(wl))]))
            for _ in range(6000)]))
        le, llo, lhi = gl.pnl.sum() / gl.n.sum() * 100, bs2[150], bs2[5849]
    else:
        le = llo = lhi = float("nan")
    verdict[c] = (pe, plo, phi, pge, le, llo, lhi, len(g), len(gl))
    print(f"{c:>6} | {len(g):>8} {pe:>+7.2f}c [{plo:>+7.2f},{phi:>+7.2f}] "
          f"{pge:>8.4f} | {len(gl):>7} {le:>+7.2f}c [{llo:>+7.2f},{lhi:>+7.2f}]")

print("\n  A coin is a DROP candidate only if BOTH intervals sit below zero.")
print("  An interval straddling zero means we cannot tell, and the default is")
print("  to keep trading it.\n")
drops = [c for c, v in verdict.items()
         if not np.isnan(v[2]) and not np.isnan(v[6]) and v[2] < 0 and v[6] < 0]
print(f"  coins failing BOTH: {drops or 'NONE'}")

print("\n" + "=" * 78)
print("DOLLARS: does dropping a coin actually help? (panel, cap 3 re-applied)")
print("=" * 78)
RAW = pd.read_parquet(RES / "postshift_nofilter.parquet")
RAW = RAW[(RAW.vol >= 2000) & (RAW.px >= 0.65) & (RAW.px < 0.80)].copy()
days = sorted(RAW.day.unique())


def sim(exclude):
    q = RAW[~RAW.coin.isin(exclude)]
    q = (q.sort_values(["wkey", "px"], ascending=[True, False])
           .groupby("wkey", as_index=False).head(3))
    d = q.groupby("day").edge.sum().reindex(days, fill_value=0) * QTY
    return d, len(q)


base, nb = sim([])
print(f"{'series set':>26} {'entries':>8} {'ent/day':>9} {'edge/ct':>9} "
      f"{'$/day':>9} {'vs keep-all':>13}")
print(f"{'KEEP ALL (deployed)':>26} {nb:>8} {nb/len(days):>9.1f} "
      f"{RAW.groupby('wkey').head(3).edge.mean()*100:>+8.2f}c "
      f"{base.mean():>+9.2f} {'-':>13}")
for ex in (["DOGE"], ["SOL"], ["DOGE", "SOL"]):
    d, n = sim(ex)
    q = RAW[~RAW.coin.isin(ex)]
    q = (q.sort_values(["wkey", "px"], ascending=[True, False])
           .groupby("wkey", as_index=False).head(3))
    diff = (d - base).values
    bs = np.sort(np.array([RNG.choice(diff, len(diff), True).mean()
                           for _ in range(8000)]))
    print(f"{'drop ' + '+'.join(ex):>26} {n:>8} {n/len(days):>9.1f} "
          f"{q.edge.mean()*100:>+8.2f}c {d.mean():>+9.2f} "
          f"{diff.mean():>+7.2f} [{bs[200]:+.0f},{bs[7799]:+.0f}]")

print("\n" + "=" * 78)
print("THE ENTRY-REPLACEMENT QUESTION")
print("=" * 78)
W = RAW.groupby("wkey").coin.nunique()
print(f"  windows with 3+ eligible coins: {(W >= 3).mean():.3f}")
for ex in (["DOGE"], ["DOGE", "SOL"]):
    W2 = RAW[~RAW.coin.isin(ex)].groupby("wkey").coin.nunique()
    print(f"  after dropping {'+'.join(ex):<9}: {(W2 >= 3).mean():.3f}   "
          f"windows losing a slot: {((W >= 3) & (W2.reindex(W.index).fillna(0) < 3)).sum()}")
print("\n  If most windows already have exactly 3 eligible coins, dropping one")
print("  does not upgrade an entry - it deletes one. That is how the hours")
print("  filter lost $34-96/day today.")
