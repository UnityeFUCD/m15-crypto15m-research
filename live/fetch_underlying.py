"""Reconstruct the true CF Benchmark BRTI series from market metadata.

WHY THIS EXISTS
  Every volatility measure used so far was a proxy built from Kalshi's implied
  probability, and it was contaminated: a favourite bought at 72c that loses
  travels 72->0 while a winner travels 72->100, so losers generate more price
  dispersion by construction. That made contemporaneous "volatility" correlate
  with edge at -0.43 for reasons that are pure accounting.

  The market object carries the real thing:

      floor_strike        = A0, the 60-second BRTI mean at window OPEN
      expiration_value    = A1, the 60-second BRTI mean at window CLOSE
      result              = yes iff A1 >= A0

  A1 of one window and A0 of the next are the same instant, so stringing them
  together gives the underlying index at 15-minute resolution, per coin.

  Underlying volatility is symmetric with respect to our outcome - a large move
  is equally large whether it goes our way or against us. So it can be used to
  predict, which the implied-probability proxy could not.

WHAT THIS ENABLES
  - true realised return per window, |A1-A0|/A0
  - true PRIOR volatility from preceding windows, strictly known at entry
  - the actual distribution of move sizes, which is what decides whether a
    standing lead survives

CACHING
  Results are written to underlying.parquet. The fetch is the expensive part
  and must not be repeated on every analysis.
"""
import sys
import time
from pathlib import Path

import pandas as pd

from acct import api

OUT = Path(__file__).resolve().parent / "underlying.parquet"
SERIES = ["KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M", "KXDOGE15M",
          "KXHYPE15M"]

rows, seen = [], set()
if OUT.exists():
    old = pd.read_parquet(OUT)
    rows = old.to_dict("records")
    seen = set(old.ticker)
    print(f"cache: {len(rows):,} markets already stored")

for s in SERIES:
    cur, got, pages = None, 0, 0
    while pages < 40:
        p = {"series_ticker": s, "limit": 1000, "status": "settled"}
        if cur:
            p["cursor"] = cur
        c, j = api("/markets", p)
        if c != 200 or not j:
            break
        b = j.get("markets") or []
        if not b:
            break
        for m in b:
            t = m.get("ticker")
            if not t or t in seen:
                continue
            fs, ev = m.get("floor_strike"), m.get("expiration_value")
            if fs is None or ev in (None, ""):
                continue
            try:
                a0, a1 = float(fs), float(ev)
            except (TypeError, ValueError):
                continue
            if a0 <= 0:
                continue
            seen.add(t)
            rows.append(dict(
                ticker=t, series=s,
                coin=s.replace("KX", "").replace("15M", ""),
                wkey=t.split("-")[1],
                close=m.get("expected_expiration_time") or m.get("close_time"),
                a0=a0, a1=a1, ret=(a1 - a0) / a0,
                result=m.get("result"),
                volume=float(m.get("volume_fp") or 0)))
            got += 1
        cur = j.get("cursor")
        pages += 1
        if not cur:
            break
        time.sleep(0.05)
    print(f"  {s:>10}: +{got} new")

D = pd.DataFrame(rows).drop_duplicates(subset=["ticker"])
D = D[D.result.isin(["yes", "no"])]
D.to_parquet(OUT)
print(f"\nstored {len(D):,} settled markets across {D.coin.nunique()} coins")
print(f"date span: {D.close.min()} .. {D.close.max()}")

print("\n" + "=" * 74)
print("SANITY - DOES A1 >= A0 REPRODUCE THE SETTLED RESULT?")
print("=" * 74)
D["implied"] = (D.a1 >= D.a0).map({True: "yes", False: "no"})
agree = (D.implied == D.result).mean()
print(f"  agreement: {agree:.6f} on {len(D):,} markets")
bad = D[D.implied != D.result]
if len(bad):
    print(f"  {len(bad)} disagreements - inspecting the closest:")
    bad = bad.assign(gap=(bad.a1 - bad.a0).abs()).nsmallest(5, "gap")
    for r in bad.itertuples():
        print(f"    {r.ticker:>34} a0 {r.a0:.2f} a1 {r.a1:.2f} "
              f"gap {r.a1-r.a0:+.4f} result {r.result}")
    print("  (ties pay YES, and values are rounded to 2dp, so exact ties are")
    print("   expected at the boundary)")
if agree < 0.99:
    print("\n  WARNING: reconstruction disagrees with settlement too often.")
    print("  Do not build anything on this until it is understood.")
    sys.exit(1)

print("\n" + "=" * 74)
print("THE MOVE-SIZE DISTRIBUTION - how thin are these decisions?")
print("=" * 74)
D["abs_bp"] = D.ret.abs() * 1e4
print(f"{'coin':>7} {'markets':>9} {'median |move|':>14} {'p25':>9} {'p75':>9} "
      f"{'p95':>9} {'YES rate':>10}")
for c, g in D.groupby("coin"):
    print(f"{c:>7} {len(g):>9} {g.abs_bp.median():>12.2f}bp "
          f"{g.abs_bp.quantile(.25):>8.2f} {g.abs_bp.quantile(.75):>8.2f} "
          f"{g.abs_bp.quantile(.95):>8.2f} {(g.result=='yes').mean():>10.4f}")
print("\n  1bp = 0.01%. If the median 15-minute move is only a few basis")
print("  points, these outcomes are decided by very thin margins - which is")
print("  exactly why volatility should dominate the win rate.")
