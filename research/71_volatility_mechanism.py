"""Is VOLATILITY the mechanism behind the intraday edge pattern?

THE HYPOTHESIS (the user's, and it is a better one than mine)
  I found that edge varies with time of day and could not deploy it, because a
  calendar rule with no mechanism fitted to two days of live data is exactly
  the kind of thing that got reversed twice yesterday.

  Volatility is different. It is not a calendar - it is the quantity that
  should MECHANICALLY drive this strategy's edge:

      settlement is A1 >= A0, two 60-second trailing means of the CF Benchmark
      we buy the favourite at ~68c, 8-14 minutes before close
      the favourite is favoured because of where the price sits RIGHT NOW

  The more the underlying moves in the remaining minutes, the more likely that
  standing lead is overturned. So edge should FALL as realised volatility
  rises, and it should do so continuously - not in blocks.

  Crypto volatility has a pronounced intraday cycle that peaks during US
  hours. If that cycle drives the edge cycle, then:
    - the time-of-day pattern is explained rather than merely observed
    - the usable variable is volatility, which is measurable BEFORE entry,
      not the hour, which is only a proxy for it
    - the response is SIZING, continuously, rather than blocking hours

WHAT IS USED AS THE VOLATILITY MEASURE
  The market's own implied probability. yesno.parquet holds the YES price at
  each minute of each window, so the dispersion of that price across the
  window's observed minutes is a direct read on how much the underlying moved
  relative to the threshold. That is exactly the quantity that decides whether
  a standing lead survives - better for this purpose than raw coin volatility,
  which ignores how far the price sits from A0.

THE TESTS, IN ORDER
  1. does the volatility proxy show the expected intraday cycle?
  2. does edge fall as volatility rises, monotonically?
  3. MEDIATION - if volatility is controlled for, does the time-of-day effect
     disappear? That is what distinguishes 'vol is the mechanism' from 'vol and
     time are two correlated symptoms'
  4. is volatility measurable EARLY enough to act on, i.e. does volatility in
     the minutes BEFORE entry predict edge?

  Only test 4 produces something deployable. 1-3 establish whether the story
  is true at all.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261050)

Y = pd.read_parquet(OUT / "yesno.parquet")
Y["yes_px"] = np.where(Y.maker_yes, Y.px, 1.0 - Y.px)
print(f"raw observations {len(Y):,}   windows {Y.wkey.nunique():,}   "
      f"days {Y.day.nunique()}")

# hour comes from grid.parquet, keyed on the same wkey
G = pd.read_parquet(OUT / "grid.parquet")[["wkey", "coin", "hour"]].drop_duplicates()
Y = Y.merge(G, on=["wkey", "coin"], how="left")
Y = Y.dropna(subset=["hour"])
Y["hour"] = Y.hour.astype(int)

# per window-coin: dispersion of the YES price across observed minutes
V = (Y.groupby(["wkey", "coin"])
       .agg(sd=("yes_px", "std"), rng=("yes_px", lambda s: s.max() - s.min()),
            k=("yes_px", "size"), hour=("hour", "first"),
            day=("day", "first"))
       .reset_index())
V = V[V.k >= 4].dropna(subset=["sd"])
print(f"windows with 4+ observations: {len(V):,}\n")

print("=" * 78)
print("TEST 1 - DOES THE VOLATILITY PROXY SHOW AN INTRADAY CYCLE?")
print("=" * 78)
print(f"{'UTC':>5} {'ET':>5} {'windows':>9} {'median sd':>11} {'median range':>13}  ")
prof = {}
for h in range(24):
    g = V[V.hour == h]
    if len(g) < 15:
        continue
    prof[h] = g.sd.median() * 100
    bar = "#" * int(g.sd.median() * 100 * 12)
    print(f"{h:>3}:00 {(h-4)%24:>3}:00 {len(g):>9} {g.sd.median()*100:>10.2f}c "
          f"{g.rng.median()*100:>12.2f}c  {bar}")
if prof:
    hi = max(prof, key=prof.get)
    lo = min(prof, key=prof.get)
    print(f"\n  most volatile hour  {hi:02d}:00 UTC ({(hi-4)%24:02d}:00 ET)  "
          f"{prof[hi]:.2f}c")
    print(f"  least volatile hour {lo:02d}:00 UTC ({(lo-4)%24:02d}:00 ET)  "
          f"{prof[lo]:.2f}c")
    print(f"  ratio {prof[hi]/max(prof[lo],1e-9):.2f}x")

print("\n" + "=" * 78)
print("TEST 2 - DOES EDGE FALL AS VOLATILITY RISES?")
print("=" * 78)
q = Y[(Y.px >= 0.65) & (Y.px < 0.80) & (Y.minute >= 8) & (Y.minute <= 14)
      & (Y.vol >= 2000)]
E = (q.sort_values("minute", ascending=False)
       .groupby(["wkey", "coin"], as_index=False).first())
E = E.merge(V[["wkey", "coin", "sd", "rng"]], on=["wkey", "coin"], how="inner")
E = (E.sort_values(["wkey", "minute"], ascending=[True, False])
       .groupby("wkey", as_index=False).head(3)).copy()
print(f"  entries with a volatility measure: {len(E):,}")
E["vb"] = pd.qcut(E.sd, 5, labels=["lowest", "low", "mid", "high", "highest"])
print(f"\n{'volatility':>12} {'entries':>9} {'median sd':>11} {'avg px':>8} "
      f"{'win':>8} {'edge/ct':>9}")
for k, g in E.groupby("vb", observed=True):
    print(f"{str(k):>12} {len(g):>9} {g.sd.median()*100:>10.2f}c "
          f"{g.px.mean()*100:>7.1f}c {g.won.mean():>8.4f} "
          f"{g.edge.mean()*100:>+8.2f}c")
r = E.sd.corr(E.edge)
days = sorted(E.day.unique())
gd = {d: x for d, x in E.groupby("day")}
bs = np.sort(np.array([
    (lambda s: s.sd.corr(s.edge))(
        pd.concat([gd[days[i]] for i in RNG.integers(0, len(days), len(days))]))
    for _ in range(6000)]))
print(f"\n  corr(volatility, edge) = {r:+.4f}   "
      f"95% CI [{bs[150]:+.4f}, {bs[5849]:+.4f}]   P(>=0) {(bs >= 0).mean():.4f}")

print("\n" + "=" * 78)
print("TEST 3 - MEDIATION: does volatility EXPLAIN the time-of-day effect?")
print("=" * 78)
E["blk"] = np.where(E.hour < 8, "00-07", np.where(E.hour < 14, "08-13", "14-23"))
print(f"{'block':>8} {'entries':>9} {'median sd':>11} {'raw edge':>10} "
      f"{'vol-adjusted edge':>19}")
E["vresid"] = E.edge - E.groupby("vb", observed=True).edge.transform("mean")
for k, g in E.groupby("blk"):
    print(f"{k:>8} {len(g):>9} {g.sd.median()*100:>10.2f}c "
          f"{g.edge.mean()*100:>+9.2f}c {g.vresid.mean()*100:>+18.2f}c")
a, b = E[E.blk == "00-07"], E[E.blk != "00-07"]
raw = (a.edge.mean() - b.edge.mean()) * 100
adj = (a.vresid.mean() - b.vresid.mean()) * 100
print(f"\n  00-07 minus rest:  raw {raw:+.2f}c   vol-adjusted {adj:+.2f}c")
if abs(raw) > 1e-9:
    print(f"  volatility explains {(1 - adj/raw)*100:.1f}% of the time effect")
print("\n  If the vol-adjusted gap collapses toward zero, volatility IS the")
print("  mechanism and the hour was only ever a proxy for it.")

print("\n" + "=" * 78)
print("TEST 4 - IS IT KNOWABLE BEFORE WE ENTER?")
print("=" * 78)
print("  Tests 1-3 use volatility measured across the WHOLE window, which")
print("  includes minutes after our entry. That is fine for explaining the")
print("  past and useless for trading. This uses only minutes 14 and earlier -")
print("  strictly before the entry decision.\n")
EARLY = (Y[Y.minute >= 12].groupby(["wkey", "coin"])
           .agg(esd=("yes_px", "std"), ek=("yes_px", "size")).reset_index())
EARLY = EARLY[EARLY.ek >= 3].dropna(subset=["esd"])
E2 = E.merge(EARLY[["wkey", "coin", "esd"]], on=["wkey", "coin"], how="inner")
print(f"  entries with an EARLY volatility measure: {len(E2):,}")
if len(E2) > 200:
    E2["eb"] = pd.qcut(E2.esd, 4, labels=["lowest", "low", "high", "highest"],
                       duplicates="drop")
    print(f"\n{'early vol':>12} {'entries':>9} {'avg px':>8} {'win':>8} "
          f"{'edge/ct':>9} {'$/day at qty20':>16}")
    nd = E2.day.nunique()
    for k, g in E2.groupby("eb", observed=True):
        print(f"{str(k):>12} {len(g):>9} {g.px.mean()*100:>7.1f}c "
              f"{g.won.mean():>8.4f} {g.edge.mean()*100:>+8.2f}c "
              f"{g.edge.sum()*20/nd:>+15.2f}")
    r2 = E2.esd.corr(E2.edge)
    gd2 = {d: x for d, x in E2.groupby("day")}
    d2 = sorted(gd2)
    bs2 = np.sort(np.array([
        (lambda s: s.esd.corr(s.edge))(
            pd.concat([gd2[d2[i]] for i in RNG.integers(0, len(d2), len(d2))]))
        for _ in range(6000)]))
    print(f"\n  corr(EARLY volatility, edge) = {r2:+.4f}   "
          f"95% CI [{bs2[150]:+.4f}, {bs2[5849]:+.4f}]   "
          f"P(>=0) {(bs2 >= 0).mean():.4f}")
    print("\n  This is the number that decides whether anything is deployable.")
    print("  Early vol predicting edge = a usable, mechanistic signal.")
    print("  Early vol NOT predicting = the volatility story is real but")
    print("  arrives too late to trade, same as the reachability finding.")
E.to_parquet(OUT / "vol_entries.parquet")
print("\nwrote vol_entries.parquet")
