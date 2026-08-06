"""Triple-check the :00 decay: is it real, and WHAT decayed?

THE DECOMPOSITION THAT DISTINGUISHES TWO FUTURES
  edge = win_rate - ask - fee

  so a falling edge has exactly two possible sources, and they imply opposite
  things about whether it comes back:

    A  THE ASK ROSE - the market repriced toward the true probability.
       That is "competed away". It does not come back on its own, because
       the price now reflects reality.

    B  THE WIN RATE FELL while the ask stayed flat - the market did NOT
       reprice; the underlying simply started resolving differently.
       That is a regime effect, and regimes revert.

  The narrative "market participants learned" is claim A. It has never been
  checked. This decomposes the observed decay into the two components.

FURTHER CHECKS, because three bins falling in order is weak evidence
  2  composition - are we trading the same coins, hours and price levels in
     July as in May? A drifting mix produces fake decay.
  3  is the "other minutes stayed flat" claim solid, or did they decay too?
  4  robustness - does the decay survive alternative edge definitions and
     alternative trade-selection rules?
  5  a crypto-wide regime control: did the whole market get harder in July?
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"
RNG = np.random.default_rng(20260807)


def fee_c(p):
    return math.ceil(0.07 * p * (1 - p) * 100) / 100.0


B = pd.read_parquet(DATA / "book_full.parquet")
B["utc"] = pd.to_datetime(B.close_ts, unit="s", utc=True)
B["day"] = B.utc.dt.date
B["date"] = pd.to_datetime(B.utc.dt.date)
B["hour"] = B.utc.dt.hour
B = B[(B.bid >= 0.65) & (B.bid < 0.80)].copy()
B["fee"] = B.ask.map(fee_c)
B["edge"] = (B.won - B.ask - B.fee) * 100
Z = B[B.minute == 0]
O = B[B.minute != 0]
half = B.date.min() + (B.date.max() - B.date.min()) / 2

print("=" * 84)
print("1. WHAT DECAYED - the ask, or the win rate?")
print("=" * 84)
print("  %-14s %7s %10s %10s %10s %11s"
      % ("period", "n", "mean ask", "win rate", "fee", "edge"))
cuts = [("May25-Jun14", "2026-05-25", "2026-06-15"),
        ("Jun15-Jun30", "2026-06-15", "2026-07-01"),
        ("Jul01-Jul15", "2026-07-01", "2026-07-16"),
        ("Jul16-Aug05", "2026-07-16", "2026-08-07")]
rows = []
for nm, a, b in cuts:
    g = Z[(Z.utc >= a) & (Z.utc < b)]
    rows.append((nm, len(g), g.ask.mean(), g.won.mean(), g.fee.mean(),
                 g.edge.mean()))
    print("  %-14s %7d %10.4f %10.4f %10.4f %+10.2fc"
          % (nm, len(g), g.ask.mean(), g.won.mean(), g.fee.mean(),
             g.edge.mean()))
first, last = rows[0], rows[-1]
d_ask = (last[2] - first[2]) * 100
d_win = (last[3] - first[3]) * 100
d_edge = last[5] - first[5]
print(f"\n  change first -> last:")
print(f"    ask       {d_ask:+.2f}c   (a RISE means the market repriced)")
print(f"    win rate  {d_win:+.2f}pp  (a FALL means outcomes changed)")
print(f"    edge      {d_edge:+.2f}c")
print(f"\n  attribution of the {d_edge:+.2f}c edge change:")
print(f"    from the ask moving        {-d_ask:+.2f}c   "
      f"({100*abs(d_ask)/max(abs(d_edge),1e-9):.0f}%)")
print(f"    from the win rate moving   {d_win:+.2f}c   "
      f"({100*abs(d_win)/max(abs(d_edge),1e-9):.0f}%)")
print("\n  -> " + ("MARKET REPRICED (competed away, does not return)"
                   if abs(d_ask) > abs(d_win) else
                   "WIN RATE FELL with the ask ~flat: the market did NOT\n"
                   "     reprice. That is a regime/outcome change, not learning."))

print("\n" + "=" * 84)
print("2. COMPOSITION - are we trading the same thing in July as in May?")
print("=" * 84)
e, l = Z[Z.date <= half], Z[Z.date > half]
print("  %-16s %14s %14s" % ("feature", "first half", "second half"))
print("  %-16s %14.4f %14.4f" % ("mean entry bid", e.bid.mean(), l.bid.mean()))
print("  %-16s %14.4f %14.4f" % ("mean ask", e.ask.mean(), l.ask.mean()))
print("  %-16s %14.4f %14.4f" % ("mean spread", (e.ask-e.bid).mean(),
                                 (l.ask-l.bid).mean()))
print("  %-16s %14.1f %14.1f" % ("markets/day", len(e)/e.day.nunique(),
                                 len(l)/l.day.nunique()))
print("  coin mix:")
for c in sorted(Z.coin.unique()):
    print("    %-6s %14.4f %14.4f"
          % (c, (e.coin == c).mean(), (l.coin == c).mean()))
print("  -> composition is %s"
      % ("STABLE" if abs(e.bid.mean()-l.bid.mean()) < 0.01
         and abs(e.ask.mean()-l.ask.mean()) < 0.02 else "DRIFTING"))

print("\n" + "=" * 84)
print("3. DID THE OTHER MINUTES DECAY TOO?")
print("=" * 84)
print("  %-8s %14s %14s %12s" % ("minute", "first half", "second half",
                                 "change"))
for mn, g in B.groupby("minute"):
    a = g[g.date <= half].edge.mean()
    b = g[g.date > half].edge.mean()
    print("  :%02d      %+13.2fc %+13.2fc %+11.2fc" % (mn, a, b, b - a))
za = Z[Z.date <= half].edge.mean() - Z[Z.date > half].edge.mean()
oa = O[O.date <= half].edge.mean() - O[O.date > half].edge.mean()
print(f"\n  :00 fell {za:+.2f}c   others fell {oa:+.2f}c   "
      f"difference {za-oa:+.2f}c")

print("\n" + "=" * 84)
print("4. ROBUSTNESS - does the decay survive other definitions?")
print("=" * 84)
print("  %-30s %12s %12s %11s" % ("edge definition", "first half",
                                  "second half", "change"))
defs = [("taker edge (ask + fee)", (Z.won - Z.ask - Z.fee) * 100),
        ("taker, no fee", (Z.won - Z.ask) * 100),
        ("maker edge (at the bid)", (Z.won - Z.bid) * 100),
        ("win rate only (pp)", Z.won * 100)]
for nm, series in defs:
    s = pd.Series(series.values, index=Z.index)
    a, b = s[Z.date <= half].mean(), s[Z.date > half].mean()
    print("  %-30s %+11.2f %+12.2f %+10.2f" % (nm, a, b, b - a))

print("\n" + "=" * 84)
print("5. WAS CRYPTO ITSELF DIFFERENT IN JULY?")
print("=" * 84)
S = pd.read_parquet(DATA / "spot_1m.parquet")
S["date"] = pd.to_datetime(S.ts, unit="s", utc=True).dt.tz_localize(None)
S = S.sort_values(["coin", "ts"])
S["ret"] = S.groupby("coin").close.pct_change()
S["half"] = np.where(S.date <= half, "first", "second")
print("  %-8s %16s %16s" % ("coin", "1m vol first (bp)", "second (bp)"))
for c, g in S.groupby("coin"):
    a = g[g.half == "first"].ret.std() * 1e4
    b = g[g.half == "second"].ret.std() * 1e4
    print("  %-8s %16.2f %16.2f" % (c, a, b))
U = pd.read_parquet(DATA / "underlying.parquet",
                    columns=["wkey", "result", "a0", "a1"])
U["cu"] = pd.to_datetime(U.wkey, format="%y%b%d%H%M", utc=True) + pd.Timedelta(hours=4)
U["date"] = pd.to_datetime(U.cu.dt.date)
U = U[U.result.isin(["yes", "no"])].dropna(subset=["a0", "a1"])
U["absmove"] = (U.a1 / U.a0 - 1).abs() * 1e4
print(f"\n  |A1/A0 - 1| across ALL windows: first half "
      f"{U[U.date <= half].absmove.mean():.2f}bp   "
      f"second {U[U.date > half].absmove.mean():.2f}bp")
print("  YES base rate: first %.4f  second %.4f"
      % ((U[U.date <= half].result == "yes").mean(),
         (U[U.date > half].result == "yes").mean()))
