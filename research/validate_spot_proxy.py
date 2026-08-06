"""Does Coinbase 1-minute spot actually reproduce the CF index endpoints?

A proxy nobody checked is a guess. This repo already holds ground truth -
a0 and a1 for 41,334 windows - so the proxy can be scored against the exact
quantity it is standing in for.

TESTS
  1. level correlation between proxy and a0/a1
  2. RETURN correlation, which is what actually matters: AUDIT C conditions on
     normalized distance to A0 and realized volatility, both of which are
     functions of returns, not levels
  3. sign agreement on a1 >= a0 - the settlement rule itself. If the proxy
     cannot call the winner, it cannot be used to study reversal.

The RTI is a 60-second trailing mean, so the closest proxy for A(t) is the
close of the 1-minute candle ending at t. Constituent-exchange differences,
index weighting and the trailing window all guarantee some error; the point
is to measure it, and to state whether it is small enough for AUDIT C's
conditioning variables.

VERDICT RULE, fixed before looking: sign agreement below 0.90 means the proxy
cannot support AUDIT C and the audit is reported as blocked on data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
SIGN_THRESHOLD = 0.90

S = pd.read_parquet(DATA / "spot_1m.parquet")
print("=" * 80)
print("VALIDATING THE COINBASE 1-MINUTE PROXY AGAINST KNOWN a0 / a1")
print("=" * 80)
print(f"  candles {len(S):,}   coins {S.coin.nunique()}")
print("  per coin:", S.groupby("coin").size().to_dict())

U = pd.read_parquet(DATA / "underlying.parquet",
                    columns=["ticker", "coin", "wkey", "result", "a0", "a1"])
U["close_utc"] = (pd.to_datetime(U.wkey, format="%y%b%d%H%M", utc=True)
                  + pd.Timedelta(hours=4))
U["open_utc"] = U.close_utc - pd.Timedelta(minutes=15)
U = U[U.result.isin(["yes", "no"])].dropna(subset=["a0", "a1"])
U = U[(U.a0 > 0) & (U.a1 > 0)]
U["close_ts"] = ((U.close_utc - EPOCH).dt.total_seconds()).astype("int64")
U["open_ts"] = ((U.open_utc - EPOCH).dt.total_seconds()).astype("int64")

# proxy for A(t) = close of the 1-minute candle ending at t
px = S.set_index(["coin", "ts"]).close
U["p0"] = [px.get((c, t), np.nan) for c, t in zip(U.coin, U.open_ts - 60)]
U["p1"] = [px.get((c, t), np.nan) for c, t in zip(U.coin, U.close_ts - 60)]
M = U.dropna(subset=["p0", "p1"]).copy()
print(f"\n  windows matched to proxy candles: {len(M):,} of {len(U):,} "
      f"({100*len(M)/len(U):.1f}%)")

print("\n" + "-" * 80)
print("1. LEVEL CORRELATION (per coin, log level)")
print("-" * 80)
print("  %-6s %8s %12s %12s" % ("coin", "n", "corr(a0,p0)", "corr(a1,p1)"))
for c, g in M.groupby("coin"):
    print("  %-6s %8d %12.6f %12.6f"
          % (c, len(g), np.corrcoef(np.log(g.a0), np.log(g.p0))[0, 1],
             np.corrcoef(np.log(g.a1), np.log(g.p1))[0, 1]))

print("\n" + "-" * 80)
print("2. RETURN CORRELATION - the quantity AUDIT C actually conditions on")
print("-" * 80)
M["r_true"] = M.a1 / M.a0 - 1.0
M["r_prox"] = M.p1 / M.p0 - 1.0
print("  %-6s %8s %14s %16s %16s"
      % ("coin", "n", "corr(returns)", "SD true (bp)", "SD proxy (bp)"))
for c, g in M.groupby("coin"):
    print("  %-6s %8d %14.6f %16.2f %16.2f"
          % (c, len(g), np.corrcoef(g.r_true, g.r_prox)[0, 1],
             g.r_true.std() * 1e4, g.r_prox.std() * 1e4))
print("  %-6s %8d %14.6f %16.2f %16.2f"
      % ("ALL", len(M), np.corrcoef(M.r_true, M.r_prox)[0, 1],
         M.r_true.std() * 1e4, M.r_prox.std() * 1e4))

print("\n" + "-" * 80)
print("3. SIGN AGREEMENT ON THE SETTLEMENT RULE  a1 >= a0")
print("-" * 80)
M["true_yes"] = M.a1 >= M.a0
M["prox_yes"] = M.p1 >= M.p0
print("  %-6s %8s %14s" % ("coin", "n", "sign agree"))
for c, g in M.groupby("coin"):
    print("  %-6s %8d %14.4f" % (c, len(g), (g.true_yes == g.prox_yes).mean()))
agree = (M.true_yes == M.prox_yes).mean()
print("  %-6s %8d %14.4f" % ("ALL", len(M), agree))

print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)
print(f"  sign agreement {agree:.4f} vs threshold {SIGN_THRESHOLD:.2f}")
if agree >= SIGN_THRESHOLD:
    print("  USABLE. The proxy tracks the index well enough to support AUDIT C's")
    print("  conditioning variables (normalized distance to A0, realized vol).")
    print("  It is still a PROXY and every AUDIT C result must say so.")
else:
    print("  NOT USABLE. The proxy cannot reproduce the settlement rule it is")
    print("  standing in for, so any reversal hazard conditioned on it would be")
    print("  measuring Coinbase-versus-CF basis, not market behaviour.")
    print("  AUDIT C is BLOCKED ON DATA and must be reported as such rather")
    print("  than run on a proxy known to be inadequate.")
