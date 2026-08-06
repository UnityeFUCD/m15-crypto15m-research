"""WHY does minute :00 work? The first mechanism test in this project.

THE QUESTION
  A :00 favourite bought 3 minutes into its window wins 84.5% of the time
  while the market charges 76c. Every report so far has said "no mechanism
  established" and offered a story about the top of the hour. This tests it.

THE DECOMPOSITION THAT MAKES IT TESTABLE
  A favourite exists at +3 min because the index has already moved away from
  A0. That favourite WINS if and only if the move HOLDS for the remaining 12
  minutes. So the whole strategy reduces to one question about the underlying:

      does an early move persist to the close?

  and the :00 claim reduces to:

      does it persist BETTER into the top of the hour?

  Crucially this can be answered WITHOUT the contract market at all, using
  only the index. That separates two very different worlds:

    A  the UNDERLYING behaves differently at :00
       -> structural, caused by hour-boundary settlement/rolls, and likely
          to persist. The decay would then be noise or crowding on top.

    B  the underlying is identical at :00 and the difference is only in the
       CONTRACT PRICE
       -> a pure market-structure mispricing with nothing anchoring it, which
          is exactly the sort of thing that gets arbitraged away - consistent
          with the observed decay to zero.

  These make opposite predictions about whether the live run will find the
  edge again, so this is the most decision-relevant test available.

DATA
  underlying.parquet  a0 and a1 for 41,334 windows - ground truth endpoints
  spot_1m.parquet     1-minute Coinbase, validated against those endpoints
                      (return corr 0.9807, sign agreement 0.9320)
  No look-ahead: the +3 minute observation is 12 minutes before the close.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"
RNG = np.random.default_rng(20260807)

U = pd.read_parquet(DATA / "underlying.parquet",
                    columns=["ticker", "coin", "wkey", "result", "a0", "a1"])
U["close_utc"] = (pd.to_datetime(U.wkey, format="%y%b%d%H%M", utc=True)
                  + pd.Timedelta(hours=4))
U["open_utc"] = U.close_utc - pd.Timedelta(minutes=15)
U = U[U.result.isin(["yes", "no"])].dropna(subset=["a0", "a1"])
U = U[(U.a0 > 0) & (U.a1 > 0)].copy()
# NEVER .astype("int64") // 10**9 on a datetime column here: this pandas
# stores datetime64[us], so that yields microseconds/1e9 - wrong by 1000x.
# It is the third time this exact trap has appeared in this project.
_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
U["close_ts"] = ((U.close_utc - _EPOCH).dt.total_seconds()).round().astype("int64")
U["open_ts"] = ((U.open_utc - _EPOCH).dt.total_seconds()).round().astype("int64")
assert U.close_ts.between(1.7e9, 1.9e9).all(), "epoch conversion is not seconds"
U["minute"] = U.close_utc.dt.minute

S = pd.read_parquet(DATA / "spot_1m.parquet").sort_values(["coin", "ts"])
px = S.set_index(["coin", "ts"]).close

print("=" * 84)
print("WHY DOES :00 WORK? - testing the UNDERLYING, not the contract market")
print("=" * 84)

# price 3 minutes into the window = the moment we buy
U["p3"] = [px.get((c, t + 180), np.nan) for c, t in zip(U.coin, U.open_ts)]
M = U.dropna(subset=["p3"]).copy()
print(f"  windows with a +3min spot observation: {len(M):,} of {len(U):,}")

# direction established by +3 min, and whether it holds to the close
M["early_up"] = M.p3 >= M.a0                 # index above A0 at +3 min
M["closed_up"] = M.a1 >= M.a0                # YES settles
M["persisted"] = M.early_up == M.closed_up   # the early move held

print("\n" + "-" * 84)
print("DOES AN EARLY MOVE PERSIST TO THE CLOSE? by close minute")
print("-" * 84)
print("  %-8s %9s %14s %16s" % ("minute", "n", "persistence", "vs other minutes"))
base = {}
for mn, g in M.groupby("minute"):
    base[mn] = g.persisted.mean()
oth = {mn: M[M.minute != mn].persisted.mean() for mn in base}
for mn in sorted(base):
    g = M[M.minute == mn]
    print("  :%02d      %9d %13.4f %+15.2fpp"
          % (mn, len(g), base[mn], (base[mn] - oth[mn]) * 100))

d = base[0] - M[M.minute != 0].persisted.mean()
days = M.close_utc.dt.date
gd = {k: v for k, v in M.groupby(days)}
ds = sorted(gd)
bs = []
for _ in range(6000):
    s = pd.concat([gd[ds[i]] for i in RNG.integers(0, len(ds), len(ds))])
    a = s[s.minute == 0].persisted.mean()
    b = s[s.minute != 0].persisted.mean()
    if a == a and b == b:
        bs.append((a - b) * 100)
bs = np.sort(np.array(bs))
print(f"\n  :00 minus the rest: {d*100:+.2f}pp   "
      f"95% CI [{bs[150]:+.2f}, {bs[5849]:+.2f}]   P(<=0) {(bs <= 0).mean():.4f}")

print("\n" + "-" * 84)
print("IS THE INDEX QUIETER INTO THE HOUR? (realised vol, last 12 minutes)")
print("-" * 84)


def tail_vol(coin, open_ts):
    lo = open_ts + 180
    seg = [px.get((coin, lo + 60 * k), np.nan) for k in range(12)]
    seg = [v for v in seg if v == v]
    if len(seg) < 6:
        return np.nan
    a = np.asarray(seg)
    return float(np.std(np.diff(a) / a[:-1]) * 1e4)


sub = M.sample(min(9000, len(M)), random_state=1).copy()
sub["tailvol_bp"] = [tail_vol(c, t) for c, t in zip(sub.coin, sub.open_ts)]
print("  %-8s %9s %16s %16s" % ("minute", "n", "tail vol (bp)", "|move| at +3 (bp)"))
sub["mv3"] = (sub.p3 / sub.a0 - 1).abs() * 1e4
for mn, g in sub.groupby("minute"):
    print("  :%02d      %9d %16.2f %16.2f"
          % (mn, len(g), g.tailvol_bp.mean(), g.mv3.mean()))

print("\n" + "-" * 84)
print("SIZE OF THE EARLY MOVE - is :00 simply moving further by +3 min?")
print("-" * 84)
print("  A bigger early move is harder to reverse, so it would explain higher")
print("  persistence without any hour-boundary effect.")
for mn, g in M.groupby("minute"):
    mv = ((g.p3 / g.a0 - 1).abs() * 1e4)
    print("  :%02d   median |move| %6.2f bp   mean %6.2f bp   n %6d"
          % (mn, mv.median(), mv.mean(), len(g)))

print("\n" + "=" * 84)
print("VERDICT")
print("=" * 84)
p = (bs <= 0).mean()
if p < 0.05:
    print("  The UNDERLYING persists better into the top of the hour.")
    print("  -> mechanism A: structural. The edge has a physical cause and")
    print("     should reappear once crowding unwinds.")
else:
    print("  The underlying does NOT persist better at :00.")
    print("  -> mechanism B: the difference lives in the CONTRACT PRICE, not")
    print("     in the index. Nothing anchors it, which is why it can be")
    print("     competed away - consistent with the decay to zero.")
    print("     The live run is then a test of whether competition has eased,")
    print("     not of a durable structural edge.")
