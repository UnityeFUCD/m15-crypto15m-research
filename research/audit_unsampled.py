"""THE DEFINITIVE AUDIT: unsampled population, exactly six coins.

TWO CORRECTIONS TO EVERY EARLIER RUN IN THIS REPO

  1. EXACTLY SIX COINS. capture/hcr.py's common_return() raises unless all six
     inputs are present, but every research script here used nc >= 4 - so the
     thing tested was not the thing specified. It turns out to cost almost
     nothing (6,883 of 6,889 windows already have all six), but "it didn't
     matter" is a result, not an assumption, and the code now matches the spec.

  2. UNSAMPLED POPULATION. Previous runs used ladder_paths (:00/:30 only) plus
     a 2,101-market random sample at :15/:45 = 7,725 of 39,453 markets. This
     uses book_full.parquet: every market with a decodable close, all four
     minutes, and NO band filter applied at fetch time, so band sensitivity is
     testable for the first time.

WHAT IS DELIBERATELY NOT HERE
  Live execution. It is a different question on different data and lives in
  prospective_execution.py. Pooling a 303-order execution record into a
  population study is how the earlier wrong answers were produced.

Tests: population edge, matched controls, permutation, chronological split,
per coin, per minute, band sensitivity. Maker, fill-corrected, and taker.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
RNG = np.random.default_rng(20260806)
P_WIN, P_LOSE = 0.843, 0.962
SPLITS = {"train": ("2026-05-25", "2026-06-30"),
          "valid": ("2026-06-30", "2026-07-18"),
          "test": ("2026-07-18", "2026-08-07")}
BAND_LO, BAND_HI = 0.65, 0.80


def fee1(p):
    return math.ceil(0.07 * p * (1 - p) * 100) / 100.0


def corr(g):
    pf = np.where(g.won == 1, P_WIN, P_LOSE)
    return (g.maker * pf).sum() / pf.sum() * 100


def boot(vals, n=8000, lo=200, hi=7799):
    b = np.sort(np.array([RNG.choice(vals, len(vals), True).mean()
                          for _ in range(n)]))
    return b[lo], b[hi], (b <= 0).mean()


# ------------------------------------------------------------------ signal
U = pd.read_parquet(DATA / "underlying.parquet",
                    columns=["ticker", "coin", "wkey", "result", "a0"])
U["close_ts"] = ((pd.to_datetime(U.wkey, format="%y%b%d%H%M", utc=True)
                  + pd.Timedelta(hours=4) - EPOCH).dt.total_seconds()
                 ).round().astype("int64")
Uu = (U[U.result.isin(["yes", "no"])].dropna(subset=["a0"]).query("a0>0")
      .sort_values(["coin", "close_ts"]).drop_duplicates(["coin", "close_ts"]))
parts = []
for c, g in Uu.groupby("coin"):
    g = g.sort_values("close_ts").reset_index(drop=True)
    g["r"] = np.where(g.close_ts.diff() == 900, g.a0 / g.a0.shift(1) - 1.0,
                      np.nan)
    parts.append(g)
S = pd.concat(parts, ignore_index=True)
CF = (S.groupby("close_ts").agg(r_common=("r", "mean"), nc=("r", "count"))
      .reset_index())

n_ge4 = int((CF.nc >= 4).sum())
CF = CF[CF.nc == 6].sort_values("close_ts").reset_index(drop=True)   # EXACTLY 6
print("=" * 78)
print("SIGNAL CONSTRUCTION - exactly six coins, matching common_return()")
print("=" * 78)
print("  windows with >=4 coins %d   with exactly 6 %d   dropped %d (%.2f%%)"
      % (n_ge4, len(CF), n_ge4 - len(CF), 100 * (1 - len(CF) / n_ge4)))

CF["contig"] = CF.close_ts.diff() == 900
CF["calm24"] = CF.r_common.abs().rolling(24).mean()
CF.loc[CF.contig.rolling(24).min() != 1, "calm24"] = np.nan
print("  windows with a valid 24-window contiguous calm estimate: %d"
      % CF.calm24.notna().sum())

# ------------------------------------------------------------------- book
BK = pd.read_parquet(DATA / "book_full.parquet")
print("\n" + "=" * 78)
print("POPULATION - unsampled, all minutes, no band filter at fetch")
print("=" * 78)
print("  markets with a valid quote at 8-14 min: %d" % len(BK))
print("  by minute: %s" % BK.minute.value_counts().sort_index().to_dict())

B = BK.merge(CF[["close_ts", "r_common", "calm24"]], on="close_ts",
             how="inner").dropna(subset=["calm24"])
B["utc"] = pd.to_datetime(B.close_ts, unit="s", utc=True)
B["day"] = B.utc.dt.date
B["week"] = B.utc.dt.isocalendar().week
B["d"] = np.where(B.side == "yes", 1.0, -1.0)
B["HCR"] = (B.d * B.r_common <= -0.0015) & (B.calm24 <= 0.0030)
B["maker"] = B.won - B.bid
B["taker"] = B.won - B.ask - B.ask.map(fee1)
B["pb"] = (B.bid * 50).round().astype(int)

print("\n" + "=" * 78)
print("BAND SENSITIVITY - testable for the first time (nothing pre-filtered)")
print("=" * 78)
print("  %-14s %7s %8s %9s %11s %11s"
      % ("band", "n", "win", "maker", "corrected", "taker"))
for lo, hi in ((0.55, 0.65), (0.60, 0.70), (0.65, 0.80), (0.70, 0.80),
               (0.80, 0.90), (0.55, 0.95)):
    g = B[(B.bid >= lo) & (B.bid < hi)]
    if len(g) < 50:
        continue
    print("  %.2f-%.2f      %7d %8.4f %+10.2fc %+10.2fc %+10.2fc"
          % (lo, hi, len(g), g.won.mean(), g.maker.mean() * 100, corr(g),
             g.taker.mean() * 100))

BB = B[(B.bid >= BAND_LO) & (B.bid < BAND_HI)].copy()
h, r = BB[BB.HCR], BB[~BB.HCR]
print("\n" + "=" * 78)
print("IN-BAND POPULATION (%.2f-%.2f)" % (BAND_LO, BAND_HI))
print("=" * 78)
print("  markets %d   HCR %d (%.1f%%)   days %d"
      % (len(BB), len(h), 100 * len(h) / len(BB), BB.day.nunique()))
print("  by minute: %s" % BB.minute.value_counts().sort_index().to_dict())
print("  %-9s %7s %8s %11s %12s %11s"
      % ("cohort", "n", "win", "maker", "corrected", "taker"))
for lbl, g in (("HCR", h), ("non-HCR", r), ("all", BB)):
    print("  %-9s %7d %8.4f %+10.2fc %+11.2fc %+10.2fc"
          % (lbl, len(g), g.won.mean(), g.maker.mean() * 100, corr(g),
             g.taker.mean() * 100))
print("  fill-corrected lift %+.2fc" % (corr(h) - corr(r)))

print("\n" + "=" * 78)
print("TEST 1 - MATCHED CONTROLS (coin x week x 2c bucket x SAME side)")
print("=" * 78)
res = []
for _, g in BB.groupby(["coin", "week", "pb", "side"]):
    a, b = g[g.HCR], g[~g.HCR]
    if len(a) >= 1 and len(b) >= 2:
        res.append((len(a), a.maker.mean() - b.maker.mean(),
                    a.won.mean() - b.won.mean()))
wts = np.array([x[0] for x in res], float)
dm = np.array([x[1] for x in res])
dw = np.array([x[2] for x in res])
bs = np.sort(np.array([np.average(dm[i], weights=wts[i]) * 100
                       for i in (RNG.integers(0, len(res), len(res))
                                 for _ in range(8000))]))
print("  strata %d covering %d HCR markets" % (len(res), int(wts.sum())))
print("  maker lift %+.2fc   win lift %+.2fpp   95%% CI [%+.2f, %+.2f]   "
      "P(<=0) %.4f" % (np.average(dm, weights=wts) * 100,
                       np.average(dw, weights=wts) * 100,
                       bs[200], bs[7799], (bs <= 0).mean()))

print("\n" + "=" * 78)
print("TEST 2 - PERMUTATION (shuffle r_common within day)")
print("=" * 78)
obs = h.maker.mean() * 100
perm = []
for _ in range(2000):
    q = BB[["day", "r_common", "d", "calm24", "maker"]].copy()
    q["rc"] = q.groupby("day").r_common.transform(
        lambda s: RNG.permutation(s.values))
    g = q[(q.d * q.rc <= -0.0015) & (q.calm24 <= 0.0030)]
    if len(g) >= 100:
        perm.append(g.maker.mean() * 100)
perm = np.array(perm)
print("  observed HCR maker %+.2fc   permuted mean %+.2fc   p95 %+.2fc"
      % (obs, perm.mean(), np.quantile(perm, .95)))
print("  P(permuted >= observed) %.4f" % (perm >= obs).mean())

print("\n" + "=" * 78)
print("TEST 3 - CHRONOLOGICAL")
print("=" * 78)
for nm, (a, b) in SPLITS.items():
    hs, rs = h[(h.utc >= a) & (h.utc < b)], r[(r.utc >= a) & (r.utc < b)]
    if len(hs) < 20:
        print("  %-6s n %d - too few" % (nm, len(hs)))
        continue
    print("  %-6s HCR n %4d maker %+6.2fc | non-HCR n %5d %+6.2fc | lift %+6.2fc"
          % (nm, len(hs), hs.maker.mean() * 100, len(rs),
             rs.maker.mean() * 100,
             (hs.maker.mean() - rs.maker.mean()) * 100))

print("\n" + "=" * 78)
print("TEST 4 - PER COIN")
print("=" * 78)
print("  %-6s %7s %9s %11s %11s" % ("coin", "n HCR", "win", "maker", "lift"))
for c, g in BB.groupby("coin"):
    a, b = g[g.HCR], g[~g.HCR]
    if len(a) < 20:
        continue
    print("  %-6s %7d %9.4f %+10.2fc %+10.2fc"
          % (c, len(a), a.won.mean(), a.maker.mean() * 100,
             (a.maker.mean() - b.maker.mean()) * 100))

print("\n" + "=" * 78)
print("TEST 5 - PER MINUTE (balanced population, so this is finally clean)")
print("=" * 78)
print("  %-8s %7s %9s %11s %11s" % ("minute", "n HCR", "win", "maker", "lift"))
for mn, g in BB.groupby("minute"):
    a, b = g[g.HCR], g[~g.HCR]
    if len(a) < 20:
        continue
    print("  %-8d %7d %9.4f %+10.2fc %+10.2fc"
          % (mn, len(a), a.won.mean(), a.maker.mean() * 100,
             (a.maker.mean() - b.maker.mean()) * 100))

print("\n" + "=" * 78)
print("TEST 6 - DAY-CLUSTERED (each cohort against zero)")
print("=" * 78)
for lbl, g in (("HCR", h), ("non-HCR", r), ("all", BB)):
    gd = {k: v for k, v in g.groupby("day")}
    ds = sorted(gd)
    b = np.sort(np.array([
        (lambda x: (x.maker * np.where(x.won == 1, P_WIN, P_LOSE)).sum()
         / np.where(x.won == 1, P_WIN, P_LOSE).sum() * 100)(
            pd.concat([gd[ds[i]] for i in RNG.integers(0, len(ds), len(ds))]))
        for _ in range(4000)]))
    print("  %-8s corrected %+6.2fc  95%% CI [%+6.2f, %+6.2f]  P(<=0) %.4f"
          % (lbl, corr(g), b[100], b[3899], (b <= 0).mean()))

print("\n" + "=" * 78)
print("TEST 6b - COIN-SUBSET ROBUSTNESS")
print("=" * 78)
print("  Does the result depend on including the two coins the live strategy")
print("  trades least? HYPE is 4%% of live orders and has the WEAKEST absolute")
print("  edge; DOGE is 11.6%% and among the strongest. Dropping coins by their")
print("  observed results would be selection on the outcome, so this is a")
print("  robustness check, not a filtering decision.")
print("  %-22s %7s %11s %11s %11s"
      % ("population", "n", "HCR", "non-HCR", "lift"))
for lbl, keep in (("all six coins", None),
                  ("drop HYPE", ["HYPE"]),
                  ("drop DOGE", ["DOGE"]),
                  ("drop HYPE + DOGE", ["HYPE", "DOGE"]),
                  ("live top-4 only", ["HYPE", "DOGE"])):
    sub = BB if keep is None else BB[~BB.coin.isin(keep)]
    a, b = sub[sub.HCR], sub[~sub.HCR]
    if len(a) < 20:
        continue
    print("  %-22s %7d %+10.2fc %+10.2fc %+10.2fc"
          % (lbl, len(sub), corr(a), corr(b), corr(a) - corr(b)))

print("\n" + "=" * 78)
print("TEST 7 - FILL-MODEL SENSITIVITY")
print("=" * 78)
print("  The 0.843/0.962 constants were estimated on 286 orders BEFORE the 17")
print("  late-closing outcomes were recovered. prospective_execution.py, run")
print("  on the corrected 303, measures 0.8848/0.9884. Both are shown so the")
print("  conclusion does not depend on which is used.")
print("  %-26s %11s %11s %11s" % ("fill model", "HCR", "non-HCR", "lift"))
for lbl, pw, pl in (("0.843 / 0.962 (old)", 0.843, 0.962),
                    ("0.8848 / 0.9884 (corrected)", 0.8848, 0.9884),
                    ("1.000 / 1.000 (no bias)", 1.0, 1.0)):
    def c2(g, pw=pw, pl=pl):
        pf = np.where(g.won == 1, pw, pl)
        return (g.maker * pf).sum() / pf.sum() * 100
    print("  %-26s %+10.2fc %+10.2fc %+10.2fc"
          % (lbl, c2(h), c2(r), c2(h) - c2(r)))
