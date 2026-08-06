"""AUDIT B - hostile population matching for the frozen minute-:00 rule.

FROZEN RULE (no tuning, no added filters)
  close minute == 00
  favourite bid in [0.65, 0.80)
  first complete valid observation with 8-14 minutes remaining
  all six coins
  original favourite side preserved

THE TEST
  :00 is compared against the other three minutes under progressively harsher
  conditioning, up to
    day x UTC hour x coin x side x 1c bid bucket x spread bucket x entry ml
  Under the full set, any comparison is between markets closing on the SAME
  day, in the SAME hour, on the SAME coin, on the SAME side, at the SAME
  price to the cent, with the SAME spread, observed at the SAME lead time.
  Nothing is left for :00 to proxy for except the minute itself.

  Fixed effects are applied by within-stratum demeaning rather than exact
  pair matching: at this conditioning depth exact matching discards almost
  everything, and demeaning uses every stratum containing both arms.

REPORTED
  raw maker, taker, win-rate calibration residual, train/valid/test,
  leave-one-coin-out, leave-one-week-out, leave-one-hour-out, and one
  PREDECLARED project-wide multiplicity sensitivity.

DECISION RULE, PREDECLARED
  :00 is NOT called profitable unless the ABSOLUTE lower 95% bound on EV per
  SUBMITTED contract exceeds zero. Beating :30 is not sufficient - :30 losing
  money says nothing about whether :00 makes any.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
RNG = np.random.default_rng(20260806)
Q = 15
FILL = {0: (0.8167, 1.0000), 15: (0.8793, 1.0000),
        30: (0.9200, 0.9615), 45: (0.9388, 1.0000)}
SPLITS = {"train": ("2026-05-25", "2026-06-30"),
          "valid": ("2026-06-30", "2026-07-18"),
          "test": ("2026-07-18", "2026-08-07")}
# PREDECLARED multiplicity budget, fixed before looking at the result below.
N_PROJECT_HYPOTHESES = 40


def fee1(p):
    return math.ceil(0.07 * p * (1 - p) * 100) / 100.0


BF = pd.read_parquet(DATA / "book_full.parquet")
BF["utc"] = pd.to_datetime(BF.close_ts, unit="s", utc=True)
B = BF[(BF.bid >= 0.65) & (BF.bid < 0.80)].copy()
B["maker"] = B.won - B.bid
B["taker"] = B.won - B.ask - B.ask.map(fee1)
B["spread"] = B.ask - B.bid
B["day"] = B.utc.dt.date
B["hour"] = B.utc.dt.hour
B["week"] = B.utc.dt.isocalendar().week
B["bid1c"] = (B.bid * 100).round().astype(int)
B["sprb"] = pd.cut(B.spread, [-1, .01, .02, .03, .05, 1],
                   labels=[0, 1, 2, 3, 4]).astype(int)
B["mlb"] = B.ml.round().astype(int)
B["is00"] = (B.minute == 0).astype(float)
# EV per SUBMITTED contract under each minute's OWN measured fill pair
pf = np.array([FILL[m][0] if w == 1 else FILL[m][1]
               for m, w in zip(B.minute, B.won)])
B["ev_sub"] = B.maker * pf

print("=" * 90)
print("AUDIT B - HOSTILE MATCHING, FROZEN :00 RULE")
print("=" * 90)
print(f"  in-band markets {len(B)}   :00 {int(B.is00.sum())}   "
      f"days {B.day.nunique()}   coins {B.coin.nunique()}")


def within(df, keys, col):
    """Within-stratum demeaned difference between :00 and the rest.

    Only strata containing BOTH arms carry information; others demean to
    zero on both sides and drop out naturally.
    """
    g = df.groupby(keys, observed=True)
    ok = g.is00.transform("nunique") == 2
    d = df[ok]
    if not len(d):
        return np.nan, 0, 0
    gg = d.groupby(keys, observed=True)
    y = d[col] - gg[col].transform("mean")
    x = d.is00 - gg.is00.transform("mean")
    denom = (x * x).sum()
    if denom <= 0:
        return np.nan, 0, 0
    return (x * y).sum() / denom, len(d), int(gg.ngroups)


LADDER = [
    ("unconditional", []),
    ("+ day", ["day"]),
    ("+ day x coin", ["day", "coin"]),
    ("+ day x coin x side", ["day", "coin", "side"]),
    ("+ ... x hour", ["day", "hour", "coin", "side"]),
    ("+ ... x 1c bid", ["day", "hour", "coin", "side", "bid1c"]),
    ("+ ... x spread", ["day", "hour", "coin", "side", "bid1c", "sprb"]),
    ("+ ... x entry ml (FULL)",
     ["day", "hour", "coin", "side", "bid1c", "sprb", "mlb"]),
]
print("\n" + "-" * 90)
print("CONDITIONING LADDER - does the :00 effect survive harsher controls?")
print("-" * 90)
print("  %-26s %9s %8s %11s %11s %11s"
      % ("conditioning", "n used", "strata", "maker", "taker", "EV_sub"))
for lbl, keys in LADDER:
    if not keys:
        a, b = B[B.is00 == 1], B[B.is00 == 0]
        print("  %-26s %9d %8s %+10.2fc %+10.2fc %+10.2fc"
              % (lbl, len(B), "-",
                 (a.maker.mean() - b.maker.mean()) * 100,
                 (a.taker.mean() - b.taker.mean()) * 100,
                 (a.ev_sub.mean() - b.ev_sub.mean()) * 100))
        continue
    cm, n, k = within(B, keys, "maker")
    ct, _, _ = within(B, keys, "taker")
    ce, _, _ = within(B, keys, "ev_sub")
    print("  %-26s %9d %8d %+10.2fc %+10.2fc %+10.2fc"
          % (lbl, n, k, cm * 100, ct * 100, ce * 100))

FULL = ["day", "hour", "coin", "side", "bid1c", "sprb", "mlb"]
print("\n" + "-" * 90)
print("FULLY CONDITIONED ESTIMATE, day-clustered")
print("-" * 90)
days = sorted(B.day.unique())
gd = {d: v for d, v in B.groupby("day")}
for col in ("maker", "taker", "ev_sub"):
    pt, n, k = within(B, FULL, col)
    bs = []
    for _ in range(2000):
        s = pd.concat([gd[days[i]] for i in RNG.integers(0, len(days),
                                                         len(days))])
        c, _, _ = within(s, FULL, col)
        if c == c:
            bs.append(c * 100)
    bs = np.sort(np.array(bs))
    print("  %-8s %+6.2fc  95%% CI [%+6.2f, %+6.2f]  P(<=0) %.4f  (n %d, %d strata)"
          % (col, pt * 100, bs[int(len(bs)*.025)], bs[int(len(bs)*.975)],
             (bs <= 0).mean(), n, k))

print("\n" + "-" * 90)
print("WIN-RATE CALIBRATION RESIDUAL  (actual win rate minus price-implied)")
print("-" * 90)
print("  %-8s %7s %10s %11s %13s" % ("minute", "n", "win", "mean bid",
                                     "residual"))
for mn, g in B.groupby("minute"):
    print("  :%02d      %7d %10.4f %11.4f %+12.2fpp"
          % (mn, len(g), g.won.mean(), g.bid.mean(),
             (g.won.mean() - g.bid.mean()) * 100))
print("  A positive residual means the favourite wins more often than its")
print("  price implies. This is the same quantity as raw maker edge.")

print("\n" + "-" * 90)
print("CHRONOLOGICAL - fully conditioned")
print("-" * 90)
for nm, (a, b) in SPLITS.items():
    s = B[(B.utc >= a) & (B.utc < b)]
    c, n, k = within(s, FULL, "ev_sub")
    z = s[s.minute == 0]
    print("  %-6s EV_sub lift %+6.2fc  (:00 n %4d, absolute EV_sub %+6.2fc)"
          % (nm, c * 100 if c == c else float("nan"), len(z),
             z.ev_sub.mean() * 100))

print("\n" + "-" * 90)
print("LEAVE-ONE-OUT (fully conditioned EV_sub lift)")
print("-" * 90)
for dim, vals in (("coin", sorted(B.coin.unique())),
                  ("week", sorted(B.week.unique())),
                  ("hour", sorted(B.hour.unique()))):
    cs = []
    for v in vals:
        c, _, _ = within(B[B[dim] != v], FULL, "ev_sub")
        if c == c:
            cs.append(c * 100)
    if cs:
        print("  drop one %-5s  n=%2d  min %+6.2fc  median %+6.2fc  max %+6.2fc"
              % (dim, len(cs), min(cs), float(np.median(cs)), max(cs)))
        print("                        all leave-one-out estimates positive: %s"
              % ("YES" if min(cs) > 0 else "NO"))

print("\n" + "-" * 90)
print("ABSOLUTE LEVEL - the predeclared decision rule")
print("-" * 90)
Z = B[B.minute == 0]
zd = {d: v for d, v in Z.groupby("day")}
zs = sorted(zd)
bs = np.sort(np.array([
    pd.concat([zd[zs[i]] for i in RNG.integers(0, len(zs), len(zs))]
              ).ev_sub.mean() * 100 for _ in range(8000)]))
lo, hi = bs[200], bs[7799]
print("  :00 EV per SUBMITTED contract  %+.2fc" % (Z.ev_sub.mean() * 100))
print("  95%% CI [%+.2f, %+.2f]   P(<=0) %.4f" % (lo, hi, (bs <= 0).mean()))
print("  lower bound > 0 ?  %s" % ("YES" if lo > 0 else "NO"))

print("\n" + "-" * 90)
print("PREDECLARED PROJECT-WIDE MULTIPLICITY SENSITIVITY")
print("-" * 90)
p_abs = (bs <= 0).mean()
print(f"  Budget fixed in advance at N = {N_PROJECT_HYPOTHESES} hypotheses")
print("  (10 dead ends + HCR + DRC + RACE + 4 minutes + 5 bands + 6 coins +")
print("   24 hours collapsed to one family + queue/index mechanisms).")
print(f"  Bonferroni alpha = 0.05/{N_PROJECT_HYPOTHESES} = "
      f"{0.05/N_PROJECT_HYPOTHESES:.5f}")
print(f"  observed P(<=0) on the absolute level = {p_abs:.4f}")
print("  survives project-wide correction: %s"
      % ("YES" if p_abs < 0.05 / N_PROJECT_HYPOTHESES else "NO"))
