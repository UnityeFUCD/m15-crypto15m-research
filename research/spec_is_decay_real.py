"""Is the decay real, and is 73 days even enough to tell?

TWO CHALLENGES, BOTH FAIR
  I reported "train +9.56c, valid +5.70c, test -0.15c" and called it decay.
  Three bins falling in order is exactly what noise does a fair fraction of
  the time, and I never tested it. Nor did I ask whether 73 days can resolve
  anything at this dispersion.

TEST 1  TREND. Regress per-trade edge on calendar time, day-clustered. If the
        slope's interval contains zero the decay is not established.

TEST 2  SIMULATION UNDER A CONSTANT EDGE. Take the observed trades, replace
        their outcomes with draws from a CONSTANT-edge process at the observed
        dispersion, split into the same three periods, and count how often the
        result declines monotonically by at least as much as observed. This is
        the direct answer: how surprising is the pattern if nothing changed?

TEST 3  SPLIT-POINT SENSITIVITY. The train/valid/test boundaries were chosen
        for other work. A real decay shows up wherever you cut; an artifact
        moves when the cut moves.

TEST 4  ROLLING WINDOW. What the edge actually does through time, rather than
        three summary numbers.

TEST 5  POWER. Given the observed per-trade dispersion, what edge can 73 days
        resolve at all, and how long would confirming +2.76c take?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "research" / "results"
from research.spec_momentum_00 import build, apply_spec, SPEC      # noqa: E402

RNG = np.random.default_rng(20260807)
FIXED = {"rise_min": 0.0, "vol_min": 0.0, "spread_widen_max": 9.0, "qty": 15}
P = dict(SPEC); P.update(FIXED)

D = build(P)
S = apply_spec(D, P).sort_values("close_ts").reset_index(drop=True)
S["t_days"] = (S.close_ts - S.close_ts.min()) / 86400.0
print("=" * 86)
print("IS THE DECAY REAL?")
print("=" * 86)
print(f"  {len(S)} trades over {S.day.nunique()} days   "
      f"full-sample edge {S.edge_c.mean():+.2f}c")
print(f"  per-trade SD {S.edge_c.std():.1f}c  "
      f"(win pays {100*(1-S.ask.mean()):.0f}c, loss costs {100*S.ask.mean():.0f}c)")

print("\n" + "-" * 86)
print("TEST 1 - TREND: slope of edge vs time, day-clustered")
print("-" * 86)
days = sorted(S.day.unique())
gd = {k: v for k, v in S.groupby("day")}


def slope(df):
    if df.t_days.nunique() < 3:
        return np.nan
    return float(np.polyfit(df.t_days, df.edge_c, 1)[0])


obs_slope = slope(S)
bs = []
for _ in range(6000):
    smp = pd.concat([gd[days[i]] for i in RNG.integers(0, len(days), len(days))])
    v = slope(smp)
    if v == v:
        bs.append(v)
bs = np.sort(np.array(bs))
print(f"  slope {obs_slope:+.4f} c per day  "
      f"({obs_slope*73:+.2f}c over the full 73 days)")
print(f"  95% CI [{bs[150]:+.4f}, {bs[5849]:+.4f}]   "
      f"P(slope >= 0) {(bs >= 0).mean():.4f}")
print("  -> " + ("decay is statistically supported" if (bs >= 0).mean() < 0.05
                 else "the downward slope is NOT distinguishable from zero"))

print("\n" + "-" * 86)
print("TEST 2 - how often does a CONSTANT edge look this decaying?")
print("-" * 86)
TR, VA = "2026-06-30", "2026-07-18"
obs = [S[S.utc < TR].edge_c.mean(),
       S[(S.utc >= TR) & (S.utc < VA)].edge_c.mean(),
       S[S.utc >= VA].edge_c.mean()]
print(f"  observed  {obs[0]:+.2f}c / {obs[1]:+.2f}c / {obs[2]:+.2f}c   "
      f"(drop {obs[0]-obs[2]:+.2f}c)")
# constant-edge null: shuffle the OUTCOMES across the whole sample, keeping
# each trade's ask, so the edge is constant in expectation by construction
mono = 0
drops = []
for _ in range(20000):
    q = S.copy()
    q["won"] = RNG.permutation(S.won.values)
    q["edge_c"] = ((P["qty"] * (q.won - q.ask)) / P["qty"]) * 100
    a = q[q.utc < TR].edge_c.mean()
    b = q[(q.utc >= TR) & (q.utc < VA)].edge_c.mean()
    c = q[q.utc >= VA].edge_c.mean()
    drops.append(a - c)
    if a > b > c and (a - c) >= (obs[0] - obs[2]):
        mono += 1
drops = np.array(drops)
print(f"  under a constant edge, P(monotone decline at least this steep) = "
      f"{mono/20000:.4f}")
print(f"  P(first-minus-last >= observed) = {(drops >= obs[0]-obs[2]).mean():.4f}")
print("  -> " + ("the pattern is unlikely under a constant edge"
                 if mono / 20000 < 0.05 else
                 "a CONSTANT edge reproduces this pattern often enough that "
                 "decay is NOT established"))

print("\n" + "-" * 86)
print("TEST 3 - does the decay survive moving the split points?")
print("-" * 86)
print("  %-26s %9s %9s %9s %11s" % ("split (thirds by...)", "p1", "p2", "p3",
                                    "p1 - p3"))
# equal-count thirds
k = len(S) // 3
a, b, c = S.iloc[:k], S.iloc[k:2 * k], S.iloc[2 * k:]
print("  %-26s %+8.2fc %+8.2fc %+8.2fc %+10.2fc"
      % ("equal trade count", a.edge_c.mean(), b.edge_c.mean(),
         c.edge_c.mean(), a.edge_c.mean() - c.edge_c.mean()))
# equal calendar thirds
q1, q2 = S.t_days.quantile([1 / 3, 2 / 3])
a, b, c = S[S.t_days <= q1], S[(S.t_days > q1) & (S.t_days <= q2)], S[S.t_days > q2]
print("  %-26s %+8.2fc %+8.2fc %+8.2fc %+10.2fc"
      % ("equal calendar time", a.edge_c.mean(), b.edge_c.mean(),
         c.edge_c.mean(), a.edge_c.mean() - c.edge_c.mean()))
# halves
h = len(S) // 2
print("  %-26s %+8.2fc %8s %+8.2fc %+10.2fc"
      % ("halves", S.iloc[:h].edge_c.mean(), "-", S.iloc[h:].edge_c.mean(),
         S.iloc[:h].edge_c.mean() - S.iloc[h:].edge_c.mean()))

print("\n" + "-" * 86)
print("TEST 4 - rolling window, 21 days")
print("-" * 86)
S["date"] = pd.to_datetime(S.day)
roll = []
for d in pd.date_range(S.date.min(), S.date.max(), freq="7D"):
    w = S[(S.date >= d) & (S.date < d + pd.Timedelta(days=21))]
    if len(w) >= 30:
        roll.append((d.date(), len(w), w.edge_c.mean(), w.won.mean()))
print("  %-12s %6s %12s %9s" % ("window start", "n", "edge", "win"))
for d, n, e, w in roll:
    bar = "#" * max(0, int(round(e))) if e > 0 else "." * max(1, int(round(-e)))
    print("  %-12s %6d %+11.2fc %9.4f  %s" % (d, n, e, w, bar))

print("\n" + "-" * 86)
print("TEST 5 - IS 73 DAYS ENOUGH?")
print("-" * 86)
sd_trade = S.edge_c.std()
daymeans = S.groupby("day").edge_c.mean()
sd_day = daymeans.std(ddof=1)
n_day = S.day.nunique()
se_day = sd_day / np.sqrt(n_day)
print(f"  per-trade SD {sd_trade:.1f}c   per-DAY SD {sd_day:.1f}c   "
      f"{len(S)/n_day:.1f} trades/day")
print(f"  SE of the mean, day-clustered: {se_day:.2f}c on {n_day} days")
print(f"  smallest edge resolvable at 80% power: {2.802*se_day:.2f}c")
print(f"  observed full-sample edge {S.edge_c.mean():+.2f}c")
print()
for target in (2.76, 5.0, 7.5):
    need = (2.802 * sd_day / target) ** 2
    print(f"    to confirm a {target:+.2f}c edge at 80% power: "
          f"{need:,.0f} days ({need/30:.1f} months)")
print()
print(f"  to detect the DECAY itself (a {obs[0]-obs[2]:.2f}c difference between")
print(f"  two periods) needs roughly {2*(2.802*sd_day/(obs[0]-obs[2]))**2:,.0f} "
      f"days split across them.")
