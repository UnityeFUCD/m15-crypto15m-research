"""INTERPLAY 2 - is the no-fill signal EVER underpriced, and do findings correlate?

THE TEST THAT MATTERS
  A predictor pays only if it is mispriced. The no-fill signal is real
  (+21pp, n=8,201, P=0.0000) and, pooled, exactly priced: worth +15.35c at a
  bid you cannot get and +0.82c at the ask you can.

  Pooled pricing efficiency does not imply pointwise efficiency. If the ask
  under-reacts to the signal in some identifiable state, that state is where
  the money is. This looks for it, and - because looking is how this project
  produced four wrong verdicts - the search is confined to TRAIN and the
  winner is evaluated on held-out data with a best-of-N correction.

DEFINITION OF UNDERPRICED
  For an unfilled probe, the ask implies P(win) = ask + fee. The signal says
  the true rate is higher. The residual

      mispricing = actual_win_rate - (ask + fee)

  is the taker edge. Positive means the ask is too low for how often that
  cohort wins - the market has not fully absorbed the information the probe
  revealed. Conditioning on states known BEFORE the commit decision only.

PART 2 - DO THE FINDINGS OVERLAP?
  Six things have been measured on this market. If they are all restatements
  of one underlying quantity, there is nothing left to combine. If any pair is
  close to orthogonal, a combination could be stronger than either. That is a
  cheap question with a decisive answer and it has never been asked.
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
OUT = ROOT / "research" / "results"
RNG = np.random.default_rng(20260807)
TRAIN = ("2026-05-25", "2026-06-30")
VALID = ("2026-06-30", "2026-07-18")
TEST = ("2026-07-18", "2026-08-07")


def fee_c(p):
    return math.ceil(0.07 * p * (1 - p) * 100) / 100.0


G = pd.read_parquet(DATA / "interplay_master.parquet")
G["utc"] = pd.to_datetime(G.close_ts, unit="s", utc=True)
G["day"] = G.utc.dt.date
G["hour"] = G.utc.dt.hour
G["minute"] = G.utc.dt.minute
G["week"] = G.utc.dt.isocalendar().week

# the commitable universe: unfilled probe, ask inside the frozen band
U = G[(~G.filled) & G.ask_after.between(0.60, 0.80)].copy()
U["fee"] = U.ask_after.map(fee_c)
U["implied"] = U.ask_after + U.fee
U["mispricing"] = U.won - U.implied            # = taker edge per contract
print("=" * 86)
print("INTERPLAY 2 - IS THE NO-FILL SIGNAL EVER UNDERPRICED?")
print("=" * 86)
print(f"  commitable markets {len(U):,} over {U.day.nunique()} days")
print(f"  pooled mispricing {U.mispricing.mean()*100:+.2f}c   "
      f"(this is the +0.82c already established)")

U["spread_b"] = pd.qcut(U.spread.rank(method="first"), 3, duplicates="drop",
                        labels=["tight", "mid", "wide"])
U["prevol_b"] = pd.qcut(U.prevol.rank(method="first"), 3, duplicates="drop",
                        labels=["calm", "mid", "wild"])
U["ask_b"] = pd.cut(U.ask_after, [0.599, 0.68, 0.74, 0.801],
                    labels=["60-68", "68-74", "74-80"])
tr = U[(U.utc >= TRAIN[0]) & (U.utc < TRAIN[1])]
va = U[(U.utc >= VALID[0]) & (U.utc < VALID[1])]
te = U[(U.utc >= TEST[0]) & (U.utc < TEST[1])]
print(f"  train {len(tr):,}   valid {len(va):,}   test {len(te):,}")

print("\n" + "-" * 86)
print("SEARCH ON TRAIN ONLY - where does the ask under-react?")
print("-" * 86)
cells = []
DIMS = ("spread_b", "prevol_b", "ask_b", "minute", "coin", "side")
for dim in DIMS:
    print(f"\n  by {dim}:")
    print("    %-10s %7s %9s %11s %13s"
          % ("level", "n", "win", "implied", "mispricing"))
    for lv, g in tr.groupby(dim, observed=True):
        if len(g) < 60:
            continue
        print("    %-10s %7d %9.4f %10.4f %+12.2fc"
              % (str(lv), len(g), g.won.mean(), g.implied.mean(),
                 g.mispricing.mean() * 100))
        cells.append({"dim": dim, "level": str(lv), "n": len(g),
                      "train_mispricing_c": g.mispricing.mean() * 100})
C = pd.DataFrame(cells).sort_values("train_mispricing_c", ascending=False)
print("\n  top train cells:")
print(C.head(5).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
best = C.iloc[0]

print("\n" + "-" * 86)
print(f"HELD-OUT TEST of the frozen winner: {best.dim} == {best.level}")
print("-" * 86)
print("  %-8s %8s %13s" % ("split", "n", "mispricing"))
for nm, s in (("train", tr), ("valid", va), ("test", te)):
    z = s[s[best.dim].astype(str) == best.level]
    print("  %-8s %8d %+12.2fc"
          % (nm, len(z), z.mispricing.mean() * 100 if len(z) else float("nan")))
ho = pd.concat([va, te])
z = ho[ho[best.dim].astype(str) == best.level]
if len(z) > 40:
    gd = {k: v for k, v in z.groupby("day")}
    ds = sorted(gd)
    bs = np.sort(np.array([
        pd.concat([gd[ds[i]] for i in RNG.integers(0, len(ds), len(ds))]
                  ).mispricing.mean() * 100 for _ in range(6000)]))
    print(f"\n  HELD OUT n={len(z)}  {z.mispricing.mean()*100:+.2f}c   "
          f"95% CI [{bs[150]:+.2f}, {bs[5849]:+.2f}]   "
          f"P(<=0) {(bs <= 0).mean():.4f}")

print("\n" + "-" * 86)
print(f"SELECTION CORRECTION - best of {len(C)} cells, labels permuted")
print("-" * 86)
obs = best.train_mispricing_c
perm = []
for _ in range(3000):
    q = tr.copy()
    q["m"] = RNG.permutation(q.mispricing.values)
    vals = []
    for dim in DIMS:
        for lv, g in q.groupby(dim, observed=True):
            if len(g) >= 60:
                vals.append(g.m.mean() * 100)
    if vals:
        perm.append(max(vals))
perm = np.array(perm)
p = float((perm >= obs).mean())
print(f"  observed best {obs:+.2f}c   permuted best mean {perm.mean():+.2f}c   "
      f"p95 {np.quantile(perm, .95):+.2f}c")
print(f"  P(permuted best >= observed) = {p:.4f}   -> "
      f"{'SURVIVES' if p < 0.05 else 'does NOT survive'}")

# --------------------------------------------------------- correlation study
print("\n" + "=" * 86)
print("DO THE PROJECT'S FINDINGS OVERLAP, OR ARE ANY ORTHOGONAL?")
print("=" * 86)
F = G.copy()
F["f_nofill"] = (~F.filled).astype(int)
F["f_min00"] = (F.minute == 0).astype(int)
F["f_calm"] = (F.prevol <= F.prevol.median()).astype(int)
F["f_tight"] = (F.spread <= F.spread.median()).astype(int)
F["f_cheap"] = (F.bid < 0.70).astype(int)
F["f_yes"] = (F.side == "yes").astype(int)
flags = ["f_nofill", "f_min00", "f_calm", "f_tight", "f_cheap", "f_yes"]
print("\n  pairwise correlation between the signals:")
print(F[flags].corr().round(3).to_string())

print("\n  each signal's standalone effect on the WIN RATE:")
print("  %-12s %8s %10s %10s %11s" % ("signal", "n on", "win|on", "win|off",
                                      "lift"))
for f in flags:
    on, off = F[F[f] == 1], F[F[f] == 0]
    print("  %-12s %8d %10.4f %10.4f %+10.2fpp"
          % (f, len(on), on.won.mean(), off.won.mean(),
             (on.won.mean() - off.won.mean()) * 100))

print("\n  CONDITIONAL on no-fill: does any second signal add anything?")
nf = F[F.f_nofill == 1]
print("  %-12s %8s %10s %10s %11s" % ("signal", "n on", "win|on", "win|off",
                                      "extra lift"))
for f in flags[1:]:
    on, off = nf[nf[f] == 1], nf[nf[f] == 0]
    if len(on) < 100 or len(off) < 100:
        continue
    print("  %-12s %8d %10.4f %10.4f %+10.2fpp"
          % (f, len(on), on.won.mean(), off.won.mean(),
             (on.won.mean() - off.won.mean()) * 100))
print("\n  If the extra lifts are near zero, every other finding is a")
print("  restatement of the fill signal and there is nothing to combine.")
C.to_csv(OUT / "interplay_02_mispricing.csv", index=False)
