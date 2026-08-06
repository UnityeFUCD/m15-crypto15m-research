"""Optimise the :00 strategy the only way that can survive: train-only.

THE TRAP THIS AVOIDS
  Optimising on all 73 days and reporting the winner is how this project
  produced four wrong verdicts. A grid search over even a modest parameter
  space will always find something that looks excellent in-sample.

  So the grid runs on TRAIN ONLY (2026-05-25 to 06-30). The single best
  configuration is FROZEN and then evaluated once on valid and test, which
  were never consulted during the search. A best-of-N permutation asks whether
  the train winner is better than what a search of that size finds in shuffled
  data, and the held-out result is reported whether or not it is flattering.

WHAT IS SEARCHED
  entry band       favourite bid floor and ceiling at the first 8-14 min quote
  wait             complete minutes between entry and decision
  ask floor/ceil   the band the IOC will actually pay into
  The inert conditions from the previous run (momentum, volume, spread
  widening) are excluded rather than re-searched: they were shown to cost
  frequency without adding edge, and re-searching them would only reintroduce
  multiplicity for no expected gain.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "research" / "results"
from research.spec_momentum_00 import build, apply_spec, dayboot, SPEC  # noqa

RNG = np.random.default_rng(20260807)
TRAIN_END, VALID_END = "2026-06-30", "2026-07-18"
MIN_TRADES_TRAIN = 40

GRID = {
    "bid_min": [0.60, 0.65, 0.70],
    "bid_max": [0.75, 0.80, 0.85],
    "wait_min": [1, 2, 3],
    "ask_lo": [0.55, 0.60, 0.65],
    "ask_hi": [0.80, 0.85, 0.90],
}
FIXED = {"rise_min": 0.0, "vol_min": 0.0, "spread_widen_max": 9.0, "qty": 15}

print("=" * 88)
print("TRAIN-ONLY OPTIMISATION OF THE :00 STRATEGY")
print("=" * 88)
combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
combos = [c for c in combos if c["bid_min"] < c["bid_max"]
          and c["ask_lo"] < c["ask_hi"]]
print(f"  grid size {len(combos)} configurations")

cache = {}


def data_for(wait):
    if wait not in cache:
        p = dict(SPEC); p.update(FIXED); p["wait_min"] = wait
        cache[wait] = build(p)
    return cache[wait]


rows = []
for c in combos:
    p = dict(SPEC); p.update(FIXED); p.update(c)
    D = data_for(c["wait_min"])
    tr = D[D.utc < TRAIN_END]
    z = apply_spec(tr, p)
    if len(z) < MIN_TRADES_TRAIN:
        continue
    rows.append({**c, "n_train": len(z), "train_edge": z.edge_c.mean(),
                 "train_dpd": z.pnl.sum() / z.day.nunique()})
R = pd.DataFrame(rows).sort_values("train_edge", ascending=False)
print(f"  configurations with >= {MIN_TRADES_TRAIN} train trades: {len(R)}")
print("\n  top 8 on TRAIN:")
print(R.head(8).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

best = R.iloc[0].to_dict()
P = dict(SPEC); P.update(FIXED)
for k in GRID:
    P[k] = best[k]
print(f"\n  FROZEN: bid [{P['bid_min']:.2f},{P['bid_max']:.2f})  "
      f"wait {P['wait_min']}min  ask [{P['ask_lo']:.2f},{P['ask_hi']:.2f}]")

D = data_for(P["wait_min"])
full = apply_spec(D, P)
print("\n" + "-" * 88)
print("HELD-OUT EVALUATION - valid and test were never consulted")
print("-" * 88)
print("  %-8s %7s %9s %13s %11s %22s %9s"
      % ("split", "n", "win", "per contract", "$/day", "95% CI", "P(<=0)"))
for nm, a, b in (("train", "2026-05-25", TRAIN_END),
                 ("valid", TRAIN_END, VALID_END),
                 ("test", VALID_END, "2026-08-07")):
    z = full[(full.utc >= a) & (full.utc < b)]
    if len(z) < 10:
        print("  %-8s %7d  too few" % (nm, len(z)))
        continue
    lo, hi, pv = dayboot(z)
    print("  %-8s %7d %9.4f %+12.2fc %+10.2f   [%+6.2f, %+6.2f] %8.4f"
          % (nm, len(z), z.won.mean(), z.edge_c.mean(),
             z.pnl.sum() / z.day.nunique(), lo, hi, pv))
ho = full[full.utc >= TRAIN_END]
lo, hi, pv = dayboot(ho)
print("\n  HELD OUT (valid+test): n=%d  %+.2fc  95%% CI [%+.2f, %+.2f]  "
      "P(<=0) %.4f" % (len(ho), ho.edge_c.mean(), lo, hi, pv))
print("  $/day held out: %+.2f" % (ho.pnl.sum() / ho.day.nunique()))

print("\n" + "-" * 88)
print("IS THE OPTIMISED VERSION BETTER THAN THE PLAIN ONE, OUT OF SAMPLE?")
print("-" * 88)
plain = dict(SPEC); plain.update(FIXED)
Dp = data_for(plain["wait_min"])
zp = apply_spec(Dp, plain)
zp_ho = zp[zp.utc >= TRAIN_END]
lo2, hi2, pv2 = dayboot(zp_ho)
print("  %-28s %7s %13s %11s %9s" % ("config", "n", "per contract", "$/day",
                                     "P(<=0)"))
print("  %-28s %7d %+12.2fc %+10.2f %9.4f"
      % ("plain (:00 + band + 2min)", len(zp_ho), zp_ho.edge_c.mean(),
         zp_ho.pnl.sum() / zp_ho.day.nunique(), pv2))
print("  %-28s %7d %+12.2fc %+10.2f %9.4f"
      % ("train-optimised", len(ho), ho.edge_c.mean(),
         ho.pnl.sum() / ho.day.nunique(), pv))
delta = ho.edge_c.mean() - zp_ho.edge_c.mean()
print(f"\n  optimisation gained {delta:+.2f}c per contract out of sample")
print("  -> " + ("the search added value"
                 if delta > 0 else
                 "the search ADDED NOTHING - it fitted train noise"))

print("\n" + "-" * 88)
print("SELECTION CORRECTION over the whole grid")
print("-" * 88)
obs = best["train_edge"]
perm = []
for _ in range(400):
    vals = []
    for wait in GRID["wait_min"]:
        Dw = data_for(wait).copy()
        Dw = Dw[Dw.utc < TRAIN_END]
        Dw["won"] = Dw.groupby("day").won.transform(
            lambda s: RNG.permutation(s.values))
        for c in combos:
            if c["wait_min"] != wait:
                continue
            p = dict(SPEC); p.update(FIXED); p.update(c)
            z = apply_spec(Dw, p)
            if len(z) >= MIN_TRADES_TRAIN:
                vals.append(z.edge_c.mean())
    if vals:
        perm.append(max(vals))
perm = np.array(perm)
pp = float((perm >= obs).mean())
print(f"  observed train best {obs:+.2f}c")
print(f"  permuted best-of-grid mean {perm.mean():+.2f}c   "
      f"p95 {np.quantile(perm, .95):+.2f}c")
print(f"  P(permuted best >= observed) = {pp:.4f}   -> "
      f"{'SURVIVES' if pp < 0.05 else 'does NOT survive'}")

R.to_csv(OUT / "spec_optimize_grid.csv", index=False)
print("\nwrote research/results/spec_optimize_grid.csv")
