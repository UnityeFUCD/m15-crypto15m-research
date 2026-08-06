"""Does the variance reduction matter, given the edge is indistinguishable from zero?

THE DECISION THIS INFORMS
  Three options: keep running the existing q15 maker strategy, switch to PTC,
  or stop. The measured edge is +0.02c per submitted contract with a 95%
  interval of [-1.57, +1.62]. PTC cuts per-window dispersion about 9x
  ($13.10 -> $1.49) without changing the mean.

  Variance reduction cannot manufacture a positive mean. What it does change
  is whether you survive long enough for a small positive mean - if one
  exists - to arrive. That is the only honest case for PTC, so it is the one
  tested here.

METHOD
  The edge is not known, so it is not assumed. Each simulated account draws
  its true per-contract edge from the measured posterior (normal, centred on
  +0.02c with the measured day-clustered SD), and then trades under that edge
  for the horizon. Averaging over draws propagates the uncertainty into the
  answer instead of hiding it behind a point estimate.

  Ruin is the $211 static floor. Paths are streamed in chunks; an earlier
  simulator in this repo allocated the whole matrix and paged to disk.

WHAT WOULD CHANGE THE ANSWER
  If PTC survives materially more often at the same expected profit, it is
  worth running even at zero mean, because survival is what buys the sample
  size needed to measure the edge at all. If both policies converge to the
  same fate, the variance reduction is cosmetic and the correct action is to
  stop rather than to switch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FLOOR = 211.0
WINDOWS_PER_DAY = 96          # in-band close windows actually available
N_PATHS = 40_000
CHUNK = 4_000
RNG = np.random.default_rng(20260806)

# measured per-window dispersion, from ptc_v3 on the complete 303-order ledger
SD_CONTROL = 13.1048
SD_PTC = 1.4860
# measured population edge per submitted contract, day-clustered
EDGE_MEAN_C = 0.02
EDGE_SD_C = 0.80              # (CI [-1.57,+1.62] -> ~0.80c per side / 1.96)
CONTRACTS_PER_WINDOW = {"CONTROL": 15.0, "PTC": 15.0 * 6 / 125}


def run(policy: str, equity0: float, days: int, edge_c: np.ndarray,
        sd_window: float, contracts: float) -> dict:
    steps = days * WINDOWS_PER_DAY // 4          # ~24 assigned windows/day
    ruined = np.zeros(N_PATHS, dtype=bool)
    final = np.empty(N_PATHS)
    for lo in range(0, N_PATHS, CHUNK):
        hi = min(lo + CHUNK, N_PATHS)
        k = hi - lo
        mu = edge_c[lo:hi, None] / 100.0 * contracts
        draws = RNG.normal(mu, sd_window, size=(k, steps))
        eq = equity0 + np.cumsum(draws, axis=1)
        ruined[lo:hi] = eq.min(axis=1) <= FLOOR
        final[lo:hi] = eq[:, -1]
    return {"policy": policy, "days": days, "equity0": equity0,
            "p_ruin": float(ruined.mean()),
            "median_final": float(np.median(final)),
            "p_profit": float((final > equity0).mean()),
            "p5": float(np.quantile(final, 0.05)),
            "p95": float(np.quantile(final, 0.95))}


print("=" * 84)
print("SURVIVAL ANALYSIS - is the variance reduction worth anything?")
print("=" * 84)
print(f"  edge drawn per account from N({EDGE_MEAN_C:+.2f}c, {EDGE_SD_C:.2f}c)"
      f" - the measured uncertainty, not a point estimate")
print(f"  ruin  = equity touches the ${FLOOR:.0f} floor")
print(f"  per-window SD: CONTROL ${SD_CONTROL:.2f}   PTC ${SD_PTC:.2f}")
print(f"  {N_PATHS:,} paths per cell")

rows = []
for equity0 in (400.0, 700.0, 1500.0):
    edge = RNG.normal(EDGE_MEAN_C, EDGE_SD_C, N_PATHS)
    for days in (30, 90, 180):
        rows.append(run("CONTROL", equity0, days, edge, SD_CONTROL,
                        CONTRACTS_PER_WINDOW["CONTROL"]))
        rows.append(run("PTC", equity0, days, edge, SD_PTC,
                        CONTRACTS_PER_WINDOW["PTC"]))
R = pd.DataFrame(rows)
R.to_csv(ROOT / "research" / "results" / "survival_decision.csv", index=False)

for equity0 in (400.0, 700.0, 1500.0):
    print(f"\n  starting equity ${equity0:,.0f}")
    print("    %-9s %6s %10s %12s %12s %12s"
          % ("policy", "days", "P(ruin)", "P(profit)", "median", "5th pct"))
    for days in (30, 90, 180):
        for pol in ("CONTROL", "PTC"):
            r = R[(R.policy == pol) & (R.days == days)
                  & (R.equity0 == equity0)].iloc[0]
            print("    %-9s %6d %10.4f %12.4f %12.2f %12.2f"
                  % (pol, days, r.p_ruin, r.p_profit, r.median_final, r.p5))

print("\n" + "=" * 84)
print("READING IT")
print("=" * 84)
a = R[(R.policy == "CONTROL") & (R.days == 90) & (R.equity0 == 700.0)].iloc[0]
b = R[(R.policy == "PTC") & (R.days == 90) & (R.equity0 == 700.0)].iloc[0]
print(f"  At $700 over 90 days:")
print(f"    CONTROL  P(ruin) {a.p_ruin:.4f}   median ${a.median_final:,.2f}")
print(f"    PTC      P(ruin) {b.p_ruin:.4f}   median ${b.median_final:,.2f}")
print()
print("  PTC survives far more often. It also stakes far less, so it earns")
print("  far less when the drawn edge is positive. Variance reduction here is")
print("  not free performance - it is a smaller bet.")
print()
print("  The honest framing: PTC is not a way to make the strategy profitable.")
print("  It is a way to stay solvent long enough to find out whether it is,")
print("  and the only reason that is worth anything is that 73 days of data")
print("  could not answer the question.")

print("\n" + "=" * 84)
print("HOW LONG UNTIL THE EDGE IS ACTUALLY MEASURABLE?")
print("=" * 84)
for pol, sd, ct in (("CONTROL", SD_CONTROL, CONTRACTS_PER_WINDOW["CONTROL"]),
                    ("PTC", SD_PTC, CONTRACTS_PER_WINDOW["PTC"])):
    for target_c in (0.5, 1.0, 2.0):
        mu = target_c / 100.0 * ct
        n = (2.802 * sd / mu) ** 2 if mu > 0 else np.inf
        print("  %-9s detect %+0.1fc/contract -> %12,.0f windows = %8,.0f days"
              .replace(",", "") % (pol, target_c, n, n / 24))
print("\n  Neither policy reaches a 0.5c resolution on any sane horizon.")
print("  That is the finding: the edge is not merely unproven, it is")
print("  unmeasurable at this account size, under either execution.")
