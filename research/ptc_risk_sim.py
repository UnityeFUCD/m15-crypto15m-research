"""PART 9 - risk and scaling simulator.

Runs >=100,000 chronological block-bootstrap paths and evaluates the scaling
gates before any size increase:

    P(hit hard floor within 30 days) < 1%
    P(hit hard floor within 90 days) < 5%
    P(drawdown >= 20% within 30 days) < 5%

INPUT HONESTY. The gates are specified against ACTUAL PROSPECTIVE PTC results.
None exist - the randomized trial has not run. This simulator therefore reads
the best available substitute, the historical PTC_120 close-window series, and
labels every output as PROVISIONAL. That substitute is optimistic in a way
that matters and the script says so rather than burying it: the historical
series contains six commitments over two days, one of which supplies 59% of
the IOC branch. Its mean is not a population mean.

The simulator is still worth running now, because the stress scenarios answer
a question that does not depend on the mean being right: at what size does
ruin become likely even if the edge is real?

STRESS SCENARIOS (all required by Part 9)
    base                observed series
    cost_1c / cost_2c   one and two cents of extra cost per committed contract
    decay_50            edge halved
    zero_edge           edge removed, dispersion retained
    adverse_blocks      repeated adverse directional blocks
    interrupted         API failures and restarts drop a share of windows
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.ptc_v3 import build, replay, PRIMARY_QTY      # noqa: E402

OUT = ROOT / "research" / "results"
OUT.mkdir(parents=True, exist_ok=True)

N_PATHS = 100_000
WINDOWS_PER_DAY = 24            # close windows assigned per arm per day
STATIC_FLOOR = 211.0
RNG_SEED = 20260806


CHUNK = 5_000          # paths held in memory at once


def simulate(series: np.ndarray, equity0: float, qty_scale: float,
             days: int, block: int, rng: np.random.Generator,
             floor: float = STATIC_FLOOR) -> dict:
    """Chronological moving-block bootstrap, streamed in chunks.

    Each path contributes only four scalars, so there is no reason to hold the
    full (paths x steps) matrix. An earlier version did: at 100,000 paths and
    a 90-day horizon that is 100,000 x 2,160 float64 = 1.73 GB per array and
    four arrays per cell, times 112 cells. It was not compute-bound, it was
    paging to disk. Streaming keeps peak memory near 86 MB for identical
    output.
    """
    steps = days * WINDOWS_PER_DAY
    n = len(series)
    block = max(1, min(block, n))
    starts = np.arange(0, n - block + 1)
    n_blocks = int(np.ceil(steps / block))

    min_eq = np.empty(N_PATHS)
    max_dd = np.empty(N_PATHS)
    final = np.empty(N_PATHS)
    off = np.arange(block)

    for lo in range(0, N_PATHS, CHUNK):
        hi = min(lo + CHUNK, N_PATHS)
        k = hi - lo
        pick = rng.choice(starts, size=(k, n_blocks))
        idx = pick[:, :, None] + off[None, None, :]
        seg = series[idx].reshape(k, -1)[:, :steps] * qty_scale
        eq = equity0 + np.cumsum(seg, axis=1)
        run = np.maximum.accumulate(eq, axis=1)
        run = np.maximum(run, equity0)
        dd = (run - eq) / np.maximum(run, 1e-9)
        min_eq[lo:hi] = eq.min(axis=1)
        max_dd[lo:hi] = dd.max(axis=1)
        final[lo:hi] = eq[:, -1]

    return {
        "p_floor": float((min_eq <= floor).mean()),
        "p_dd20": float((max_dd >= 0.20).mean()),
        "median_final": float(np.median(final)),
        "p5_final": float(np.quantile(final, 0.05)),
        "mean_max_dd_pct": float(max_dd.mean()),
    }


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    data = build()
    x = replay(data, 120, qty=PRIMARY_QTY)
    w = x.groupby("close_dt").pnl.sum().sort_index()
    base = w.to_numpy(float)

    print("=" * 86)
    print("PART 9 - RISK AND SCALING SIMULATION  (PROVISIONAL INPUT)")
    print("=" * 86)
    print(f"  source series: historical PTC_120, {len(base)} close windows, "
          f"2 calendar days")
    print(f"  mean {base.mean():+.4f} $/window   SD {base.std(ddof=1):.4f}")
    print("  *** This is NOT a prospective series. Six commitments, one of")
    print("  *** which is 59% of the IOC branch. Treat the mean as unknown;")
    print("  *** the dispersion and the ruin geometry are the usable parts.")

    scen = {
        "base": base,
        "cost_1c": base - 0.01 * PRIMARY_QTY * (x.selected.sum() / len(w)),
        "cost_2c": base - 0.02 * PRIMARY_QTY * (x.selected.sum() / len(w)),
        "decay_50": base.mean() * 0.5 + (base - base.mean()),
        "zero_edge": base - base.mean(),
        "adverse_blocks": np.concatenate([base, np.sort(base)[:max(1, len(base)//5)]]),
        "interrupted": base * (rng.random(len(base)) > 0.10),
    }

    rows = []
    for qty_scale, label in ((1 / 3, "q5"), (2 / 3, "q10"), (1.0, "q15"),
                             (4 / 3, "q20")):
        for name, s in scen.items():
            for days, tag in ((30, "30d"), (90, "90d")):
                for equity0 in (500.0,):
                    r = simulate(np.asarray(s, float), equity0, qty_scale,
                                 days, 4, rng)
                    rows.append({"qty": label, "scenario": name,
                                 "horizon": tag, "equity0": equity0, **r})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "ptc_risk_sim.csv", index=False)

    print("\n" + "-" * 86)
    print(f"SCALING GATES at $500 start ({N_PATHS:,} paths per cell)")
    print("-" * 86)
    print("  gates: P(floor,30d) < 1%   P(floor,90d) < 5%   P(dd>=20%,30d) < 5%")
    print("  %-6s %-16s %14s %14s %14s %8s"
          % ("qty", "scenario", "P(floor 30d)", "P(floor 90d)", "P(dd20 30d)",
             "verdict"))
    passes = {}
    for qty in ("q5", "q10", "q15", "q20"):
        for name in scen:
            a = R[(R.qty == qty) & (R.scenario == name) & (R.horizon == "30d")
                  & (R.equity0 == 500.0)]
            b = R[(R.qty == qty) & (R.scenario == name) & (R.horizon == "90d")
                  & (R.equity0 == 500.0)]
            if not len(a) or not len(b):
                continue
            p30, p90 = float(a.p_floor.iloc[0]), float(b.p_floor.iloc[0])
            d30 = float(a.p_dd20.iloc[0])
            ok = (p30 < 0.01) and (p90 < 0.05) and (d30 < 0.05)
            passes[(qty, name)] = ok
            print("  %-6s %-16s %13.4f %14.4f %14.4f %8s"
                  % (qty, name, p30, p90, d30, "PASS" if ok else "FAIL"))

    print("\n" + "-" * 86)
    print("LARGEST SIZE PASSING EVERY SCENARIO")
    print("-" * 86)
    best = None
    for qty in ("q5", "q10", "q15", "q20"):
        if all(passes.get((qty, n), False) for n in scen):
            best = qty
    print(f"  {best if best else 'NONE - no size clears every stress scenario'}")
    if best is None:
        for qty in ("q5", "q10", "q15", "q20"):
            bad = [n for n in scen if not passes.get((qty, n), False)]
            print(f"    {qty:4s} fails: {', '.join(bad) if bad else '-'}")

    print("\n" + "-" * 86)
    print("THE CONSTRAINT THAT ACTUALLY BINDS TODAY")
    print("-" * 86)
    print(f"  account equity   $136.27")
    print(f"  static floor     ${STATIC_FLOOR:.2f}")
    print(f"  shortfall        ${STATIC_FLOOR - 136.27:.2f}")
    print("  commit_quantity() returns 0 at this equity and the machine")
    print("  reports KILL. No simulated size is reachable until the account")
    print("  is funded above the floor, and the verdict is FAIL regardless.")

    (OUT / "ptc_risk_sim_summary.json").write_text(
        json.dumps({"paths_per_cell": N_PATHS,
                    "source": "historical PTC_120 (PROVISIONAL)",
                    "windows": len(base),
                    "largest_passing_size": best,
                    "gate_results": {f"{k[0]}|{k[1]}": v
                                     for k, v in passes.items()}},
                   indent=2), encoding="utf-8")
    print("\nwrote research/results/ptc_risk_sim.csv")


if __name__ == "__main__":
    main()
