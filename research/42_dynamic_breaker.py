"""A breaker that stops on EVIDENCE, not on bad luck.

THE FLAW IN THE CURRENT DESIGN
  All three existing layers trigger on DRAWDOWN. Drawdown is mostly bad luck:
  window P&L is serially independent (autocorrelation null at every lag), so a
  bad run carries no information about the next window. A drawdown-triggered
  stop therefore fires mostly when nothing has changed - which is exactly why
  it costs $178 of expected wealth in the good case.

  What we actually want to stop for is the edge having DIED. That is a
  statistical question with a statistical answer, and it is a different signal
  from drawdown.

THE ALTERNATIVE
  A sequential test. Accumulate evidence for "edge is healthy" against "edge is
  dead" and stop when the evidence crosses a threshold. This is the classic
  SPRT: with per-settlement outcomes x, accumulate the log-likelihood ratio
  under the two hypotheses and stop when it favours 'dead' strongly enough.

  Crucially, a bad LUCK run raises the LLR only slightly and it decays as
  normal results resume; a genuinely dead edge drives it monotonically. So the
  test distinguishes the two cases the drawdown rule cannot.

WHAT IS COMPARED, over the same simulated paths
    A. no breaker at all
    B. the deployed static 30%-of-peak drawdown envelope
    C. an SPRT edge test
    D. both together
  in three regimes: edge alive, edge weak, edge dead. A dynamic breaker is only
  worth deploying if it beats the static one in the DEAD regime without costing
  materially more in the ALIVE regime.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260909)
D = pd.read_parquet(OUT / "grid.parquet")
QTY, BANK = 30, 447.31
NPATH, NWIN = 8000, 96 * 14

q = D[(D.px >= 0.65) & (D.px < 0.85) & (D.minute >= 8) & (D.minute <= 14)
      & (D.vol >= 2000)]
E = (q.sort_values("minute", ascending=False)
       .groupby(["wkey", "coin"], as_index=False).first())
E = (E.sort_values(["wkey", "minute"], ascending=[True, False])
       .groupby("wkey", as_index=False).head(3)).copy()
E["pnl"] = np.where(E.won == 1, (1 - E.px) * QTY, -E.px * QTY)
WP = E.groupby("wkey").pnl.sum().to_numpy()
BASE = WP.mean()
print(f"window pool: mean ${BASE:+.2f}  sd ${WP.std():.2f}  n {len(WP)}")

# SPRT calibration: healthy vs dead, in dollars per WINDOW
MU_OK = BASE                      # healthy
MU_DEAD = -2.0                    # dead: about -5c/contract at 1.8 positions
SD = WP.std()
A_STOP = np.log((1 - 0.05) / 0.02)   # stop when strongly favouring 'dead'
print(f"SPRT: H_ok mu=${MU_OK:+.2f}/window  H_dead mu=${MU_DEAD:+.2f}/window  "
      f"sd ${SD:.2f}  boundary {A_STOP:.2f}\n")


def llr_step(x):
    """log f(x|dead) - log f(x|ok) for a normal with common sd."""
    return (-(x - MU_DEAD) ** 2 + (x - MU_OK) ** 2) / (2 * SD ** 2)


def simulate(shift, use_dd, use_sprt):
    pool = WP + shift
    idx = RNG.integers(0, len(pool), size=(NPATH, NWIN))
    draws = pool[idx]
    eq = np.full(NPATH, BANK)
    peak = np.full(NPATH, BANK)
    llr = np.zeros(NPATH)
    dead = np.zeros(NPATH, dtype=bool)
    stop_at = np.full(NPATH, NWIN, dtype=int)
    for t in range(NWIN):
        live = ~dead
        if not live.any():
            break
        x = draws[:, t]
        eq[live] += x[live]
        peak = np.maximum(peak, eq)
        llr[live] += llr_step(x[live])
        newdead = np.zeros(NPATH, dtype=bool)
        if use_dd:
            newdead |= (eq < 0.70 * peak)
        if use_sprt:
            newdead |= (llr > A_STOP)
        newdead &= live
        stop_at[newdead] = t
        dead |= newdead
    return eq, dead, stop_at


print(f"{'regime':>14} {'breaker':>16} {'mean final':>12} {'median':>10} "
      f"{'5th pct':>10} {'P(ruin)':>9} {'P(stop)':>9} {'med stop win':>13}")
for lbl, shift in (("ALIVE", 0.0), ("WEAK", -4.0), ("DEAD", -8.0)):
    for bl, dd, sp in (("none", False, False), ("static 30%dd", True, False),
                       ("SPRT edge", False, True), ("both", True, True)):
        eq, dead, sa = simulate(shift, dd, sp)
        ms = int(np.median(sa[dead])) if dead.any() else -1
        print(f"{lbl:>14} {bl:>16} {eq.mean():>12,.0f} {np.median(eq):>10,.0f} "
              f"{np.percentile(eq, 5):>10,.0f} {(eq < 0.5*BANK).mean():>9.4f} "
              f"{dead.mean():>9.4f} {ms if ms>0 else '-':>13}")
    print()

print("=== THE DECISION CRITERION ===")
a_none, _, _ = simulate(0.0, False, False)
a_dd, d_dd, _ = simulate(0.0, True, False)
a_sp, d_sp, _ = simulate(0.0, False, True)
d_none, _, _ = simulate(-8.0, False, False)
d_ddx, _, _ = simulate(-8.0, True, False)
d_spx, _, s_spx = simulate(-8.0, False, True)
print(f"  cost when ALIVE : static 30%dd ${a_none.mean()-a_dd.mean():>8,.0f}   "
      f"SPRT ${a_none.mean()-a_sp.mean():>8,.0f}")
print(f"  saved when DEAD : static 30%dd ${d_ddx.mean()-d_none.mean():>8,.0f}   "
      f"SPRT ${d_spx.mean()-d_none.mean():>8,.0f}")
print(f"  false-stop rate when ALIVE: static {d_dd.mean():.4f}   "
      f"SPRT {d_sp.mean():.4f}")
print(f"  windows to detect a DEAD edge (median): SPRT "
      f"{int(np.median(s_spx)) if len(s_spx) else '-'}  "
      f"(~{int(np.median(s_spx))/96:.1f} days)")
print("\n  A dynamic breaker is worth deploying ONLY if it costs less when")
print("  alive AND saves at least as much when dead. If the static rule wins")
print("  on both, added complexity in a live risk control is a liability.")
