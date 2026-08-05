"""When a halt fires, when should we restart - and should the threshold scale?

TWO DEFECTS IN THE CURRENT SETUP
  1. FIXED DOLLARS DO NOT SCALE. $75 was 22% of the bankroll when it was set at
     $341. It is 14% now at $527. At $2,000 it would be 3.8%, which is well
     inside ordinary noise - the strategy would halt constantly on nothing. A
     fixed dollar stop silently tightens into uselessness as the account grows.
  2. NO RESTART RULE EXISTS. The halt is defined; the resume is not. In
     practice it has been "whenever a human notices", which is not a policy.

THE KEY MEASURED FACT THAT DETERMINES THE ANSWER
  Window P&L is serially INDEPENDENT (autocorrelation null at every lag,
  permutation p = 0.87, 0.28, 0.095, 0.68, 0.77). A drawdown therefore carries
  NO information about the next window. That splits halts into two kinds which
  need opposite restart rules:

    BAD LUCK halt   -> nothing has changed -> restart IMMEDIATELY. Waiting is
                       a pure tax on a positive-expectation edge.
    BAD EDGE halt   -> something HAS changed -> do not restart until the
                       evidence says otherwise.

  A drawdown trigger cannot distinguish them, which is why it needs a
  companion evidence test to decide the RESTART, even if the evidence test is
  not trusted to decide the HALT.

SIMULATED HERE
  Fixed vs fractional thresholds across bankrolls, and four restart rules, over
  paths where the edge is alive and paths where it is dead.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260911)
D = pd.read_parquet(OUT / "grid.parquet")
QTY = 30
NPATH, NWIN = 6000, 96 * 20

q = D[(D.px >= 0.65) & (D.px < 0.85) & (D.minute >= 8) & (D.minute <= 14)
      & (D.vol >= 2000)]
E = (q.sort_values("minute", ascending=False)
       .groupby(["wkey", "coin"], as_index=False).first())
E = (E.sort_values(["wkey", "minute"], ascending=[True, False])
       .groupby("wkey", as_index=False).head(3)).copy()
E["pnl"] = np.where(E.won == 1, (1 - E.px) * QTY, -E.px * QTY)
WP = E.groupby("wkey").pnl.sum().to_numpy()
print(f"window pool: mean ${WP.mean():+.2f}  sd ${WP.std():.2f}\n")

print("=== PART 1: HOW OFTEN DOES A FIXED $75 STOP FIRE, BY BANKROLL? ===")
print("   (edge ALIVE the whole time - every one of these is a FALSE stop)\n")
print(f"{'bankroll':>10} {'$75 as %':>10} {'P(halt in 20d)':>15} "
      f"{'median windows to halt':>24}")
for bank in (341, 527, 750, 1000, 2000, 5000):
    idx = RNG.integers(0, len(WP), size=(NPATH, NWIN))
    eq = bank + np.cumsum(WP[idx], axis=1)
    run = np.concatenate([np.full((NPATH, 1), float(bank)), eq], axis=1)
    peak = np.maximum.accumulate(run, axis=1)
    hit = (peak - run) > 75.0
    ever = hit.any(axis=1)
    first = np.where(ever, hit.argmax(axis=1), NWIN)
    med = int(np.median(first[ever])) if ever.any() else -1
    print(f"{bank:>10} {75/bank*100:>9.1f}% {ever.mean():>15.4f} "
          f"{med if med > 0 else '-':>24}")
print("\n  As the bankroll grows the SAME dollar stop becomes a smaller")
print("  fraction, so it fires more often on ordinary noise. A fixed stop is")
print("  not a constant risk policy - it is a tightening one.")

print("\n=== PART 2: FRACTIONAL THRESHOLD - constant risk at any bankroll ===")
print(f"{'threshold':>12} " + "".join(f"{'$'+str(b):>12}" for b in
                                      (341, 527, 1000, 2000, 5000)))
for frac in (0.10, 0.15, 0.20, 0.30):
    row = ""
    for bank in (341, 527, 1000, 2000, 5000):
        idx = RNG.integers(0, len(WP), size=(2500, NWIN))
        eq = bank + np.cumsum(WP[idx], axis=1)
        run = np.concatenate([np.full((2500, 1), float(bank)), eq], axis=1)
        peak = np.maximum.accumulate(run, axis=1)
        p = (((peak - run) / peak) > frac).any(axis=1).mean()
        row += f"{p:>12.4f}"
    print(f"{frac:>11.0%} {row}")
print("\n  P(false halt in 20 days) with the edge ALIVE. A fractional rule")
print("  gives the SAME risk profile at every bankroll, which is the point.")

print("\n=== PART 3: RESTART RULES, on paths where the edge is ALIVE ===")
print("   halt at 15% below peak, then resume by rule\n")


def sim_restart(rule, shift=0.0, frac=0.15, bank=527.0, npath=4000):
    pool = WP + shift
    idx = RNG.integers(0, len(pool), size=(npath, NWIN))
    draws = pool[idx]
    eq = np.full(npath, bank)
    peak = np.full(npath, bank)
    halted = np.zeros(npath, dtype=bool)
    cooldown = np.zeros(npath, dtype=int)
    trough = np.zeros(npath)
    nhalt = np.zeros(npath, dtype=int)
    traded = np.zeros(npath, dtype=int)
    for t in range(NWIN):
        live = ~halted
        eq[live] += draws[live, t]
        traded += live.astype(int)
        peak = np.maximum(peak, eq)
        newh = live & ((peak - eq) / peak > frac)
        nhalt += newh.astype(int)
        trough[newh] = eq[newh]
        halted |= newh
        if rule == "never":
            pass
        elif rule == "immediate":
            cooldown[halted] = 0
            halted[:] = False
        elif rule.startswith("wait"):
            k = int(rule.split("_")[1])
            cooldown[halted] += 1
            back = halted & (cooldown >= k)
            halted[back] = False
            cooldown[back] = 0
        elif rule == "recover_half":
            # resume only once equity has recovered half the drawdown
            back = halted & (eq >= trough + 0.5 * (peak - trough))
            halted[back] = False
    return eq, nhalt, traded


for lbl, shift in (("ALIVE", 0.0), ("DEAD (-8/window)", -8.0)):
    print(f"  --- edge {lbl} ---")
    print(f"{'restart rule':>16} {'mean final':>12} {'median':>10} "
          f"{'5th pct':>10} {'halts':>7} {'% windows traded':>18}")
    for rule in ("never", "immediate", "wait_96", "wait_288", "recover_half"):
        eq, nh, tr = sim_restart(rule, shift=shift)
        print(f"{rule:>16} {eq.mean():>12,.0f} {np.median(eq):>10,.0f} "
              f"{np.percentile(eq, 5):>10,.0f} {nh.mean():>7.2f} "
              f"{tr.mean()/NWIN*100:>17.1f}%")
    print()

print("=== THE POINT ===")
print("  On ALIVE paths, any rule that keeps you out costs money, because the")
print("  windows you skip are as good as any other - that is what serial")
print("  independence means. On DEAD paths, staying out is everything.")
print("  So the restart decision must be driven by EVIDENCE about the edge,")
print("  not by elapsed time and not by equity recovery - neither of which")
print("  carries information about whether the edge is still there.")
