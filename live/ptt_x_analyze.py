"""PTT-X readout. Primary observable is implementation shortfall, which needs
no settlement, so it is measurable at n in the tens rather than the thousands.

  IS = C_ptt - C_immediate_taker     per contract, NEGATIVE = cheaper

Errors are clustered on the 15-minute close window: three coins in one window
are one macro draw, not three, and treating them as independent is the mistake
that made every earlier headline result look significant.
"""
import glob, json, sys
import numpy as np
import pandas as pd

rows, drops = [], 0
for fn in sorted(glob.glob("ptt_*.jsonl")):
    for line in open(fn):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("ev") == "shortfall":
            rows.append(r)
        elif r.get("ev") == "dropped_row":
            drops += 1
if not rows:
    sys.exit("no shortfall rows yet")

D = pd.DataFrame(rows)
D = D[D.acquired > 0].copy()
print(f"opportunities scored {len(D)}   windows {D.window.nunique()}   "
      f"rows dropped as unresolved {drops}")
if not len(D):
    sys.exit("nothing acquired yet")

D["is_c"] = D.IS_per_contract_c.astype(float)


def cluster_ci(d, col="is_c", B=4000, seed=20260804):
    """Block bootstrap over close windows. Weight by contracts acquired."""
    g = list(d.groupby("window"))
    if len(g) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    for b in range(B):
        pick = [g[i][1] for i in rng.integers(0, len(g), len(g))]
        s = pd.concat(pick)
        out[b] = np.average(s[col], weights=s.acquired)
    out.sort()
    return out[int(.025 * B)], out[int(.975 * B)]


def wm(d, col="is_c"):
    return float(np.average(d[col], weights=d.acquired))


print("\n=== PRIMARY: implementation shortfall by arm (negative = cheaper) ===")
print(f"{'arm':>5} {'opps':>5} {'contracts':>10} {'F_Q':>7} {'complete':>9} "
      f"{'IS c/contract':>14} {'95% CI (window-clustered)':>28}")
for arm, d in D.groupby("arm"):
    lo, hi = cluster_ci(d)
    fq = d.passive_qty.sum() / d.target.sum()
    ci = f"[{lo:+7.3f},{hi:+7.3f}]" if np.isfinite(lo) else "  (need >1 window)"
    print(f"{arm:>5} {len(d):>5} {int(d.acquired.sum()):>10} {fq:>7.3f} "
          f"{d.completed.mean():>9.3f} {wm(d):>+14.3f} {ci:>28}")

ctl = D[D.arm == 0]
if len(ctl):
    lo, hi = cluster_ci(ctl)
    print(f"\n  CONTROL CHECK (arm 0 crosses for real; IS should be ~0):"
          f" {wm(ctl):+.3f}c  CI[{lo:+.3f},{hi:+.3f}]")
    print("  a control materially away from zero is depth-model bias and it"
          "\n  contaminates every other arm by the same amount.")

print("\n=== the quantity that decides it:  IS = F_Q*S - (1-F_Q)*D ===")
tre = D[D.arm > 0]
if len(tre):
    fq = tre.passive_qty.sum() / tre.target.sum()
    S = np.average(tre.c_imm_per - tre.rest_px, weights=tre.target) * 100
    chased = tre[tre.taker_qty > 0]
    Dch = (np.average(chased.c_ptt / chased.acquired - chased.c_imm_per,
                      weights=chased.acquired) * 100) if len(chased) else np.nan
    print(f"  F_Q  quantity-weighted passive fill   {fq:.4f}   "
          f"(order-count fill would say {(tre.passive_qty>0).mean():.4f})")
    print(f"  S    saving on a passive fill         {S:+.3f}c")
    print(f"  D    chase penalty on the residual    {Dch:+.3f}c  (n={len(chased)} chased)")
    if np.isfinite(Dch) and fq < 1:
        print(f"  breakeven chase penalty  D* = F/(1-F)*S = {fq/(1-fq)*S:+.3f}c"
              f"   -> {'SURVIVES' if Dch < fq/(1-fq)*S else 'FAILS'}")
    elif fq >= 1:
        print("  D* undefined: nothing has failed to fill yet")
    print(f"\n  abandoned residual {int(tre.abandoned.sum())} of "
          f"{int(tre.target.sum())} contracts "
          f"({tre.abandoned.sum()/tre.target.sum():.4f})")
    print("  abandonment is the same-population leak: any contract we wanted and"
          "\n  never acquired puts the maker selection bias back.")

print("\n=== by placement ===")
for pl, d in D.groupby("placement"):
    print(f"  {pl:14s} opps {len(d):4d}  F_Q {d.passive_qty.sum()/d.target.sum():.3f}  "
          f"IS {wm(d):+.3f}c  median spread {d.spread0_c.median():.2f}c")

print("\n=== depth: does the top of book actually hold our size? ===")
print(f"  median depth available at decision {D.depth_at_decision.median():.0f} "
      f"of {D.target.median():.0f} requested")
print(f"  cost source: {D.cost_src.value_counts().to_dict()}")
