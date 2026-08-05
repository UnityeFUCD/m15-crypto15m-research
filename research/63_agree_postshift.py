"""Does the agreement rule hold POST-SHIFT? It is now the only filter running.

WHY THIS ONE MATTERS MOST
  Two of the three things deployed today were reversed after re-testing:
  the dead-entry filter was costing $14.89/day, and the edge-weighted sizing
  model had stopped being contract-neutral and stopped ranking. Both failed for
  the same reason - fitted pre-shift, and the distributions moved.

  The agreement rule is the survivor. It is also the only remaining filter, so
  if it is stale too, the system is discarding entries for no reason.

  Its pre-shift evidence is strong: discarded cohort -19.80c with CI
  [-33.47,-5.00], +$18.12/day against a keep-all baseline, better on 8/10 days.
  The post-shift evidence was one line in a larger table and was never checked
  against a correct baseline or on the exact live band.

THE BAR, applied the same way that caught the other two
  1. baseline is KEEP-ALL, never discard-all
  2. the discarded cohort must be clearly NEGATIVE, not merely smaller
  3. dollars, day-clustered
  4. and specifically on the live configuration: 65-80c, 8-14 min, vol>=2000,
     cap 3, in the runner's own consideration order
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261026)
QTY = 20
SER = {"BTC": 0, "ETH": 1, "SOL": 2, "XRP": 3, "DOGE": 4, "HYPE": 5}


def rule(C, agree, cap=3):
    keep = []
    for wk, g in C.groupby("wkey", sort=True):
        fd, n = None, 0
        for r in g.itertuples():
            if n >= cap:
                break
            if fd is None:
                fd = bool(r.maker_yes)
            elif agree and bool(r.maker_yes) != fd:
                n += 1          # burn the slot, do not backfill
                continue
            keep.append(r)
            n += 1
    return pd.DataFrame([r._asdict() for r in keep]) if keep else pd.DataFrame()


def report(C, label):
    days = sorted(C.day.unique())
    a, b = rule(C, False), rule(C, True)
    if not len(a) or not len(b):
        print(f"  {label}: insufficient data")
        return
    ka = set(zip(a.wkey, a.coin))
    kb = set(zip(b.wkey, b.coin))
    drop = ka - kb
    dd = a[[(w, c) in drop for w, c in zip(a.wkey, a.coin)]]
    pa = a.groupby("day").edge.sum().reindex(days, fill_value=0) * QTY
    pb = b.groupby("day").edge.sum().reindex(days, fill_value=0) * QTY
    d = (pb - pa).values
    print(f"\n  {label}   ({len(days)} days)")
    print(f"    keep-all       : n {len(a):>4}  edge {a.edge.mean()*100:>+6.2f}c  "
          f"${pa.mean():>+8.2f}/day")
    print(f"    agreement rule : n {len(b):>4}  edge {b.edge.mean()*100:>+6.2f}c  "
          f"${pb.mean():>+8.2f}/day")
    if len(dd):
        print(f"    DISCARDED      : n {len(dd):>4} ({len(dd)/len(a):.3f})  "
              f"win {dd.won.mean():.4f}  edge {dd.edge.mean()*100:>+6.2f}c")
        g = {x: gg for x, gg in dd.groupby("day")}
        ds = sorted(g)
        if len(ds) >= 2:
            bs = np.sort(np.array([
                np.concatenate([g[ds[i]].edge.values
                                for i in RNG.integers(0, len(ds), len(ds))]).mean() * 100
                for _ in range(6000)]))
            print(f"    discarded-cohort 95% CI [{bs[150]:+.2f}, {bs[5849]:+.2f}]   "
                  f"P(>=0) {(bs >= 0).mean():.4f}")
    if len(days) >= 3:
        bs2 = np.sort(np.array([RNG.choice(d, len(d), True).mean()
                                for _ in range(8000)]))
        print(f"    GAIN ${d.mean():+.2f}/day  95% CI [{bs2[200]:+.2f},{bs2[7799]:+.2f}]"
              f"  better {(pb > pa).sum()}/{len(days)}")
    else:
        print(f"    GAIN ${d.mean():+.2f}/day  better {(pb > pa).sum()}/{len(days)} "
              f"(too few days for an interval)")


# PRE-SHIFT, exact live band
Y = pd.read_parquet(OUT / "yesno.parquet")
q = Y[(Y.px >= 0.65) & (Y.px < 0.80) & (Y.minute >= 8) & (Y.minute <= 14)
      & (Y.vol >= 2000)]
C1 = (q.sort_values("minute", ascending=False)
        .groupby(["wkey", "coin"], as_index=False).first()).copy()
C1["ord"] = C1.coin.map(SER).fillna(9)
C1 = C1.sort_values(["wkey", "ord", "minute"], ascending=[True, True, False])

# POST-SHIFT, exact live band
P = pd.read_parquet(OUT / "postshift_nofilter.parquet")
C2 = P[(P.vol >= 2000) & (P.px >= 0.65) & (P.px < 0.80)].copy()
C2["ord"] = C2.coin.map(SER).fillna(9)
C2 = C2.sort_values(["wkey", "ord", "px"], ascending=[True, True, False])

print("=" * 74)
print("AGREEMENT RULE ON THE EXACT LIVE CONFIG (65-80c, 8-14, vol>=2000, cap 3)")
print("=" * 74)
report(C1, "PRE-SHIFT")
report(C2, "POST-SHIFT")

print("\n" + "=" * 74)
print("HOW OFTEN DOES THE RULE EVEN BIND?")
print("=" * 74)
for lbl, C in (("PRE", C1), ("POST", C2)):
    W = C.groupby("wkey").agg(n=("maker_yes", "size"),
                              y=("maker_yes", "sum")).reset_index()
    M = W[W.n >= 2]
    mixed = ((M.y > 0) & (M.y < M.n)).mean() if len(M) else float("nan")
    print(f"  {lbl:>5}: windows with 2+ candidates {len(M):>4}   "
          f"direction-MIXED {mixed:.3f}")
print("\n  The rule can only act on mixed windows. If mixing is rare it cannot")
print("  help or hurt much either way - and its measured effect should be")
print("  read in that light.")
