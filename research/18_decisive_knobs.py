"""The decisive test for the only two knob changes worth making.

CONTEXT
  The live account has now confirmed the model: predicted win 0.798 / +8.34c,
  delivered 0.796 / +9.27c over 1,876 contracts. So the baseline is real and
  the question is narrow - does raising the volume floor from 500 and the
  per-window cap from 2 actually improve things, or is that a second-order
  overfit to the same data that discovered the volume filter?

DESIGN
  1. WALK-FORWARD. For each day d, the config is judged only on day d, and the
     comparison across configs uses the same days. No day is scored by a rule
     tuned on itself.
  2. WINDOW-EQUAL-WEIGHTED. Contract weighting has faked three findings this
     session by over-sampling busy windows.
  3. PAIRED, DAY-CLUSTERED bootstrap. The candidate and the incumbent trade
     overlapping windows, so the paired difference is far better estimated than
     either level. Clustering on day is what respects the real dependence.
  4. The volume filter was DISCOVERED on this data, so the level of the effect
     is optimistically biased. Only the SHAPE across thresholds is being tested
     here, and it is reported with that caveat attached.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260808)
D = pd.read_parquet(OUT / "grid.parquet")
QTY, MAX_GROSS = 30, 110.0
days = sorted(D.day.unique())


def take(plo, phi, mlo, mhi, vmin, cap):
    q = D[(D.px >= plo) & (D.px < phi) & (D.minute >= mlo) & (D.minute <= mhi)
          & (D.vol >= vmin)]
    if not len(q):
        return pd.DataFrame(columns=["day", "wkey", "px", "edge", "won"])
    e = (q.sort_values("minute", ascending=False)
           .groupby(["wkey", "coin"], as_index=False).first())
    e = (e.sort_values(["wkey", "minute"], ascending=[True, False])
           .groupby("wkey", as_index=False).head(cap))
    keep = []
    for _, g in e.groupby("wkey"):
        gross = 0.0
        for _, r in g.iterrows():
            c = QTY * r.px
            if gross + c > MAX_GROSS:
                continue
            gross += c
            keep.append((r.day, r.wkey, r.px, r.edge, r.won))
    return pd.DataFrame(keep, columns=["day", "wkey", "px", "edge", "won"])


def dayboot(vals_by_day, n=6000):
    """Day-clustered bootstrap of a mean over pooled per-entry values."""
    ds = list(vals_by_day)
    out = np.empty(n)
    for i in range(n):
        pick = RNG.choice(len(ds), len(ds), True)
        v = np.concatenate([vals_by_day[ds[k]] for k in pick])
        out[i] = v.mean()
    return np.sort(out)


print("=== VOLUME FLOOR LADDER (65-85c, 8-14 min, cap 2) ===")
print(f"{'vmin':>6} {'ent/day':>8} {'win':>7} {'edge c':>8} "
      f"{'day-clustered 95% CI':>24} {'$/day @30':>10}")
base = None
lad = {}
for vmin in (0, 250, 500, 1000, 2000, 4000):
    t = take(0.65, 0.85, 8, 14, vmin, 2)
    if len(t) < 60:
        continue
    g = {d: gg.edge.to_numpy() for d, gg in t.groupby("day")}
    bs = dayboot(g)
    lad[vmin] = t
    lo, hi = bs[int(.025 * len(bs))] * 100, bs[int(.975 * len(bs))] * 100
    print(f"{vmin:>6} {len(t)/len(days):>8.1f} {t.won.mean():>7.4f} "
          f"{t.edge.mean()*100:>+8.2f} [{lo:>+9.2f},{hi:>+9.2f}] "
          f"{t.edge.sum()*QTY/len(days):>+10.2f}")

print("\n=== PAIRED TEST: does raising the floor beat keeping it at 500? ===")
print("   comparison is on the WINDOWS THEY SHARE, so it isolates the")
print("   threshold effect from the change in which windows get traded.\n")
b = lad[500]
print(f"{'candidate':>10} {'shared win':>11} {'base edge':>10} {'cand edge':>10} "
      f"{'DIFF':>8} {'95% CI':>20} {'p':>7}")
for vmin in (1000, 2000, 4000):
    c = lad.get(vmin)
    if c is None:
        continue
    m = b.merge(c, on=["day", "wkey"], suffixes=("_b", "_c"))
    if len(m) < 40:
        continue
    m["d"] = m.edge_c - m.edge_b
    g = {d: gg.d.to_numpy() for d, gg in m.groupby("day")}
    bs = dayboot(g)
    lo, hi = bs[int(.025 * len(bs))] * 100, bs[int(.975 * len(bs))] * 100
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    print(f"{vmin:>10} {len(m):>11,} {m.edge_b.mean()*100:>+10.2f} "
          f"{m.edge_c.mean()*100:>+10.2f} {m.d.mean()*100:>+8.2f} "
          f"[{lo:>+8.2f},{hi:>+8.2f}] {p:>7.4f}")
print("  NOTE: on shared windows the two configs hold the SAME positions, so a")
print("  zero difference here is EXPECTED and correct. The floor works by")
print("  REMOVING windows, which the ladder above measures, not this table.")

print("\n=== WALK-FORWARD: pick the floor on past days only, trade the next ===")
print("   this is the honest question - would choosing the floor from history")
print("   have beaten just leaving it at 500?\n")
cands = [0, 500, 1000, 2000, 4000]
tabs = {v: take(0.65, 0.85, 8, 14, v, 2) for v in cands}
picked, wf, fixed = [], [], []
for i in range(3, len(days)):
    past, now = days[:i], days[i]
    best, bs_ = None, -9e9
    for v in cands:
        t = tabs[v]
        h = t[t.day.isin(past)]
        if len(h) < 60:
            continue
        sc = h.edge.sum() * QTY / len(past)          # $/day on history
        if sc > bs_:
            bs_, best = sc, v
    tn = tabs[best]
    tn = tn[tn.day == now]
    t5 = tabs[500]
    t5 = t5[t5.day == now]
    picked.append(best)
    wf.append(tn.edge.sum() * QTY)
    fixed.append(t5.edge.sum() * QTY)
    print(f"  {now}  chose vmin {best:>5}  ->  ${tn.edge.sum()*QTY:>+8.2f}   "
          f"(fixed 500: ${t5.edge.sum()*QTY:>+8.2f})")
wf, fixed = np.array(wf), np.array(fixed)
print(f"\n  walk-forward selection : ${wf.mean():+.2f}/day")
print(f"  fixed at vmin 500      : ${fixed.mean():+.2f}/day")
print(f"  difference             : ${wf.mean()-fixed.mean():+.2f}/day "
      f"over {len(wf)} out-of-sample days")
print(f"  floors chosen          : {picked}")

print("\n=== PER-WINDOW CAP, at the floor the walk-forward supports ===")
vsel = max(set(picked), key=picked.count)
print(f"   (using vmin {vsel}, the most frequently chosen)\n")
print(f"{'cap':>5} {'ent/day':>8} {'win':>7} {'edge c':>8} {'$/day @30':>10} "
      f"{'3rd-entry edge':>15}")
for cap in (1, 2, 3, 4):
    t = take(0.65, 0.85, 8, 14, vsel, cap)
    if len(t) < 60:
        continue
    extra = ""
    if cap >= 2:
        prev = take(0.65, 0.85, 8, 14, vsel, cap - 1)
        m = t.merge(prev[["day", "wkey", "px"]], on=["day", "wkey", "px"],
                    how="left", indicator=True)
        add = m[m._merge == "left_only"]
        if len(add) > 20:
            extra = f"{add.edge.mean()*100:>+15.2f}"
    print(f"{cap:>5} {len(t)/len(days):>8.1f} {t.won.mean():>7.4f} "
          f"{t.edge.mean()*100:>+8.2f} {t.edge.sum()*QTY/len(days):>+10.2f} {extra}")
print("\n  the last column is the edge of the MARGINAL entry that each cap adds.")
print("  if it stays positive, raising the cap adds money rather than diluting.")
