"""Should we enter LATER - 4-6 minutes out instead of 8-14?

WHY THIS IS GENUINELY UNTESTED
  Every timing test so far narrowed WITHIN the deployed 8-14 minute band:
  8-11, 8-10, 8-9, 11-14. All were worse, and I concluded "enter early". But
  narrowing inside a band is not the same as MOVING the band. Nothing below 8
  minutes has ever been tested, and the grid carries observations down to
  minute 2.

THE TRADE-OFF, and why the sign is not obvious
  Entering later means:
    + the favourite has had longer to establish itself, so the win rate should
      be HIGHER - less time remains for it to be overturned
    - the price is higher for exactly that reason, so the edge per contract
      should be LOWER
    - less time remains to fill, and the measured fill rate falls hard with
      less time (0.909 at 13.5-14 min, 0.550 at 10-12)
    + capital is tied up for less time, which matters if capital binds

  Whether the higher win rate outruns the higher price is empirical. The
  earlier tests answered it for 8-11 vs 8-14; they did NOT answer it for 4-6
  vs 8-14, where the price/probability trade-off is much steeper.

MEASURED HONESTLY
  - a chronological capital ledger, since a shorter hold changes turnover
  - the fill-rate penalty applied explicitly, because a band we cannot fill is
    worth nothing regardless of its edge
  - window-equal weighting and day-clustered intervals throughout
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261014)
D = pd.read_parquet(OUT / "grid.parquet")
QTY, LAG = 15, 45.0
days = sorted(D.day.unique())

print("=== RAW ECONOMICS BY MINUTE REMAINING (65-85c, vol>=2000) ===")
q = D[(D.px >= 0.65) & (D.px < 0.85) & (D.vol >= 2000) & (D.minute >= 2)
      & (D.minute <= 14)]
print(f"{'min left':>9} {'n':>6} {'avg px':>8} {'win':>8} {'required':>10} "
      f"{'SURPLUS':>9} {'edge/ct':>9}")
for m, g in q.groupby("minute"):
    if len(g) < 40:
        continue
    print(f"{int(m):>9} {len(g):>6,} {g.px.mean()*100:>7.1f}c {g.won.mean():>8.4f} "
          f"{g.px.mean()*100:>9.1f}% {(g.won.mean()-g.px.mean())*100:>+8.2f}pp "
          f"{g.edge.mean()*100:>+8.2f}c")
print("\n  Note what happens to PRICE as the window closes - if it rises faster")
print("  than the win rate, the surplus shrinks and later entry is worse.")

W = D.copy()
W["close"] = pd.to_datetime(W.wkey, format="%y%b%d%H%M", utc=True,
                            errors="coerce")
W = W.dropna(subset=["close"]).copy()
W["t_entry"] = W.close - pd.to_timedelta(W.secs, unit="s")
W["t_free"] = W.close + pd.to_timedelta(LAG, unit="s")

# measured fill rate by time remaining, from LIVE data
FILL = {14: 0.909, 13: 0.909, 12: 0.835, 11: 0.835, 10: 0.550, 9: 0.550,
        8: 0.550, 7: 0.45, 6: 0.40, 5: 0.35, 4: 0.30, 3: 0.25, 2: 0.20}


def sim(lo, hi, gross_cap=110.0, cap=3, use_fill=False):
    s = W[(W.px >= 0.65) & (W.px < 0.85) & (W.vol >= 2000)
          & (W.minute >= lo) & (W.minute <= hi)]
    if len(s) < 50:
        return None
    e = (s.sort_values("minute", ascending=False)
           .groupby(["wkey", "coin"], as_index=False).first())
    e = (e.sort_values(["wkey", "minute"], ascending=[True, False])
           .groupby("wkey", as_index=False).head(cap)).sort_values("t_entry")
    openp, taken = [], []
    for r in e.itertuples():
        openp = [p for p in openp if p[0] > r.t_entry]
        g = sum(c for _, c in openp)
        cost = QTY * r.px
        if g + cost > gross_cap:
            continue
        openp.append((r.t_free, cost))
        f = FILL.get(int(r.minute), 0.5) if use_fill else 1.0
        taken.append((r.day, r.edge * QTY * f, r.minute, r.px, r.won, f))
    if len(taken) < 30:
        return None
    t = pd.DataFrame(taken, columns=["day", "pnl", "minute", "px", "won", "f"])
    per = t.groupby("day").pnl.sum().reindex(days, fill_value=0.0)
    return dict(n=len(t), ent_day=len(t) / len(days), day=per.mean(),
                sd=per.std(ddof=1), edge=t.pnl.sum() / (len(t) * QTY) * 100,
                px=t.px.mean() * 100, win=t.won.mean(),
                mean_min=t.minute.mean(), fill=t.f.mean())


print("\n" + "=" * 78)
print("ENTRY BANDS, IGNORING the fill-rate penalty (upper bound on late entry)")
print("=" * 78)
print(f"{'band':>10} {'ent/day':>8} {'avg min':>8} {'avg px':>8} {'win':>7} "
      f"{'edge/ct':>9} {'$/day':>9} {'sd':>8}")
BANDS = [(8, 14, "8-14 LIVE"), (6, 14, "6-14"), (4, 14, "4-14"),
         (2, 14, "2-14"), (4, 8, "4-8"), (4, 6, "4-6"), (5, 7, "5-7"),
         (2, 6, "2-6"), (6, 8, "6-8"), (2, 4, "2-4")]
res = {}
for lo, hi, lbl in BANDS:
    r = sim(lo, hi)
    if not r:
        print(f"{lbl:>10}  -- too few entries --")
        continue
    res[lbl] = r
    print(f"{lbl:>10} {r['ent_day']:>8.1f} {r['mean_min']:>8.2f} "
          f"{r['px']:>7.1f}c {r['win']:>7.4f} {r['edge']:>+9.2f} "
          f"{r['day']:>+9.2f} {r['sd']:>8.2f}")

print("\n" + "=" * 78)
print("THE SAME BANDS, WITH the measured fill-rate penalty applied")
print("=" * 78)
print("  Fill rate collapses as the window closes - 0.909 at 13.5-14 min but")
print("  0.550 at 10-12 and lower below that. A band we cannot fill is worth")
print("  nothing no matter how good its edge looks.\n")
print(f"{'band':>10} {'ent/day':>8} {'mean fill':>10} {'edge/ct':>9} "
      f"{'$/day':>9} {'vs LIVE':>9}")
base = None
for lo, hi, lbl in BANDS:
    r = sim(lo, hi, use_fill=True)
    if not r:
        continue
    if lbl.startswith("8-14"):
        base = r["day"]
    print(f"{lbl:>10} {r['ent_day']:>8.1f} {r['fill']:>10.3f} "
          f"{r['edge']:>+9.2f} {r['day']:>+9.2f} "
          f"{(r['day']-base) if base else 0:>+9.2f}")

print("\n" + "=" * 78)
print("DAY-CLUSTERED TEST: best late band vs the live band")
print("=" * 78)
for cand in ("4-8", "4-6", "6-8", "6-14"):
    a = sim(8, 14, use_fill=True)
    lo, hi = (int(x) for x in cand.split("-"))
    b = sim(lo, hi, use_fill=True)
    if not a or not b:
        continue
    sa = W[(W.px >= 0.65) & (W.px < 0.85) & (W.vol >= 2000)
           & (W.minute >= 8) & (W.minute <= 14)]
    sb = W[(W.px >= 0.65) & (W.px < 0.85) & (W.vol >= 2000)
           & (W.minute >= lo) & (W.minute <= hi)]
    ea = (sa.sort_values("minute", ascending=False)
            .groupby(["wkey", "coin"], as_index=False).first())
    eb = (sb.sort_values("minute", ascending=False)
            .groupby(["wkey", "coin"], as_index=False).first())
    ea["pnl"] = ea.edge * QTY * ea.minute.map(FILL).fillna(0.5)
    eb["pnl"] = eb.edge * QTY * eb.minute.map(FILL).fillna(0.5)
    da = ea.groupby("day").pnl.sum().reindex(days, fill_value=0)
    db = eb.groupby("day").pnl.sum().reindex(days, fill_value=0)
    d = (db - da).values
    bs = np.sort(np.array([RNG.choice(d, len(d), True).mean()
                           for _ in range(8000)]))
    print(f"  {cand:>5} minus 8-14: ${d.mean():+7.2f}/day   "
          f"95% CI [{bs[200]:+7.2f}, {bs[7799]:+7.2f}]   "
          f"better {(db > da).sum()}/{len(d)} days")
