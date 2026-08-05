"""Is there a YES/NO asymmetry in the favourite we buy?

THE HYPOTHESIS, AND WHY IT IS SHARP
  Settlement is A1 >= A0 with TIES PAYING YES, so YES carries a structural
  tie-break. Separately, retail crypto flow is structurally bullish - people
  buy "it goes up".

  Put together those predict OPPOSITE things, which is what makes this worth
  testing rather than assuming:

    tie-break        -> holding YES is slightly better
    bullish retail   -> when the favourite is NO, the longshot is YES, and
                        bullish retail buying that YES longshot is exactly the
                        impatient flow we are paid to absorb. So holding NO
                        should be better.

  If holding NO wins, the mechanism is retail bullishness and it should show
  up as HIGHER longshot share in NO-favourite windows. That is a second,
  independent prediction the data can check - so this is not just a split.

DISCIPLINE
  price is controlled throughout (it drives win rate mechanically), entries are
  window-equal-weighted, CIs are day-clustered, and anything that survives gets
  a walk-forward test.
"""
import calendar
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("C:/Users/fycin/m15-claude-independent")
OUT = Path(__file__).resolve().parent
MON = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT",
     "NOV", "DEC"])}
RNG = np.random.default_rng(20260819)


def pts(ct):
    if not ct or len(ct) < 19:
        return None
    try:
        base = calendar.timegm(time.strptime(ct[:19], "%Y-%m-%dT%H:%M:%S"))
    except Exception:
        return None
    r = ct[19:].rstrip("Zz")
    if not r:
        return float(base)
    if not r.startswith(".") or not r[1:].isdigit():
        return None
    return base + int(r[1:]) / (10.0 ** len(r[1:]))


def cu(tk):
    p = tk.split("-")[1]
    return calendar.timegm((2000 + int(p[:2]), MON[p[2:5]], int(p[5:7]),
                            int(p[7:9]), int(p[9:11]), 0, 0, 0, 0)) + 4 * 3600


CACHE = OUT / "yesno.parquet"
if CACHE.exists():
    D = pd.read_parquet(CACHE)
    print(f"loaded cache: {len(D):,} rows")
else:
    sys.path.insert(0, str(ROOT))
    from claude_frame_ext import load_wide
    fr = load_wide(blocks=("A013", "E", "H"))
    fr = fr[fr.source != "A013"]
    fr = fr[np.isin(fr["y"].to_numpy(), (0, 1))]
    Y = dict(zip(fr.ticker.values, fr["y"].to_numpy().astype(int)))
    rows = []
    for coin in ["BTC", "ETH", "SOL", "XRP", "DOGE"]:
        for fp in sorted(glob.glob(str(ROOT / f"data/sare/trades/KX{coin}15M-*.json"))):
            tk = os.path.basename(fp)[:-5]
            y = Y.get(tk)
            if y is None:
                continue
            T = cu(tk)
            try:
                tr = json.loads(Path(fp).read_text())
            except Exception:
                continue
            seq = []
            for t in tr:
                ts = pts(t.get("created_time", ""))
                if ts is None:
                    continue
                s = T - ts
                if s <= 0 or s > 900:
                    continue
                try:
                    yp = float(t["yes_price_dollars"])
                    n = float(t["count_fp"])
                    sd = t.get("taker_side")
                except Exception:
                    continue
                if sd not in ("yes", "no") or n <= 0:
                    continue
                # maker price of the favourite; maker holds YES iff taker took NO
                mp = yp if sd == "no" else 1.0 - yp
                seq.append((s, mp, sd == "no", n,
                            yp if sd == "yes" else 1.0 - yp))
            if len(seq) < 10:
                continue
            seq.sort(key=lambda x: -x[0])
            cum_v = cum_l = 0.0
            seen = set()
            for (s, mp, my, n, tp) in seq:
                m = int(s // 60)
                if 2 <= m <= 14 and m not in seen and 0.50 <= mp < 0.95:
                    seen.add(m)
                    rows.append((coin, tk.split("-")[1],
                                 time.strftime("%Y-%m-%d", time.gmtime(T)),
                                 m, mp, cum_v, cum_l / max(cum_v, 1),
                                 bool(my),                    # maker holds YES
                                 1 if (my == bool(y)) else 0,
                                 (1 - mp) if (my == bool(y)) else -mp))
                cum_v += n
                if tp < 0.50:
                    cum_l += n
        print(f"  {coin}: {len(rows):,}", flush=True)
    D = pd.DataFrame(rows, columns=["coin", "wkey", "day", "minute", "px",
                                    "vol", "frac_long", "maker_yes", "won",
                                    "edge"])
    D.to_parquet(CACHE, index=False)

q = D[(D.px >= 0.65) & (D.px < 0.85) & (D.minute >= 8) & (D.minute <= 14)
      & (D.vol >= 2000)]
E = (q.sort_values("minute", ascending=False)
       .groupby(["wkey", "coin"], as_index=False).first()).copy()
E = (E.sort_values(["wkey", "minute"], ascending=[True, False])
       .groupby("wkey", as_index=False).head(3))
E["pb"] = (E.px * 20).astype(int)
days = sorted(E.day.unique())
print(f"\nentries {len(E):,}  windows {E.wkey.nunique():,}  days {len(days)}")
print(f"maker holds YES on {E.maker_yes.mean()*100:.1f}% of entries "
      f"(favourite is YES that often)\n")

print("=== THE SPLIT (inside the live filter) ===")
print(f"{'we hold':>9} {'n':>6} {'avg px':>8} {'win':>8} {'excess':>9} "
      f"{'edge/ct':>9} {'$/day @30':>10}")
for lbl, g in (("YES", E[E.maker_yes]), ("NO", E[~E.maker_yes])):
    print(f"{lbl:>9} {len(g):>6,} {g.px.mean()*100:>7.1f}c {g.won.mean():>8.4f} "
          f"{(g.won.mean()-g.px.mean())*100:>+8.2f}c {g.edge.mean()*100:>+8.2f}c "
          f"{g.edge.sum()*30/len(days):>+10.2f}")

a, b = E[~E.maker_yes], E[E.maker_yes]
ga = {d: g for d, g in a.groupby("day")}
gb = {d: g for d, g in b.groupby("day")}
bs = []
for _ in range(6000):
    pick = RNG.choice(len(days), len(days), True)
    sa = pd.concat([ga[days[k]] for k in pick if days[k] in ga])
    sb = pd.concat([gb[days[k]] for k in pick if days[k] in gb])
    if len(sa) > 20 and len(sb) > 20:
        bs.append((sb.edge.mean() - sa.edge.mean()) * 100)
bs = np.sort(np.array(bs))
d = (b.edge.mean() - a.edge.mean()) * 100
p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
print(f"\n  YES minus NO: {d:+.2f}c   95% CI "
      f"[{bs[int(.025*len(bs))]:+.2f}, {bs[int(.975*len(bs))]:+.2f}]   p {p:.4f}")

print("\n=== CONTROLLING FOR PRICE (the split must not just be a price effect) ===")
print(f"{'bucket':>8} {'n YES':>7} {'n NO':>6} {'edge YES':>10} {'edge NO':>9} "
      f"{'DIFF':>8}")
for pb, g in E.groupby("pb"):
    gy, gn = g[g.maker_yes], g[~g.maker_yes]
    if len(gy) < 20 or len(gn) < 20:
        continue
    print(f"{pb*5:>6}-{pb*5+5:<2} {len(gy):>7,} {len(gn):>6,} "
          f"{gy.edge.mean()*100:>+10.2f} {gn.edge.mean()*100:>+9.2f} "
          f"{(gy.edge.mean()-gn.edge.mean())*100:>+8.2f}")

print("\n=== THE MECHANISM CHECK (independent prediction) ===")
print("  If retail bullishness drives this, NO-favourite windows should carry")
print("  a HIGHER longshot share, because the longshot there is YES.\n")
for lbl, g in (("we hold YES (favourite=YES, longshot=NO)", E[E.maker_yes]),
               ("we hold NO  (favourite=NO,  longshot=YES)", E[~E.maker_yes])):
    print(f"  {lbl}  frac_long {g.frac_long.mean():.4f}  vol {g.vol.mean():,.0f}")

print("\n=== BY COIN ===")
print(f"{'coin':>6} {'n YES':>7} {'n NO':>6} {'edge YES':>10} {'edge NO':>9} "
      f"{'DIFF':>8}")
for c, g in E.groupby("coin"):
    gy, gn = g[g.maker_yes], g[~g.maker_yes]
    if len(gy) < 15 or len(gn) < 15:
        continue
    print(f"{c:>6} {len(gy):>7,} {len(gn):>6,} {gy.edge.mean()*100:>+10.2f} "
          f"{gn.edge.mean()*100:>+9.2f} {(gy.edge.mean()-gn.edge.mean())*100:>+8.2f}")

print("\n=== WALK-FORWARD ===")
CUT = days[len(days) // 2]
te = E[E.day >= CUT]
fit = E[E.day < CUT]
better_in_fit = "YES" if (fit[fit.maker_yes].edge.mean()
                          > fit[~fit.maker_yes].edge.mean()) else "NO"
sel = te[te.maker_yes] if better_in_fit == "YES" else te[~te.maker_yes]
nd = te.day.nunique()
print(f"  first half says hold {better_in_fit} is better")
print(f"  out of sample: all {te.edge.mean()*100:+.2f}c "
      f"(${te.edge.sum()*30/nd:+.2f}/day)  vs  {better_in_fit}-only "
      f"{sel.edge.mean()*100:+.2f}c (${sel.edge.sum()*30/nd:+.2f}/day)")
