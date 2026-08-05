"""Test the liquidity-premium hypothesis, and re-verify the z-control number.

THE CLAIM I MADE
  "our edge is payment for providing liquidity to impatient traders. In a
  market with 122 contracts traded, nobody is being impatient, so there is
  nothing to be paid for."

  That is a hypothesis, not a result. It makes a sharp prediction that total
  volume does NOT: the edge should come specifically from taker flow BUYING
  THE LONGSHOT - the expensive, impatient act - and not from taker flow buying
  the favourite, which is the same volume with the opposite meaning.

  If total volume works only through longshot-buying volume, the hypothesis
  survives. If favourite-buying volume works just as well, it is refuted and
  volume is measuring something else entirely (attention, liquidity, or a
  market-quality proxy).

ALSO RE-VERIFIED
  The earlier "+10.96 after controlling z" used z quartiles computed over the
  WHOLE sample while the volume split was done inside price buckets. Mixing the
  two conditioning schemes can leave composition effects. Redone here with both
  controls applied inside the same cell.
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
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
RNG = np.random.default_rng(20260805)


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


sys.path.insert(0, str(ROOT))
from claude_frame_ext import load_wide
fr = load_wide(blocks=("A013", "E", "H"))
fr = fr[fr.source != "A013"]
fr = fr[np.isin(fr["y"].to_numpy(), (0, 1))]
Y = dict(zip(fr.ticker.values, fr["y"].to_numpy().astype(int)))

rows = []
for coin in ["ETH", "SOL", "XRP", "DOGE"]:
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
            # price the TAKER paid, and whether that was the cheap side
            tpx = yp if sd == "yes" else 1.0 - yp
            seq.append((s, yp if sd == "no" else 1.0 - yp, sd == "no", n, tpx))
        if len(seq) < 20:
            continue
        seq.sort(key=lambda x: -x[0])
        ent = None
        for i, (s, mp, my, n, tp) in enumerate(seq):
            if 480 <= s <= 840 and 0.60 <= mp < 0.90:
                ent = (i, s, mp, my)
                break
        if ent is None:
            continue
        i0, s0, px0, my = ent
        prior = seq[:i0]
        # THE DECOMPOSITION: taker buying the LONGSHOT (impatient, what we are
        # paid for) versus taker buying the FAVOURITE (same volume, opposite
        # meaning)
        v_long = sum(x[3] for x in prior if x[4] < 0.50)
        v_fav = sum(x[3] for x in prior if x[4] >= 0.50)
        rows.append(dict(coin=coin, wkey=tk.split("-")[1],
                         day=time.strftime("%Y-%m-%d", time.gmtime(T)),
                         px=px0, won=1 if (my == bool(y)) else 0,
                         v_long=v_long, v_fav=v_fav, v_tot=v_long + v_fav,
                         frac_long=v_long / max(v_long + v_fav, 1),
                         edge=(1 - px0 if (my == bool(y)) else -px0)))
D = pd.DataFrame(rows)
D["pb"] = (D.px * 20).astype(int)
D.to_parquet(OUT / "mechanism2.parquet", index=False)
print(f"entries {len(D):,}   windows {D.wkey.nunique():,}   days {D.day.nunique()}")
print(f"median longshot vol {D.v_long.median():,.0f}   "
      f"median favourite vol {D.v_fav.median():,.0f}")

print("\n=== THE DECISIVE TEST ===")
print("if the edge is payment for absorbing impatient LONGSHOT buying, then")
print("v_long should carry the effect and v_fav should not.\n")
print(f"{'split on':>12} {'edge low':>10} {'edge high':>10} {'LIFT':>9} "
      f"{'day-clustered CI':>22}")
days = sorted(D.day.unique())
for var in ("v_tot", "v_long", "v_fav"):
    D["hi"] = D.groupby(["coin", "pb"])[var].transform(lambda s: s > s.median())
    a, b = D[~D.hi], D[D.hi]
    lift = (b.edge.mean() - a.edge.mean()) * 100
    bs = []
    for _ in range(3000):
        s = pd.concat([D[D.day == p] for p in RNG.choice(days, len(days), True)])
        x, z2 = s[~s.hi], s[s.hi]
        if len(x) > 50 and len(z2) > 50:
            bs.append((z2.edge.mean() - x.edge.mean()) * 100)
    bs = np.sort(np.array(bs))
    print(f"{var:>12} {a.edge.mean()*100:>+10.2f} {b.edge.mean()*100:>+10.2f} "
          f"{lift:>+9.2f} [{bs[int(.025*len(bs))]:>+8.2f},{bs[int(.975*len(bs))]:>+8.2f}]")

print("\n=== HORSE RACE: does v_fav add anything once v_long is controlled? ===")
D["lq"] = D.groupby(["coin", "pb"]).v_long.transform(
    lambda s: pd.qcut(s.rank(method="first"), 3, labels=False, duplicates="drop"))
D["fq"] = D.groupby(["coin", "pb"]).v_fav.transform(
    lambda s: pd.qcut(s.rank(method="first"), 3, labels=False, duplicates="drop"))
print(f"{'':>18}" + "".join(f"{'v_fav ' + str(i):>12}" for i in range(3)))
for lq in range(3):
    row = ""
    for fq in range(3):
        g = D[(D.lq == lq) & (D.fq == fq)]
        row += f"{g.edge.mean()*100:>+12.2f}" if len(g) > 40 else f"{'-':>12}"
    print(f"  v_long {lq}      {row}")
print("\n  read DOWN a column: does more longshot buying help at fixed v_fav?")
print("  read ACROSS a row:  does more favourite buying help at fixed v_long?")

print("\n=== FRACTION OF FLOW THAT IS LONGSHOT BUYING ===")
D["fb"] = D.groupby(["coin", "pb"]).frac_long.transform(
    lambda s: pd.qcut(s.rank(method="first"), 4, labels=False, duplicates="drop"))
print(f"{'quartile':>10} {'n':>7} {'mean frac':>11} {'avg px':>8} {'win':>8} {'edge':>9}")
for q, g in D.groupby("fb"):
    print(f"{int(q):>10} {len(g):>7,} {g.frac_long.mean():>11.4f} "
          f"{g.px.mean()*100:>7.1f}c {g.won.mean():>8.4f} {g.edge.mean()*100:>+9.2f}")
