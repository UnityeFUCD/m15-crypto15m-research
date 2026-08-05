"""Are the longshot buyers INFORMED? And can we see it before we commit?

THE QUESTION
  We are paid to absorb people buying the cheap side. If that flow is pure
  impatience, the payment is free money. If some of it comes from traders who
  know the favourite is about to flip, then that subset is adverse selection -
  and it is precisely our 15.4% losing tail, which costs -73c against +27c
  wins. Separating the two would cut drawdown at its source.

WHAT MAKES FLOW LOOK INFORMED (all measured BEFORE we enter)
  permanence of price impact  - informed trades move the price and it STAYS
                                moved; noise trades move it and it reverts.
                                This is the cleanest discriminator available
                                without counterparty identity.
  flow acceleration           - a burst right before we commit
  trade size and its tail     - one large ticket vs many small ones
  flip history                - how unstable the favourite has already been
  price drift / volatility    - the state the flow has already produced

DISCIPLINE
  - every feature uses ONLY trades strictly before the entry timestamp
  - price mechanically predicts the win rate, so every test is run INSIDE
    price buckets; an unconditional result here would be worthless
  - window-equal-weighted (contract weighting has faked three findings)
  - day-clustered bootstrap, and a fit/test split on days for anything that
    looks predictive
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
            # maker price of the favourite, whether maker held YES, taker's price
            mp = yp if sd == "no" else 1.0 - yp
            seq.append((s, mp, sd == "no", n, yp if sd == "yes" else 1.0 - yp))
        if len(seq) < 25:
            continue
        seq.sort(key=lambda x: -x[0])              # chronological
        # entry = first trade in the live band, exactly as the runner scans
        ent = None
        for i, (s, mp, my, n, tp) in enumerate(seq):
            if 480 <= s <= 840 and 0.55 <= mp < 0.90:
                ent = i
                break
        if ent is None or ent < 8:
            continue
        s0, px0, my0 = seq[ent][0], seq[ent][1], seq[ent][2]
        prior = seq[:ent]
        cumv = sum(x[3] for x in prior)
        # volume floor applied in ANALYSIS, not extraction, so the floor
        # itself can be treated as a control rather than baked in

        ls = [x for x in prior if x[4] < 0.50]     # taker bought the cheap side
        lsv = sum(x[3] for x in ls)
        # ---- flow timing ----
        rec = [x for x in prior if x[0] - s0 <= 120]      # last 120s before entry
        rec_ls = [x for x in rec if x[4] < 0.50]
        v60a = sum(x[3] for x in ls if x[0] - s0 <= 60)
        v60b = sum(x[3] for x in ls if 60 < x[0] - s0 <= 120)
        # ---- impact permanence: informed flow moves price and it STAYS ----
        perm, npm = 0.0, 0
        for (s, mp, my, n, tp) in ls:
            if s - s0 < 90 or n < 5:
                continue
            after = [z for z in prior if s - 15 <= z[0] < s]      # ~15s later
            later = [z for z in prior if s - 75 <= z[0] < s - 45]  # ~60s later
            if not after or not later:
                continue
            # signed toward the LONGSHOT buyer's direction (favourite falling)
            d_imm = mp - np.mean([z[1] for z in after])
            d_late = mp - np.mean([z[1] for z in later])
            if abs(d_imm) > 1e-9:
                perm += d_late / d_imm
                npm += 1
        # ---- favourite instability ----
        flips, last_flip = 0, None
        for j in range(1, len(prior)):
            if prior[j][2] != prior[j - 1][2]:
                flips += 1
                last_flip = prior[j][0]
        pxs = np.array([x[1] for x in prior[-40:]])
        p120 = [x[1] for x in prior if x[0] - s0 <= 120]
        rows.append(dict(
            coin=coin, wkey=tk.split("-")[1],
            day=time.strftime("%Y-%m-%d", time.gmtime(T)),
            hour=time.gmtime(T).tm_hour, px=px0, secs=s0, vol=cumv,
            won=1 if (my0 == bool(y)) else 0,
            edge=(1 - px0) if (my0 == bool(y)) else -px0,
            frac_long=lsv / max(cumv, 1),
            ls_recent=sum(x[3] for x in rec_ls) / max(sum(x[3] for x in rec), 1)
                      if rec else 0.0,
            ls_accel=v60a / max(v60b, 1.0),
            max_ls=max([x[3] for x in ls], default=0.0),
            max_ls_share=max([x[3] for x in ls], default=0.0) / max(lsv, 1),
            n_ls=len(ls), mean_ls=lsv / max(len(ls), 1),
            impact_perm=(perm / npm) if npm >= 3 else np.nan, n_perm=npm,
            flips=flips,
            secs_since_flip=(last_flip - s0) if last_flip else 900.0,
            px_vol=float(pxs.std()) if len(pxs) > 5 else np.nan,
            px_drift=(px0 - p120[0]) if p120 else 0.0,
            dist50=abs(px0 - 0.5)))
    print(f"  {coin}: {len(rows):,} entries", flush=True)

D = pd.DataFrame(rows)
D.to_parquet(OUT / "adverse.parquet", index=False)
print(f"\nentries {len(D):,}  windows {D.wkey.nunique():,}  days {D.day.nunique()}")
print(f"win {D.won.mean():.4f}  edge {D.edge.mean()*100:+.2f}c  "
      f"mean px {D.px.mean()*100:.1f}c")
print(f"impact_perm available on {D.impact_perm.notna().sum():,} entries")
