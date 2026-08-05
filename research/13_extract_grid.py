"""Extract a per-minute entry grid so every knob can be optimised on one dataset.

For each (window, coin) and each whole minute of time-remaining, record the
FIRST trade in that minute along with the state a live runner would have seen
at that instant: maker price, cumulative volume, longshot-share of that volume,
and the eventual settlement.

That single table supports slicing by price band, time window, volume
threshold, per-window cap and coin without re-reading 4,490 trade files each
time - and every field is causal, computed only from what had already happened
when the trade printed.
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
print(f"settlements {len(Y):,}")

rows = []
for coin in ["BTC", "ETH", "SOL", "XRP", "DOGE"]:
    files = sorted(glob.glob(str(ROOT / f"data/sare/trades/KX{coin}15M-*.json")))
    for fp in files:
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
            tpx = yp if sd == "yes" else 1.0 - yp      # price the taker paid
            seq.append((s, yp if sd == "no" else 1.0 - yp, sd == "no", n, tpx))
        if len(seq) < 10:
            continue
        seq.sort(key=lambda x: -x[0])                   # chronological
        cum_v = 0.0
        cum_l = 0.0
        seen_min = set()
        for (s, mp, my, n, tp) in seq:
            m = int(s // 60)                            # whole minutes remaining
            if 2 <= m <= 14 and m not in seen_min and 0.50 <= mp < 0.95:
                seen_min.add(m)
                rows.append((coin, tk.split("-")[1], tk,
                             time.strftime("%Y-%m-%d", time.gmtime(T)),
                             time.gmtime(T).tm_hour, m, s, mp,
                             cum_v, cum_l / max(cum_v, 1),
                             1 if (my == bool(y)) else 0,
                             (1 - mp) if (my == bool(y)) else -mp))
            cum_v += n
            if tp < 0.50:
                cum_l += n
    print(f"  {coin}: {len(rows):,} rows", flush=True)

D = pd.DataFrame(rows, columns=["coin", "wkey", "ticker", "day", "hour",
                                "minute", "secs", "px", "vol", "frac_long",
                                "won", "edge"])
D.to_parquet(OUT / "grid.parquet", index=False)
print(f"\nrows {len(D):,}  windows {D.wkey.nunique():,}  days {D.day.nunique()}  "
      f"coins {D.coin.nunique()}")
print(f"minutes covered {sorted(D.minute.unique())}")
print(f"price range {D.px.min()*100:.0f}-{D.px.max()*100:.0f}c   "
      f"median vol {D.vol.median():,.0f}")
print(f"\nrows per coin: {D.coin.value_counts().to_dict()}")
