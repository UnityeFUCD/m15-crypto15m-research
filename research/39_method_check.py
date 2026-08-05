"""Is my API replay method biased, or did the market level-shift?

WHY THIS MUST BE SETTLED
  The API replay says the recent premium is +2.32c. The backtest says +11.29c
  on what is supposed to be the same measure - maker price of the favourite,
  8-14 minutes out, 65-85c, volume >= 2000, scored against settlement. A 5x
  gap. Two possibilities with very different consequences:

    method bias   -> every per-coin number from the replay is wrong, including
                     the +1.01c that got BNB removed from live trading.
    level shift   -> the method is sound and the market genuinely paid less
                     after Aug 1, which makes the backtest stale as a guide.

  Guessing between them is not acceptable when one branch means I removed a
  coin on bad evidence.

THE TEST
  Run the REPLAY logic - trade-by-trade scan, first qualifying print - against
  the LOCAL backtest files, i.e. the exact period the backtest covers. If it
  reproduces ~+11c, the method is sound and the recent drop is real. If it
  returns ~+2c on the backtest period too, the method is biased and the
  backtest/replay difference is an artifact of how entries are selected.
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
for coin in ["DOGE", "SOL", "XRP", "ETH"]:
    n0 = len(rows)
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
            mp = yp if sd == "no" else 1.0 - yp
            seq.append((s, mp, sd == "no", n))
        if len(seq) < 10:
            continue
        seq.sort(key=lambda x: -x[0])
        cum = 0.0
        # EXACTLY the replay rule: first qualifying PRINT, not first of a minute
        for (s, mp, my, n) in seq:
            if 480 <= s <= 840 and 0.65 <= mp < 0.85 and cum >= 2000:
                won = 1 if (my == bool(y)) else 0
                rows.append((coin, time.strftime("%Y-%m-%d", time.gmtime(T)),
                             mp, won, (1 - mp) if won else -mp))
                break
            cum += n
    print(f"  {coin}: {len(rows)-n0} entries", flush=True)

R = pd.DataFrame(rows, columns=["coin", "day", "px", "won", "edge"])
print(f"\nREPLAY LOGIC on the BACKTEST PERIOD: {len(R):,} entries, "
      f"{R.day.nunique()} days")
print(f"  win {R.won.mean():.4f}   avg px {R.px.mean()*100:.1f}c   "
      f"edge {R.edge.mean()*100:+.2f}c")

print("\n=== SIDE BY SIDE ===")
G = pd.read_parquet(OUT / "grid.parquet")
q = G[(G.px >= 0.65) & (G.px < 0.85) & (G.minute >= 8) & (G.minute <= 14)
      & (G.vol >= 2000)]
E = (q.sort_values("minute", ascending=False)
       .groupby(["wkey", "coin"], as_index=False).first())
E = (E.sort_values(["wkey", "minute"], ascending=[True, False])
       .groupby("wkey", as_index=False).head(3))
print(f"  grid/backtest method, backtest period : {E.edge.mean()*100:+.2f}c "
      f"(n {len(E):,})")
print(f"  REPLAY method,        backtest period : {R.edge.mean()*100:+.2f}c "
      f"(n {len(R):,})")
print(f"  REPLAY method,        recent period   : +2.32c  (measured earlier)")

d = R.edge.mean() * 100 - E.edge.mean() * 100
print(f"\n  method difference on the SAME period: {d:+.2f}c")
if abs(d) < 2.5:
    print("  -> methods AGREE. The recent +2.32c is a genuine level shift,")
    print("     and the backtest is stale as a guide to today's market.")
else:
    print("  -> methods DISAGREE. The replay understates edge, so every")
    print("     per-coin replay number is biased low - including the +1.01c")
    print("     that removed BNB. That decision needs revisiting.")

print("\n=== PER COIN, replay logic on the backtest period ===")
for c, g in R.groupby("coin"):
    print(f"  {c:>5}: n {len(g):>4}  win {g.won.mean():.4f}  "
          f"edge {g.edge.mean()*100:>+7.2f}c")
