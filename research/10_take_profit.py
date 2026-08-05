"""Should we take profit early instead of holding to settlement?

THE ARGUMENT FOR
  We buy at ~67c. If the price runs to 85c we could sell for +18c certain,
  instead of holding a coin-flip-with-odds to a 0/100 settlement. That would
  cut variance enormously - and variance, not edge, is what makes this feel
  inconsistent.

  It also frees CAPITAL, which is our actual binding constraint: the drawdown
  envelope blocked 161 orders tonight. A dollar returned early can be redeployed
  into a fresh cheap contract.

THE ARGUMENT AGAINST
  Our entire thesis is that favourites are UNDERPRICED. Window-weighted, every
  band from 60c to 90c shows a positive edge. If a contract at 85c is really
  worth ~92c, selling at 85c hands 7c to the buyer. Worse, selling means
  CROSSING - we become the taker, pay the spread and the taker fee, on a
  strategy whose whole edge comes from never crossing.

  And it is the classic "cut winners, ride losers" shape: we would exit the
  positions that are going well and keep every position that is going badly.

MEASUREMENT
  For each entry at 65-70c, walk the subsequent trade prices in that market and
  find whether it ever touches a target. Compare hold-to-settlement against
  exit-at-target, on identical positions. Exit is priced at the BID (we cross)
  with the exact taker fee.
"""
import calendar
import glob
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("C:/Users/fycin/m15-claude-independent")
OUT = Path(__file__).resolve().parent
W = 900
MON = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
RNG = np.random.default_rng(20260805)
fee = lambda p: math.ceil(0.07 * p * (1 - p) * 1e4 - 1e-12) / 1e4


def parse_ts(ct):
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


def close_utc(tk):
    p = tk.split("-")[1]
    return calendar.timegm((2000 + int(p[:2]), MON[p[2:5]], int(p[5:7]),
                            int(p[7:9]), int(p[9:11]), 0, 0, 0, 0)) + 4 * 3600


# settlement from the validated frame
import sys
sys.path.insert(0, str(ROOT))
from claude_frame_ext import load_wide
fr = load_wide(blocks=("A013", "E", "H"))
fr = fr[fr.source != "A013"]
fr = fr[np.isin(fr["y"].to_numpy(), (0, 1))]
Y = dict(zip(fr.ticker.values, fr["y"].to_numpy().astype(int)))
print(f"settlements available: {len(Y):,}")

TARGETS = [0.78, 0.82, 0.86, 0.90, 0.94]
rows = []
for coin in ["ETH", "SOL", "XRP", "DOGE"]:
    files = sorted(glob.glob(str(ROOT / f"data/sare/trades/KX{coin}15M-*.json")))
    for fi, fp in enumerate(files):
        tk = os.path.basename(fp)[:-5]
        y = Y.get(tk)
        if y is None:
            continue
        T = close_utc(tk)
        try:
            tr = json.loads(Path(fp).read_text())
        except Exception:
            continue
        seq = []
        for t in tr:
            ts = parse_ts(t.get("created_time", ""))
            if ts is None:
                continue
            s = T - ts
            if s <= 0 or s > W:
                continue
            try:
                yp = float(t["yes_price_dollars"])
                side = t.get("taker_side")
            except Exception:
                continue
            if side not in ("yes", "no"):
                continue
            seq.append((s, yp if side == "no" else 1.0 - yp, side == "no"))
        if len(seq) < 20:
            continue
        seq.sort(key=lambda x: -x[0])          # chronological
        # ENTRY: first trade in 8-14 min with maker price 65-70c
        ent = None
        for i, (s, mp, myes) in enumerate(seq):
            if 480 <= s <= 840 and 0.65 <= mp < 0.70:
                ent = (i, s, mp, myes)
                break
        if ent is None:
            continue
        i0, s0, px0, myes = ent
        won = 1 if (myes == bool(y)) else 0
        # subsequent prices ON OUR SIDE
        after = [(s, (mp if mo == myes else 1.0 - mp))
                 for s, mp, mo in seq[i0 + 1:]]
        peak = max((p for _, p in after), default=px0)
        rec = dict(coin=coin, ticker=tk, day=time.strftime("%Y-%m-%d", time.gmtime(T)),
                   px=px0, won=won, peak=peak,
                   hold=(1.0 - px0) if won else -px0)
        for tg in TARGETS:
            hit = next((p for _, p in after if p >= tg), None)
            if hit is None:
                rec[f"e{int(tg*100)}"] = rec["hold"]        # never touched: hold
            else:
                # we CROSS to exit: sell into the bid, roughly one tick under,
                # and pay the taker fee
                exit_px = hit - 0.01
                rec[f"e{int(tg*100)}"] = (exit_px - px0) - fee(exit_px)
        rows.append(rec)
    print(f"  {coin}: {len(rows):,} entries so far", flush=True)

D = pd.DataFrame(rows)
D.to_parquet(OUT / "take_profit.parquet", index=False)
print(f"\nentries {len(D):,}   windows {D.ticker.nunique():,}   days {D.day.nunique()}")
print(f"entry price {D.px.mean()*100:.1f}c   win rate {D.won.mean():.4f}   "
      f"hold P&L {D.hold.mean()*100:+.2f}c")

print(f"\n=== HOW OFTEN DOES THE PRICE REACH EACH TARGET? ===")
for tg in TARGETS:
    print(f"  reaches {tg*100:.0f}c : {(D.peak >= tg).mean():.4f} of entries")

print(f"\n=== HOLD vs TAKE-PROFIT (per contract, and variance) ===")
print(f"{'policy':>14} {'mean c':>9} {'sd c':>8} {'Sharpe':>8} {'worst':>9} "
      f"{'day-clustered CI':>22}")


def dayci(v, days, B=6000):
    df = pd.DataFrame({"v": v, "d": days})
    g = df.groupby("d").v.mean().values
    idx = RNG.integers(0, len(g), size=(B, len(g)))
    b = np.sort(g[idx].mean(1))
    return g.mean(), b[int(.025 * B)], b[int(.975 * B)]


for lab, col in [("HOLD", "hold")] + [(f"exit at {int(t*100)}c", f"e{int(t*100)}")
                                      for t in TARGETS]:
    v = D[col].values * 100
    m, lo, hi = dayci(v, D.day.values)
    sd = v.std(ddof=1)
    print(f"{lab:>14} {m:>+9.2f} {sd:>8.2f} {m/sd:>8.4f} {v.min():>+9.2f} "
          f"[{lo:>+8.2f},{hi:>+8.2f}]")

print("\n=== CAPITAL VIEW: is an early exit worth it if we redeploy? ===")
print("edge per DOLLAR is what matters when capital is the constraint.\n")
hold_pd = D.hold.mean() / D.px.mean()
print(f"  hold to settlement : {D.hold.mean()*100:+.2f}c on {D.px.mean()*100:.1f}c "
      f"= {hold_pd:.4f} per $1")
for tg in TARGETS:
    c = f"e{int(tg*100)}"
    print(f"  exit at {tg*100:.0f}c        : {D[c].mean()*100:+.2f}c on "
          f"{D.px.mean()*100:.1f}c = {D[c].mean()/D.px.mean():.4f} per $1")
