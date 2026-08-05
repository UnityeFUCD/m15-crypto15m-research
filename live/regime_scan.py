"""Does the longshot premium have a REGIME structure we can act on?

WHY THIS MATTERS NOW
  The premium fell from +10.40c (Jul 23 - Aug 1) to +2.32c (Aug 2-5), verified
  as a real level shift rather than a measurement artifact. That raises the
  question of whether such shifts are predictable at all.

THE DECISION THIS DRIVES - and it hinges on ONE property
  PERSISTENCE. If today's premium predicts tomorrow's, a regime is a state we
  can detect and adapt to: trade bigger in good regimes, stand down in bad
  ones. If the premium is serially uncorrelated, then every adaptive rule is
  chasing noise - it will cut size after unlucky days and add it after lucky
  ones, which is strictly worse than doing nothing.

  So autocorrelation is tested FIRST, because a null there kills every
  adaptive scheme regardless of what the hour-of-day table looks like.

MULTIPLE COMPARISONS
  Testing 24 hours guarantees roughly one "significant" hour at p<0.05 by
  chance. So the hour analysis is judged by whether the PATTERN survives a
  split-half test, not by whether any single hour has a small p-value.

Data: local backtest files (Jul 23 - Aug 1) plus API replay of everything
since, using the replay rule that was verified to agree with the grid method
to within 0.89c.
"""
import calendar
import glob
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("C:/Users/fycin/m15-claude-independent")
OUT = Path(__file__).resolve().parent
CACHE = OUT / "regime_data.parquet"
MON = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT",
     "NOV", "DEC"])}
RNG = np.random.default_rng(20260903)


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


def scan(seq, T):
    """The verified replay rule: first qualifying print."""
    seq.sort(key=lambda x: -x[0])
    cum = 0.0
    for (s, mp, my, n) in seq:
        if 480 <= s <= 840 and 0.65 <= mp < 0.85 and cum >= 2000:
            return (s, mp, my)
        cum += n
    return None


if CACHE.exists():
    D = pd.read_parquet(CACHE)
    print(f"loaded cache: {len(D):,} entries")
else:
    rows = []
    # --- local history ---
    sys.path.insert(0, str(ROOT))
    from claude_frame_ext import load_wide
    fr = load_wide(blocks=("A013", "E", "H"))
    fr = fr[fr.source != "A013"]
    fr = fr[np.isin(fr["y"].to_numpy(), (0, 1))]
    Y = dict(zip(fr.ticker.values, fr["y"].to_numpy().astype(int)))
    for coin in ["DOGE", "SOL", "XRP", "ETH"]:
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
                seq.append((s, yp if sd == "no" else 1.0 - yp, sd == "no", n))
            if len(seq) < 10:
                continue
            r = scan(seq, T)
            if not r:
                continue
            _, mp, my = r
            won = 1 if (my == bool(y)) else 0
            g = time.gmtime(T)
            rows.append((coin, time.strftime("%Y-%m-%d", g), g.tm_hour,
                         g.tm_wday, mp, won, (1 - mp) if won else -mp, "hist"))
        print(f"  local {coin}: {len(rows)}", flush=True)

    # --- recent, via API ---
    from acct import api

    def tsz(s):
        try:
            return datetime.fromisoformat((s or "").replace("Z", "+00:00"))
        except Exception:
            return None

    for ser in ["KXDOGE15M", "KXSOL15M", "KXXRP15M", "KXETH15M"]:
        mk, cur = [], None
        while len(mk) < 400:
            p = {"series_ticker": ser, "status": "settled", "limit": 100}
            if cur:
                p["cursor"] = cur
            c, j = api("/markets", p)
            if c != 200 or not j:
                break
            b = j.get("markets") or []
            mk += b
            cur = j.get("cursor")
            if not cur or not b:
                break
        n0 = len(rows)
        for m in mk[:400]:
            if m.get("result") not in ("yes", "no"):
                continue
            y = 1 if m["result"] == "yes" else 0
            close = tsz(m.get("close_time"))
            if not close:
                continue
            out, cur2 = [], None
            for _ in range(4):
                p = {"ticker": m.get("ticker"), "limit": 1000}
                if cur2:
                    p["cursor"] = cur2
                c, j = api("/markets/trades", p)
                if c != 200 or not j:
                    break
                bb = j.get("trades") or []
                out += bb
                cur2 = j.get("cursor")
                if not cur2 or not bb:
                    break
            seq = []
            for t in out:
                tt = tsz(t.get("created_time"))
                if not tt:
                    continue
                s = (close - tt).total_seconds()
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
                seq.append((s, yp if sd == "no" else 1.0 - yp, sd == "no", n))
            if len(seq) < 10:
                continue
            r = scan(seq, None)
            if not r:
                continue
            _, mp, my = r
            won = 1 if (my == bool(y)) else 0
            rows.append((ser.replace("KX", "").replace("15M", ""),
                         close.strftime("%Y-%m-%d"), close.hour,
                         close.weekday(), mp, won,
                         (1 - mp) if won else -mp, "api"))
        print(f"  api {ser}: {len(rows)-n0}", flush=True)

    D = pd.DataFrame(rows, columns=["coin", "day", "hour", "wday", "px",
                                    "won", "edge", "src"])
    D = D.drop_duplicates(subset=["coin", "day", "hour", "px"])
    D.to_parquet(CACHE, index=False)

days = sorted(D.day.unique())
print(f"\nentries {len(D):,}   days {len(days)} "
      f"({days[0]} .. {days[-1]})   coins {sorted(D.coin.unique())}\n")

print("=== DAILY PREMIUM ===")
dm = D.groupby("day").edge.agg(["mean", "size"])
for d, r in dm.iterrows():
    bar = "#" * max(0, int(r["mean"] * 100 / 1.5)) if r["mean"] > 0 else ""
    print(f"  {d}  n {int(r['size']):>4}  {r['mean']*100:>+7.2f}c  {bar}")

print("\n=== TEST 1: IS THE PREMIUM PERSISTENT? (this decides everything) ===")
v = dm["mean"].values * 100
if len(v) >= 6:
    r1 = np.corrcoef(v[:-1], v[1:])[0, 1]
    bs = []
    for _ in range(6000):
        p = RNG.permutation(v)
        bs.append(np.corrcoef(p[:-1], p[1:])[0, 1])
    bs = np.sort(np.array(bs))
    pv = (np.abs(bs) >= abs(r1)).mean()
    print(f"  lag-1 autocorrelation of the daily premium: {r1:+.4f}")
    print(f"  permutation p = {pv:.4f}   (null: days are exchangeable)")
    print(f"  day-to-day sd of the premium: {v.std():.2f}c")
    if pv < 0.05 and r1 > 0:
        print("  -> PERSISTENT. Yesterday informs today; adaptation is viable.")
    else:
        print("  -> NOT persistent. Yesterday does not predict today, so any")
        print("     adaptive rule would be chasing noise and would cut size")
        print("     after unlucky days - strictly worse than holding steady.")

print("\n=== TEST 2: HOUR OF DAY (UTC) ===")
print(f"{'hour':>6} {'n':>6} {'edge':>9} {'win':>8} {'avg px':>8}")
hh = D.groupby("hour").agg(n=("edge", "size"), e=("edge", "mean"),
                           w=("won", "mean"), p=("px", "mean"))
for h, r in hh.iterrows():
    if r["n"] < 15:
        continue
    print(f"{int(h):>6} {int(r['n']):>6} {r['e']*100:>+9.2f} {r['w']:>8.4f} "
          f"{r['p']*100:>7.1f}c")

print("\n  split-half check: does the hour pattern repeat on held-out days?")
half = len(days) // 2
A = D[D.day.isin(days[:half])].groupby("hour").edge.mean() * 100
B = D[D.day.isin(days[half:])].groupby("hour").edge.mean() * 100
J = pd.concat([A, B], axis=1, keys=["first", "second"]).dropna()
if len(J) >= 6:
    c = J["first"].corr(J["second"])
    print(f"  corr(first-half hour means, second-half hour means) = {c:+.4f}")
    print("  near zero = the hour pattern is noise, not a regime")

print("\n=== TEST 3: DAY OF WEEK ===")
WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
wd = D.groupby("wday").agg(n=("edge", "size"), e=("edge", "mean"))
for w, r in wd.iterrows():
    if r["n"] < 20:
        continue
    print(f"  {WD[int(w)]:>4} n {int(r['n']):>5}  {r['e']*100:>+7.2f}c")

print("\n=== TEST 4: DOES ANY SAME-DAY OBSERVABLE TRACK THE PREMIUM? ===")
agg = D.groupby("day").agg(edge=("edge", "mean"), px=("px", "mean"),
                           won=("won", "mean"), n=("edge", "size"))
for col in ("px", "n"):
    if len(agg) >= 6:
        c = np.corrcoef(agg[col], agg["edge"])[0, 1]
        print(f"  corr(daily {col:>3}, daily premium) = {c:+.4f}")
print("\n  A strong correlation with a SAME-DAY observable is only useful if")
print("  it is available before we trade - otherwise it is hindsight.")
