"""Is the wipeout structure DIFFERENT after the regime shift?

THE GAP THIS CLOSES
  Every correlated-loss test so far ran on grid.parquet - Jul 23 to Aug 1. But
  the longshot premium level-shifted after Aug 1 (+10.40c -> +2.32c on the
  population, verified with the replay method reproducing the grid method to
  within 0.89c). If the market changed, the correlation structure may have
  changed with it, and every "no pattern" conclusion would only apply to the
  OLD regime.

  That is a real hole, and it is the most defensible remaining reason to keep
  looking rather than accept the null.

WHAT IS REBUILT HERE
  A post-shift window dataset from the exchange directly, with the same fields
  the pre-shift tests used, so the two periods can be compared like for like:
  wipeout rate, correlation ratio, direction alignment, and whether any of the
  entry-time features separate wipeouts NOW even though they did not THEN.
"""
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Users/fycin/m15-claude-independent/live")
from acct import api

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261008)
CACHE = OUT / "postshift_windows.parquet"
SERIES = ["KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M", "KXDOGE15M",
          "KXHYPE15M"]


def ts(s):
    try:
        return datetime.fromisoformat((s or "").replace("Z", "+00:00"))
    except Exception:
        return None


if CACHE.exists():
    D = pd.read_parquet(CACHE)
    print(f"loaded cache: {len(D):,} entries")
else:
    rows = []
    for ser in SERIES:
        mk, cur = [], None
        while len(mk) < 220:
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
        for m in mk[:220]:
            if m.get("result") not in ("yes", "no"):
                continue
            y = 1 if m["result"] == "yes" else 0
            close = ts(m.get("close_time"))
            if not close:
                continue
            out, c2 = [], None
            for _ in range(3):
                p = {"ticker": m.get("ticker"), "limit": 1000}
                if c2:
                    p["cursor"] = c2
                c, j = api("/markets/trades", p)
                if c != 200 or not j:
                    break
                bb = j.get("trades") or []
                out += bb
                c2 = j.get("cursor")
                if not c2 or not bb:
                    break
            seq = []
            for t in out:
                tt = ts(t.get("created_time"))
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
                mp = yp if sd == "no" else 1.0 - yp
                tp = yp if sd == "yes" else 1.0 - yp
                seq.append((s, mp, sd == "no", n, tp))
            if len(seq) < 10:
                continue
            seq.sort(key=lambda x: -x[0])
            cum = lon = 0.0
            hit = None
            for (s, mp, my, n, tp) in seq:
                if 480 <= s <= 840 and 0.65 <= mp < 0.85 and cum >= 2000:
                    hit = (s, mp, my, cum, lon / max(cum, 1))
                    break
                cum += n
                if tp < 0.50:
                    lon += n
            if not hit:
                continue
            s, mp, my, cv, fl = hit
            rows.append(dict(
                coin=ser.replace("KX", "").replace("15M", ""),
                wkey=m["ticker"].split("-")[1], ticker=m["ticker"],
                close=close, day=close.strftime("%Y-%m-%d"), hour=close.hour,
                px=mp, maker_yes=my, vol=cv, frac_long=fl,
                won=1 if (my == bool(y)) else 0,
                edge=(1 - mp) if (my == bool(y)) else -mp))
        print(f"  {ser}: {len(rows)-n0} entries", flush=True)
    D = pd.DataFrame(rows)
    D.to_parquet(CACHE, index=False)

D["day"] = pd.to_datetime(D.close).dt.strftime("%Y-%m-%d")
D["post"] = (D.day >= "2026-08-02").astype(int)
print(f"\nentries {len(D):,}   days {D.day.nunique()}   "
      f"pre-shift {int((D.post==0).sum())}  post-shift {int((D.post==1).sum())}")


def wstats(sub, label):
    """Window-level correlation statistics for one period."""
    sub = sub.sort_values(["wkey", "px"], ascending=[True, False])
    S = sub.groupby("wkey", as_index=False).head(3)
    W = S.groupby("wkey").agg(n=("won", "size"),
                              k=("won", "sum"),
                              nyes=("maker_yes", "sum"),
                              edge=("edge", "mean")).reset_index()
    M = W[W.n >= 2]
    if len(M) < 20:
        print(f"  {label}: only {len(M)} multi-position windows - skipped")
        return None
    p = S.won.mean()
    obs2 = ((M.n == 2) & (M.k == 0)).sum() / max((M.n == 2).sum(), 1)
    obs3 = ((M.n == 3) & (M.k == 0)).sum() / max((M.n == 3).sum(), 1)
    ind2, ind3 = (1 - p) ** 2, (1 - p) ** 3
    al = ((M.nyes == M.n) | (M.nyes == 0)).mean()
    wipe = (M.k == 0).mean()
    print(f"\n  {label}")
    print(f"    entries {len(S):,}   multi-position windows {len(M)}")
    print(f"    per-entry win rate      {p:.4f}")
    print(f"    edge/contract           {S.edge.mean()*100:+.2f}c")
    print(f"    wipeout rate            {wipe:.4f}  ({int((M.k==0).sum())} events)")
    print(f"    P(both lose | n=2)      {obs2:.4f}   indep {ind2:.4f}   "
          f"ratio {obs2/ind2 if ind2 else float('nan'):.2f}x")
    print(f"    P(all lose  | n=3)      {obs3:.4f}   indep {ind3:.4f}   "
          f"ratio {obs3/ind3 if ind3 else float('nan'):.2f}x")
    print(f"    direction-aligned share {al:.4f}")
    return M


print("\n" + "=" * 74)
print("HAS THE CORRELATION STRUCTURE CHANGED ACROSS THE REGIME SHIFT?")
print("=" * 74)
pre = wstats(D[D.post == 0], "PRE-SHIFT  (to Aug 1)")
post = wstats(D[D.post == 1], "POST-SHIFT (Aug 2 on)")

if post is not None and len(post) >= 30:
    print("\n" + "=" * 74)
    print("DO ANY FEATURES SEPARATE WIPEOUTS *NOW*? (post-shift only)")
    print("=" * 74)
    S = (D[D.post == 1].sort_values(["wkey", "px"], ascending=[True, False])
         .groupby("wkey", as_index=False).head(3))
    W = S.groupby("wkey").agg(n=("won", "size"), k=("won", "sum"),
                              px=("px", "mean"), vol=("vol", "mean"),
                              fl=("frac_long", "mean"),
                              nyes=("maker_yes", "sum"),
                              hour=("hour", "first"),
                              day=("day", "first")).reset_index()
    W = W[W.n >= 2].copy()
    W["wipe"] = (W.k == 0).astype(int)
    W["aligned"] = ((W.nyes == W.n) | (W.nyes == 0)).astype(int)
    print(f"  windows {len(W)}   wipeouts {int(W.wipe.sum())} "
          f"({W.wipe.mean():.4f})\n")
    print(f"{'feature':>10} {'low half':>10} {'high half':>10} {'ratio':>8} "
          f"{'p (day-clustered)':>19}")
    for f in ("px", "vol", "fl", "hour", "aligned"):
        d = W.dropna(subset=[f])
        if d[f].nunique() < 3:
            a = d[d[f] == 0].wipe.mean() if (d[f] == 0).any() else np.nan
            b = d[d[f] == 1].wipe.mean() if (d[f] == 1).any() else np.nan
            med = 0.5
        else:
            med = d[f].median()
            a, b = d[d[f] <= med].wipe.mean(), d[d[f] > med].wipe.mean()
        real = b - a
        null = []
        for _ in range(4000):
            S2 = d.copy()
            S2["wipe"] = S2.groupby("day").wipe.transform(
                lambda s: RNG.permutation(s.values))
            if d[f].nunique() < 3:
                null.append(S2[S2[f] == 1].wipe.mean() - S2[S2[f] == 0].wipe.mean())
            else:
                null.append(S2[S2[f] > med].wipe.mean()
                            - S2[S2[f] <= med].wipe.mean())
        null = np.array(null)
        p = (np.abs(null) >= abs(real)).mean() if np.isfinite(real) else np.nan
        print(f"{f:>10} {a:>10.4f} {b:>10.4f} "
              f"{b/a if a else float('nan'):>8.2f} {p:>19.4f}")
