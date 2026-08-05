"""Is it the RANK that is bad, or the WINDOW? And is that knowable in advance?

THE OBSERVATION
  On live fills, edge falls hard with how many entries a window contains:
      1-entry windows   +18.56c   (14 windows)
      2-entry windows    +7.43c   (66)
      3-entry windows    -5.97c   (32)
  and by rank: rank1 +8.14c, rank2 -2.51c, rank3 -2.65c.

  Pre-shift this ran the other way. Rank-3 entries carried +16.65c and busy
  windows were the good ones, because a 3rd entry only existed when three coins
  independently cleared a volume floor that kept just 31.1% of candidates.
  Post-shift the floor keeps 58.2%, so three coins clearing it is routine.

THE CONFOUND THAT MUST BE SEPARATED FIRST
  The rank-1 cohort CONTAINS the 1-entry windows. So "rank 1 beats rank 2"
  may be nothing more than "quiet windows beat busy windows" re-expressed. The
  two readings have completely different consequences:

    rank effect   -> lower the cap; later entries in a window are worse
    window effect -> the cap is innocent; BREADTH is a window-level signal

  Separating them is simple: compare rank 1 against rank 2 ONLY inside windows
  that actually have two or more entries. That holds window type fixed.

WHY THE WINDOW READING WOULD MATTER MORE
  Breadth is observable BEFORE the entry. The runner already knows how many
  coins cleared the volume floor when it evaluates a window - it logs n_peers.
  Every adverse-selection predictor tested so far (16 features, path dynamics,
  cross-sectional z, queue depth) has failed. A window-level breadth signal
  would be the first that works, and it costs nothing to observe.

  It also has a mechanism that fits the correlated-loss result: wipeouts are
  windows where all three positions lose together. If breadth marks windows
  where every coin is moving as one, breadth is a wipeout-risk proxy.

THE BAR IS UNCHANGED
  Discards must be verifiably NEGATIVE, baseline is keep-all, dollars, and it
  must beat what is deployed. Breadth reducing edge is not enough - trading
  fewer broad windows has to make MORE money.
"""
import glob
import json
from datetime import datetime

import numpy as np
import pandas as pd

from acct import api

RNG = np.random.default_rng(20261047)


def ts(x):
    try:
        return datetime.fromisoformat((x or "").replace("Z", "+00:00"))
    except Exception:
        return None


POST, PEERS = {}, {}
for fp in sorted(glob.glob("lsm_*.jsonl")) + sorted(glob.glob("lsm_*.log")):
    try:
        for ln in open(fp, errors="ignore"):
            ln = ln.strip()
            if not ln.startswith("{"):
                continue
            try:
                d = json.loads(ln)
            except Exception:
                continue
            t = d.get("ticker")
            if not t:
                continue
            if d.get("ev") == "post" and d.get("price"):
                u = ts(d.get("utc"))
                if t not in POST or (u and u < POST[t][1]):
                    POST[t] = (float(d["price"]), u)
            if d.get("n_peers") is not None:
                PEERS.setdefault(t.split("-")[1], []).append(int(d["n_peers"]))
    except Exception:
        pass

rows, cur, seen = [], None, set()
for _ in range(12):
    p = {"limit": 200, "settlement_status": "settled"}
    if cur:
        p["cursor"] = cur
    c, j = api("/portfolio/settlements", p)
    if c != 200 or not j:
        break
    b = j.get("settlements") or []
    if not b:
        break
    for s in b:
        if s.get("ticker") and s["ticker"] not in seen:
            seen.add(s["ticker"])
            rows.append(s)
    cur = j.get("cursor")
    if not cur:
        break
o = []
for s in rows:
    t = s["ticker"]
    if t not in POST or POST[t][1] is None:
        continue
    n = float(s.get("yes_count_fp") or 0) + float(s.get("no_count_fp") or 0)
    cost = float(s.get("yes_total_cost_dollars") or 0) + float(
        s.get("no_total_cost_dollars") or 0)
    if n <= 0 or cost <= 0:
        continue
    o.append(dict(ticker=t, window=t.split("-")[1], posted=POST[t][1], n=n,
                  px=cost / n,
                  pnl=float(s.get("revenue") or 0) / 100.0 - cost
                      - float(s.get("fee_cost") or 0)))
D = pd.DataFrame(o).sort_values(["window", "posted"])
D["rank"] = D.groupby("window").cumcount() + 1
D["breadth"] = D.groupby("window")["rank"].transform("max")
print(f"fills {len(D)}   windows {D.window.nunique()}   "
      f"net ${D.pnl.sum():+.2f}\n")


def wboot(df, fn, n=8000):
    ws = sorted(df.window.unique())
    gw = {w: g for w, g in df.groupby("window")}
    v = []
    for _ in range(n):
        s = pd.concat([gw[ws[i]] for i in RNG.integers(0, len(ws), len(ws))])
        r = fn(s)
        if r is not None and not np.isnan(r):
            v.append(r)
    return np.sort(np.array(v))


print("=" * 76)
print("STEP 1 - RANK 1 vs RANK 2, INSIDE WINDOWS THAT HAVE BOTH")
print("=" * 76)
M = D[D.breadth >= 2]
a = M[M["rank"] == 1]
b = M[M["rank"] == 2]
ea = a.pnl.sum() / a.n.sum() * 100
eb = b.pnl.sum() / b.n.sum() * 100
print(f"  windows with 2+ entries: {M.window.nunique()}")
print(f"  rank 1 within them: n {len(a):>3}  {ea:+7.2f}c")
print(f"  rank 2 within them: n {len(b):>3}  {eb:+7.2f}c")
bs = wboot(M, lambda s: (
    s[s["rank"] == 1].pnl.sum() / max(s[s["rank"] == 1].n.sum(), 1)
    - s[s["rank"] == 2].pnl.sum() / max(s[s["rank"] == 2].n.sum(), 1)) * 100)
print(f"  difference {ea-eb:+.2f}c   95% CI [{bs[200]:+.2f}, {bs[7799]:+.2f}]   "
      f"P(rank1 not better) {(bs <= 0).mean():.4f}")
print("\n  If this is ~zero, rank is innocent and BREADTH is the real variable.")

print("\n" + "=" * 76)
print("STEP 2 - BREADTH AS A WINDOW-LEVEL SIGNAL")
print("=" * 76)
Wg = D.groupby("window").agg(pnl=("pnl", "sum"), n=("n", "sum"),
                             breadth=("breadth", "first"))
print(f"{'breadth':>9} {'windows':>9} {'contracts':>11} {'edge/ct':>9} "
      f"{'net $':>10} {'losing windows':>16}")
for k, g in Wg.groupby("breadth"):
    lose = (g.pnl < 0).mean()
    print(f"{int(k):>9} {len(g):>9} {g.n.sum():>11.0f} "
          f"{g.pnl.sum()/g.n.sum()*100:>+8.2f}c {g.pnl.sum():>+10.2f} "
          f"{lose:>16.3f}")
b3 = D[D.breadth >= 3]
b12 = D[D.breadth < 3]
if len(b3) >= 10 and len(b12) >= 10:
    e3 = b3.pnl.sum() / b3.n.sum() * 100
    e12 = b12.pnl.sum() / b12.n.sum() * 100
    bs2 = wboot(D, lambda s: (
        s[s.breadth >= 3].pnl.sum() / max(s[s.breadth >= 3].n.sum(), 1)
        - s[s.breadth < 3].pnl.sum() / max(s[s.breadth < 3].n.sum(), 1)) * 100)
    print(f"\n  breadth>=3 minus breadth<3: {e3-e12:+.2f}c   "
          f"95% CI [{bs2[200]:+.2f}, {bs2[7799]:+.2f}]   "
          f"P(broad windows WORSE) {(bs2 < 0).mean():.4f}")

print("\n" + "=" * 76)
print("STEP 3 - IS BREADTH JUST A WIPEOUT PROXY?")
print("=" * 76)
Wg["allwin"] = D.groupby("window").apply(
    lambda g: (g.pnl > 0).all()).reindex(Wg.index)
Wg["alllose"] = D.groupby("window").apply(
    lambda g: (g.pnl < 0).all()).reindex(Wg.index)
print(f"{'breadth':>9} {'windows':>9} {'all win':>9} {'all lose':>10} "
      f"{'mixed':>8}")
for k, g in Wg.groupby("breadth"):
    mixed = 1 - g.allwin.mean() - g.alllose.mean()
    print(f"{int(k):>9} {len(g):>9} {g.allwin.mean():>9.3f} "
          f"{g.alllose.mean():>10.3f} {mixed:>8.3f}")
print("\n  Broad windows cannot be 'mixed' by accident - if they all move")
print("  together, breadth is measuring correlation, which is the known")
print("  wipeout mechanism rather than a new signal.")

print("\n" + "=" * 76)
print("STEP 4 - WOULD SKIPPING BROAD WINDOWS MAKE MONEY?")
print("=" * 76)
print(f"{'rule':>28} {'fills':>7} {'net $':>10} {'c/ct':>9} {'vs keep-all':>13}")
base = D.pnl.sum()
print(f"{'keep all (deployed)':>28} {len(D):>7} {base:>10.2f} "
      f"{base/D.n.sum()*100:>+8.2f}c {'-':>13}")
for bmax in (1, 2):
    k = D[D.breadth <= bmax]
    if len(k) < 20:
        continue
    bs3 = wboot(D, lambda s, m=bmax: s[s.breadth <= m].pnl.sum() - s.pnl.sum())
    print(f"{'only breadth<=' + str(bmax):>28} {len(k):>7} {k.pnl.sum():>10.2f} "
          f"{k.pnl.sum()/k.n.sum()*100:>+8.2f}c "
          f"{k.pnl.sum()-base:>+8.2f} [{bs3[200]:+.0f},{bs3[7799]:+.0f}]")
print("\n  NOTE: this is NOT deployable as written. Breadth is only fully known")
print("  after the window's entries are placed - the 3rd entry knows it, the")
print("  1st does not. A usable version would key on how many coins CLEAR the")
print("  filters at decision time, which is a different quantity.")
