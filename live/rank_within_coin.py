"""Is 'rank' a real effect, or just the series order spelling out COIN?

THE NEAR-MISS THIS EXISTS TO CATCH
  A difference-in-differences said the rank structure inverted between regimes:
  rank3-minus-rank1 went from +2.95c pre-shift to -13.55c post-shift, DiD
  -16.49c with CI [-31.07,-1.87] and P=0.0127. On its own that reads as a
  clean result and would justify lowering MAX_PER_WINDOW to 2.

  But the runner assigns rank by a FIXED series order - BTC, ETH, SOL, XRP,
  DOGE, HYPE. Rank is therefore not an independent property of an entry; it is
  very nearly a relabelling of which coin it is:

      PRE  rank3 = 96% DOGE          PRE  rank1 = SOL55 XRP30 DOGE12
      POST rank3 = DOGE48 XRP27 HYPE25
      LIVE rank3 = XRP32 DOGE26 HYPE18 ETH9

  So "rank3 got worse relative to rank1" and "the coins that happen to land in
  rank3 changed, and coins have different edges" predict the SAME observation.
  The DiD cannot distinguish them.

THE SEPARATING TEST
  Hold the coin fixed. For a given coin, does the rank it happened to occupy
  predict its edge? A coin lands in different ranks depending on which OTHER
  coins qualified in that window, which is close to arbitrary from its own
  point of view.

    - if rank predicts edge WITHIN a coin, rank is real and the cap is at issue
    - if it does not, rank is a proxy for coin, the DiD is an artifact, and the
      cap is innocent. The right lever would then be coin selection, which was
      tested earlier today and found not to support dropping anything.

  This is the same discipline that killed the intraday tilt: an effect has to
  survive the confound that most plausibly generates it.
"""
import glob
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from acct import api

RNG = np.random.default_rng(20261049)
RES = Path("../research/final_minute_favorite_maker_validation")
SER = {"BTC": 0, "ETH": 1, "SOL": 2, "XRP": 3, "DOGE": 4, "HYPE": 5}


def ts(x):
    try:
        return datetime.fromisoformat((x or "").replace("Z", "+00:00"))
    except Exception:
        return None


def rank_panel(df, order_col):
    d = df.copy()
    d["ord"] = d.coin.map(SER).fillna(9)
    d = d.sort_values(["wkey", "ord", order_col], ascending=[True, True, False])
    d["rank"] = d.groupby("wkey").cumcount() + 1
    return d[d["rank"] <= 3]


G = pd.read_parquet(RES / "grid.parquet")
q = G[(G.px >= 0.65) & (G.px < 0.80) & (G.minute >= 8) & (G.minute <= 14)
      & (G.vol >= 2000)]
PRE = rank_panel((q.sort_values("minute", ascending=False)
                    .groupby(["wkey", "coin"], as_index=False).first()), "minute")
P = pd.read_parquet(RES / "postshift_nofilter.parquet")
POST = rank_panel(P[(P.vol >= 2000) & (P.px >= 0.65) & (P.px < 0.80)], "px")

POSTED = {}
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
            if d.get("ev") == "post" and d.get("ticker") and d.get("price"):
                t, u = d["ticker"], ts(d.get("utc"))
                if t not in POSTED or (u and u < POSTED[t]):
                    POSTED[t] = u
    except Exception:
        pass
rows, cur, seen = [], None, set()
for _ in range(12):
    pr = {"limit": 200, "settlement_status": "settled"}
    if cur:
        pr["cursor"] = cur
    c, j = api("/portfolio/settlements", pr)
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
    if t not in POSTED or POSTED[t] is None:
        continue
    n = float(s.get("yes_count_fp") or 0) + float(s.get("no_count_fp") or 0)
    cost = float(s.get("yes_total_cost_dollars") or 0) + float(
        s.get("no_total_cost_dollars") or 0)
    if n <= 0 or cost <= 0:
        continue
    o.append(dict(wkey=t.split("-")[1], posted=POSTED[t], n=n,
                  coin=t.split("-")[0].replace("KX", "").replace("15M", ""),
                  edge=(float(s.get("revenue") or 0) / 100.0 - cost
                        - float(s.get("fee_cost") or 0)) / n))
L = pd.DataFrame(o).sort_values(["wkey", "posted"])
L["rank"] = L.groupby("wkey").cumcount() + 1
LIVE = L[L["rank"] <= 3]

print("=" * 78)
print("HOW COLLINEAR ARE RANK AND COIN?")
print("=" * 78)
for nm, D in (("PRE", PRE), ("POST", POST), ("LIVE", LIVE)):
    ct = pd.crosstab(D.coin, D["rank"])
    pred = ct.max(axis=1).sum() / len(D)
    print(f"  {nm:>5}: knowing the coin predicts its rank "
          f"{pred:.3f} of the time  (chance ~0.33-0.50)")

print("\n" + "=" * 78)
print("THE SEPARATING TEST - WITHIN EACH COIN, DOES RANK PREDICT EDGE?")
print("=" * 78)
for nm, D in (("PRE", PRE), ("POST", POST), ("LIVE", LIVE)):
    print(f"\n  {nm}")
    print(f"{'coin':>7} {'rank1':>10} {'rank2':>10} {'rank3':>10} "
          f"{'r3 - r1':>10} {'n':>6}")
    gaps = []
    for c, g in D.groupby("coin"):
        e = {}
        for r in (1, 2, 3):
            gg = g[g["rank"] == r]
            e[r] = gg.edge.mean() * 100 if len(gg) >= 5 else np.nan
        if not np.isnan(e[1]) and not np.isnan(e[3]):
            gaps.append(e[3] - e[1])
        f = lambda v: f"{v:>+9.2f}c" if not np.isnan(v) else f"{'-':>10}"
        gp = (f"{e[3]-e[1]:>+9.2f}c"
              if not np.isnan(e[1]) and not np.isnan(e[3]) else f"{'-':>10}")
        print(f"{c:>7} {f(e[1])} {f(e[2])} {f(e[3])} {gp} {len(g):>6}")
    if len(gaps) >= 2:
        print(f"    coins where rank3 < rank1: "
              f"{sum(1 for x in gaps if x < 0)}/{len(gaps)}   "
              f"mean within-coin gap {np.mean(gaps):+.2f}c")
    else:
        print("    too few coins have both ranks populated to compare")

print("\n" + "=" * 78)
print("POOLED WITHIN-COIN CONTRAST (coin fixed effects removed)")
print("=" * 78)
for nm, D in (("PRE", PRE), ("POST", POST), ("LIVE", LIVE)):
    d = D.copy()
    d["resid"] = d.edge - d.groupby("coin").edge.transform("mean")
    a, b = d[d["rank"] == 3], d[d["rank"] == 1]
    if len(a) < 5 or len(b) < 5:
        print(f"  {nm:>5}: too few")
        continue
    obs = (a.resid.mean() - b.resid.mean()) * 100
    ws = sorted(d.wkey.unique())
    gw = {w: g for w, g in d.groupby("wkey")}
    bs = []
    for _ in range(6000):
        s = pd.concat([gw[ws[i]] for i in RNG.integers(0, len(ws), len(ws))])
        x, y = s[s["rank"] == 3], s[s["rank"] == 1]
        if len(x) >= 3 and len(y) >= 3:
            bs.append((x.resid.mean() - y.resid.mean()) * 100)
    bs = np.sort(np.array(bs))
    raw_a, raw_b = D[D["rank"] == 3], D[D["rank"] == 1]
    raw = (raw_a.edge.mean() - raw_b.edge.mean()) * 100
    print(f"  {nm:>5}: raw contrast {raw:+7.2f}c   "
          f"coin-adjusted {obs:+7.2f}c   "
          f"95% CI [{bs[int(.025*len(bs))]:+7.2f}, "
          f"{bs[int(.975*len(bs))]:+7.2f}]")

print("\n" + "=" * 78)
print("VERDICT")
print("=" * 78)
print("  If the coin-adjusted contrast collapses toward zero while the raw one")
print("  is large, rank was standing in for coin and the difference-in-")
print("  differences was an artifact. In that case cap 3 is NOT implicated and")
print("  must stay - the earlier coin-selection test already showed no coin is")
print("  a drop candidate either.")
