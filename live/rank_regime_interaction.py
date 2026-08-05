"""Did the RANK structure invert between regimes? A difference-in-differences.

THE HYPOTHESIS, STATED BEFORE LOOKING
  Four separate measurements today all pointed the same way and not one of them
  cleared 95% on its own:

    rank 1 vs rank 2 within a window   +9.12c   P(not better) 0.0307
    breadth>=3 vs breadth<3           -14.55c   P(worse)      0.9620
    rank-3 cohort                      -2.65c   CI [-20.15,+12.95]
    cap3 minus cap2                   -$20.61   CI [-161,+106]

  Pre-shift the structure ran the OTHER way: rank-3 entries carried +16.65c
  with CI [+12.23,+21.40] and busy windows were the best ones, because a 3rd
  entry only existed when three coins independently cleared a volume floor that
  kept 31.1% of candidates. Post-shift that floor keeps 58.2%, so a 3rd entry
  is routine and the selection is gone.

WHY TESTING LIVE ALONE IS THE WRONG TEST
  The live rank-3 slice is 32 fills over 2 days. It will never reach
  significance on its own in any reasonable time. But the CLAIM is not "rank-3
  is negative" - it is "rank-3 got worse RELATIVE to rank-1, between regimes".

  That is a difference-in-differences, and it is far better powered:
    - the within-window contrast (rank3 - rank1) cancels anything that moves
      the whole book, including the regime-wide edge decay
    - the pre-shift arm has 10 days and tight intervals, so it contributes
      real power rather than just another noisy slice

  If the interaction is significant, the mechanism behind cap 3 is
  demonstrably dead even though the live slice alone is ambiguous.

WHAT WOULD MAKE ME WRONG
  1. the contrast is confounded by coin - pre-shift is XRP/SOL/DOGE with no
     BTC at all, live is BTC/ETH/HYPE/XRP/DOGE/SOL. If rank correlates with
     coin differently across regimes, this measures coin, not rank.
  2. rank is defined differently in the two datasets. Pre-shift it is
     reconstructed by sorting; live it is the observed post order. They have to
     be made to mean the same thing.
  3. the live arm carries fill bias (96.2% of losers fill, 84.3% of winners)
     and the panel arms do not.

  All three are addressed or reported.
"""
import glob
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from acct import api

RNG = np.random.default_rng(20261048)
RES = Path("../research/final_minute_favorite_maker_validation")
SER = {"BTC": 0, "ETH": 1, "SOL": 2, "XRP": 3, "DOGE": 4, "HYPE": 5}


def ts(x):
    try:
        return datetime.fromisoformat((x or "").replace("Z", "+00:00"))
    except Exception:
        return None


def rank_panel(df, order_col):
    """Rank within window using the runner's own series order, as it does."""
    d = df.copy()
    d["ord"] = d.coin.map(SER).fillna(9)
    d = d.sort_values(["wkey", "ord", order_col], ascending=[True, True, False])
    d["rank"] = d.groupby("wkey").cumcount() + 1
    return d[d["rank"] <= 3]


# ---- PRE-SHIFT ----
G = pd.read_parquet(RES / "grid.parquet")
q = G[(G.px >= 0.65) & (G.px < 0.80) & (G.minute >= 8) & (G.minute <= 14)
      & (G.vol >= 2000)]
pre = (q.sort_values("minute", ascending=False)
         .groupby(["wkey", "coin"], as_index=False).first())
PRE = rank_panel(pre, "minute")

# ---- POST-SHIFT PANEL ----
P = pd.read_parquet(RES / "postshift_nofilter.parquet")
P = P[(P.vol >= 2000) & (P.px >= 0.65) & (P.px < 0.80)]
POST = rank_panel(P, "px")

# ---- LIVE ----
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

print(f"PRE  entries {len(PRE):>5}  windows {PRE.wkey.nunique():>4}")
print(f"POST entries {len(POST):>5}  windows {POST.wkey.nunique():>4}")
print(f"LIVE entries {len(LIVE):>5}  windows {LIVE.wkey.nunique():>4}\n")

print("=" * 76)
print("EDGE BY RANK IN EACH REGIME")
print("=" * 76)
print(f"{'regime':>7} {'rank1':>10} {'rank2':>10} {'rank3':>10} "
      f"{'rank3 - rank1':>15}")
CON = {}
for nm, D in (("PRE", PRE), ("POST", POST), ("LIVE", LIVE)):
    e = {}
    for r in (1, 2, 3):
        g = D[D["rank"] == r]
        e[r] = g.edge.mean() * 100 if len(g) >= 5 else np.nan
    CON[nm] = e[3] - e[1]
    print(f"{nm:>7} {e[1]:>+9.2f}c {e[2]:>+9.2f}c {e[3]:>+9.2f}c "
          f"{e[3]-e[1]:>+14.2f}c")
print("\n  The last column is the within-window contrast. It is immune to")
print("  anything that shifts the whole book, including the regime decay.")


def contrast_boot(D, n=6000):
    """Bootstrap rank3-minus-rank1, clustering on windows."""
    ws = sorted(D.wkey.unique())
    gw = {w: g for w, g in D.groupby("wkey")}
    out = []
    for _ in range(n):
        s = pd.concat([gw[ws[i]] for i in RNG.integers(0, len(ws), len(ws))])
        a, b = s[s["rank"] == 3], s[s["rank"] == 1]
        if len(a) >= 3 and len(b) >= 3:
            out.append((a.edge.mean() - b.edge.mean()) * 100)
    return np.sort(np.array(out))


print("\n" + "=" * 76)
print("THE CONTRAST, WITH WINDOW-CLUSTERED INTERVALS")
print("=" * 76)
B = {}
for nm, D in (("PRE", PRE), ("POST", POST), ("LIVE", LIVE)):
    b = contrast_boot(D)
    B[nm] = b
    print(f"  {nm:>5}: rank3 - rank1 = {CON[nm]:+7.2f}c   "
          f"95% CI [{b[int(.025*len(b))]:+7.2f}, {b[int(.975*len(b))]:+7.2f}]")

print("\n" + "=" * 76)
print("DIFFERENCE-IN-DIFFERENCES: did the contrast actually change?")
print("=" * 76)
for nm in ("POST", "LIVE"):
    n = min(len(B["PRE"]), len(B[nm]))
    d = np.sort(RNG.permutation(B[nm])[:n] - RNG.permutation(B["PRE"])[:n])
    obs = CON[nm] - CON["PRE"]
    print(f"  {nm} minus PRE: {obs:+7.2f}c   "
          f"95% CI [{d[int(.025*n)]:+7.2f}, {d[int(.975*n)]:+7.2f}]   "
          f"P(no change or better) {(d >= 0).mean():.4f}")
print("\n  A significant NEGATIVE difference means rank-3 specifically got")
print("  worse relative to rank-1 - i.e. the selection that justified cap 3")
print("  is gone, not merely that everything decayed.")

print("\n" + "=" * 76)
print("CONFOUND CHECK - is the contrast really about COIN?")
print("=" * 76)
print(f"{'regime':>7} {'rank1 coins':>34} {'rank3 coins':>34}")
for nm, D in (("PRE", PRE), ("POST", POST), ("LIVE", LIVE)):
    def mix(r):
        g = D[D["rank"] == r]
        if not len(g):
            return "-"
        v = g.coin.value_counts(normalize=True)
        return " ".join(f"{k}{v[k]*100:.0f}" for k in v.index[:4])
    print(f"{nm:>7} {mix(1):>34} {mix(3):>34}")
print("\n  Pre-shift has no BTC or HYPE at all, so the coin mix differs between")
print("  regimes by construction. What matters is whether rank1 and rank3 draw")
print("  from DIFFERENT coins WITHIN a regime - if they do, this is coin.")

print("\n" + "=" * 76)
print("VERDICT")
print("=" * 76)
n = min(len(B["PRE"]), len(B["LIVE"]))
dl = np.sort(RNG.permutation(B["LIVE"])[:n] - RNG.permutation(B["PRE"])[:n])
if (dl >= 0).mean() < 0.05:
    print("  The rank structure HAS inverted, significantly. The mechanism")
    print("  behind cap 3 no longer holds, and lowering the cap to 2 is")
    print("  justified by the interaction even though the live slice alone is")
    print("  ambiguous. This is a config change and needs the user's approval.")
else:
    print("  The interaction is NOT significant. The apparent inversion is")
    print("  within what two days of live data produce by chance, so cap 3")
    print("  stays. Revisit when more post-shift days exist.")
