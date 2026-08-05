"""Does QUEUE DEPTH at order time predict a bad fill?

THE OPENING
  Today established that our fill process is adversely biased: we fill 96.2% of
  losers and 84.3% of winners, because a losing favourite must fall back
  through our resting bid while a winning one runs away. That bias is worth
  roughly 27.6% of pre-shift edge and 58.9% post-shift.

  Every attempt to predict WHICH fills go bad has failed - 15 features, path
  dynamics, cross-sectional z-scores, multivariate AUC 0.6972 that loses money
  at every threshold. All of those used market state. None used our own QUEUE
  POSITION, which the runner logs on every opportunity and which is known
  BEFORE the order is placed.

  lsm_report shows fill rate varies a lot with it:
      queue <100   0.8571
      100-400      0.8245
      400-1k       0.7545
      1k-3k        0.9000
      >3k          0.6679

  Fill rate alone is not interesting - today's work showed a LOW fill rate is
  often good news, because it means the favourite ran away and won. The
  question is whether deep queue positions produce fills that LOSE.

THE MECHANISM THAT WOULD MAKE IT TRUE
  Sitting behind 3,000 contracts means the whole queue ahead has to be consumed
  before we trade. That only happens on a genuine push through the level - real
  selling pressure, not a stray order. So a deep-queue fill should be much more
  strongly selected on 'the market moved against us' than a front-of-queue fill,
  which can happen on noise.

  If true, queue depth is the first adverse-selection predictor available at
  DECISION time rather than after the fact, and skipping deep-queue
  opportunities would be a real filter.

THE BAR, unchanged and applied strictly
  1. baseline is KEEP-ALL, never discard-all
  2. the discarded cohort must be verifiably NEGATIVE, not merely smaller
  3. dollars, and paired against what is actually deployed
  4. and it must not simply be re-expressing price or volume, both of which
     correlate with book depth
"""
import glob
import json
from datetime import datetime

import numpy as np
import pandas as pd

from acct import api

RNG = np.random.default_rng(20261040)


def ts(s):
    try:
        return datetime.fromisoformat((s or "").replace("Z", "+00:00"))
    except Exception:
        return None


OPP = {}
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
            if d.get("ev") == "opportunity":
                OPP.setdefault(t, {}).update(
                    qa=d.get("queue_ahead"), px=d.get("post_at"),
                    side=d.get("side"), spread=d.get("spread_c"),
                    flong=d.get("frac_long"), minleft=d.get("min_left"))
            if d.get("ev") == "terminal":
                OPP.setdefault(t, {}).update(
                    filled=d.get("FILLED"), target=d.get("target"),
                    fq=d.get("F_Q"))
    except Exception:
        pass
A = {t: o for t, o in OPP.items() if o.get("qa") is not None and o.get("px")}
print(f"opportunities with a logged queue_ahead: {len(A)}")

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

out = []
for s in rows:
    t = s.get("ticker")
    if t not in A:
        continue
    n = float(s.get("yes_count_fp") or 0) + float(s.get("no_count_fp") or 0)
    cost = float(s.get("yes_total_cost_dollars") or 0) + float(
        s.get("no_total_cost_dollars") or 0)
    if n <= 0 or cost <= 0:
        continue
    rev = float(s.get("revenue") or 0) / 100.0
    fee = float(s.get("fee_cost") or 0)
    st = ts(s.get("settled_time"))
    o = A[t]
    out.append(dict(ticker=t, qa=float(o["qa"]), n=n, px=cost / n,
                    pnl=rev - cost - fee, edge=(rev - cost - fee) / n,
                    won=1 if rev > 0 else 0,
                    spread=float(o.get("spread") or 0),
                    flong=float(o.get("flong") or 0),
                    window=t.split("-")[1],
                    day=str(st.date()) if st else "?"))
D = pd.DataFrame(out)
if len(D) < 50:
    raise SystemExit(f"only {len(D)} matched - too few")
print(f"matched settled fills: {len(D)}   contracts {D.n.sum():.0f}   "
      f"net ${D.pnl.sum():+.2f} ({D.pnl.sum()/D.n.sum()*100:+.2f}c/ct)\n")

print("=" * 76)
print("EDGE BY QUEUE DEPTH AT ORDER TIME")
print("=" * 76)
D["qb"] = pd.cut(D.qa, [-1, 100, 400, 1000, 3000, 1e9],
                 labels=["<100", "100-400", "400-1k", "1k-3k", ">3k"])
print(f"{'queue ahead':>12} {'fills':>7} {'contracts':>11} {'avg px':>8} "
      f"{'win':>7} {'edge/ct':>9} {'net $':>10}")
for k, g in D.groupby("qb", observed=True):
    print(f"{str(k):>12} {len(g):>7} {g.n.sum():>11.0f} "
          f"{g.px.mean()*100:>7.1f}c {g.won.mean():>7.4f} "
          f"{g.pnl.sum()/g.n.sum()*100:>+8.2f}c {g.pnl.sum():>+10.2f}")

print("\n" + "=" * 76)
print("IS DEEP QUEUE ACTUALLY NEGATIVE? (window-clustered)")
print("=" * 76)
wins = sorted(D.window.unique())
gw = {w: g for w, g in D.groupby("window")}


def boot(mask_fn, n=8000):
    v = []
    for _ in range(n):
        s = pd.concat([gw[wins[i]] for i in RNG.integers(0, len(wins), len(wins))])
        s = s[mask_fn(s)]
        if len(s) and s.n.sum() > 0:
            v.append(s.pnl.sum() / s.n.sum() * 100)
    return np.sort(np.array(v))


for thr in (1000, 2000, 3000):
    deep = D[D.qa >= thr]
    shal = D[D.qa < thr]
    if len(deep) < 8:
        continue
    bs = boot(lambda s, t=thr: s.qa >= t)
    e = deep.pnl.sum() / deep.n.sum() * 100
    print(f"  queue >= {thr:>5}: n {len(deep):>4}  edge {e:>+7.2f}c  "
          f"95% CI [{bs[int(.025*len(bs))]:+.2f}, {bs[int(.975*len(bs))]:+.2f}]  "
          f"P(>=0) {(bs >= 0).mean():.4f}")
    print(f"  {'':>13} shallow n {len(shal):>4}  edge "
          f"{shal.pnl.sum()/shal.n.sum()*100:>+7.2f}c")

print("\n" + "=" * 76)
print("KEEP-ALL vs SKIP-DEEP, IN DOLLARS (the only comparison that counts)")
print("=" * 76)
days = sorted(D.day.unique())
print(f"{'rule':>22} {'fills':>7} {'$ total':>10} {'$/day':>9} {'edge/ct':>9}")
print(f"{'KEEP ALL (deployed)':>22} {len(D):>7} {D.pnl.sum():>10.2f} "
      f"{D.pnl.sum()/len(days):>9.2f} {D.pnl.sum()/D.n.sum()*100:>+8.2f}c")
for thr in (1000, 2000, 3000):
    k = D[D.qa < thr]
    if len(k) < 20:
        continue
    print(f"{'skip queue >= ' + str(thr):>22} {len(k):>7} {k.pnl.sum():>10.2f} "
          f"{k.pnl.sum()/len(days):>9.2f} {k.pnl.sum()/k.n.sum()*100:>+8.2f}c")

print("\n" + "=" * 76)
print("CONFOUND - is queue depth just price or volume in disguise?")
print("=" * 76)
print(f"  corr(queue_ahead, entry price) = {D.qa.corr(D.px):+.4f}")
print(f"  corr(queue_ahead, frac_long)   = {D.qa.corr(D.flong):+.4f}")
print(f"  corr(queue_ahead, spread)      = {D.qa.corr(D.spread):+.4f}")
D["pb"] = pd.cut(D.px, [0.60, 0.70, 0.75, 0.90])
print(f"\n{'price bucket':>16} {'shallow edge':>14} {'deep edge':>12} {'gap':>9}")
for b, g in D.groupby("pb", observed=True):
    a, d2 = g[g.qa < 1000], g[g.qa >= 1000]
    if len(a) < 4 or len(d2) < 4:
        continue
    ea = a.pnl.sum() / a.n.sum() * 100
    ed = d2.pnl.sum() / d2.n.sum() * 100
    print(f"{str(b):>16} {ea:>+13.2f}c {ed:>+11.2f}c {ed-ea:>+8.2f}c")
print("\n  If the deep-queue penalty vanishes inside price buckets, queue depth")
print("  is a proxy for price and adds nothing the band does not already do.")
