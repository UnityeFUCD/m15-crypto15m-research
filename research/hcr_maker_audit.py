"""HCR audited on MAKER edge - the metric the production design actually pays.

WHY THIS RUN EXISTS
  hcr_final_audit.py scores every test on TAKER edge (cross the spread, pay
  the fee). But the production spec in capture/hcr.py is post-only maker: it
  rests at the bid and pays no fee. Judging a maker strategy by taker economics
  understates it by the full spread, which in this band is 3-6c.

  So the taker verdict answers "would HCR work if we crossed?" and the maker
  verdict answers "does HCR work as specified?". Both are reported.

FILL CORRECTION
  Maker edge assumes the resting order fills. It does not fill at random:
  measured on 303 live orders, P(fill | loser) = 0.962 and P(fill | winner)
  = 0.843. Weighting each market by its fill probability is what the strategy
  would actually realise, and it is always worse than the raw maker number.
"""
from __future__ import annotations

import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("hfa", HERE / "hcr_final_audit.py")

# hcr_final_audit runs its report at import time; re-use only its builder by
# executing the module source up to the report section.
src = (HERE / "hcr_final_audit.py").read_text().split("B = build()")[0]
ns: dict = {"__file__": str(HERE / "hcr_final_audit.py")}
exec(compile(src, str(HERE / "hcr_final_audit.py"), "exec"), ns)

B = ns["build"]()
RNG = np.random.default_rng(413)
P_FILL_WIN, P_FILL_LOSE = 0.843, 0.962
h, r = B[B.HCR], B[~B.HCR]


def corrected(g):
    """Fill-probability-weighted maker edge, in cents."""
    pf = np.where(g.won == 1, P_FILL_WIN, P_FILL_LOSE)
    return (g.maker * pf).sum() / pf.sum() * 100


print("=" * 78)
print("MAKER AUDIT - the production design is post-only maker, not taker")
print("=" * 78)
print("  %-10s %6s %9s %11s %16s" % ("cohort", "n", "win", "maker", "fill-corrected"))
for lbl, g in (("HCR", h), ("non-HCR", r)):
    print("  %-10s %6d %9.4f %+10.2fc %+15.2fc"
          % (lbl, len(g), g.won.mean(), g.maker.mean() * 100, corrected(g)))
print("  raw maker lift %+.2fc      fill-corrected lift %+.2fc"
      % ((h.maker.mean() - r.maker.mean()) * 100, corrected(h) - corrected(r)))

print("\n" + "=" * 78)
print("TEST 1m - MATCHED CONTROLS on maker edge")
print("=" * 78)
res = []
for (c, w, pb, sd), g in B.groupby(["coin", "week", "pb", "side"]):
    a, b = g[g.HCR], g[~g.HCR]
    if len(a) >= 1 and len(b) >= 2:
        res.append((len(a), a.maker.mean() - b.maker.mean(),
                    a.won.mean() - b.won.mean()))
wts = np.array([x[0] for x in res], dtype=float)
dm = np.array([x[1] for x in res])
dw = np.array([x[2] for x in res])
bs = np.sort(np.array([
    np.average(dm[i], weights=wts[i]) * 100
    for i in (RNG.integers(0, len(res), len(res)) for _ in range(8000))]))
print("  strata %d covering %d HCR markets" % (len(res), int(wts.sum())))
print("  weighted MAKER lift %+.2fc   win lift %+.2fpp"
      % (np.average(dm, weights=wts) * 100, np.average(dw, weights=wts) * 100))
print("  95%% CI [%+.2f, %+.2f]   P(<=0) %.4f"
      % (bs[200], bs[7799], (bs <= 0).mean()))

print("\n" + "=" * 78)
print("TEST 2m - DAY-CLUSTERED maker edge (is each cohort itself profitable?)")
print("=" * 78)
for lbl, g in (("HCR", h), ("non-HCR", r), ("all", B)):
    gd = {k: v for k, v in g.groupby("day")}
    ds = sorted(gd)
    b = np.sort(np.array([
        pd.concat([gd[ds[i]] for i in RNG.integers(0, len(ds), len(ds))]
                  ).maker.mean() * 100 for _ in range(6000)]))
    print("  %-8s %+6.2fc  95%% CI [%+6.2f, %+6.2f]  P(<=0) %.4f"
          % (lbl, g.maker.mean() * 100, b[150], b[5849], (b <= 0).mean()))

print("\n" + "=" * 78)
print("TEST 3m - CHRONOLOGICAL on maker edge")
print("=" * 78)
for nm, (a, b) in ns["SPLITS"].items():
    hs, rs = h[(h.utc >= a) & (h.utc < b)], r[(r.utc >= a) & (r.utc < b)]
    if len(hs) < 20:
        continue
    print("  %-6s HCR n %3d maker %+6.2fc (corr %+6.2fc) | non-HCR %+6.2fc "
          "| lift %+6.2fc"
          % (nm, len(hs), hs.maker.mean() * 100, corrected(hs),
             rs.maker.mean() * 100,
             (hs.maker.mean() - rs.maker.mean()) * 100))

print("\n" + "=" * 78)
print("TEST 4m - DOLLARS PER DAY at q15 (%d days), fill-corrected maker"
      % B.day.nunique())
print("=" * 78)
days = B.day.nunique()
print("  %-14s %6s %14s %14s" % ("policy", "n", "raw $/day", "corrected $/day"))
for lbl, g in (("all markets", B), ("HCR only", h), ("non-HCR", r)):
    pf = np.where(g.won == 1, P_FILL_WIN, P_FILL_LOSE)
    print("  %-14s %6d %+14.2f %+14.2f"
          % (lbl, len(g), g.maker.sum() * 15 / days,
             (g.maker * pf).sum() * 15 / days))
print("\n  NOTE: 'n' is the SAMPLED population, not the true one. Per-contract")
print("  cents are unbiased; $/day scales with how much of the day we sampled.")
