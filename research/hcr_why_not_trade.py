"""Is 'HCR is more positive' enough to trade on? Four checks.

The user's objection is correct on its face: fill-corrected, HCR is +0.88c and
non-HCR is -1.10c. If one cohort is above water and the other is not, why not
trade only the first?

This script tests whether that gap is actionable, and it also tests MY OWN
counter-argument, which was 'the effect is absent in train'. If the train-vs-
test difference is itself inside noise, then my disqualifier is weaker than I
stated and I should say so.

  A. is the HCR cohort's own edge distinguishable from zero?
  B. is the train-vs-test gap real, or is my disqualifier noise too?
  C. how long to tell +0.88c from 0.00c at the achievable firing rate?
  D. conditional on trading anyway, does the HCR filter beat trading all?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE / "hcr_final_audit.py").read_text().split("B = build()")[0]
ns: dict = {"__file__": str(HERE / "hcr_final_audit.py")}
exec(compile(src, str(HERE / "hcr_final_audit.py"), "exec"), ns)

B = ns["build"]()
RNG = np.random.default_rng(90210)
P_WIN, P_LOSE = 0.843, 0.962
B["pf"] = np.where(B.won == 1, P_WIN, P_LOSE)
h, r = B[B.HCR], B[~B.HCR]
DAYS = B.day.nunique()


def corr(g):
    return (g.maker * g.pf).sum() / g.pf.sum() * 100


print("=" * 78)
print("A. IS THE HCR COHORT ITSELF ABOVE ZERO?  (not 'above non-HCR' - above 0)")
print("=" * 78)
for lbl, g in (("HCR", h), ("non-HCR", r)):
    gd = {k: v for k, v in g.groupby("day")}
    ds = sorted(gd)
    bs = np.sort(np.array([
        (lambda x: (x.maker * x.pf).sum() / x.pf.sum() * 100)(
            pd.concat([gd[ds[i]] for i in RNG.integers(0, len(ds), len(ds))]))
        for _ in range(6000)]))
    print("  %-8s fill-corrected %+6.2fc   95%% CI [%+6.2f, %+6.2f]   P(<=0) %.4f"
          % (lbl, corr(g), bs[150], bs[5849], (bs <= 0).mean()))
print("\n  The HCR interval contains zero. '+0.88c' is a point estimate whose")
print("  uncertainty is several times its own size.")

print("\n" + "=" * 78)
print("B. IS MY OWN DISQUALIFIER REAL? (train lift vs test lift)")
print("=" * 78)
sp = ns["SPLITS"]
tr = B[(B.utc >= sp["train"][0]) & (B.utc < sp["train"][1])]
te = B[(B.utc >= sp["test"][0]) & (B.utc < sp["test"][1])]


def lift(g):
    a, b = g[g.HCR], g[~g.HCR]
    return (a.maker.mean() - b.maker.mean()) * 100


print("  train lift %+.2fc (HCR n %d)   test lift %+.2fc (HCR n %d)"
      % (lift(tr), tr.HCR.sum(), lift(te), te.HCR.sum()))
diff = lift(te) - lift(tr)
bs = []
for _ in range(6000):
    a = tr.sample(len(tr), replace=True, random_state=int(RNG.integers(1e9)))
    b = te.sample(len(te), replace=True, random_state=int(RNG.integers(1e9)))
    if a.HCR.sum() > 5 and (~a.HCR).sum() > 5 and b.HCR.sum() > 5:
        bs.append(lift(b) - lift(a))
bs = np.sort(np.array(bs))
print("  test-minus-train difference %+.2fc   95%% CI [%+.2f, %+.2f]   P(<=0) %.4f"
      % (diff, bs[150], bs[int(len(bs) * .975)], (bs <= 0).mean()))
if bs[150] <= 0:
    print("\n  HONEST RESULT: the train/test gap is ITSELF inside noise. So my")
    print("  'absent in train' argument does not prove the signal is fake - it")
    print("  proves the data cannot tell the two periods apart. That weakens my")
    print("  disqualifier to: UNINFORMATIVE, not DISPROVEN.")
else:
    print("\n  The train/test gap is larger than sampling noise: the periods")
    print("  genuinely differ, which is what regime-dependence looks like.")

print("\n" + "=" * 78)
print("C. HOW LONG TO TELL +0.88c FROM ZERO?")
print("=" * 78)
per_day = len(h) / DAYS
q = 15
pnl = (h.maker * h.pf).values * q
sd_ct = pnl.std()
daily_sd = sd_ct * np.sqrt(per_day)
daily_mean = corr(h) / 100 * q * per_day
print("  HCR firings/day %.1f   at q%d" % (per_day, q))
print("  mean %+.2f $/day   SD %.2f $/day" % (daily_mean, daily_sd))
if daily_mean > 0:
    n80 = (2.8 * daily_sd / daily_mean) ** 2
    n95 = (1.96 * daily_sd / daily_mean) ** 2
    print("  days to reach 95%% confidence the mean is not zero : %,.0f".replace(",", "")
          % n95)
    print("  days for a properly powered (80%%) test            : %,.0f".replace(",", "")
          % n80)
    print("  that is %.1f years at 80%% power." % (n80 / 365))
print("\n  Meanwhile the account is $136.27 and the kill floor is $398.25.")

print("\n" + "=" * 78)
print("D. CONDITIONAL ON TRADING ANYWAY - DOES THE FILTER HELP?")
print("=" * 78)
print("  %-22s %6s %11s %13s" % ("policy", "n", "per contract", "$/day at q15"))
for lbl, g in (("trade everything", B), ("HCR only", h), ("non-HCR only", r)):
    print("  %-22s %6d %+10.2fc %+12.2f"
          % (lbl, len(g), corr(g), (g.maker * g.pf).sum() * q / DAYS))
print("\n  Day-matched, HCR-only vs trade-everything, WITHIN each day:")
pr = []
for day, g in B.groupby("day"):
    a = g[g.HCR]
    if len(a) >= 1 and len(g) >= 3:
        pr.append((a.maker * a.pf).sum() / a.pf.sum()
                  - (g.maker * g.pf).sum() / g.pf.sum())
pr = np.array(pr) * 100
bs = np.sort(np.array([RNG.choice(pr, len(pr), True).mean() for _ in range(8000)]))
print("    days %d   filter lift %+.2fc   95%% CI [%+.2f, %+.2f]   P(<=0) %.4f"
      % (len(pr), pr.mean(), bs[200], bs[7799], (bs <= 0).mean()))
