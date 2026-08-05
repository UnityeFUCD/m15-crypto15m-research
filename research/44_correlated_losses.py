"""Can a correlated all-lose window be PREDICTED at entry time?

WHY THIS IS THE HIGHEST-VALUE QUESTION LEFT
  The worst 5% of windows carry more loss than everything else makes in
  profit, and they are not "mostly losses" - they average 2.30 positions and
  2.30 losses. EVERY position in a bad window loses. P(all 3 lose) runs 6.2x
  independence. The $73.50 wipeout on 2026-08-05 was exactly this event.

  If such windows can be spotted BEFORE committing, the single largest source
  of drawdown becomes avoidable. If they cannot, then size is the only lever
  and that should be stated plainly rather than searched for indefinitely.

  A previous test of others_px against the all-lose event gave a ratio of
  1.01x - no signal. That was ONE feature. This tests everything available,
  and tests the mechanism rather than just screening.

ALSO ANSWERED HERE
  The user's observation that losses concentrate in the 80c+ band. That is
  checked directly, on both backtest and live data, because it is cheap and it
  determines whether the band should be trimmed.

DISCIPLINE
  Everything is measured at entry time, no lookahead. Screening many features
  guarantees false positives, so anything that survives is re-tested
  out-of-sample on held-out days.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260914)
D = pd.read_parquet(OUT / "grid.parquet")
QTY = 30

# cross-sectional state at each (window, minute)
G = D.groupby(["wkey", "minute"]).agg(
    s_px=("px", "sum"), n=("px", "size"), s_sq=("px", lambda s: (s ** 2).sum()),
    s_fl=("frac_long", "sum"), s_vol=("vol", "sum")).reset_index()
W = D.merge(G, on=["wkey", "minute"], how="left")
k = (W.n - 1).clip(lower=1)
W["others_px"] = np.where(W.n >= 3, (W.s_px - W.px) / k, np.nan)
W["others_fl"] = np.where(W.n >= 3, (W.s_fl - W.frac_long) / k, np.nan)
var = np.where(W.n >= 3, (W.s_sq - W.px ** 2) / k - W["others_px"] ** 2, np.nan)
W["others_disp"] = np.sqrt(np.clip(var, 0, None))
W["mkt_px"] = W.s_px / W.n                       # whole-market mean favourite
W["mkt_disp"] = np.sqrt(np.clip(W.s_sq / W.n - W.mkt_px ** 2, 0, None))

q = W[(W.px >= 0.65) & (W.px < 0.85) & (W.minute >= 8) & (W.minute <= 14)
      & (W.vol >= 2000)]
E = (q.sort_values("minute", ascending=False)
       .groupby(["wkey", "coin"], as_index=False).first())
E = (E.sort_values(["wkey", "minute"], ascending=[True, False])
       .groupby("wkey", as_index=False).head(3)).copy()
E["pnl"] = np.where(E.won == 1, (1 - E.px) * QTY, -E.px * QTY)

print("=" * 70)
print("PART 1 - ARE LOSSES CONCENTRATED IN THE HIGH BAND?")
print("=" * 70)
b = W[(W.minute >= 8) & (W.minute <= 14) & (W.vol >= 2000) &
      (W.px >= 0.55) & (W.px < 0.95)].copy()
b = (b.sort_values("minute", ascending=False)
       .groupby(["wkey", "coin"], as_index=False).first())
b["bucket"] = pd.cut(b.px, [.55, .65, .70, .75, .80, .85, .95])
print(f"{'bucket':>12} {'n':>6} {'win':>8} {'loss rate':>10} "
      f"{'avg loss $':>11} {'edge/ct':>9} {'share of all losses':>20}")
tot_loss = (b[b.won == 0].px * QTY).sum()
for bk, g in b.groupby("bucket", observed=True):
    L = g[g.won == 0]
    lsum = (L.px * QTY).sum()
    print(f"{str(bk):>12} {len(g):>6} {g.won.mean():>8.4f} "
          f"{1-g.won.mean():>10.4f} {(L.px*QTY).mean() if len(L) else 0:>11.2f} "
          f"{g.edge.mean()*100:>+9.2f} {lsum/tot_loss*100:>19.1f}%")
print("\n  A high bucket loses LESS OFTEN but each loss costs MORE (you pay")
print("  more per contract). 'share of all losses' shows where the dollars")
print("  actually go. Check whether the high band is really the problem, or")
print("  simply the most visible one.")

print("\n" + "=" * 70)
print("PART 2 - CAN THE ALL-LOSE WINDOW BE PREDICTED?")
print("=" * 70)
WD = E.groupby("wkey").agg(
    n=("won", "size"), losses=("won", lambda s: int((s == 0).sum())),
    pnl=("pnl", "sum"), px=("px", "mean"), vol=("vol", "mean"),
    fl=("frac_long", "mean"), opx=("others_px", "mean"),
    odisp=("others_disp", "mean"), mpx=("mkt_px", "mean"),
    mdisp=("mkt_disp", "mean"), minute=("minute", "mean"),
    day=("day", "first"), hour=("hour", "first")).reset_index()
M = WD[WD.n >= 2].copy()
M["allbad"] = (M.losses == M.n).astype(int)
base = M.allbad.mean()
print(f"windows with 2+ positions: {len(M)}   all-lose rate: {base:.4f}")
print(f"  (independence at win 0.846 would give ~{(1-0.846)**2:.4f} for n=2)")

FEATS = ["px", "vol", "fl", "opx", "odisp", "mpx", "mdisp", "minute"]
print(f"\n{'feature':>10} {'low half':>10} {'high half':>10} {'ratio':>8} "
      f"{'p (day-clustered)':>19}")
hits = []
for f in FEATS:
    d = M.dropna(subset=[f])
    if len(d) < 60:
        continue
    hi = d[f] > d[f].median()
    a, bb = d[~hi].allbad.mean(), d[hi].allbad.mean()
    ratio = bb / a if a > 0 else np.nan
    days = sorted(d.day.unique())
    g = {x: gg for x, gg in d.groupby("day")}
    bs = []
    for _ in range(3000):
        s = pd.concat([g[days[i]] for i in RNG.integers(0, len(days), len(days))])
        h2 = s[f] > s[f].median()
        if h2.sum() > 15 and (~h2).sum() > 15:
            bs.append(s[h2].allbad.mean() - s[~h2].allbad.mean())
    bs = np.sort(np.array(bs))
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean()) if len(bs) else np.nan
    print(f"{f:>10} {a:>10.4f} {bb:>10.4f} {ratio:>8.2f} {p:>19.4f}")
    if p < 0.05:
        hits.append(f)

print("\n=== DOES THE MARKET-WIDE STATE PREDICT IT? (the mechanism test) ===")
print("  An all-lose window means every favourite moved against us at once -")
print("  a market-wide move. If those are foreseeable, mkt_px or mkt_disp")
print("  should separate them.\n")
M["mq"] = pd.qcut(M.mpx.rank(method="first"), 4, labels=False, duplicates="drop")
print(f"{'mkt_px quartile':>16} {'n':>6} {'mean mkt px':>12} {'all-lose':>10} "
      f"{'window $':>10}")
for qq, g in M.dropna(subset=["mq"]).groupby("mq"):
    print(f"{int(qq):>16} {len(g):>6} {g.mpx.mean()*100:>11.1f}c "
          f"{g.allbad.mean():>10.4f} {g.pnl.mean():>+10.2f}")

print("\n=== TIME-CLUSTERING: do bad windows arrive in bursts? ===")
M2 = M.copy()
M2["close"] = pd.to_datetime(M2.wkey, format="%y%b%d%H%M", utc=True,
                             errors="coerce")
M2 = M2.dropna(subset=["close"]).sort_values("close").reset_index(drop=True)
ab = M2.allbad.values
for lag in (1, 2, 3, 4):
    if len(ab) > lag + 20:
        r = np.corrcoef(ab[:-lag], ab[lag:])[0, 1]
        nul = np.array([np.corrcoef(p[:-lag], p[lag:])[0, 1]
                        for p in (RNG.permutation(ab) for _ in range(2000))])
        print(f"  lag {lag}: autocorr of all-lose = {r:+.4f}   "
              f"permutation p = {(np.abs(nul) >= abs(r)).mean():.4f}")
print("  If null at every lag, wipeouts do not cluster in TIME either -")
print("  they are independent draws, and only size controls their impact.")

if hits:
    print(f"\n=== OUT-OF-SAMPLE CHECK ON SURVIVORS: {hits} ===")
    days = sorted(M.day.unique())
    cut = days[len(days) // 2]
    for f in hits:
        d = M.dropna(subset=[f])
        thr = d[d.day < cut][f].median()
        te = d[d.day >= cut]
        a = te[te[f] <= thr].allbad.mean()
        bb = te[te[f] > thr].allbad.mean()
        print(f"  {f:>10}: OOS low {a:.4f}  high {bb:.4f}  "
              f"ratio {bb/a if a>0 else float('nan'):.2f}")
else:
    print("\n=== NOTHING PREDICTS THE ALL-LOSE WINDOW ===")
    print("  No entry-time feature separates wipeout windows from normal ones.")
    print("  Combined with the null time-autocorrelation, that means these")
    print("  events are unforecastable and SIZE is the only control.")
