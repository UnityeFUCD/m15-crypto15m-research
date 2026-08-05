"""Does TRUE underlying volatility predict our edge - and can we optimise on it?

WHAT CHANGED
  Every earlier volatility test used Kalshi's implied probability as a proxy,
  and it was contaminated: a 72c favourite that loses travels 72->0 while a
  winner travels 72->100, so losers manufacture dispersion. The -0.43
  correlation was mostly that identity.

  The market metadata carries the real underlying:
      floor_strike     = A0, the 60-second BRTI mean at window OPEN
      expiration_value = A1, the 60-second BRTI mean at window CLOSE
  reconstructing A1 >= A0 reproduces the settled result 99.98% of the time on
  41,334 markets over 73 days. Underlying volatility is SYMMETRIC with respect
  to our outcome - a big move is big whichever way it goes - so unlike the
  proxy it can legitimately predict.

THE MECHANISM BEING TESTED
  We buy the favourite 8-14 minutes before close. The favourite leads because
  the index currently sits on one side of A0. Whether that lead survives is a
  race between the size of the lead and the movement still to come. Median BTC
  moves are ~9bp over a full 15 minutes, so these are thin decisions and
  volatility should dominate the win rate.

  If true, then in a high-volatility regime a 68c favourite is worth LESS than
  68c, and in a calm regime it is worth MORE - and the market, quoting a single
  price, cannot be right in both.

THE FOUR QUESTIONS
  1. does underlying volatility actually cluster? (with 73 days, properly)
  2. does PRIOR underlying volatility predict our realised edge?
  3. how does the WIN RATE of the 65-70c band vary with it - the user's
     specific question
  4. is the market's price CALIBRATED across volatility regimes, or does it
     misprice systematically in one of them? That is where edge comes from.
"""
from pathlib import Path

import numpy as np
import pandas as pd

LIVE = Path(__file__).resolve().parent
RES = LIVE.parent / "research" / "final_minute_favorite_maker_validation"
RNG = np.random.default_rng(20261053)

U = pd.read_parquet(LIVE / "underlying.parquet")
U["close"] = pd.to_datetime(U.close, format="mixed", utc=True)
U = U.sort_values(["coin", "close"]).reset_index(drop=True)
print(f"underlying: {len(U):,} windows, {U.coin.nunique()} coins, "
      f"{U.close.dt.date.nunique()} days")

# realised volatility from PRIOR windows only - strictly known at entry
for w in (4, 8, 16, 48):
    U[f"rv{w}"] = (U.groupby("coin").ret
                     .transform(lambda s: s.abs().shift(1).rolling(w).mean()))
U["rv_long"] = (U.groupby("coin").ret
                  .transform(lambda s: s.abs().shift(1).rolling(96).mean()))
U["rv_ratio"] = U.rv8 / U.rv_long          # short vs long: is it flaring up?

print("\n" + "=" * 78)
print("Q1 - DOES UNDERLYING VOLATILITY CLUSTER? (73 days, true data)")
print("=" * 78)
U["abs_ret"] = U.ret.abs()
for w in (4, 8, 16, 48):
    d = U.dropna(subset=[f"rv{w}"])
    print(f"  corr(|this move|, mean |move| over prior {w:>2} windows) = "
          f"{d.abs_ret.corr(d[f'rv{w}']):+.4f}   n {len(d):,}")
print("\n  The proxy gave +0.20 on 10 days. This is the true series on 73 days.")
print("  Strong clustering is what makes a PRIOR-volatility signal possible.")

print("\n" + "=" * 78)
print("Q2 - DOES PRIOR UNDERLYING VOLATILITY PREDICT OUR EDGE?")
print("=" * 78)
G = pd.read_parquet(RES / "grid.parquet")
q = G[(G.px >= 0.65) & (G.px < 0.80) & (G.minute >= 8) & (G.minute <= 14)
      & (G.vol >= 2000)]
E = (q.sort_values("minute", ascending=False)
       .groupby(["wkey", "coin"], as_index=False).first())
E = (E.sort_values(["wkey", "minute"], ascending=[True, False])
       .groupby("wkey", as_index=False).head(3)).copy()
M = E.merge(U[["wkey", "coin", "rv4", "rv8", "rv16", "rv48", "rv_long",
               "rv_ratio", "abs_ret"]], on=["wkey", "coin"], how="inner")
M = M.dropna(subset=["rv8", "rv_long"]).copy()
print(f"  entries matched to true volatility: {len(M):,} over "
      f"{M.day.nunique()} days")
days = sorted(M.day.unique())


def dayboot(df, fn, n=6000):
    gd = {d: g for d, g in df.groupby("day")}
    ds = sorted(gd)
    out = []
    for _ in range(n):
        s = pd.concat([gd[ds[i]] for i in RNG.integers(0, len(ds), len(ds))])
        v = fn(s)
        if v is not None and not np.isnan(v):
            out.append(v)
    return np.sort(np.array(out))


print(f"\n{'measure':>12} {'corr with edge':>16} {'95% CI':>22} {'P(>=0)':>9}")
for col in ("rv4", "rv8", "rv16", "rv48", "rv_ratio"):
    r = M[col].corr(M.edge)
    b = dayboot(M, lambda s, c=col: s[c].corr(s.edge))
    print(f"{col:>12} {r:>+15.4f} [{b[150]:>+8.4f},{b[5849]:>+8.4f}] "
          f"{(b >= 0).mean():>9.4f}")
print("\n  rv_ratio is short-horizon vol divided by long - it asks whether")
print("  volatility is FLARING UP right now, not whether the coin is")
print("  generally volatile. Those are different signals.")

print("\n" + "=" * 78)
print("Q3 - HOW DOES THE WIN RATE MOVE WITH VOLATILITY?")
print("=" * 78)
M["vq"] = pd.qcut(M.rv8, 5, labels=["calmest", "calm", "mid", "busy", "wildest"])
print(f"{'vol regime':>11} {'entries':>9} {'median rv8':>12} {'avg px':>8} "
      f"{'WIN RATE':>10} {'implied':>9} {'edge/ct':>9}")
for k, g in M.groupby("vq", observed=True):
    print(f"{str(k):>11} {len(g):>9} {g.rv8.median()*1e4:>10.2f}bp "
          f"{g.px.mean()*100:>7.1f}c {g.won.mean():>10.4f} "
          f"{g.px.mean():>9.4f} {g.edge.mean()*100:>+8.2f}c")
print("\n  'implied' is what the market charged. Edge is win rate minus that.")

print("\n" + "=" * 78)
print("Q4 - IS THE MARKET CALIBRATED ACROSS VOLATILITY REGIMES?")
print("=" * 78)
print("  This is where any real edge must come from: the market quotes ONE")
print("  price, but the true probability depends on the volatility regime.\n")
M["pxb"] = pd.cut(M.px, [0.65, 0.70, 0.75, 0.80])
print(f"{'price band':>14} {'vol regime':>11} {'n':>6} {'paid':>8} "
      f"{'actual win':>12} {'edge':>9}")
for pb, g in M.groupby("pxb", observed=True):
    if len(g) < 60:
        continue
    g = g.copy()
    g["v2"] = pd.qcut(g.rv8, 3, labels=["calm", "mid", "wild"],
                      duplicates="drop")
    for k, h in g.groupby("v2", observed=True):
        if len(h) < 15:
            continue
        print(f"{str(pb):>14} {str(k):>11} {len(h):>6} "
              f"{h.px.mean():>8.4f} {h.won.mean():>12.4f} "
              f"{h.edge.mean()*100:>+8.2f}c")
print("\n  If 'actual win' tracks 'paid' in every cell, the market is well")
print("  calibrated and there is nothing to exploit. If the calm cells beat")
print("  their price and the wild cells miss it, the market is quoting an")
print("  average and we can trade the difference.")

M.to_parquet(LIVE / "true_vol_entries.parquet")
print(f"\nwrote true_vol_entries.parquet ({len(M):,} rows)")
