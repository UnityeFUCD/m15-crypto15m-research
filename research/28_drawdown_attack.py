"""Where does drawdown actually come from, and what is the cheapest way to cut it?

WHY EARLIER DRAWDOWN NUMBERS WERE USELESS
  Every previous bootstrap reported maxDD ~$0 because all 10 backtest days were
  positive and the backtest drift (+11.29c/contract) swamps the variance. But
  the LIVE edge is +7.41c, and live has already produced a real $61 drawdown.
  A model that says drawdowns essentially never happen is contradicted by the
  only live sample we have.

  So this runs the simulation at the LIVE edge, not the backtest edge, by
  stressing outcomes until the win rate matches what the exchange actually
  paid. That is the regime we are trading in.

STRUCTURE PRESERVED
  Whole WINDOWS are resampled, never individual entries, because the coins
  inside a window fail together (P(all 3 lose) = 6.2x independence). Resampling
  entries independently is what manufactured the fake zero-drawdown result.

THEN: rank interventions by drawdown reduced per dollar of profit given up.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260815)
D = pd.read_parquet(OUT / "grid.parquet")
QTY, MAX_GROSS = 30, 110.0
NPATH, NDAY = 20000, 30
BANK = 367.31
days = sorted(D.day.unique())


def entries(cap, vmin=2000, band=(0.65, 0.85), tw=(8, 14), qty_rule=None):
    q = D[(D.px >= band[0]) & (D.px < band[1]) & (D.minute >= tw[0])
          & (D.minute <= tw[1]) & (D.vol >= vmin)]
    e = (q.sort_values("minute", ascending=False)
           .groupby(["wkey", "coin"], as_index=False).first())
    e = (e.sort_values(["wkey", "minute"], ascending=[True, False])
           .groupby("wkey", as_index=False).head(cap)).copy()
    e["rank"] = e.groupby("wkey").cumcount() + 1
    e["qty"] = qty_rule(e) if qty_rule else QTY
    keep = []
    for _, g in e.groupby("wkey"):
        gross = 0.0
        for r in g.itertuples():
            c = r.qty * r.px
            if gross + c > MAX_GROSS or r.qty <= 0:
                continue
            gross += c
            keep.append((r.wkey, r.day, r.px, r.won, r.qty))
    return pd.DataFrame(keep, columns=["wkey", "day", "px", "won", "qty"])


def stress(K, delta):
    """Degrade the win rate by a fixed DECREMENT, not to a fixed level.

    Stressing every config to a common win rate would be wrong: win rate rises
    mechanically with entry price, so a cheap-band config starts lower and
    would be stressed less, manufacturing a free lunch for narrow bands. The
    live shortfall is a degradation, so it is applied as one - the same
    decrement to every configuration.

    Flips are applied at the WINDOW level so the correlation structure that
    drives clustered losses is preserved rather than diluted."""
    K = K.copy()
    cur = K.won.mean()
    target_win = cur - delta
    if target_win >= cur:
        return K
    wins = K[K.won == 1]
    need = int(round((cur - target_win) * len(K)))
    # flip whole windows, worst-case-realistic: a bad window takes everything
    order = RNG.permutation(wins.wkey.unique())
    flip, got = [], 0
    for w in order:
        idx = wins.index[wins.wkey == w]
        flip += list(idx)
        got += len(idx)
        if got >= need:
            break
    K.loc[flip, "won"] = 0
    return K


def pnl_rows(K):
    return np.where(K.won == 1, (1 - K.px) * K.qty, -K.px * K.qty)


def sim(K, label):
    K = K.copy()
    K["pnl"] = pnl_rows(K)
    wp = K.groupby("wkey").pnl.sum().to_numpy()          # whole-window P&L
    nwin_day = K.wkey.nunique() / len(days)
    k = max(int(round(nwin_day)), 1)
    idx = RNG.integers(0, len(wp), size=(NPATH, NDAY, k))
    daily = wp[idx].sum(2)
    eq = BANK + np.cumsum(daily, axis=1)
    run = np.concatenate([np.full((NPATH, 1), BANK), eq], axis=1)
    dd = (np.maximum.accumulate(run, axis=1) - run).max(1)
    return dict(label=label, n=len(K), contracts=K.qty.sum(),
                edge=K.pnl.sum() / K.qty.sum() * 100,
                day=daily.mean(), sd=daily.std(),
                pneg=(daily < 0).mean(), p5=np.percentile(daily, 5),
                medDD=np.median(dd), dd95=np.percentile(dd, 95),
                dd99=np.percentile(dd, 99),
                ruin=(run.min(1) < 0.5 * BANK).mean())


BT_WIN   = 0.8462          # backtest win rate at the live config
LIVE_WIN = 0.7788          # what the exchange actually paid, 104 settlements
DELTA    = BT_WIN - LIVE_WIN   # the degradation, applied equally to all configs
print(f"bank ${BANK:.2f}  qty {QTY}  {NPATH:,} paths x {NDAY} days")
print(f"stressing win rate to the LIVE {LIVE_WIN:.4f} "
      f"(backtest is {entries(3).won.mean():.4f})\n")

base = entries(3)
print("=== BACKTEST EDGE vs LIVE EDGE: same config, different regime ===")
for lbl, K in (("backtest win 0.846", base),
               ("LIVE (stressed)", stress(base, DELTA))):
    r = sim(K, lbl)
    print(f"  {lbl:>20}  edge {r['edge']:>+6.2f}c  ${r['day']:>+7.2f}/day  "
          f"P(day<0) {r['pneg']:.3f}  medDD ${r['medDD']:>6.2f}  "
          f"95%DD ${r['dd95']:>6.2f}  99%DD ${r['dd99']:>7.2f}")
print("\n  the entire drawdown profile lives in that win-rate gap.")

print("\n=== INTERVENTIONS, all evaluated AT THE LIVE EDGE ===")
CANDS = [
    ("cap 3 (current)", lambda: entries(3)),
    ("cap 2", lambda: entries(2)),
    ("cap 1", lambda: entries(1)),
    ("cap 3, rank2/3 at qty 15", lambda: entries(
        3, qty_rule=lambda e: np.where(e["rank"] == 1, QTY, 15))),
    ("cap 3, rank3 at qty 15", lambda: entries(
        3, qty_rule=lambda e: np.where(e["rank"] <= 2, QTY, 15))),
    ("cap 3, band 65-80c", lambda: entries(3, band=(0.65, 0.80))),
    ("cap 3, band 65-75c", lambda: entries(3, band=(0.65, 0.75))),
    ("cap 3, vmin 4000", lambda: entries(3, vmin=4000)),
]
res = []
for lbl, fn in CANDS:
    K = stress(fn(), DELTA)
    if len(K) < 100:
        continue
    res.append(sim(K, lbl))
R = pd.DataFrame(res)
print(f"{'intervention':>26} {'ent/day':>8} {'edge':>7} {'$/day':>9} "
      f"{'P(day<0)':>9} {'medDD':>8} {'95%DD':>8} {'99%DD':>9} {'P(ruin)':>8}")
for r in res:
    print(f"{r['label']:>26} {r['n']/len(days):>8.1f} {r['edge']:>+7.2f} "
          f"{r['day']:>+9.2f} {r['pneg']:>9.3f} {r['medDD']:>8.2f} "
          f"{r['dd95']:>8.2f} {r['dd99']:>9.2f} {r['ruin']:>8.4f}")

b = res[0]
print(f"\n=== EFFICIENCY: dollars of 95%-drawdown removed per dollar/day given up ===")
for r in res[1:]:
    dgive = b["day"] - r["day"]
    dcut = b["dd95"] - r["dd95"]
    ratio = (dcut / dgive) if dgive > 0.01 else float("inf")
    print(f"  {r['label']:>26}  gives up ${dgive:>6.2f}/day  "
          f"cuts 95%DD ${dcut:>7.2f}  ratio {ratio:>7.2f}"
          f"{'   <-- free' if dgive <= 0.01 and dcut > 0 else ''}")

print("\n=== WHERE THE LEFT TAIL ACTUALLY SITS ===")
K = stress(entries(3), DELTA)
K["pnl"] = pnl_rows(K)
wp = K.groupby("wkey").agg(pnl=("pnl", "sum"), n=("pnl", "size"),
                           px=("px", "mean"), losses=("won", lambda s: (s == 0).sum()))
worst = wp.nsmallest(int(len(wp) * 0.05), "pnl")
print(f"  worst 5% of windows ({len(worst)}) carry ${worst.pnl.sum():,.0f} of "
      f"${wp.pnl.sum():,.0f} total")
print(f"  they average {worst.n.mean():.2f} positions vs {wp.n.mean():.2f} overall")
print(f"  and {worst.losses.mean():.2f} losses vs {wp.losses.mean():.2f} overall")
print(f"  mean entry price in them {worst.px.mean()*100:.1f}c vs "
      f"{wp.px.mean()*100:.1f}c overall")
print("\n  If the worst windows are simply the FULL ones, the lever is the cap.")
print("  If they are the EXPENSIVE ones, the lever is the band.")
