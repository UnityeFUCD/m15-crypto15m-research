"""HCR + DRC + RACE on the actual LSM book: overlap and combined policies.

Uses the fully corrected ledger:
  - LSM orders only (client_order_id prefix)
  - held side derived from action (buy X -> X, sell X -> opposite)
  - all 17 late-closing markets recovered
  - rank from the true submission sequence over ALL orders
  - fill-level latency from the exchange created_time
  - HCR with the minute-00 filter DROPPED (t=0.92, unjustified)
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture.kalshi import KalshiClient          # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
RNG = np.random.default_rng(42)
api = KalshiClient()

U = pd.read_parquet(DATA / "underlying.parquet")
U["close_ts"] = ((pd.to_datetime(U.wkey, format="%y%b%d%H%M", utc=True)
                  + pd.Timedelta(hours=4) - EPOCH).dt.total_seconds()
                 ).round().astype("int64")
Uu = (U[U.result.isin(["yes", "no"])].dropna(subset=["a0"]).query("a0>0")
      .sort_values(["coin", "close_ts"]).drop_duplicates(["coin", "close_ts"]))
parts = []
for c, g in Uu.groupby("coin"):
    g = g.sort_values("close_ts").reset_index(drop=True)
    contig = g.close_ts.diff() == 900
    g["r"] = np.where(contig, g.a0 / g.a0.shift(1) - 1.0, np.nan)
    g["sig4"] = pd.Series(g.r).rolling(4).std(ddof=1)
    g.loc[contig.rolling(4).min() != 1, "sig4"] = np.nan
    g["z"] = g.r / (g.sig4 + 1e-12)
    parts.append(g)
S = pd.concat(parts, ignore_index=True)
CF = (S.groupby("close_ts").agg(r_common=("r", "mean"), nc=("r", "count"))
      .reset_index())
CF = CF[CF.nc >= 4].sort_values("close_ts").reset_index(drop=True)
CF["calm24"] = CF.r_common.abs().rolling(24).mean()

O = pd.read_parquet(DATA / "orders_history.parquet")
F = pd.read_parquet(DATA / "fills_history.parquet")
LSM = O[O.client_order_id.astype(str).str.startswith("lsm")].copy()
LSM["sub_t"] = pd.to_datetime(LSM.created_time, utc=True, format="mixed")
LSM["held"] = np.where(LSM.action == "buy", LSM.side,
                       np.where(LSM.side == "yes", "no", "yes"))
LSM["close_ts"] = ((pd.to_datetime(LSM.ticker.str.split("-").str[1],
                                   format="%y%b%d%H%M", utc=True)
                    + pd.Timedelta(hours=4) - EPOCH).dt.total_seconds()
                   ).round().astype("int64")
need = sorted(set(LSM.ticker) - set(U.ticker))
add = []
for i in range(0, len(need), 20):
    r = api.get("/markets", {"tickers": ",".join(need[i:i + 20])})
    for m in ((r.body or {}).get("markets") or []):
        if m.get("result") in ("yes", "no"):
            add.append(dict(ticker=m["ticker"], result=m["result"]))
FR = (pd.concat([U[["ticker", "result"]], pd.DataFrame(add)], ignore_index=True)
      .drop_duplicates("ticker"))
LSM = LSM.merge(FR, on="ticker", how="inner").sort_values(["close_ts", "sub_t"])
LSM["rank"] = LSM.groupby("close_ts").cumcount() + 1

Fl = F[F.order_id.isin(set(LSM.order_id))].copy()
Fl["fill_t"] = pd.to_datetime(Fl.created_time, utc=True, format="mixed")
Fl = Fl.merge(LSM[["order_id", "sub_t", "close_ts", "rank", "result"]],
              on="order_id")
Fl["lat"] = (Fl.fill_t - Fl.sub_t).dt.total_seconds()
Fl["n"] = Fl.count_fp.astype(float)
Fl["px"] = np.where(Fl.side == "yes", Fl.yes_price_dollars.astype(float),
                    Fl.no_price_dollars.astype(float))
Fl["won"] = (Fl.result == Fl.side).astype(int)
Fl["pnl"] = Fl.won * Fl.n - Fl.px * Fl.n - Fl.fee_cost.astype(float)
Fl = Fl.merge(CF[["close_ts", "r_common", "calm24"]], on="close_ts", how="left")
Fl["d"] = np.where(Fl.side == "yes", 1.0, -1.0)
Fl["opp"] = Fl.d * Fl.r_common
Fl["HCR"] = ((Fl.opp <= -0.0015) & (Fl.calm24 <= 0.0030)).fillna(False)
Fl["coin"] = (Fl.ticker.str.split("-").str[0]
              .str.replace("KX", "", regex=False)
              .str.replace("15M", "", regex=False))
zmap = S.set_index(["coin", "close_ts"]).z
Fl["z"] = [zmap.get((c, t), np.nan) for c, t in zip(Fl.coin, Fl.close_ts)]
Fl["DRC"] = ((Fl.z >= 1.0) & (Fl.side == "no")
             & (Fl.coin.isin(["DOGE", "ETH", "SOL", "XRP"]))).fillna(False)

print("=" * 78)
print("OVERLAP ON THE 275 ACTUAL LSM MARKETS")
print("=" * 78)
M = (Fl.groupby("ticker")
     .agg(HCR=("HCR", "max"), DRC=("DRC", "max"), rank=("rank", "first"),
          n=("n", "sum"), pnl=("pnl", "sum"), won=("won", "max")).reset_index())
M[["HCR", "DRC"]] = M[["HCR", "DRC"]].fillna(False)
print("  %-22s %8s %10s %8s %11s %9s"
      % ("cohort", "markets", "contracts", "win", "P&L", "c/ct"))
for lbl, g in (("ALL (actual)", M), ("HCR only", M[M.HCR & ~M.DRC]),
               ("DRC only", M[M.DRC & ~M.HCR]), ("BOTH", M[M.HCR & M.DRC]),
               ("NEITHER", M[~M.HCR & ~M.DRC])):
    if not len(g):
        print("  %-22s %8d   none" % (lbl, 0))
        continue
    print("  %-22s %8d %10.0f %8.4f %+10.2f %+8.2f"
          % (lbl, len(g), g.n.sum(), g.won.mean(), g.pnl.sum(),
             g.pnl.sum() / g.n.sum() * 100))
print("\n  HCR %d markets | DRC %d | both %d -> largely INDEPENDENT signals"
      % (M.HCR.sum(), M.DRC.sum(), (M.HCR & M.DRC).sum()))


def policy(sel=None, cap=99, t1=1e9, t2=1e9):
    f = Fl if sel is None else Fl[sel(Fl)]
    f = f[f["rank"] <= cap]
    if cap <= 2:
        k = (((f["rank"] == 1) & (f.lat <= t1))
             | ((f["rank"] == 2) & (f.lat <= t2)))
    else:
        k = (((f["rank"] == 1) & (f.lat <= t1))
             | ((f["rank"] == 2) & (f.lat <= t2)) | (f["rank"] > 2))
    s = f[k]
    w = s.groupby("close_ts").pnl.sum().sort_index()
    eq = w.cumsum()
    dd = (eq.cummax() - eq).max() if len(w) else 0.0
    return s.pnl.sum(), s.n.sum(), dd, (w.min() if len(w) else 0.0), len(w)


print("\n" + "=" * 78)
print("COMBINED POLICIES ON THE SAME BOOK")
print("=" * 78)
ROWS = [("1  actual (everything, full wait)", None, 99, 1e9, 1e9),
        ("2  HCR only", lambda d: d.HCR, 99, 1e9, 1e9),
        ("3  HCR or DRC", lambda d: d.HCR | d.DRC, 99, 1e9, 1e9),
        ("4  cap2 only", None, 2, 1e9, 1e9),
        ("5  cap2 + 8s/2s (RACE)", None, 2, 8, 2),
        ("6  HCR + cap2", lambda d: d.HCR, 2, 1e9, 1e9),
        ("7  HCR + cap2 + 8s/2s", lambda d: d.HCR, 2, 8, 2),
        ("8  (HCR|DRC) + cap2 + 8s/2s", lambda d: d.HCR | d.DRC, 2, 8, 2)]
print("  %-32s %10s %10s %9s %10s %8s"
      % ("policy", "P&L", "contracts", "max DD", "worst wdw", "windows"))
for lbl, sel, cap, a, b in ROWS:
    p, n, dd, ww, nw = policy(sel, cap, a, b)
    print("  %-32s %+10.2f %10.0f %9.2f %+10.2f %8d"
          % (lbl, p, n, dd, ww, nw))

print("\n" + "=" * 78)
print("ROBUSTNESS - close-window bootstrap vs the actual book")
print("=" * 78)
wins = sorted(Fl.close_ts.unique())
gw = {w: g for w, g in Fl.groupby("close_ts")}


def polx(df, sel, cap, t1, t2):
    f = df if sel is None else df[sel(df)]
    f = f[f["rank"] <= cap]
    if cap <= 2:
        return f[((f["rank"] == 1) & (f.lat <= t1))
                 | ((f["rank"] == 2) & (f.lat <= t2))].pnl.sum()
    return f[((f["rank"] == 1) & (f.lat <= t1))
             | ((f["rank"] == 2) & (f.lat <= t2))
             | (f["rank"] > 2)].pnl.sum()


for lbl, sel, cap, a, b in (("HCR only", lambda d: d.HCR, 99, 1e9, 1e9),
                            ("HCR|DRC", lambda d: d.HCR | d.DRC, 99, 1e9, 1e9),
                            ("HCR+cap2+8s/2s", lambda d: d.HCR, 2, 8, 2)):
    obs = polx(Fl, sel, cap, a, b) - Fl.pnl.sum()
    bs = []
    for _ in range(3000):
        s = pd.concat([gw[wins[i]]
                       for i in RNG.integers(0, len(wins), len(wins))])
        bs.append(polx(s, sel, cap, a, b) - s.pnl.sum())
    bs = np.sort(np.array(bs))
    print("  %-18s gain %+9.2f  95%% CI [%+9.2f, %+9.2f]  P(<=0) %.4f"
          % (lbl, obs, bs[75], bs[2924], (bs <= 0).mean()))
