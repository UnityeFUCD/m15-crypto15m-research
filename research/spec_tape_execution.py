"""Would the :00 strategy actually have EXECUTED? Real tape, Aug 4-5.

WHAT THE BACKTEST ASSUMES AND CANNOT CHECK
  spec_momentum_00.py buys q15 at the displayed ask and books
  qty * (won - ask) - fee. That silently assumes the displayed ask holds 15
  contracts. A one-lot quote and a 500-lot quote look identical in candle
  data. If the real book is thin, the strategy earns a fraction of the
  backtested amount, or pays up through several price levels.

  data/trades_lsm.parquet holds 2,114,639 real public trades across the 301
  markets traded on Aug 4-5, with millisecond timestamps, size, price and
  taker side. That is an actual tape, and it can answer the question the
  candles cannot.

WHAT IS MEASURED
  For every trade the strategy would have taken on those two days:

    depth_at_or_better   contracts that traded at our price or better, on our
                         side, in the 60s after our decision - an estimate of
                         what we could have lifted
    time_to_first_print  how long until ANY trade happened at all
    price_at_fill        the volume-weighted price we would actually have paid
    slippage             that price minus the displayed ask

  Then P&L is recomputed under realistic partial fills and compared with the
  backtest's assumption.

WHY THIS IS THE RIGHT CHECK NOW
  The strategy's mean ask is 78.60c, so a loss costs 3.7x what a win pays.
  Execution shortfall at that ratio is not a rounding error. And a taker
  strategy is exactly where displayed-price backtests flatter themselves.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"
OUT = ROOT / "research" / "results"
from research.spec_momentum_00 import build, apply_spec, SPEC   # noqa: E402

QTY = 15
WINDOW_S = 60


def fee_total(qty, price):
    if qty <= 0:
        return 0.0
    return math.ceil(0.07 * qty * price * (1 - price) * 10_000 - 1e-12) / 10_000


print("=" * 88)
print("TAPE EXECUTION TEST - would the :00 strategy have filled?")
print("=" * 88)

D = build()
# best configuration from the previous analysis: drop volume + momentum
BEST = dict(SPEC); BEST.update({"vol_min": 0.0, "rise_min": 0.0})
S = apply_spec(D, BEST)
S["utc"] = pd.to_datetime(S.close_ts, unit="s", utc=True)

T = pd.read_parquet(DATA / "trades_lsm.parquet")
tape_tickers = set(T.ticker.astype(str))
S["has_tape"] = S.ticker.isin(tape_tickers)
live = S[S.has_tape].copy()
print(f"  strategy signals overall: {len(S)}")
print(f"  of those with real tape (Aug 4-5): {len(live)}")
if not len(live):
    sys.exit("no overlap between strategy signals and the tape")
print(f"  days covered: {sorted(set(live.utc.dt.date))}")

# decision time = entry_ml - wait minutes before close
live["decision_ts"] = live.close_ts - 60.0 * (14 - BEST["wait_min"])
T["t"] = T.ts.astype(float)   # tape ts is epoch SECONDS.
# It was documented as milliseconds. That was wrong and it is the same
# unit-confusion class that invalidated PTC v1: the compaction step did
# .astype("int64") // 10**6 assuming datetime64[ns], but this pandas stores
# datetime64[us], so the division already produced seconds. Verified by
# checking that trades land 0-15 minutes before each close.
by = {k: v.sort_values("t") for k, v in T.groupby("ticker")}

rows = []
for r in live.itertuples():
    tp = by.get(r.ticker)
    if tp is None:
        continue
    w = tp[(tp.t >= r.decision_ts) & (tp.t < r.decision_ts + WINDOW_S)]
    # to BUY our side we lift the ask. On the YES scale our price is:
    our_yes_c = round(r.ask * 100) if r.side == "yes" else round((1 - r.ask) * 100)
    if r.side == "yes":
        # buying YES: trades at yes price <= our ask are at or better
        ok = w[(w.yes_c <= our_yes_c) & (w.taker_book_side == "ask")]
        px = ok.yes_c / 100.0
    else:
        # buying NO at (1 - yes_bid): trades at yes price >= our threshold
        ok = w[(w.yes_c >= our_yes_c) & (w.taker_book_side == "bid")]
        px = 1.0 - ok.yes_c / 100.0
    depth = float(ok["count"].sum())
    vwap = float((px * ok["count"]).sum() / depth) if depth > 0 else np.nan
    first = float(w.t.min() - r.decision_ts) if len(w) else np.nan
    rows.append(dict(ticker=r.ticker, coin=r.coin, side=r.side, won=int(r.won),
                     ask=float(r.ask), depth=depth, vwap=vwap,
                     first_print_s=first, any_prints=int(len(w))))
E = pd.DataFrame(rows)

print("\n" + "-" * 88)
print("LIQUIDITY AT OUR PRICE, 60s after the decision")
print("-" * 88)
print(f"  signals examined            {len(E)}")
print(f"  with ANY print in the window {int((E.any_prints > 0).sum())}")
print(f"  with depth at our price      {int((E.depth > 0).sum())}")
print(f"  median depth                 {E.depth.median():.0f} contracts")
print(f"  25th / 75th pct depth        {E.depth.quantile(.25):.0f} / "
      f"{E.depth.quantile(.75):.0f}")
print(f"  share with depth >= {QTY}       "
      f"{(E.depth >= QTY).mean():.4f}")
print(f"  median time to first print   {E.first_print_s.median():.1f}s")

print("\n" + "-" * 88)
print("REALISTIC FILL vs THE BACKTEST ASSUMPTION")
print("-" * 88)
E["filled_qty"] = np.minimum(E.depth, QTY)
E["exec_px"] = np.where(E.depth > 0, E.vwap, np.nan)
E["slip_c"] = (E.exec_px - E.ask) * 100
got = E[E.filled_qty > 0].copy()
got["fee"] = [fee_total(int(q), p) for q, p in zip(got.filled_qty, got.exec_px)]
got["pnl_real"] = got.filled_qty * (got.won - got.exec_px) - got.fee
E["fee_bt"] = [fee_total(QTY, a) for a in E.ask]
E["pnl_backtest"] = QTY * (E.won - E.ask) - E.fee_bt

print("  %-30s %10s %12s" % ("", "contracts", "P&L"))
print("  %-30s %10.0f %+12.2f"
      % ("backtest (q15 at displayed ask)", QTY * len(E), E.pnl_backtest.sum()))
print("  %-30s %10.0f %+12.2f"
      % ("realistic (capped by depth)", got.filled_qty.sum(),
         got.pnl_real.sum()))
print("  %-30s %10s %+12.2f"
      % ("shortfall", "", got.pnl_real.sum() - E.pnl_backtest.sum()))
print(f"\n  fill ratio {got.filled_qty.sum()/(QTY*len(E)):.4f}")
print(f"  mean slippage vs displayed ask {got.slip_c.mean():+.2f}c")
print(f"  win rate on realised fills {got.won.mean():.4f} "
      f"(backtest cohort {E.won.mean():.4f})")

print("\n" + "-" * 88)
print("PER-TRADE DETAIL")
print("-" * 88)
print("  %-28s %5s %7s %8s %9s %9s %9s"
      % ("market", "side", "ask", "depth", "vwap", "won", "P&L"))
for r in got.sort_values("ticker").itertuples():
    print("  %-28s %5s %7.4f %8.0f %9.4f %9d %+9.2f"
          % (r.ticker, r.side, r.ask, r.depth, r.exec_px, r.won, r.pnl_real))

print("\n" + "-" * 88)
print("WHAT SIZE IS ACTUALLY SUPPORTABLE?")
print("-" * 88)
print("  %-8s %12s %14s %12s" % ("qty", "fill ratio", "P&L", "per contract"))
for q in (1, 5, 10, 15, 20, 30):
    fq = np.minimum(E.depth, q)
    m = fq > 0
    fee = np.array([fee_total(int(a), b) for a, b in
                    zip(fq[m], E.exec_px[m])])
    pnl = float((fq[m] * (E.won[m] - E.exec_px[m]) - fee).sum())
    print("  q%-7d %12.4f %+13.2f %+11.2fc"
          % (q, fq.sum() / (q * len(E)), pnl,
             pnl / fq.sum() * 100 if fq.sum() else float("nan")))

E.to_csv(OUT / "spec_tape_execution.csv", index=False)
print("\nwrote research/results/spec_tape_execution.csv")
