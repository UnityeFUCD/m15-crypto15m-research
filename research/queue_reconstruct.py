"""Reconstruct queue-ahead per LSM order from the public trade tape.

THE IDEA
  /portfolio/orders/{id}/queue_position is live-only, so the direct
  measurement is gone forever for historical orders. The QUANTITY is not.

  In a FIFO book an order resting at price P fills only after everything
  ahead of it at P has traded. So for each of our orders:

    queue_ahead = volume traded at OUR price, on the side that CONSUMES our
                  resting order, between our placement and our first fill

  and for an order that never filled, the same volume over its whole resting
  life is a LOWER BOUND on what stood in front of it.

MECHANICS
  We rest a passive BUY of the favourite side S at price P. We are consumed
  when a taker SELLS S into the bid, i.e. taker_outcome_side == S and
  taker_book_side == 'bid'. Prices are matched on the YES scale: a NO buy at
  p is a YES sell at 1-p, so the tape's yes_px is compared against the YES
  price of our resting order.

ASSUMPTIONS (this is a PROXY, never the exchange's field)
  1. Strict FIFO at a price level. Kalshi does not document priority; under
     pro-rata this measure is wrong.
  2. Cancellations ahead of us are invisible - they remove queue without
     trading - so this UNDERSTATES the queue we joined. Lower bound.
  3. Our own fills are excluded.

OUTPUT  data/queue_ahead.parquet, one row per LSM order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"

O = pd.read_parquet(DATA / "orders_history.parquet")
F = pd.read_parquet(DATA / "fills_history.parquet")
T = pd.read_parquet(DATA / "trades_lsm.parquet")
U = pd.read_parquet(DATA / "underlying.parquet", columns=["ticker", "result"])
U = U[U.result.isin(["yes", "no"])].drop_duplicates("ticker")
mx = DATA / "lsm_missing_outcomes.parquet"
if mx.exists():
    ex = pd.read_parquet(mx)
    U = pd.concat([U, ex[~ex.ticker.isin(U.ticker)][["ticker", "result"]]],
                  ignore_index=True)

l = O[O.client_order_id.astype(str).str.startswith("lsm")].copy()
l["held"] = np.where(l.action.eq("sell"),
                     np.where(l.side.eq("yes"), "no", "yes"), l.side)
l["filled"] = pd.to_numeric(l.fill_count_fp, errors="coerce").fillna(0)
l["submitted"] = pd.to_numeric(l.initial_count_fp, errors="coerce").fillna(0)
l["did_fill"] = l.filled > 0
l["t0"] = pd.to_datetime(l.created_time, format="mixed", utc=True)
l["t_end"] = pd.to_datetime(l.last_update_time, format="mixed", utc=True)
yp = pd.to_numeric(l.yes_price_dollars, errors="coerce")
npx = pd.to_numeric(l.no_price_dollars, errors="coerce")
# YES-scale price of our resting order
l["yes_px"] = np.where(l.held.eq("yes"), yp, 1.0 - npx)
l = l.merge(U, on="ticker", how="left")
l["would_win"] = l.held == l.result

ff = (F.assign(ft=pd.to_datetime(F.created_time, format="mixed", utc=True))
      .groupby("order_id").ft.min().rename("first_fill"))
l = l.merge(ff, left_on="order_id", right_index=True, how="left")

T["t"] = pd.to_datetime(T.created_time, format="mixed", utc=True)
by_ticker = {k: v.sort_values("t") for k, v in T.groupby("ticker")}

rows = []
for r in l.itertuples():
    tp = by_ticker.get(r.ticker)
    if tp is None or not len(tp):
        continue
    stop = r.first_fill if pd.notna(r.first_fill) else r.t_end
    if pd.isna(stop):
        continue
    # trades that would consume a resting BUY of `held` at our price
    m = (tp.t >= r.t0) & (tp.t < stop)
    m &= tp.taker_outcome_side.eq(r.held)
    m &= tp.taker_book_side.eq("bid")
    m &= (tp.yes_px - r.yes_px).abs() < 1e-9
    w = tp[m]
    rows.append(dict(
        order_id=r.order_id, ticker=r.ticker, held=r.held,
        yes_px=r.yes_px, submitted=r.submitted, filled=r.filled,
        did_fill=bool(r.did_fill), would_win=bool(r.would_win),
        rest_seconds=(stop - r.t0).total_seconds(),
        queue_ahead=float(w["count"].sum()),
        n_trades_ahead=int(len(w)),
        tape_trades_in_market=int(len(tp)),
    ))

Q = pd.DataFrame(rows)
Q.to_parquet(DATA / "queue_ahead.parquet")

print("=" * 84)
print("QUEUE-AHEAD RECONSTRUCTION (proxy - FIFO assumed, cancels invisible)")
print("=" * 84)
print(f"  orders reconstructed: {len(Q)} of {len(l)}")
print(f"  tape: {len(T):,} trades over {T.ticker.nunique()} markets")
print(f"\n  queue_ahead distribution (contracts ahead of us at our price):")
print(Q.queue_ahead.describe().to_string())
print(f"\n  median rest time: {Q.rest_seconds.median():.1f}s")

print("\n" + "-" * 84)
print("THE AUDIT D QUESTION: does queue position predict adverse fills?")
print("-" * 84)
f = Q[Q.did_fill]
print("  %-22s %7s %10s %12s" % ("queue-ahead bucket", "n", "P(win)",
                                 "median q"))
if len(f) > 20:
    f = f.copy()
    try:
        f["qb"] = pd.qcut(f.queue_ahead.rank(method="first"), 4,
                          labels=["Q1 front", "Q2", "Q3", "Q4 back"])
        for b, g in f.groupby("qb", observed=True):
            print("  %-22s %7d %10.4f %12.0f"
                  % (b, len(g), g.would_win.mean(), g.queue_ahead.median()))
    except ValueError:
        print("  (too few distinct values to bucket)")
print("\n  never-filled orders: n=%d, would-win rate %.4f"
      % ((~Q.did_fill).sum(), Q[~Q.did_fill].would_win.mean()))
print("  filled orders      : n=%d, would-win rate %.4f"
      % (Q.did_fill.sum(), Q[Q.did_fill].would_win.mean()))
print("\n  NOTE: median rest is short, so most fills have near-zero measured")
print("  queue ahead. The informative cases are the never-filled orders,")
print("  where queue_ahead is a lower bound on what blocked us.")
