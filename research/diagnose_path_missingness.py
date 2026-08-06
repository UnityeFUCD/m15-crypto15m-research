"""Diagnose why 83 of 303 LSM orders lack paths_full coverage.

The timestamp-corrected PTC audit found that matched orders made +$153.78 while
unmatched orders lost -$144.40.  Any path-based execution result is unusable
until this missingness is explained.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "research" / "results"
OUT.mkdir(parents=True, exist_ok=True)

O = pd.read_parquet(DATA / "orders_history.parquet")
F = pd.read_parquet(DATA / "fills_history.parquet")
U = pd.read_parquet(DATA / "underlying.parquet")
P = pd.read_parquet(DATA / "paths_full.parquet", columns=["ticker", "n_pts"])
B = pd.read_parquet(DATA / "book_full.parquet", columns=["ticker"])

l = O[O.client_order_id.astype(str).str.startswith("lsm")].copy()
l["held"] = np.where(l.action.eq("sell"), np.where(l.side.eq("yes"), "no", "yes"), l.side)
l["filled_qty"] = pd.to_numeric(l.fill_count_fp, errors="coerce").fillna(0)
l["submitted_qty"] = pd.to_numeric(l.initial_count_fp, errors="coerce").fillna(0)
l["created_dt"] = pd.to_datetime(l.created_time, format="mixed", utc=True)
l["close_dt"] = (
    pd.to_datetime(l.ticker.str.split("-").str[1], format="%y%b%d%H%M", utc=True)
    + pd.Timedelta(hours=4)
)
l["day"] = l.close_dt.dt.date.astype(str)
l["hour"] = l.close_dt.dt.hour
l["minute"] = l.close_dt.dt.minute
l["coin"] = l.ticker.str.extract(r"^KX([A-Z]+)15M")[0]
yp = pd.to_numeric(l.yes_price_dollars, errors="coerce")
np_ = pd.to_numeric(l.no_price_dollars, errors="coerce")
l["price"] = np.where(l.held.eq("yes"), yp, np_)

u = U.drop_duplicates("ticker").copy()
u["underlying_result_present"] = u.result.isin(["yes", "no"])
u["underlying_a0_present"] = u.a0.notna()
u["expected_close"] = pd.to_datetime(u.close, format="mixed", utc=True, errors="coerce")
u["ticker_close"] = (
    pd.to_datetime(u.wkey, format="%y%b%d%H%M", utc=True, errors="coerce")
    + pd.Timedelta(hours=4)
)
u["expected_minus_true_min"] = (u.expected_close - u.ticker_close).dt.total_seconds() / 60

missing_outcomes = DATA / "lsm_missing_outcomes.parquet"
if missing_outcomes.exists():
    mo = pd.read_parquet(missing_outcomes)[["ticker", "result"]].drop_duplicates("ticker")
    mo = mo.rename(columns={"result": "recovered_result"})
else:
    mo = pd.DataFrame(columns=["ticker", "recovered_result"])

l = l.merge(
    u[["ticker", "result", "underlying_result_present", "underlying_a0_present", "expected_minus_true_min"]],
    on="ticker",
    how="left",
)
l = l.merge(mo, on="ticker", how="left")
l["final_result"] = l.result.where(l.result.isin(["yes", "no"]), l.recovered_result)
l["won"] = l.held.eq(l.final_result)
l["pnl"] = l.filled_qty * (l.won.astype(float) - l.price)
l["in_paths"] = l.ticker.isin(set(P.ticker))
l["in_book"] = l.ticker.isin(set(B.ticker))
l = l.merge(P.drop_duplicates("ticker"), on="ticker", how="left")

# Was the market available to the fetch script's source pool?
l["path_fetch_pool_eligible"] = (
    l.underlying_result_present.fillna(False)
    & l.expected_minus_true_min.round(1).eq(5.0)
)

summary = {
    "orders": len(l),
    "matched_paths": int(l.in_paths.sum()),
    "unmatched_paths": int((~l.in_paths).sum()),
    "matched_pnl": float(l.loc[l.in_paths, "pnl"].sum()),
    "unmatched_pnl": float(l.loc[~l.in_paths, "pnl"].sum()),
    "unmatched_in_book": int(l.loc[~l.in_paths, "in_book"].sum()),
    "unmatched_pool_eligible": int(l.loc[~l.in_paths, "path_fetch_pool_eligible"].sum()),
    "unmatched_missing_underlying_result": int(
        (~l.in_paths & ~l.underlying_result_present.fillna(False)).sum()
    ),
    "unmatched_recovered_result": int((~l.in_paths & l.recovered_result.notna()).sum()),
}

by_reason = (
    l[~l.in_paths]
    .assign(
        reason=np.select(
            [
                ~l.loc[~l.in_paths, "underlying_result_present"].fillna(False),
                ~l.loc[~l.in_paths, "expected_minus_true_min"].round(1).eq(5.0),
                l.loc[~l.in_paths, "in_book"],
            ],
            [
                "underlying_result_missing_at_fetch",
                "expected_close_filter_failed",
                "book_exists_but_path_missing",
            ],
            default="eligible_but_missing_path",
        )
    )
    .groupby("reason")
    .agg(
        orders=("order_id", "size"),
        unique_markets=("ticker", "nunique"),
        pnl=("pnl", "sum"),
        win_rate=("won", "mean"),
        fill_rate=("filled_qty", lambda s: (s > 0).mean()),
    )
    .reset_index()
)

by_day = (
    l.groupby(["day", "in_paths"])
    .agg(orders=("order_id", "size"), pnl=("pnl", "sum"), win_rate=("won", "mean"))
    .reset_index()
)
by_coin = (
    l.groupby(["coin", "in_paths"])
    .agg(orders=("order_id", "size"), pnl=("pnl", "sum"), win_rate=("won", "mean"))
    .reset_index()
)

cols = [
    "order_id", "ticker", "coin", "day", "hour", "minute", "held", "final_result",
    "won", "filled_qty", "submitted_qty", "price", "pnl", "in_book", "in_paths",
    "underlying_result_present", "recovered_result", "expected_minus_true_min",
    "path_fetch_pool_eligible", "n_pts",
]
l[cols].to_csv(OUT / "path_missingness_orders.csv", index=False)
by_reason.to_csv(OUT / "path_missingness_reasons.csv", index=False)
by_day.to_csv(OUT / "path_missingness_by_day.csv", index=False)
by_coin.to_csv(OUT / "path_missingness_by_coin.csv", index=False)
(OUT / "path_missingness_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

report = f"""# Full-path missingness diagnosis

The PTC direct audit matched {summary['matched_paths']} of {summary['orders']}
LSM orders.  Matched orders made ${summary['matched_pnl']:.2f}; unmatched orders
lost ${summary['unmatched_pnl']:.2f}.  This missingness must be explained before
any path-based result is generalized.

## Reason waterfall

{by_reason.to_markdown(index=False, floatfmt='.4f')}

## By day

{by_day.to_markdown(index=False, floatfmt='.4f')}

## By coin

{by_coin.to_markdown(index=False, floatfmt='.4f')}

## Interpretation

- ``underlying_result_missing_at_fetch`` means ``fetch_paths_full.py`` excluded
  the market before calling the API because it required a settled result from
  the earlier ``underlying.parquet`` snapshot.
- ``book_exists_but_path_missing`` means the market has a valid entry quote in
  ``book_full`` but the full-path fetch did not retain a usable path.
- ``eligible_but_missing_path`` means the fetch source pool included the market
  but no retained path row exists; this can reflect API failure or insufficient
  valid candles.

A cohort excluded because the historical snapshot lacked its result is a pure
survivorship bug.  It must be refetched from the API or no direct PTC conclusion
can be made about it.
"""
(OUT / "path_missingness_report.md").write_text(report, encoding="utf-8")
print(json.dumps(summary, indent=2))
print(by_reason.to_string(index=False))
