"""Fetch settlement for LSM markets missing from underlying.parquet.

This bug has now bitten twice. underlying.parquet was snapshotted before some
LSM markets closed, so a naive merge silently DROPS them - and they are not
missing at random. The first time, 17 markets absent from the snapshot were
worth -$134 and moved the measured result from +$143.38 to +$9.38.

prospective_execution.py reintroduced the same error by dropping rows with no
settled outcome. This script closes the hole permanently by resolving those
tickers from the API and caching them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture.kalshi import KalshiClient          # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
CACHE = DATA / "lsm_missing_outcomes.parquet"

O = pd.read_parquet(DATA / "orders_history.parquet")
U = pd.read_parquet(DATA / "underlying.parquet", columns=["ticker", "result"])
U = U[U.result.isin(["yes", "no"])]

lsm = O[O.client_order_id.astype(str).str.startswith("lsm")]
missing = sorted(set(lsm.ticker) - set(U.ticker))
print(f"LSM tickers: {lsm.ticker.nunique()}   missing an outcome: {len(missing)}")

known = {}
if CACHE.exists():
    old = pd.read_parquet(CACHE)
    known = dict(zip(old.ticker, old.result))
    print(f"cached: {len(known)}")

api = KalshiClient()
rows = [{"ticker": t, "result": r} for t, r in known.items()]
for t in missing:
    if t in known:
        continue
    try:
        res = api.get(f"/markets/{t}")
        m = (res.body or {}).get("market") or {}
        r = (m.get("result") or "").lower()
        st = m.get("status")
        print(f"  {t:32s} status={st:12s} result={r or '(none)'}")
        if r in ("yes", "no"):
            rows.append({"ticker": t, "result": r})
    except Exception as e:
        print(f"  {t:32s} ERROR {e}")

D = pd.DataFrame(rows).drop_duplicates("ticker")
if len(D):
    D.to_parquet(CACHE)
print(f"\nresolved {len(D)} of {len(missing)} missing outcomes -> {CACHE.name}")
