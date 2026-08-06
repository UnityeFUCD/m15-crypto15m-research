"""Fetch full paths for LSM markets missing from paths_full.parquet.

This deliberately does NOT use the old underlying snapshot as the source pool.
That snapshot omitted late-closing outcomes and created a survivorship bug: the
220 path-matched LSM orders made +$153.78 while the 83 unmatched orders lost
-$144.40.  The source of truth here is the actual LSM order ticker.

Run locally with credentials kept outside the repository:

    export KALSHI_CRED_DIR=/path/to/credentials
    python research/fetch_missing_lsm_paths.py

The script is resumable, appends only successfully reconstructed rows, preserves
the paths_full schema, and never places an order.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from capture.kalshi import KalshiClient

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PATHS = DATA / "paths_full.parquet"


def outcomes() -> pd.DataFrame:
    u = pd.read_parquet(DATA / "underlying.parquet", columns=["ticker", "result"])
    u = u[u.result.isin(["yes", "no"])].drop_duplicates("ticker")
    missing = DATA / "lsm_missing_outcomes.parquet"
    if missing.exists():
        x = pd.read_parquet(missing)
        x = x[x.result.isin(["yes", "no"]) & ~x.ticker.isin(u.ticker)]
        u = pd.concat([u, x[["ticker", "result"]]], ignore_index=True)
    return u.drop_duplicates("ticker")


def true_close(ticker: str) -> pd.Timestamp:
    return (
        pd.to_datetime(ticker.split("-")[1], format="%y%b%d%H%M", utc=True)
        + pd.Timedelta(hours=4)
    )


def fetch_path(api: KalshiClient, ticker: str, result: str) -> dict | None:
    series = ticker.split("-")[0]
    coin = ticker.removeprefix("KX").split("15M", 1)[0]
    close = true_close(ticker)
    close_ts = int(close.timestamp())

    body = None
    for attempt in range(4):
        try:
            response = api.get(
                f"/series/{series}/markets/{ticker}/candlesticks",
                {
                    "period_interval": 1,
                    "start_ts": close_ts - 16 * 60,
                    "end_ts": close_ts,
                },
            )
            if response.ok:
                body = response.body
                break
        except Exception:
            pass
        time.sleep(0.8 * (attempt + 1))
    if body is None:
        return None

    path: list[dict] = []
    for candle in (body or {}).get("candlesticks", []):
        try:
            ml = (close_ts - int(candle["end_period_ts"])) / 60.0
            yb = float(candle["yes_bid"]["close_dollars"])
            ya = float(candle["yes_ask"]["close_dollars"])
            vol = float(candle.get("volume_fp") or 0)
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= ml <= 15 and 0 < yb < ya < 1:
            path.append(
                {
                    "ml": ml,
                    "yb": round(yb, 4),
                    "ya": round(ya, 4),
                    "v": vol,
                }
            )
    path.sort(key=lambda p: -p["ml"])
    entry = next((p for p in path if 8 <= p["ml"] <= 14), None)
    if entry is None or len(path) < 2:
        return None

    favorite_yes = entry["yb"] >= 0.5
    side = "yes" if favorite_yes else "no"
    bid = entry["yb"] if favorite_yes else 1.0 - entry["ya"]
    ask = entry["ya"] if favorite_yes else 1.0 - entry["yb"]
    return {
        "ticker": ticker,
        "coin": coin,
        "close_ts": close_ts,
        "minute": int(close.minute),
        "side": side,
        "bid": bid,
        "ask": ask,
        "entry_ml": entry["ml"],
        "n_pts": len(path),
        "won": int(side == result),
        "path": json.dumps(path),
    }


def main() -> None:
    orders = pd.read_parquet(DATA / "orders_history.parquet")
    lsm = orders[orders.client_order_id.astype(str).str.startswith("lsm")]
    tickers = sorted(set(lsm.ticker.astype(str)))
    result_map = outcomes().set_index("ticker").result.to_dict()

    existing = pd.read_parquet(PATHS) if PATHS.exists() else pd.DataFrame()
    have = set(existing.ticker.astype(str)) if len(existing) else set()
    todo = [t for t in tickers if t not in have]
    print(f"LSM markets {len(tickers)}; existing paths {len(have & set(tickers))}; to fetch {len(todo)}")

    api = KalshiClient()
    rows: list[dict] = []
    failures: list[str] = []
    for i, ticker in enumerate(todo, start=1):
        result = result_map.get(ticker)
        if result not in ("yes", "no"):
            failures.append(ticker)
            print(f"[{i}/{len(todo)}] outcome missing: {ticker}")
            continue
        row = fetch_path(api, ticker, result)
        if row is None:
            failures.append(ticker)
            print(f"[{i}/{len(todo)}] path failed: {ticker}")
        else:
            rows.append(row)
            print(f"[{i}/{len(todo)}] kept {ticker} ({row['n_pts']} points)")

        if i % 20 == 0 and rows:
            merged = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
            merged = merged.drop_duplicates("ticker", keep="last")
            merged.to_parquet(PATHS, index=False)

    if rows:
        merged = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
        merged = merged.drop_duplicates("ticker", keep="last")
        merged.to_parquet(PATHS, index=False)
    else:
        merged = existing

    remaining = sorted(set(tickers) - set(merged.ticker.astype(str)))
    pd.DataFrame({"ticker": remaining}).to_csv(
        DATA / "missing_lsm_paths_after_fetch.csv", index=False
    )
    print(f"\nadded {len(rows)}; remaining {len(remaining)}")
    if failures:
        print("failures:")
        print("\n".join(failures))


if __name__ == "__main__":
    main()
