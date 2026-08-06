"""Fetch an independent FR2 validation sample from :15/:45 close slots.

Run locally from the repository root:

    export KALSHI_CRED_DIR=/path/to/credentials
    python research/fetch_fr2_odd_paths.py --limit 5000
    python research/fetch_fr2_odd_paths.py --all

The script is resumable and never places orders. It stores real one-minute
candlesticks, including decision-time volume, for a frozen out-of-sample FR2
test. Credentials remain outside the repository.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from capture.kalshi import KalshiClient

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = DATA / "fr2_odd_paths.parquet"
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int, default=20260816)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = KalshiClient()

    underlying = pd.read_parquet(DATA / "underlying.parquet")
    expected = pd.to_datetime(underlying["close"], format="mixed", utc=True)
    underlying["close_utc"] = (
        pd.to_datetime(underlying["wkey"], format="%y%b%d%H%M", utc=True)
        + pd.Timedelta(hours=4)
    )
    clean = (
        ((expected - underlying["close_utc"]).dt.total_seconds() / 60)
        .round(1)
        .eq(5.0)
    )
    pool = underlying[
        clean & underlying["close_utc"].dt.minute.isin([15, 45])
    ].copy()

    rows: list[dict] = []
    done: set[str] = set()
    if CACHE.exists():
        old = pd.read_parquet(CACHE)
        rows = old.to_dict("records")
        done = set(old["ticker"])

    todo = pool[~pool["ticker"].isin(done)].copy()
    if not args.all and len(todo) > args.limit:
        todo = todo.sample(args.limit, random_state=args.seed)

    started = time.time()
    errors = 0
    for i, market in enumerate(todo.itertuples(index=False), start=1):
        close_ts = int((market.close_utc - EPOCH).total_seconds())
        result = api.get(
            f"/series/{market.series}/markets/{market.ticker}/candlesticks",
            {
                "period_interval": 1,
                "start_ts": close_ts - 16 * 60,
                "end_ts": close_ts,
            },
        )
        if not result.ok:
            errors += 1
            continue

        path: list[dict] = []
        for candle in (result.body or {}).get("candlesticks", []):
            try:
                path.append(
                    {
                        "ml": (
                            close_ts - int(candle["end_period_ts"])
                        ) / 60.0,
                        "bc": float(candle["yes_bid"]["close_dollars"]),
                        "bh": float(candle["yes_bid"]["high_dollars"]),
                        "ac": float(candle["yes_ask"]["close_dollars"]),
                        "al": float(candle["yes_ask"]["low_dollars"]),
                        "volume_fp": float(candle.get("volume_fp") or 0),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue

        if len(path) >= 10:
            rows.append(
                {
                    "ticker": market.ticker,
                    "series": market.series,
                    "coin": market.coin,
                    "close_utc": market.close_utc,
                    "result": market.result,
                    "path": json.dumps(path),
                }
            )

        if i % 200 == 0:
            pd.DataFrame(rows).drop_duplicates("ticker").to_parquet(
                CACHE, index=False
            )
            elapsed = time.time() - started
            print(
                f"{i}/{len(todo)} kept={len(rows)} errors={errors} "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )

    output = pd.DataFrame(rows).drop_duplicates("ticker")
    output.to_parquet(CACHE, index=False)
    print(
        f"stored {len(output):,} markets across "
        f"{pd.to_datetime(output.close_utc, utc=True).dt.date.nunique()} days"
    )
    print(f"cache: {CACHE}")


if __name__ == "__main__":
    main()
